"""AST end-to-end: Amazon 2014 data -> AST semantic IDs -> T5 generative recommender -> Recall/NDCG.

    python train.py experiment=beauty
    python train.py data.category=Toys_and_Games tokenization.m=4 tokenization.min_share_prob=0.01

Optional stages (the full run needs none of them):
    stop_after=embed                 only encode + cache item embeddings
    stop_after=sids                  stop after writing item2tokseq_ids.pkl (CPU is enough when embeddings are cached)
    tokenization.sids_path=<pkl>     train on previously built semantic IDs
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Seq2SeqTrainingArguments, TrainerCallback

from data import build_amazon2014_seq_splits, build_asin2text, load_amazon2014_meta, print_split_stats
from evaluation import build_code_metrics
from model import (
    TrieConstrainedSeq2SeqTrainer,
    build_item_code_trie,
    build_t5_seq2seq,
    make_prefix_allowed_tokens_fn,
)
from silhouette_tree import build_item2tokseq_ids_silhouette_tree, encode_items
from utils import (
    DEFAULT_IGN_ID,
    EvalSaveEveryNEpochsCallback,
    build_contiguous_wordlevel_tokenizer,
    build_id2text_from_item2token,
    compute_item_probs_from_dd_train,
    compute_leaf_popularity_stats,
    make_seq2seq_tokenize_fn,
    normalize_item2token,
    print_model_parameters,
    set_global_seed,
    tokenize_splits,
)


class _StopAtStep(TrainerCallback):
    """Stop after `step` optimizer steps without changing the learning-rate schedule."""

    def __init__(self, step: int):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.step:
            control.should_training_stop = True
        return control


def load_or_encode_items(cfg: DictConfig, id2text, data_root: Path) -> np.ndarray:
    """Unit-norm item embeddings, cached under <data.root>/embeddings (keyed by encoder + item texts)."""
    t = cfg.tokenization
    key = hashlib.sha1(f"{t.encoder_name}|{t.max_seq_length}|".encode() + "\n".join(id2text).encode()).hexdigest()[:16]
    path = data_root / "embeddings" / f"{cfg.data.category}_{str(t.encoder_name).replace('/', '_')}_{key}.npy"
    if t.cache_embeddings and path.exists():
        print(f"[embeddings] loaded {path}")
        return np.load(path)
    embs = encode_items(
        id2text, encoder_name=str(t.encoder_name), batch_size=int(t.batch_size), max_seq_length=int(t.max_seq_length)
    )
    if t.cache_embeddings:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}.npy")
        np.save(tmp, embs)
        os.replace(tmp, path)
        print(f"[embeddings] saved {path}")
    return embs


def build_semantic_ids(cfg: DictConfig, id2text, tokenizer, item_probs, item_embs):
    t = cfg.tokenization
    return build_item2tokseq_ids_silhouette_tree(
        id2text=id2text,
        tokenizer=tokenizer,
        base=int(t.base),
        m=int(t.m),
        seed=int(cfg.seed),
        encoder_name=str(t.encoder_name),
        batch_size=int(t.batch_size),
        max_seq_length=int(t.max_seq_length),
        item_embs=item_embs,
        item_embs_normalized=True,
        item_probs=item_probs if bool(t.use_item_mass) else None,
        # KMeans / silhouette model selection
        max_iter_kmeans=int(t.max_iter_kmeans),
        n_init_kmeans=int(t.n_init_kmeans),
        silhouette_metric=str(t.silhouette_metric),
        patience=int(t.patience),
        silhouette_sample_size=None if t.silhouette_sample_size is None else int(t.silhouette_sample_size),
        k_search_step=int(t.k_search_step),
        weighted_silhouette=bool(t.weighted_silhouette),
        # tree
        fixed_k=(
            None
            if getattr(t, "fixed_k", None) is None
            else list(t.fixed_k) if isinstance(t.fixed_k, (list, tuple, ListConfig))
            else (str(t.fixed_k) if any(sep in str(t.fixed_k) for sep in (",", "-")) else int(t.fixed_k))
        ),
        density_driven_k=bool(getattr(t, "density_driven_k", False)),
        target_items_per_leaf=float(getattr(t, "target_items_per_leaf", 3.0)),
        k_window_delta=int(getattr(t, "k_window_delta", 2)),
        k_cap=int(getattr(t, "k_cap", 64)),
        min_items_to_split=int(getattr(t, "min_items_to_split", 3)),
        adaptive_depth=bool(getattr(t, "adaptive_depth", False)),
        max_depth=int(getattr(t, "max_depth", int(t.m) - 1 if bool(getattr(t, "adaptive_depth", False)) else 6)),
        max_items_per_leaf=int(getattr(t, "max_items_per_leaf", 4)),
        capacity_aware_min_k=bool(t.capacity_aware_min_k),
        stop_recursion_if_no_split=bool(t.stop_recursion_if_no_split),
        # minimum-mass constraint
        min_share_prob=None if t.min_share_prob is None else float(t.min_share_prob),
        enforce_min_share_hard=bool(t.enforce_min_share_hard),
        # MILP
        MILP=bool(t.MILP),
        milp_only_if_needed=bool(t.milp_only_if_needed),
        milp_candidate_topk=None if t.milp_candidate_topk is None else int(t.milp_candidate_topk),
        milp_time_limit=None if t.milp_time_limit is None else float(t.milp_time_limit),
        milp_gap=float(t.milp_gap),
        milp_solver_name=str(t.milp_solver_name),
        milp_verbose=bool(t.milp_verbose),
        milp_use_weighted_centers=bool(t.milp_use_weighted_centers),
        milp_refine_rounds=int(t.milp_refine_rounds),
        # logging
        print_each_clustering=bool(t.print_each_clustering),
        print_stats=bool(t.print_stats),
        return_tree_stats=True,
    )


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_global_seed(int(cfg.seed))

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    data_root = Path(get_original_cwd()) / str(cfg.data.root)
    category = str(cfg.data.category)

    # ---- data: leave-last-out splits over item ids 3..N+2 (0/1/2 = pad/eos/unk) ----
    dd, item2token, _ = build_amazon2014_seq_splits(
        category, root=str(data_root), download_if_missing=bool(cfg.data.download_if_missing)
    )
    dd = dd.map(lambda ex: {"history": ex["history"][-int(cfg.data.max_history):]})
    print_split_stats(dd)
    item2token = normalize_item2token(item2token, start=int(cfg.tokenization.start_token))

    asin_set = set(item2token.keys())
    meta = load_amazon2014_meta(
        category,
        root=str(data_root),
        only_asins=asin_set,
        download_if_missing=bool(cfg.data.download_if_missing),
        keep_columns=list(cfg.data.keep_columns),
    )
    dd = tokenize_splits(dd, item2token)

    # ---- code-token vocabulary: <c{level}_{k}> for level < m, k < base ----
    BASE = int(cfg.tokenization.base)
    M = int(cfg.tokenization.m)
    is_ragged = bool(getattr(cfg.tokenization, "adaptive_depth", False))
    if is_ragged:
        user_max_depth = getattr(cfg.tokenization, "max_depth", None)
        if user_max_depth is not None:
            max_depth = int(user_max_depth)
        else:
            max_depth = M - 1
        TOTAL_LEVELS = max_depth + 1
        eff_M = None
        max_new_tokens = TOTAL_LEVELS + 1
    else:
        max_depth = M - 1
        TOTAL_LEVELS = M
        eff_M = M
        max_new_tokens = M

    tokenizer = build_contiguous_wordlevel_tokenizer(
        [f"<c{lvl}_{i}>" for lvl in range(TOTAL_LEVELS) for i in range(BASE)], padding_side="left"
    )
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id

    # ---- AST semantic IDs ----
    asin2text = build_asin2text(meta, asin_set=asin_set, cols=list(cfg.data.cols_to_use))
    id2text = build_id2text_from_item2token(item2token, asin2text)
    sids_path = cfg.tokenization.sids_path
    if sids_path:  # reuse semantic IDs built by an earlier run (e.g. stop_after=sids)
        with open(Path(get_original_cwd()) / str(sids_path), "rb") as f:
            item2tokseq_ids = pickle.load(f)
        print(f"[AST] loaded semantic IDs from {sids_path}")
        # Try to load item_probs saved alongside the original sids
        _probs_path = (Path(get_original_cwd()) / str(sids_path)).parent / "item_probs.pkl"
        if _probs_path.exists():
            with open(_probs_path, "rb") as pf:
                item_probs = pickle.load(pf)
            print(f"[AST] loaded item_probs from {_probs_path}")
        else:
            item_probs, _, _ = compute_item_probs_from_dd_train(dd)
            if len(item_probs) < len(id2text):
                item_probs = np.pad(item_probs, (0, len(id2text) - len(item_probs)))
    else:
        item_embs = load_or_encode_items(cfg, id2text, data_root)
        if cfg.stop_after == "embed":
            return
        item_probs, _, _ = compute_item_probs_from_dd_train(dd)  # interaction frequency in train
        if len(item_probs) < len(id2text):  # items never seen in train get zero mass
            item_probs = np.pad(item_probs, (0, len(id2text) - len(item_probs)))
        t0 = time.time()
        res = build_semantic_ids(cfg, id2text, tokenizer, item_probs, item_embs)
        if isinstance(res, tuple):
            item2tokseq_ids, tree_stats = res
        else:
            item2tokseq_ids, tree_stats = res, {}
        sid_seconds = time.time() - t0
        with open(out_dir / "item2tokseq_ids.pkl", "wb") as f:
            pickle.dump(item2tokseq_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
        codes = list(item2tokseq_ids.values())
        leaf_prefixes = [tuple(c[:-1]) for c in codes]
        leaf_counts = list(Counter(leaf_prefixes).values())
        stats = {
            "n_items": len(codes),
            "n_unique_sids": len({tuple(c) for c in codes}),
            **({f"prefixes_len{l}": len({tuple(c[:l]) for c in codes}) for l in range(1, TOTAL_LEVELS)} if not is_ragged else {
                "sid_len_min": int(np.min([len(c) for c in codes])),
                "sid_len_median": float(np.median([len(c) for c in codes])),
                "sid_len_max": int(np.max([len(c) for c in codes])),
            }),
            "build_seconds": round(sid_seconds, 1),
            "n_leaf_clusters": len(leaf_counts),
            "items_per_leaf_mean": round(float(np.mean(leaf_counts)), 2),
            "items_per_leaf_std": round(float(np.std(leaf_counts)), 2),
            "items_per_leaf_min": int(np.min(leaf_counts)),
            "items_per_leaf_median": float(np.median(leaf_counts)),
            "items_per_leaf_max": int(np.max(leaf_counts)),
            "items_per_leaf_p25": float(np.percentile(leaf_counts, 25)),
            "items_per_leaf_p75": float(np.percentile(leaf_counts, 75)),
            "leaf_sizes": leaf_counts,
        }
        stats.update(tree_stats)
        if item_probs is not None:
            pop_stats = compute_leaf_popularity_stats(item2tokseq_ids, item_probs)
            stats["popularity_composition"] = pop_stats
            with open(out_dir / "item_probs.pkl", "wb") as pf:
                pickle.dump(item_probs, pf, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"[AST PopStats] Mixed Head-Tail Leaves: {pop_stats['leaf_cluster_breakdown']['pct_leaves_mixed_head_tail']}% | "
                f"Tail with Head: {pop_stats['tail_item_experience']['pct_tail_items_sharing_with_head']}% | "
                f"Tail Isolated: {pop_stats['tail_item_experience']['pct_tail_items_isolated_in_pure_tail']}%"
            )

        summary = {k: v for k, v in stats.items() if k not in ("leaf_sizes", "split_records", "popularity_composition")}
        print(f"[AST] {summary}")
        with open(out_dir / "sids_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        if cfg.stop_after == "sids":
            return

    # ---- seq2seq examples ----
    # Compute item_tiers (head/body/tail by interaction frequency) for grouped eval metrics
    item_tiers: Optional[dict] = None
    if item_probs is not None:
        _items = sorted(item2tokseq_ids.keys())
        _N = len(_items)
        if _N > 0:
            _n_head = max(1, int(round(_N * 0.20)))
            _n_tail = max(1, int(round(_N * 0.20)))
            _masses = sorted(
                [(it, float(item_probs[it]) if it < len(item_probs) else 0.0) for it in _items],
                key=lambda x: (x[1], -x[0]),
                reverse=True,
            )
            _head_set = {x[0] for x in _masses[:_n_head]}
            _tail_set  = {x[0] for x in _masses[-_n_tail:]} - _head_set
            item_tiers = {}
            for it in _items:
                if it in _head_set:
                    item_tiers[it] = "head"
                elif it in _tail_set:
                    item_tiers[it] = "tail"
                else:
                    item_tiers[it] = "body"
            print(f"[AST] item_tiers: head={len(_head_set)} body={_N - len(_head_set) - len(_tail_set)} tail={len(_tail_set)}")

    tokenize_fn = make_seq2seq_tokenize_fn(item2tokseq_ids=item2tokseq_ids, M=eff_M, eos_id=eos_id if is_ragged else None)
    tokenized = {
        split: dd[split].map(tokenize_fn, remove_columns=list(cfg.data.remove_cols))
        for split in ["train", "validation", "test"]
    }

    # ---- model ----
    mc = cfg.model
    model = build_t5_seq2seq(
        vocab_size=len(tokenizer),
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        encoder_layers=int(mc.encoder_layers),
        decoder_layers=int(mc.decoder_layers),
        d_model=int(mc.d_model),
        d_ff=int(mc.d_ff),
        num_heads=int(mc.num_heads),
        dropout_rate=float(mc.dropout_rate),
        initializer_factor=float(mc.initializer_factor),
        layer_norm_epsilon=float(mc.layer_norm_epsilon),
        feed_forward_proj=str(mc.feed_forward_proj),
    )
    model.resize_token_embeddings(len(tokenizer))
    print_model_parameters(model)

    gen = model.generation_config
    gen.num_beams = int(cfg.generation.num_beams)
    gen.num_return_sequences = int(cfg.generation.num_return_sequences)
    gen.do_sample = bool(cfg.generation.do_sample)
    gen.max_new_tokens = max_new_tokens
    gen.min_new_tokens = 2 if is_ragged else eff_M

    prefix_allowed_tokens = make_prefix_allowed_tokens_fn(
        build_item_code_trie(item2tokseq_ids, eos_id=eos_id if is_ragged else None),
        pad_id=pad_id,
        eos_id=eos_id,
        decoder_start_id=model.config.decoder_start_token_id,
        M=eff_M,
    )

    # ---- training ----
    train_ds = tokenized["train"].shuffle(seed=int(cfg.seed))
    if cfg.debug.train_select is not None:
        train_ds = train_ds.select(list(cfg.debug.train_select))
    if cfg.debug.eval_limit is not None:  # quick checks only: evaluate on the first n users
        for split in ("validation", "test"):
            tokenized[split] = tokenized[split].select(range(min(int(cfg.debug.eval_limit), len(tokenized[split]))))

    callbacks = [
        EvalSaveEveryNEpochsCallback(
            eval_n=cfg.training.schedule.eval_every_n_epochs,
            save_n=cfg.training.schedule.save_every_n_epochs,
            eval_on_last=cfg.training.schedule.always_eval_on_last,
            save_on_last=cfg.training.schedule.always_save_on_last,
        )
    ]
    if cfg.training.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(cfg.training.early_stopping.patience),
                early_stopping_threshold=float(cfg.training.early_stopping.threshold),
            )
        )

    trainer = TrieConstrainedSeq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(**OmegaConf.to_container(cfg.training.hf, resolve=True)),
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=DEFAULT_IGN_ID, return_tensors="pt"
        ),
        train_dataset=train_ds,
        eval_dataset=tokenized["validation"],
        compute_metrics=build_code_metrics(
            item2tokseq_ids=item2tokseq_ids,
            pad_id=pad_id,
            M=eff_M,
            eos_id=eos_id if is_ragged else None,
            k_values=(5, 10),
            item_tiers=item_tiers,
        ),
        callbacks=callbacks,
        prefix_allowed_tokens_fn=prefix_allowed_tokens,
        gen_M=eff_M,
        max_new_tokens=max_new_tokens,
    )
    if cfg.debug.stop_at_step is not None:
        trainer.add_callback(_StopAtStep(int(cfg.debug.stop_at_step)))
    trainer.train()

    if cfg.debug.stop_at_step is not None:  # quick check: report the training-loss curve and stop
        losses = {f"loss@{h['step']}": h["loss"] for h in trainer.state.log_history if "loss" in h}
        print("Train losses:", losses)
        with open(out_dir / "train_log.json", "w") as f:
            json.dump(losses, f, indent=2)
        return

    # best checkpoint (by validation NDCG@10) is loaded at the end of training
    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"], metric_key_prefix="test")
    print("Test:", test_metrics)
    # validation metrics of the selected checkpoint (the Trainer's best step; else the first best evaluation)
    evals = [h for h in trainer.state.log_history if "eval_ndcg@10" in h]
    best_step = getattr(trainer.state, "best_global_step", None)
    best_eval = next((h for h in evals if best_step and h.get("step") == best_step), None)
    if best_eval is None:
        best_eval = max(evals, key=lambda h: float(h["eval_ndcg@10"])) if evals else {}
    val_metrics = {k.replace("eval_", "val_"): float(v) for k, v in best_eval.items() if k.startswith("eval_") and "@" in k}
    if best_eval:
        val_metrics["best_epoch"] = float(best_eval["epoch"])
    with open(out_dir / "test_metrics.json", "w") as f:
        # Separate int-valued count keys from float-valued metric keys
        _test_floats = {k: float(v) for k, v in test_metrics.items() if "_count" not in k}
        _test_counts = {k: int(v) for k, v in test_metrics.items() if "_count" in k}
        json.dump(
            {
                "category": category,
                "variant": str(cfg.tokenization.variant),
                "m": "ragged" if is_ragged else M,
                "max_depth": max_depth if is_ragged else M - 1,
                "base": BASE,
                "min_share_prob": cfg.tokenization.min_share_prob,
                "use_item_mass": bool(cfg.tokenization.use_item_mass),
                **_test_floats,
                **_test_counts,
                **val_metrics,
            },
            f,
            indent=2,
        )

    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()

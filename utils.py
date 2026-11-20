"""Seeding, item-id bookkeeping, item mass, the code-token vocabulary, and seq2seq encoding."""
from __future__ import annotations

import os
import random
from collections import Counter
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from datasets import DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, TrainerCallback, set_seed

DEFAULT_SPECIAL_TOKENS = ("<pad>", "<eos>", "<unk>")
DEFAULT_IGN_ID = -100


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------
def normalize_item2token(item2token: dict, start: int = 1) -> dict:
    """Re-index item ids to the contiguous range start..start+N-1 (must be 1-1)."""
    vals = list(item2token.values())
    uniq = sorted(set(vals))
    if len(uniq) != len(vals):
        raise ValueError("item2token has duplicate token ids (must be 1-1).")
    old2new = {old: new for new, old in enumerate(uniq, start=start)}
    return {k: old2new[v] for k, v in item2token.items()}


def tokenize_splits(dd: DatasetDict, item2token: dict, unk_token: int = 2) -> DatasetDict:
    """Map ASIN histories/labels to integer item ids."""
    get = item2token.get

    def _tok_batch(batch):
        return {
            "history": [[get(x, unk_token) for x in h] for h in batch["history"]],
            "label": [get(x, unk_token) for x in batch["label"]],
        }

    return dd.map(_tok_batch, batched=True, desc="Mapping ASINs to item ids")


def build_id2text_from_item2token(item2token: Dict[str, int], asin2text: Dict[str, str]) -> List[str]:
    """id2text[item_id] = item text ('missing_meta' for ids without an item)."""
    id2text = ["missing_meta"] * (int(max(item2token.values())) + 1)
    for asin, item_id in item2token.items():
        id2text[int(item_id)] = asin2text.get(asin, "missing_meta")
    return id2text


def compute_item_probs_from_dd_train(dd: DatasetDict, *, include_history: bool = True, include_label: bool = True):
    """
    Empirical item mass p_i from the training split (history items + labels).

    Returns (item_probs [V], item_counts [V], total) with V = max_item_id + 1.
    """
    counts: Counter = Counter()
    max_id = -1
    total = 0
    for ex in dd["train"]:
        if include_history:
            h = ex["history"]
            if h:
                counts.update(h)
                total += len(h)
                max_id = max(max_id, max(h))
        if include_label:
            y = ex["label"]
            counts[y] += 1
            total += 1
            max_id = max(max_id, y)

    if max_id < 0 or total == 0:
        raise ValueError("No items found in train (empty histories and labels).")

    item_counts = np.zeros(max_id + 1, dtype=np.int64)
    for i, c in counts.items():
        item_counts[int(i)] = int(c)
    return item_counts.astype(np.float64) / total, item_counts, total


def compute_leaf_popularity_stats(
    item2tokseq_ids: Dict[int, List[int]],
    item_probs: np.ndarray,
    head_pct: float = 0.20,
    tail_pct: float = 0.20,
) -> dict:
    """
    Analyze leaf cluster composition w.r.t. item interaction frequency:
    - Head: top `head_pct` (e.g. 20%) most interacted items in training split
    - Tail: bottom `tail_pct` (e.g. 20%) least interacted items
    - Body: middle items
    
    Computes:
    - Pure Head, Pure Body, Pure Tail leaves
    - Mixed Head-Tail leaves (tail items piggybacking on head prefixes)
    - Tail protection / isolation rates
    """
    items = sorted(item2tokseq_ids.keys())
    N = len(items)
    if N == 0:
        return {}

    # Extract interaction mass for each item
    item_masses = []
    for it in items:
        m = float(item_probs[it]) if it < len(item_probs) else 0.0
        item_masses.append((it, m))

    # Sort items by interaction frequency descending (tie-breaking by item id)
    sorted_by_mass = sorted(item_masses, key=lambda x: (x[1], -x[0]), reverse=True)

    n_head = max(1, int(round(N * head_pct)))
    n_tail = max(1, int(round(N * tail_pct)))

    head_set = {x[0] for x in sorted_by_mass[:n_head]}
    tail_set = {x[0] for x in sorted_by_mass[-n_tail:]}
    tail_set = tail_set - head_set
    body_set = set(items) - head_set - tail_set

    # Group items by leaf prefix
    prefix2items: Dict[tuple, List[int]] = {}
    item2prefix = {}
    for it, seq in item2tokseq_ids.items():
        prefix = tuple(seq[:-1])
        prefix2items.setdefault(prefix, []).append(it)
        item2prefix[it] = prefix

    total_leaves = len(prefix2items)
    pure_head_leaves = 0
    pure_body_leaves = 0
    pure_tail_leaves = 0
    mixed_head_tail_leaves = 0
    mixed_any_leaves = 0

    leaf_type = {}
    for pfx, cluster_items in prefix2items.items():
        h_cnt = sum(1 for x in cluster_items if x in head_set)
        b_cnt = sum(1 for x in cluster_items if x in body_set)
        t_cnt = sum(1 for x in cluster_items if x in tail_set)

        is_pure_h = (h_cnt > 0 and b_cnt == 0 and t_cnt == 0)
        is_pure_b = (b_cnt > 0 and h_cnt == 0 and t_cnt == 0)
        is_pure_t = (t_cnt > 0 and h_cnt == 0 and b_cnt == 0)
        is_mixed_ht = (h_cnt > 0 and t_cnt > 0)
        is_mixed_any = (int(h_cnt > 0) + int(b_cnt > 0) + int(t_cnt > 0)) > 1

        if is_pure_h: pure_head_leaves += 1
        if is_pure_b: pure_body_leaves += 1
        if is_pure_t: pure_tail_leaves += 1
        if is_mixed_ht: mixed_head_tail_leaves += 1
        if is_mixed_any: mixed_any_leaves += 1

        leaf_type[pfx] = {
            "h_cnt": h_cnt,
            "b_cnt": b_cnt,
            "t_cnt": t_cnt,
            "has_head": h_cnt > 0,
            "has_body": b_cnt > 0,
            "is_pure_tail": is_pure_t,
        }

    # Tail item experience
    tail_with_head = 0
    tail_with_body_only = 0
    tail_isolated = 0
    for t_it in tail_set:
        pfx = item2prefix[t_it]
        info = leaf_type[pfx]
        if info["has_head"]:
            tail_with_head += 1
        elif info["has_body"]:
            tail_with_body_only += 1
        else:
            tail_isolated += 1

    # Head item experience
    head_with_tail = 0
    head_isolated = 0
    for h_it in head_set:
        pfx = item2prefix[h_it]
        info = leaf_type[pfx]
        if info["t_cnt"] > 0:
            head_with_tail += 1
        elif info["b_cnt"] == 0:
            head_isolated += 1

    n_tail_total = max(1, len(tail_set))
    n_head_total = max(1, len(head_set))

    return {
        "thresholds": {"head_top_pct": head_pct, "tail_bottom_pct": tail_pct},
        "item_counts": {
            "total_items": N,
            "head_items": len(head_set),
            "body_items": len(body_set),
            "tail_items": len(tail_set),
        },
        "leaf_cluster_breakdown": {
            "total_leaves": total_leaves,
            "pure_head_leaves": pure_head_leaves,
            "pure_body_leaves": pure_body_leaves,
            "pure_tail_leaves": pure_tail_leaves,
            "mixed_head_tail_leaves": mixed_head_tail_leaves,
            "mixed_any_leaves": mixed_any_leaves,
            "pct_leaves_mixed_head_tail": round(100.0 * mixed_head_tail_leaves / total_leaves, 2) if total_leaves else 0.0,
            "pct_leaves_pure_tail": round(100.0 * pure_tail_leaves / total_leaves, 2) if total_leaves else 0.0,
        },
        "tail_item_experience": {
            "tail_items_sharing_with_head": tail_with_head,
            "pct_tail_items_sharing_with_head": round(100.0 * tail_with_head / n_tail_total, 2),
            "tail_items_sharing_with_body_only": tail_with_body_only,
            "pct_tail_items_sharing_with_body_only": round(100.0 * tail_with_body_only / n_tail_total, 2),
            "tail_items_isolated_in_pure_tail": tail_isolated,
            "pct_tail_items_isolated_in_pure_tail": round(100.0 * tail_isolated / n_tail_total, 2),
        },
        "head_item_experience": {
            "head_items_sharing_with_tail": head_with_tail,
            "pct_head_items_sharing_with_tail": round(100.0 * head_with_tail / n_head_total, 2),
            "head_items_isolated_in_pure_head": head_isolated,
            "pct_head_items_isolated_in_pure_head": round(100.0 * head_isolated / n_head_total, 2),
        },
    }


# ---------------------------------------------------------------------------
# Code-token vocabulary and seq2seq examples
# ---------------------------------------------------------------------------
def build_contiguous_wordlevel_tokenizer(
    tokens: Union[Dict[str, int], List[str]],
    special_tokens: tuple = DEFAULT_SPECIAL_TOKENS,
    padding_side: str = "left",
) -> PreTrainedTokenizerFast:
    """WordLevel tokenizer with ids: <pad>=0, <eos>=1, <unk>=2, then `tokens` in order."""
    pad_tok, eos_tok, unk_tok = special_tokens
    vocab: Dict[str, int] = {t: i for i, t in enumerate([pad_tok, eos_tok, unk_tok])}
    for t in (list(tokens.keys()) if isinstance(tokens, dict) else list(tokens)):
        if t not in vocab:
            vocab[t] = len(vocab)

    backend = Tokenizer(WordLevel(vocab=vocab, unk_token=unk_tok))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token=pad_tok,
        eos_token=eos_tok,
        unk_token=unk_tok,
    )
    tokenizer.padding_side = padding_side
    return tokenizer


def make_seq2seq_tokenize_fn(*, item2tokseq_ids: Dict[int, List[int]], M: Optional[int] = None, eos_id: Optional[int] = None):
    """Encoder input = concatenated semantic IDs of the history; decoder target = SID of the next item."""

    def fn(ex):
        input_ids: List[int] = []
        for h in ex.get("history") or []:
            code = item2tokseq_ids[h]
            if M is not None and len(code) != M:
                raise ValueError(f"history item {h} code length={len(code)} != M={M}")
            input_ids.extend(code)

        code_label = item2tokseq_ids[ex["label"]]
        if M is not None and len(code_label) != M:
            raise ValueError(f"target item {ex['label']} code length={len(code_label)} != M={M}")
        labels = list(code_label)
        if eos_id is not None:
            labels.append(int(eos_id))

        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    return fn


# ---------------------------------------------------------------------------
# Training schedule
# ---------------------------------------------------------------------------
class EvalSaveEveryNEpochsCallback(TrainerCallback):
    """Evaluate / checkpoint only every N epochs (and on the last epoch)."""

    def __init__(self, eval_n: int = 10, save_n: int = 10, eval_on_last: bool = True, save_on_last: bool = True):
        self.eval_n = int(eval_n)
        self.save_n = int(save_n)
        self.eval_on_last = bool(eval_on_last)
        self.save_on_last = bool(save_on_last)

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is None:
            return control
        epoch = int(round(state.epoch))
        last_epoch = int(round(args.num_train_epochs))
        control.should_evaluate = bool((epoch % self.eval_n == 0) or (self.eval_on_last and epoch == last_epoch))
        control.should_save = bool((epoch % self.save_n == 0) or (self.save_on_last and epoch == last_epoch))
        return control


def print_model_parameters(model: torch.nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return int(trainable)

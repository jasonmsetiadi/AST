"""Differentiable semantic tree (Naive STE) and joint T5 trainer."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput

from evaluation import build_code_metrics
from model import TrieConstrainedSeq2SeqTrainer, build_item_code_trie, make_prefix_allowed_tokens_fn


class DifferentiableSemanticTree(nn.Module):
    """Fixed-topology hierarchical tree with learnable per-node centroids and STE / Gumbel-Softmax routing."""

    def __init__(
        self,
        *,
        item_embs: np.ndarray,
        tree_structure: dict,
        tokenizer,
        init_item2tokseq_ids: Dict[int, List[int]],
        method: str = "ste",
        temperature: float = 1.0,
        min_temperature: float = 0.1,
        gumbel_noise_scale: float = 1.0,
        min_gumbel_noise_scale: float = 0.0,
        learn_item_embs: bool = True,
    ):
        super().__init__()
        method = str(method).lower()
        if method not in ("ste", "gumbel"):
            raise ValueError(f"Unknown joint_training.method={method!r}; expected 'ste' or 'gumbel'.")
        self.method = method
        self.L = int(tree_structure["L"])
        self.m = int(tree_structure["m"])
        self.base = int(tree_structure["base"])
        self.init_temperature = float(temperature)
        self.min_temperature = float(min_temperature)
        self.temperature = float(temperature)
        self.init_noise_scale = float(gumbel_noise_scale)
        self.min_noise_scale = float(min_gumbel_noise_scale)
        self.noise_scale = float(gumbel_noise_scale)
        self.learn_item_embs = bool(learn_item_embs)

        embs_t = torch.tensor(np.asarray(item_embs, dtype=np.float32), dtype=torch.float32)
        embs_t = F.normalize(embs_t, dim=-1)
        self.N, self.d = embs_t.shape
        self.register_buffer("base_item_embs", embs_t)
        if self.learn_item_embs:
            self.item_emb_delta = nn.Parameter(torch.zeros_like(embs_t))
        else:
            self.item_emb_delta = None

        nodes_by_level = tree_structure["nodes_by_level"]
        self.centroids = nn.ParameterList()
        self.k_max_per_level: List[int] = []

        for lvl in range(self.L):
            nodes = nodes_by_level[lvl]
            M_l = len(nodes)
            K_max_l = max((int(n["K"]) for n in nodes), default=1)
            self.k_max_per_level.append(K_max_l)

            c_tensor = torch.zeros((M_l, K_max_l, self.d), dtype=torch.float32)
            v_mask = torch.zeros((M_l, K_max_l), dtype=torch.bool)
            child_nid = torch.zeros((M_l, K_max_l), dtype=torch.long)

            for nid, node in enumerate(nodes):
                K_n = int(node["K"])
                c_tensor[nid, :K_n] = torch.tensor(node["centroids"], dtype=torch.float32)
                v_mask[nid, :K_n] = True
                for c_idx, ch in enumerate(node["children"][:K_n]):
                    child_nid[nid, c_idx] = max(0, int(ch))

            self.centroids.append(nn.Parameter(c_tensor))
            self.register_buffer(f"valid_mask_{lvl}", v_mask)
            self.register_buffer(f"child_node_id_{lvl}", child_nid)

            tok_ids_l = torch.tensor(
                [tokenizer.convert_tokens_to_ids(f"<c{lvl}_{k}>") for k in range(K_max_l)],
                dtype=torch.long,
            )
            self.register_buffer(f"level_tok_ids_{lvl}", tok_ids_l)

        suffix_lvl = self.m - 1
        suffix_tok_ids = torch.tensor(
            [tokenizer.convert_tokens_to_ids(f"<c{suffix_lvl}_{k}>") for k in range(self.base)],
            dtype=torch.long,
        )
        self.register_buffer("suffix_tok_ids", suffix_tok_ids)

        init_suffix = torch.zeros((self.N,), dtype=torch.long)
        for item_id in range(self.N):
            init_suffix[item_id] = int(init_item2tokseq_ids[item_id][-1])
        self.register_buffer("item_suffix_tok", init_suffix)

    def _valid_mask(self, lvl: int) -> torch.Tensor:
        return getattr(self, f"valid_mask_{lvl}")

    def _child_node_id(self, lvl: int) -> torch.Tensor:
        return getattr(self, f"child_node_id_{lvl}")

    def level_tok_ids(self, lvl: int) -> torch.Tensor:
        return getattr(self, f"level_tok_ids_{lvl}")

    def get_item_embs(self, item_ids: torch.Tensor) -> torch.Tensor:
        x = self.base_item_embs[item_ids]
        if self.item_emb_delta is not None:
            x = F.normalize(x + self.item_emb_delta[item_ids], dim=-1)
        return x

    def update_anneal_state(self, progress: float) -> None:
        """Linearly anneal temperature and Gumbel noise scale over progress in [0, 1]."""
        p = max(0.0, min(1.0, float(progress)))
        self.temperature = self.init_temperature + p * (self.min_temperature - self.init_temperature)
        self.noise_scale = self.init_noise_scale + p * (self.min_noise_scale - self.init_noise_scale)

    def forward(
        self,
        item_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            item_ids: LongTensor of shape [...]
            mask: optional BoolTensor/FloatTensor of shape [...] (1 = valid, 0 = pad)
        Returns:
            p_ste_list: list of L tensors [..., K_max_l]
            c_hard_tensor: LongTensor [..., L]
            commit_loss: scalar Tensor
            balance_loss: scalar Tensor
        """
        x = self.get_item_embs(item_ids)
        node_id = torch.zeros(item_ids.shape, dtype=torch.long, device=item_ids.device)

        p_ste_list: List[torch.Tensor] = []
        c_hard_list: List[torch.Tensor] = []
        commit_losses: List[torch.Tensor] = []
        balance_losses: List[torch.Tensor] = []

        valid_weight = mask.float() if mask is not None else None

        for lvl in range(self.L):
            C = self.centroids[lvl][node_id]  # [..., K_max_l, d]
            vmask = self._valid_mask(lvl)[node_id]  # [..., K_max_l]

            dist2 = ((x.unsqueeze(-2) - C) ** 2).sum(dim=-1)
            raw_logits = -dist2
            if self.training and self.method == "gumbel" and self.noise_scale > 0.0:
                u = torch.rand_like(raw_logits).clamp_(1e-6, 1.0 - 1e-6)
                gumbel = -torch.log(-torch.log(u))
                raw_logits = raw_logits + self.noise_scale * gumbel
            logits = (raw_logits / max(self.temperature, 1e-6)).masked_fill(~vmask, -1e9)
            p_soft = F.softmax(logits, dim=-1)

            c_hard = logits.argmax(dim=-1)
            p_hard = F.one_hot(c_hard, num_classes=C.size(-2)).to(p_soft.dtype)
            p_ste = p_hard + (p_soft - p_soft.detach())  # ST forward, soft/Gumbel-soft backward

            # Commitment loss: anchor items and assigned centroids
            assigned_c = (p_hard.unsqueeze(-1) * C).sum(dim=-2)
            item_commit = ((x - assigned_c) ** 2).sum(dim=-1)
            if valid_weight is not None:
                denom = valid_weight.sum().clamp(min=1.0)
                commit_losses.append((item_commit * valid_weight).sum() / denom)
            else:
                commit_losses.append(item_commit.mean())

            # Optional balance loss: maximize entropy of batch-average soft usage
            flat_p = p_soft.reshape(-1, p_soft.size(-1))
            if valid_weight is not None:
                fw = valid_weight.reshape(-1, 1)
                mean_p = (flat_p * fw).sum(dim=0) / fw.sum().clamp(min=1.0)
            else:
                mean_p = flat_p.mean(dim=0)
            neg_entropy = (mean_p * (mean_p.clamp(min=1e-8)).log()).sum()
            balance_losses.append(neg_entropy)

            p_ste_list.append(p_ste)
            c_hard_list.append(c_hard)
            node_id = self._child_node_id(lvl)[node_id, c_hard]

        c_hard_tensor = torch.stack(c_hard_list, dim=-1)
        commit_loss = torch.stack(commit_losses).mean() if commit_losses else x.new_zeros(())
        balance_loss = torch.stack(balance_losses).mean() if balance_losses else x.new_zeros(())
        return p_ste_list, c_hard_tensor, commit_loss, balance_loss

    @torch.no_grad()
    def export_item2tokseq_ids(self, batch_size: int = 4096) -> Dict[int, List[int]]:
        """Hard nearest-centroid lookup for all items + deterministic collision suffix."""
        was_training = self.training
        self.eval()
        device = self.base_item_embs.device

        all_prefixes: List[Tuple[int, ...]] = []
        for start in range(0, self.N, batch_size):
            end = min(start + batch_size, self.N)
            item_ids = torch.arange(start, end, dtype=torch.long, device=device)
            _, c_hard, _, _ = self.forward(item_ids)
            c_np = c_hard.cpu().numpy()
            tok_maps = [self.level_tok_ids(lvl).cpu().numpy() for lvl in range(self.L)]
            for i in range(end - start):
                all_prefixes.append(tuple(int(tok_maps[lvl][int(c_np[i, lvl])]) for lvl in range(self.L)))

        prefix2items: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        for item_id, prefix in enumerate(all_prefixes):
            prefix2items[prefix].append(item_id)

        max_bucket = max((len(v) for v in prefix2items.values()), default=0)
        if max_bucket > self.base:
            print(
                f"[JointTree WARNING] max prefix bucket size {max_bucket} > base={self.base}; "
                "suffix tokens will wrap and cause collisions."
            )

        suffix_cpu = self.suffix_tok_ids.cpu().tolist()
        item2tokseq_ids: Dict[int, List[int]] = {}
        new_suffix = torch.zeros((self.N,), dtype=torch.long)

        for prefix, items in prefix2items.items():
            for j, item_id in enumerate(sorted(items)):
                suf_tok = int(suffix_cpu[j % self.base])
                item2tokseq_ids[item_id] = list(prefix) + [suf_tok]
                new_suffix[item_id] = suf_tok

        self.item_suffix_tok.copy_(new_suffix.to(device))
        if was_training:
            self.train()
        return item2tokseq_ids


class JointDataCollator:
    """Left-pads raw item-ID histories and packs target item IDs."""

    def __call__(self, features: List[dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        max_hist = max(len(f["history_item_ids"]) for f in features)
        history_item_ids = torch.zeros((batch_size, max_hist), dtype=torch.long)
        history_mask = torch.zeros((batch_size, max_hist), dtype=torch.long)
        target_item_id = torch.zeros((batch_size,), dtype=torch.long)

        for i, f in enumerate(features):
            h = f["history_item_ids"]
            L_h = len(h)
            if L_h > 0:
                history_item_ids[i, max_hist - L_h :] = torch.tensor(h, dtype=torch.long)
                history_mask[i, max_hist - L_h :] = 1
            target_item_id[i] = int(f["target_item_id"])

        return {
            "history_item_ids": history_item_ids,
            "history_mask": history_mask,
            "target_item_id": target_item_id,
        }


class JointT5Wrapper(nn.Module):
    """Wraps DifferentiableSemanticTree + T5ForConditionalGeneration for joint end-to-end training."""

    def __init__(
        self,
        *,
        tree: DifferentiableSemanticTree,
        t5,
        pad_id: int,
        commit_loss_weight: float = 0.0,
        balance_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.tree = tree
        self.t5 = t5
        self.pad_id = int(pad_id)
        self.commit_loss_weight = float(commit_loss_weight)
        self.balance_loss_weight = float(balance_loss_weight)

    def _ste_embeds_for_items(
        self, p_ste_list: List[torch.Tensor], suffix_tok_ids: torch.Tensor
    ) -> torch.Tensor:
        """Build [..., m, d_model] embeddings using STE weights for levels 0..L-1 and discrete lookup for L."""
        level_embs: List[torch.Tensor] = []
        for lvl, p_ste in enumerate(p_ste_list):
            tok_table = self.t5.shared(self.tree.level_tok_ids(lvl))  # [K_max_l, d_model]
            emb_l = torch.matmul(p_ste.to(tok_table.dtype), tok_table)
            level_embs.append(emb_l)
        level_embs.append(self.t5.shared(suffix_tok_ids))
        return torch.stack(level_embs, dim=-2)

    def update_anneal_state(self, progress: float) -> None:
        """Linearly anneal temperature and Gumbel noise scale over progress in [0, 1]."""
        p = max(0.0, min(1.0, float(progress)))
        self.temperature = self.init_temperature + p * (self.min_temperature - self.init_temperature)
        self.noise_scale = self.init_noise_scale + p * (self.min_noise_scale - self.init_noise_scale)

    def forward(
        self,
        history_item_ids: torch.Tensor,
        history_mask: torch.Tensor,
        target_item_id: torch.Tensor,
    ) -> Seq2SeqLMOutput:
        B, H = history_item_ids.shape
        m = self.tree.m
        L = self.tree.L

        # 1. Encoder inputs_embeds via STE
        p_ste_hist, _, commit_hist, balance_hist = self.tree(history_item_ids, mask=history_mask)
        suffix_hist = self.tree.item_suffix_tok[history_item_ids]
        hist_embeds = self._ste_embeds_for_items(p_ste_hist, suffix_hist)  # [B, H, m, d_model]
        inputs_embeds = hist_embeds.reshape(B, H * m, -1)
        attention_mask = history_mask.unsqueeze(-1).expand(-1, -1, m).reshape(B, H * m)

        # 2. Target STE distributions + shifted decoder_inputs_embeds
        p_ste_tgt, _, commit_tgt, balance_tgt = self.tree(target_item_id)
        suffix_tgt = self.tree.item_suffix_tok[target_item_id]
        tgt_embeds = self._ste_embeds_for_items(p_ste_tgt, suffix_tgt)  # [B, m, d_model]

        pad_tok = torch.full((B, 1), self.pad_id, dtype=torch.long, device=target_item_id.device)
        pad_emb = self.t5.shared(pad_tok)  # [B, 1, d_model]
        decoder_inputs_embeds = torch.cat([pad_emb, tgt_embeds[:, :-1, :]], dim=1)  # [B, m, d_model]

        # 3. T5 forward
        outputs = self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=False,
        )
        logits = outputs.logits  # [B, m, vocab_size]
        log_probs = F.log_softmax(logits.float(), dim=-1)

        # 4. STE cross-entropy over levels 0..L-1 + standard CE on suffix level L
        rec_loss = logits.new_zeros((), dtype=torch.float32)
        for lvl in range(L):
            lp_l = log_probs[:, lvl, self.tree.level_tok_ids(lvl)]  # [B, K_max_l]
            rec_loss = rec_loss - (p_ste_tgt[lvl].float() * lp_l).sum(dim=-1).mean()
        rec_loss = rec_loss + F.cross_entropy(logits[:, L, :].float(), suffix_tgt)
        rec_loss = rec_loss / float(m)

        total_loss = rec_loss
        if self.commit_loss_weight > 0.0:
            total_loss = total_loss + self.commit_loss_weight * 0.5 * (commit_hist + commit_tgt).float()
        if self.balance_loss_weight > 0.0:
            total_loss = total_loss + self.balance_loss_weight * 0.5 * (balance_hist + balance_tgt).float()

        return Seq2SeqLMOutput(loss=total_loss, logits=logits)


class JointTrieConstrainedSeq2SeqTrainer(TrieConstrainedSeq2SeqTrainer):
    """Seq2SeqTrainer that refreshes the trie & metrics from the differentiable tree before eval."""

    def __init__(
        self,
        *args,
        pad_id: int,
        m: int,
        k_values=(5, 10),
        item_tiers: Optional[Dict[int, str]] = None,
        suffix_refresh_steps: int = 200,
        anneal_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._joint_pad_id = int(pad_id)
        self._joint_m = int(m)
        self._joint_k_values = tuple(k_values)
        self._joint_item_tiers = item_tiers
        self._suffix_refresh_steps = int(suffix_refresh_steps)
        self._anneal_steps = None if anneal_steps is None else int(anneal_steps)
        self._last_refresh_step: int = -1
        self._last_anneal_step: int = -1
        self._current_item2tokseq_ids: Optional[Dict[int, List[int]]] = None

    def _unwrap_joint(self, model=None) -> JointT5Wrapper:
        m = model if model is not None else self.model
        return m.module if hasattr(m, "module") else m

    def _sync_codebook_for_eval(self) -> None:
        joint = self._unwrap_joint()
        item2tokseq_ids = joint.tree.export_item2tokseq_ids()
        self._current_item2tokseq_ids = item2tokseq_ids
        trie = build_item_code_trie(item2tokseq_ids, eos_id=None)
        self._prefix_allowed_tokens_fn = make_prefix_allowed_tokens_fn(
            trie,
            pad_id=self._joint_pad_id,
            eos_id=joint.t5.config.eos_token_id,
            decoder_start_id=joint.t5.config.decoder_start_token_id,
            M=self._joint_m,
        )
        self.compute_metrics = build_code_metrics(
            item2tokseq_ids=item2tokseq_ids,
            pad_id=self._joint_pad_id,
            M=self._joint_m,
            eos_id=None,
            k_values=self._joint_k_values,
            item_tiers=self._joint_item_tiers,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        joint = self._unwrap_joint(model)
        if model.training and self.state.global_step != self._last_anneal_step:
            self._last_anneal_step = self.state.global_step
            if joint.tree.method == "gumbel":
                total_steps = self._anneal_steps if self._anneal_steps is not None else max(1, int(self.state.max_steps))
                progress = min(1.0, float(self.state.global_step) / float(max(1, total_steps)))
                joint.tree.update_anneal_state(progress)

        if (
            model.training
            and self._suffix_refresh_steps > 0
            and self.state.global_step > 0
            and self.state.global_step != self._last_refresh_step
            and self.state.global_step % self._suffix_refresh_steps == 0
        ):
            self._last_refresh_step = self.state.global_step
            joint.tree.export_item2tokseq_ids()
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        self._sync_codebook_for_eval()
        return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix: str = "test"):
        self._sync_codebook_for_eval()
        return super().predict(test_dataset=test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def _materialize_discrete_batch(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert self._current_item2tokseq_ids is not None
        hist_ids = inputs["history_item_ids"].cpu().tolist()
        hist_mask = inputs["history_mask"].cpu().tolist()
        tgt_ids = inputs["target_item_id"].cpu().tolist()
        device = inputs["history_item_ids"].device

        B = len(hist_ids)
        H = len(hist_ids[0]) if B > 0 else 0
        m = self._joint_m

        input_ids = torch.full((B, H * m), self._joint_pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, H * m), dtype=torch.long, device=device)
        labels = torch.zeros((B, m), dtype=torch.long, device=device)

        for i in range(B):
            for h_idx in range(H):
                if hist_mask[i][h_idx] == 1:
                    code = self._current_item2tokseq_ids[int(hist_ids[i][h_idx])]
                    start = h_idx * m
                    input_ids[i, start : start + m] = torch.tensor(code, dtype=torch.long, device=device)
                    attention_mask[i, start : start + m] = 1
            labels[i] = torch.tensor(self._current_item2tokseq_ids[int(tgt_ids[i])], dtype=torch.long, device=device)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        inputs = self._prepare_inputs(inputs)
        if self._current_item2tokseq_ids is None:
            self._sync_codebook_for_eval()

        discrete = self._materialize_discrete_batch(inputs)
        t5 = self._unwrap_joint(model).t5

        with torch.no_grad():
            outputs = t5(
                input_ids=discrete["input_ids"],
                attention_mask=discrete["attention_mask"],
                labels=discrete["labels"],
            )
            loss = outputs.loss.detach()

        if prediction_loss_only or not self.args.predict_with_generate:
            return (loss, None, discrete["labels"])

        gen_kwargs = getattr(self, "_gen_kwargs", {}).copy()
        if self._prefix_allowed_tokens_fn is not None:
            gen_kwargs["prefix_allowed_tokens_fn"] = self._prefix_allowed_tokens_fn
        gen_kwargs["max_new_tokens"] = self._joint_m
        gen_kwargs["min_new_tokens"] = self._joint_m

        with torch.no_grad():
            generated = t5.generate(
                input_ids=discrete["input_ids"],
                attention_mask=discrete["attention_mask"],
                **gen_kwargs,
            )
        return (loss, generated, discrete["labels"])

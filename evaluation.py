"""Recall@k / NDCG@k over items decoded from generated semantic IDs."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


def build_code_metrics(
    *,
    item2tokseq_ids: Dict[int, List[int]],
    pad_id: int,
    M: Optional[int] = None,
    eos_id: Optional[int] = None,
    k_values=(5, 10),
    ign_id: int = -100,
    debug: bool = False,
):
    """
    Returns a `compute_metrics` fn for the HF trainer.

    Each of the R returned beams is mapped back to its item (codes are unique, which is
    asserted); duplicates are dropped while preserving beam order, then Recall@k and
    NDCG@k are computed against the ground-truth next item.
    """
    code2items: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for item, code in item2tokseq_ids.items():
        code2items[tuple(int(x) for x in code)].append(int(item))
    collisions = [c for c, items in code2items.items() if len(items) > 1]
    if collisions:
        raise ValueError(f"Codebook has {len(collisions)} collisions; evaluation is ill-defined.")
    code2item = {code: items[0] for code, items in code2items.items()}

    def _clean_to_code(seq: List[int]) -> Optional[Tuple[int, ...]]:
        seq = [int(x) for x in seq]
        while seq and seq[0] == pad_id:  # strip the decoder-start token / pad
            seq = seq[1:]
        if M is not None:
            if len(seq) < M:
                return None
            return tuple(seq[:M])
        else:
            cleaned = []
            for t in seq:
                if t == pad_id or (eos_id is not None and t == eos_id):
                    break
                cleaned.append(t)
            return tuple(cleaned) if cleaned else None

    def _ndcg_at_k(pred_items: List[int], gt: int, k: int) -> float:
        try:
            rank = pred_items[:k].index(gt)
        except ValueError:
            return 0.0
        return 1.0 / math.log2(rank + 2.0)

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids

        if (
            isinstance(preds, np.ndarray)
            and preds.ndim == 3
            and np.issubdtype(preds.dtype, np.floating)
            and preds.shape[-1] > 256
        ):
            preds = preds.argmax(axis=-1)

        N = labels.shape[0]

        gt_items: List[Optional[int]] = [None] * N
        for i in range(N):
            lab = [int(x) for x in labels[i].tolist() if int(x) != ign_id and (eos_id is None or int(x) != eos_id)]
            if M is not None:
                if len(lab) == M:
                    gt_items[i] = code2item.get(tuple(lab))
            else:
                gt_items[i] = code2item.get(tuple(lab))

        if preds.ndim == 3:  # (N, R, L)
            if preds.shape[0] != N:
                raise ValueError(f"Unexpected preds shape {preds.shape} vs labels N={N}")
            R_eff = preds.shape[1]
            flat = preds.reshape(N * R_eff, preds.shape[2])
        else:  # (N*R, L)
            flat = preds
            if flat.shape[0] % N != 0:
                raise ValueError(f"Cannot infer num_return_sequences from preds shape={flat.shape} with N={N}")
            R_eff = flat.shape[0] // N

        recalls = {k: [] for k in k_values}
        ndcgs = {k: [] for k in k_values}
        for i in range(N):
            gt = gt_items[i]
            cand_items: List[int] = []
            for r in range(R_eff):
                code = _clean_to_code(flat[i * R_eff + r].tolist())
                if code is None:
                    continue
                item = code2item.get(code)
                if item is not None and item not in cand_items:
                    cand_items.append(item)

            for k in k_values:
                if gt is None:
                    recalls[k].append(0.0)
                    ndcgs[k].append(0.0)
                else:
                    recalls[k].append(1.0 if gt in cand_items[:k] else 0.0)
                    ndcgs[k].append(_ndcg_at_k(cand_items, gt, k))

        out = {f"recall@{k}": float(np.mean(recalls[k])) for k in k_values}
        out.update({f"ndcg@{k}": float(np.mean(ndcgs[k])) for k in k_values})
        if debug:
            print(f"[metrics] N={N} R={R_eff} gt_mapped={sum(g is not None for g in gt_items)}/{N} {out}")
        return out

    return compute_metrics

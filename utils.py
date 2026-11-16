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

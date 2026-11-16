"""T5 generative recommender with trie-constrained beam search over valid semantic IDs."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from transformers import Seq2SeqTrainer, T5Config, T5ForConditionalGeneration


def build_t5_seq2seq(
    vocab_size: int,
    pad_token_id: int,
    eos_token_id: int,
    *,
    encoder_layers: int = 6,
    decoder_layers: int = 6,
    d_model: int = 128,
    d_ff: int = 1024,
    num_heads: int = 6,
    dropout_rate: float = 0.15,
    initializer_factor: float = 0.02,
    layer_norm_epsilon: float = 1e-6,
    feed_forward_proj: str = "relu",
) -> T5ForConditionalGeneration:
    config = T5Config(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=encoder_layers,
        num_decoder_layers=decoder_layers,
        num_heads=num_heads,                 # d_kv keeps the T5 default (64)
        initializer_factor=initializer_factor,
        layer_norm_epsilon=layer_norm_epsilon,
        dropout_rate=dropout_rate,
        feed_forward_proj=feed_forward_proj,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        decoder_start_token_id=pad_token_id,  # T5 convention: PAD starts decoding
    )
    return T5ForConditionalGeneration(config)


# ---------------------------------------------------------------------------
# Constrained decoding: only real item codes can be generated
# ---------------------------------------------------------------------------
def build_item_code_trie(item2tokseq_ids: Dict[int, List[int]], eos_id: Optional[int] = None) -> Dict:
    trie: Dict = {}
    for item, seq in item2tokseq_ids.items():
        node = trie
        for t in seq:
            node = node.setdefault(int(t), {})
        node["_item"] = int(item)  # terminal marker (not a token id)
        if eos_id is not None:
            node[int(eos_id)] = {"_item": int(item)}
    return trie


def trie_allowed_next(trie: Dict, prefix: List[int]) -> Optional[List[int]]:
    node = trie
    for t in prefix:
        t = int(t)
        if t not in node:
            return None
        node = node[t]
    return [k for k in node.keys() if isinstance(k, int)]


def make_prefix_allowed_tokens_fn(
    trie: Dict, *, pad_id: int, eos_id: int, decoder_start_id: Optional[int], M: Optional[int] = None
):
    """`prefix_allowed_tokens_fn` for `generate`: next tokens that keep the prefix inside the trie."""

    def fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
        if input_ids.numel() > 1:
            first = int(input_ids[0].item())
            if decoder_start_id is not None and first == int(decoder_start_id):
                input_ids = input_ids[1:]
            elif first == int(pad_id):
                input_ids = input_ids[1:]

        prefix = input_ids.tolist()
        while prefix and prefix[0] == pad_id:
            prefix = prefix[1:]

        if M is not None and len(prefix) >= M:
            allowed = [int(eos_id)]
        else:
            allowed = trie_allowed_next(trie, prefix) or [int(eos_id)]

        if M is not None:
            # generation length is fixed to M, so EOS is never needed
            return [t for t in allowed if int(t) != int(eos_id)]
        else:
            return allowed

    return fn


class TrieConstrainedSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer whose generation (eval / predict) is always trie-constrained and exactly M tokens long."""

    def __init__(
        self, *args, prefix_allowed_tokens_fn=None, gen_M: Optional[int] = 4, max_new_tokens: int = 10, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._gen_M = gen_M
        self._max_new_tokens = max_new_tokens

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        inputs = self._prepare_inputs(inputs)

        loss = None
        if "labels" in inputs:
            with torch.no_grad():
                outputs = model(**inputs)
            loss = outputs.loss.detach()

        if prediction_loss_only or not self.args.predict_with_generate:
            return (loss, None, inputs.get("labels"))

        gen_kwargs = getattr(self, "_gen_kwargs", {}).copy()
        if self._prefix_allowed_tokens_fn is not None:
            gen_kwargs["prefix_allowed_tokens_fn"] = self._prefix_allowed_tokens_fn
        if self._gen_M is not None:
            gen_kwargs["max_new_tokens"] = self._gen_M
            gen_kwargs["min_new_tokens"] = self._gen_M
        else:
            gen_kwargs["max_new_tokens"] = self._max_new_tokens
            gen_kwargs["min_new_tokens"] = 2

        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                **gen_kwargs,
            )
        return (loss, generated, inputs.get("labels"))

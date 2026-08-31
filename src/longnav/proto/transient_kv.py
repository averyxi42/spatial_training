"""PROTOTYPE - transient (hand-camera) KV mechanism. No production code is modified.

Design: docs/MOBILE_MANIP_KV_DESIGN.md. Mobile manipulation adds a hand-camera image
per turn. Hand tokens must ATTEND the full history (joint-training semantics) but must
never PERSIST as history (manipulation is locally Markovian; the cache stays lean and
the pretrain->finetune attention geometry stays identical).

Mechanism (measured at parity with append-all in tools/bench_kv_cache.py):
  * ``TransientFilteredCache.update`` stores only the persistent slice of each new K/V
    block but returns cached-history ++ full-current-block, which is what the attention
    layers consume. No rollback, no crop, cache append-only.
  * RoPE "rollback": the tokens AFTER a hand block take the positions they would have
    had without it (computed with the production ``get_rope_index``, twice per turn),
    so the cache never contains a positional gap.
  * Sparse integration: the transient mask is declared in PRE-sparsification
    coordinates; ``_TransientShim`` (inserted between the production ``TextMixin`` and
    the HF text model purely via MRO) re-slices it with the ``seq_keep_mask`` the mixin
    just computed, arms the cache, and drops hand embeds from ``kept_visual_embeds`` so
    the similarity DB (``past_image_embeds``) stays head-camera-only.

Install onto an already-loaded production sparse model with ``install_transient(model)``
(instance ``__class__`` swap of the language model -- weights untouched).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import DynamicCache

from longnav.utils.modeling import Qwen3VLSparseTextModel, TextMixin
from transformers import Qwen3VLTextModel


class TransientFilteredCache(DynamicCache):
    """DynamicCache that stores only persistent tokens but exposes the full block.

    Set ``transient_mask`` (bool, shape (S_post_sparse,), True = transient) before a
    forward; it is consumed (reset to None) afterwards by the shim. ``update`` is
    called once per layer within the forward, so the mask must stay stable across the
    whole forward -- the shim owns that discipline.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.transient_mask: Optional[torch.Tensor] = None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        m = self.transient_mask
        if m is None or not bool(m.any()):
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        keep = ~m.to(key_states.device)
        k_hist, v_hist = super().update(
            key_states[:, :, keep], value_states[:, :, keep], cache_kwargs=cache_kwargs,
            layer_idx=layer_idx)
        hist_len = k_hist.shape[2] - int(keep.sum())
        k_full = torch.cat([k_hist[:, :, :hist_len], key_states], dim=2)
        v_full = torch.cat([v_hist[:, :, :hist_len], value_states], dim=2)
        return k_full, v_full


class _TransientShim:
    """Sits between TextMixin and the HF text model in the MRO.

    Runs at the exact point where TextMixin has finished sparsification (its
    ``seq_keep_mask`` / ``kept_visual_embeds`` are set) and is about to call the real
    text model. Consumes ``self._pending_transient`` (PRE-sparse bool mask over the
    current chunk, set externally by the rollout/test harness).
    """

    _pending_transient: Optional[torch.Tensor] = None

    def forward(self, *args, past_key_values=None, **kwargs):
        pending = getattr(self, "_pending_transient", None)
        self._pending_transient = None
        if pending is not None and isinstance(past_key_values, TransientFilteredCache):
            keep = getattr(self, "seq_keep_mask", None)
            if keep is not None:
                post_mask = pending[keep.to(pending.device)]
            else:
                post_mask = pending
            past_key_values.transient_mask = post_mask
            # Keep the similarity DB head-only: drop kept visual embeds that fall in
            # the transient span. kept_visual_embeds rows correspond, in order, to the
            # True entries of vis_keep_mask over this chunk's visual tokens; map those
            # back to sequence positions via visual_pos_masks BEFORE sparsification.
            vpm = kwargs.get("visual_pos_masks", None)
            if getattr(self, "kept_visual_embeds", None) and vpm is not None:
                # post-sparse visual positions, aligned with post_mask
                vis_positions = torch.nonzero(vpm[0].cpu()).squeeze(-1)
                vis_transient = post_mask.cpu()[vis_positions]
                for b, emb in enumerate(self.kept_visual_embeds):
                    if emb.shape[0] == int(vis_transient.numel()):
                        self.kept_visual_embeds[b] = emb[~vis_transient]
                    elif bool(vis_transient.any()):
                        n_trans = int(vis_transient.sum())
                        # kept embeds are ordered like vis positions; drop the
                        # transient tail/segment by index intersection
                        keep_rows = torch.ones(emb.shape[0], dtype=torch.bool)
                        idx = torch.nonzero(vis_transient).squeeze(-1)
                        idx = idx[idx < emb.shape[0]]
                        keep_rows[idx] = False
                        self.kept_visual_embeds[b] = emb[keep_rows]
        out = super().forward(*args, past_key_values=past_key_values, **kwargs)
        if isinstance(past_key_values, TransientFilteredCache):
            past_key_values.transient_mask = None
        return out


class ProtoTransientTextModel(TextMixin, _TransientShim, Qwen3VLTextModel):
    """TextMixin -> _TransientShim -> Qwen3VLTextModel (pure MRO composition)."""


def install_transient(model) -> None:
    """Swap the loaded sparse model's language model class in place (weights untouched)."""
    lm = model.model.language_model
    assert isinstance(lm, Qwen3VLSparseTextModel), type(lm)
    lm.__class__ = ProtoTransientTextModel
    lm._pending_transient = None


def stitched_positions(vl_model, turn_inputs_full: dict, turn_inputs_nohand: dict,
                       transient_mask: torch.Tensor, offset: int
                       ) -> Tuple[torch.Tensor, int]:
    """Per-turn mrope positions with the hand block positionally transparent.

    ``turn_inputs_full`` is the real chunk (head + hand images); ``turn_inputs_nohand``
    is the same chunk with the hand image's tokens and grid removed. Pre-hand tokens
    take the full chunk's positions; the hand block keeps its natural positions; the
    tokens AFTER the hand block take the no-hand chunk's positions (RoPE rolled back).
    The returned offset advances by the no-hand extent -- the cache never sees a gap.
    """
    pos_full, _ = vl_model.get_rope_index(
        input_ids=turn_inputs_full["input_ids"],
        image_grid_thw=turn_inputs_full.get("image_grid_thw"),
        video_grid_thw=None, attention_mask=turn_inputs_full.get("attention_mask"))
    pos_nh, delta_nh = vl_model.get_rope_index(
        input_ids=turn_inputs_nohand["input_ids"],
        image_grid_thw=turn_inputs_nohand.get("image_grid_thw"),
        video_grid_thw=None, attention_mask=turn_inputs_nohand.get("attention_mask"))
    m = transient_mask
    n = m.shape[0]
    first = int(torch.nonzero(m)[0]) if bool(m.any()) else n
    stitched = torch.empty_like(pos_full)
    stitched[..., :first + int(m.sum())] = pos_full[..., :first + int(m.sum())]
    # after-hand tokens: the no-hand computation's positions for the same tokens
    stitched[..., first + int(m.sum()):] = pos_nh[..., first:]
    stitched = stitched + offset
    new_offset = offset + turn_inputs_nohand["input_ids"].shape[1] + int(delta_nh.item())
    return stitched, new_offset


def image_token_spans(input_ids: torch.Tensor, image_token_id: int):
    """Contiguous runs of image placeholder tokens, as (start, end) pairs."""
    m = (input_ids[0] == image_token_id).int()
    pad = torch.cat([torch.zeros(1, dtype=torch.int), m.cpu(), torch.zeros(1, dtype=torch.int)])
    diff = pad[1:] - pad[:-1]
    starts = torch.nonzero(diff == 1).squeeze(-1)
    ends = torch.nonzero(diff == -1).squeeze(-1)
    return list(zip(starts.tolist(), ends.tolist()))

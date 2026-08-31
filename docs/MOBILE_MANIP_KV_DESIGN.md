# Mobile manipulation: hand-camera tokens and the KV cache

Design discussion 2026-08-31. Context: extending the nav VLA with an arm readout and a
hand-camera observation. Manipulation is locally Markovian (MS-HAB baselines: 128x128
depth, frame-stack 3, no long history); the head camera's history is the nav memory.
Desired semantics: head tokens durable, hand tokens transient -- hand tokens ATTEND the
full history (joint training requirement) but are never attended AS history.

## Constraint discovered in the rollout code

`vlm_worker.infer_step` forwards per-turn deltas `[prev_postfix-1 : cur_postfix-1]`, so
sampled-action tokens enter the cache as the head of the next delta: action history is
cache-consistent BY CONSTRUCTION, append-only, no rollback. Any eviction design breaks
this and pays DynamicCache.crop's per-layer copies.

## Microbench (tools/bench_kv_cache.py; Qwen3-VL-2B text path, sdpa, GPU5, 3 repeats,
100 turns x [120 persistent + 16 hand + 30 readout] tokens)

| arm | total | median/turn | final cache |
|---|---|---|---|
| A append-all (absorb hand into history) | 4.9-5.9 s | 43-50 ms | 16,600 |
| B rollback (crop + re-forward) | 9.1-9.5 s | 78-92 ms | 15,000 |
| C filtered cache.update | 5.0-5.7 s | 45-58 ms | 15,000 |

B is ~1.9x A and grows with cache length: rollback is ruled out. C matches A.

## Chosen mechanism (option 2, "filtered update")

`FilteredCache(DynamicCache)`: `update()` stores only the persistent slice of the new
K/V but RETURNS cached-history ++ full-current-block for attention. One forward per
turn contains head+hand tokens; hand K/V never lands in the cache. No attention-layer
surgery (the return value of cache.update is what layers attend over). Positions: the
tokens AFTER the hand block reuse the hand block's positions, so RoPE is rolled back
natively and the cache is gap-free; train-time uses a 4D mask reproducing the same
geometry (future turns cannot attend past hand blocks; hand blocks attend everything
before them).

Pretraining consistency: Markovian manip pretraining = history-length-1 episodes in the
SAME format; under filtered-update the hand block's attention pattern is identical in
pretrain and joint finetune (reader of its turn, never history), so nothing is unlearned.

## Open engineering items before trusting it

1. Production path adds mrope, the sparsifying patch's image-embed DB (hand images must
   not enter `past_image_embeds`), and flash-attention: build behind a flag.
2. Bit-parity test: filtered-cache rollout logits vs a full no-cache forward with the
   equivalent 4D mask, fixed episode, exact match (the project's echo-fix discipline).
3. Readout order: base/nav readout first (preserves pretrained positional habits), arm
   readout after; coordination decoder-side via zero-init context tokens
   (CodeContextMixer precedent), not backbone cross-attention.
4. Proprio (qpos/gripper/is_grasped) belongs in the same transient block.

## Prototype + parity results (2026-08-31, no production code modified)

`src/longnav/proto/transient_kv.py`: `TransientFilteredCache` (store persistent slice,
return full concat), `_TransientShim` (MRO insert between the production TextMixin and
the HF text model: re-slices the pre-sparse hand mask with the just-computed
seq_keep_mask, arms the cache, keeps the similarity DB head-only), `install_transient`
(instance class-swap on a loaded sparse model), `stitched_positions` (production
get_rope_index called on the chunk with and without the hand image; post-hand tokens
take the hand-free positions -- RoPE rolled back with the real mrope arithmetic).

`tests/proto/parity_transient_kv.py` (real sparse checkpoint, sdpa, 3 turns x
[head 256px + hand 128px]): noise floor (plain incremental vs full forward, matched
mask path) 0.28-0.38 max|dlogit|; transient arm (FilteredCache + stitched RoPE vs full
forward under the equivalent 4D mask) 0.31-0.34 -- WITHIN the floor. Cache holds
exactly tokens-minus-hand (462-192=270); sparse-mode smoke: composes with the live
sparsify path (cache 230), DB excludes hand embeds.

Incidental finding that cost a debugging cycle: under sdpa+bf16, a CACHED prefill and
an UNCACHED forward of identical inputs differ by ~5.9 max|dlogit| (argmax intact) --
sdpa's is_causal fast path vs its additive-mask path (`dump/probe_prefill_ab.py`,
exactly reproducible). Parity comparisons must pin the mask path on both sides; and it
is one more reason production standardizes on flash_attention_2 for rollout/training
forward agreement.

Remaining before training on it: flash-attention variant of the reference (or accept
sdpa-only parity), wiring into a VLMWorker fork for real rollout, and the training-side
4D-mask builder reusing the same turn/transient bookkeeping.

## Flash-path validation (2026-08-31, second pass -- "using sdpa is pointless")

Rebuilt the validation on the production attention path. Chronicle, each step
evidence-backed (probes under dump/probe_*.py, tests under tests/proto/):

1. HF's flash integration packed-detects supplied position_ids (`_is_packed_sequence`)
   and crashes on mrope image chunks -- production already neutralizes this with a
   monkeypatch in VLMWorker.__init__ (vlm_worker.py:54-57), previously undocumented and
   load-bearing for every flash rollout. Harnesses must mirror it.
2. Position convention revised: contiguous WITHIN a turn, RoPE rollback applied to the
   next turn's START OFFSET (hand-free extent). Flash-admissible, identical cross-turn
   geometry to a packed training forward, and within-turn geometry exactly matches
   single-turn (Markovian) manipulation pretraining.
3. Semantics correction: the mechanism guarantees CACHE GEOMETRY, not information
   erasure -- the turn's readout attends the hand (intended) and its K/V persist, so
   hand info reaches later turns THROUGH the readout: measured ~3.0-3.4 max|dlogit| on
   later turns (production layout). This is the learned bottleneck, a feature.
4. Bitwise validation (hand-at-suffix layout, nothing persistent after the hand,
   determinism control passing): DENSE mode -- later-turn readouts bitwise EQUAL under
   hand-content perturbation, cache tensors bitwise identical at every probed layer.
   The filter is exact on flash.
5. Sparse mode residual (~0.4 max|dlogit|) fully attributed: keep-selection identical
   (layer-0 K bitwise equal), similarity DB bitwise equal (shim strip works), vision
   tower per-image isolated (co-batched head embeds bitwise equal under hand swap);
   the residual is SHAPE-dependent flash accumulation -- hand KEEP COUNTS differ with
   content (post-sparse lengths 151/131/126 vs 147/134/131), changing kernel block
   partitioning. Numeric, not informational (only the count leaks).

Verdict: mechanism validated for rollout on the production stack. Training-side open
item stands: the transient 4D mask needs flex_attention (or a chunked-with-cache
training forward) -- flash cannot express it; microbench before the joint-finetune
design is finalized.

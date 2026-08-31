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

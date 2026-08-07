# Head-only training on a frozen context

Freeze a trunk's output to disk, then iterate on the action head against it in minutes
instead of hours. Three entry points and one module:

| | |
|---|---|
| `data_scripts/harvest_context.py` | run the trunk once, write `(pooled, context, targets, pose)` per observation |
| `data_scripts/train_head_only.py` | train the head, and only the head, on that store |
| `data_scripts/reattach_head.py` | put the trained head back on its own trunk -> an ordinary policy checkpoint |
| `src/longnav/utils/head_only.py` | the store, the model, the trainer, the collapse diagnostics |

## Why it exists

`dump/cnf_sigma_ablation/FINDINGS.md` established, on cached `(context, chunk)` pairs,
that the CNF head's failures are not the head's:

* the flow reaches 91-98% of a ridge regression's R^2 on the same split -- "no cell fails
  at its own job";
* the dequantisation sigma is not the problem (`3e-5, warm=none`, the current default, was
  the best cell of fifteen on every conditioning metric);
* `scale_saturation = 0` and `|logdet|` tracking the noise-floor budget are the *expected*
  regime, not pathology;
* the diagnosed cause is **context collapse** -- the conditioning vector going
  near-constant, 93% of every row the dataset mean, ~2 effective dimensions.

That result was produced by an ad-hoc harness that could not be compared to a real run and
could not produce a runnable policy. This pipeline is the same experiment made first-class,
so the two questions

    can the head learn?              (head on a frozen, known-good context)
    is the trunk giving it anything? (the same head, end to end)

are asked with the *same code* and answered by comparing the *same metric names*.

## The seam

Everything upstream of the pooled vector is the trunk: backbone, LoRA, modality encoders,
and the pooling rule. Everything downstream is the head. The cut is
`TurnVectorHead.pooled_context` -> `TurnVectorHead.project`, and it is a real seam in the
production code, not a copy of one:

```
TurnVectorRegressor.forward  = encode_turns -> head.project -> loss
TurnFlowPolicy.forward       = encode_turns -> head.project -> FlowObjectiveMixin.flow_objective
harvest_context.py           = encode_turns -> head.project -> disk
FrozenContextFlowPolicy      =                 head.project -> FlowObjectiveMixin.flow_objective
                                 ^ from disk
```

`tests/test_head_only.py` pins that `FrozenContextFlowPolicy.flow_objective` **is**
`TurnFlowPolicy.flow_objective` -- the same function object, not an equivalent one. A
divergence between the two would not raise; it would produce two numbers that look
comparable, are not, and would invalidate every conclusion drawn by comparing them.

## What the store holds

Two context columns, because they cut the model at two places:

* **`pooled`** `(N, pooled_dim)` -- the frozen trunk state. Training on this trains the
  projection *and* the flow: exactly `train_flow_sft.py`'s trainable set minus the LoRA
  adapters and the modality encoders. This is the default and the only reattachable one.
* **`context`** `(N, context_dim)` -- the source head's own output. Training on this
  freezes the projection too and trains only the flow, which is the
  `dump/cnf_sigma_ablation` setting. Useful as a control; not reattachable, because the
  result is a flow bolted to someone else's projection rather than a single head.

Plus `targets` (the anchor-relative action chunk) and `pose` (the value actually injected,
after `relative_se2` -- i.e. what the encoder saw, not the scene-frame column).

### Storage format, and why

Sharded `.npy`, opened `mmap_mode="r"`, with a JSON manifest. **Measured: 12,540 bytes per
row** (`pooled` 2048 + `context` 1024 + `targets` 20x3 + `pose` 3, all float32) over a
221,090-row / 4,672-episode harvest occupying 2.58 GiB. The full 39,061-episode train split
extrapolates to ~1.85M rows and ~23 GB.

Harvest throughput, four workers on four H100s: **~26 rows/s per worker** (~1.9 s/episode,
~104 rows/s aggregate) with the cards otherwise idle, falling to **~16 rows/s** once
head-only training runs share them. A completed 1500-episode worker averaged 16.0 rows/s /
2.96 s/episode end to end under that contention, for 70,981 rows in 0.829 GiB. So the full
39k train split is roughly five hours on four dedicated GPUs and closer to eight if you are
training on them at the same time -- either way a once-only cost, and part of why the
harvest is resumable.

* `.npz` -- what the prior work used -- is a zip container. It cannot be memory-mapped and
  every read decompresses whole arrays into RAM. Fine at 11k rows, impossible at 2M.
* A HuggingFace/Arrow dataset stores each 2048-float row as a variable-length list with
  offset overhead and rebuilds a numpy array per access. Correct, but the wrong shape for
  a fixed-width float matrix that a training run will index a few hundred thousand times.
* Plain `.npy` is self-describing in dtype and shape, gives zero-copy random access, needs
  nothing beyond numpy, and lets the page cache do the caching.

Sharding buys the resume: arrays are written to a temporary name and renamed, and the
manifest entry -- which is what makes an episode "done" -- is appended only afterwards. A
crash costs at most one shard and never leaves a half-written row. Harvesting is
embarrassingly parallel but a single manifest written by four processes would race, so
each worker takes a disjoint `--episode-start/--episode-stop` range into its own `--out`
and `ContextStore.open_many` joins the parts, refusing any pair that disagrees about the
trunk or shares an episode id.

### Provenance

A frozen-context dataset whose source trunk is unknown is worthless, because the trunk is
the independent variable of every experiment run on it. The manifest records the
checkpoint path and step, a **content hash of its adapter and head weights**, the corpus,
the split, the modality spec, the pooling mode, the chunk shape and the observation rate.

The hash deliberately excludes `optimizer.pt`, `rng_state_*.pth` and `trainer_state.json`:
those change with resume bookkeeping that changes no forward pass, and hashing them would
make a legitimate reattachment fail after a restart.

## The three silent failures it gates on

**1. The pose modality.** It has three separate wiring points, and two fail silently:
`modality_specs` on the `ModelConfig`, `ModalityBatch.pop_from` + `pending()` around the
backbone call, and `modality_specs=` on the `TurnVectorCollator`. In this pipeline point 1
comes from the checkpoint's own config (so it cannot disagree with what was trained), point
2 lives inside the shared `encode_turns`, and point 3 is taken off the model rather than
re-parsed from a flag.

But a wiring check only proves the plumbing is connected. `--verify-pose` (on by default)
proves the water is flowing: it re-runs one batch with every pose shifted by 1.0 and
requires the context to move. Measured on the v9 trunk: a +1.0 shift moves the context by
1.8-7.6 against a context scale of ~17. A harvest with the pose silently absent looks
completely normal and is wrong.

**2. Reattaching to the wrong trunk.** Checked against the stored hash; a mismatch names
both sides and stops. A head trained on trunk A and bolted onto trunk B loads, runs, and is
meaningless -- there is no observable that would go wrong, which is exactly why the check
must be here and not downstream.

**3. An unfaithful harvest.** After reattachment the assembled model's own forward is run
on real turns from the harvest corpus and the pooled state it computes is compared with
what the store recorded for those rows. It is a hard gate, not a warning: if it fails, the
trunk is not the one the context came from, or the modality encoders are not loaded, or the
pooling/affix configuration differs, and every conclusion drawn from the store is void.

Pooled rather than context, deliberately: the head has been replaced by construction so its
output is *expected* to differ, while the pooled state is exactly the quantity that was
frozen. It is also the stronger statement -- any head is a deterministic function of it.

## Metric parity

Every number is produced by inherited code, so the names and the arithmetic are the full
run's:

| where it comes from | keys |
|---|---|
| `FlowSFTTrainer._drain_metrics` (inherited unchanged) | `eval_nll_per_dim`, `eval_min_nll_per_dim`, `eval_nll_margin`, `eval_nll_headroom`, `eval_logdet_absmax`, `eval_scale_saturation`, `eval_stop_*`, `eval_creep_*`, `eval_stop_ratio_dx`, `eval_turn_loss` |
| `TurnVectorSFTTrainer._drain_metrics` | `eval_rmse_*`, `eval_mae_*`, `eval_rmse_mean`, `eval_turns` |
| HF `Trainer` | `loss`, `grad_norm`, `learning_rate` |
| `attach_model_metrics` | the per-module grad/weight/activation series |
| added here | `eval_ctx_*`, `eval_in_ctx_*`, `eval_w_ctx_over_w_x` |

One difference is deliberate. A real run is under bf16 autocast because a 2B backbone has
to be; there is no backbone here, so the head runs in fp32 by default and `--bf16` restores
the exact numerical path. The flow itself is fp32 in both -- `flow_objective` disables
autocast around it -- so the objective is unaffected either way and only the projection
differs.

`eval_sparse_tokens` and `eval_dense_tokens` are 0 here rather than absent, because there
was no sequence. That is true, and it is a useful marker that a log came from this pipeline.

## Two analysis traps

Both of these reverse the conclusion if read carelessly, and both bit during this
pipeline's own first analysis.

**`nll_headroom` is measured against a moving floor during the sigma anneal.**
`min_nll_per_dim` falls ~4.6 nats over a 1000-step anneal from `sigma_start = 100`, so
headroom grows during the anneal whatever the model does. Read across the boundary, v12
looks like it goes 1.67 -> 3.96 and is losing ground; read only after the floor settles at
-9.495, v12's headroom **shrinks**, 4.07 -> 3.94. `dump/head_only/compare_to_v12.py`
splits the two phases and labels every row `ANNEAL` or `post`.

**`grad_norm` is confounded by the learning-rate schedule.** With `--lr-scheduler cosine`,
a short run spends most of its length at a decaying rate and its gradient norm falls for
that reason alone. A 2000-step run of this pipeline peaks at 256 around step 1050 and
descends to 48 -- which looks like stability and is mostly the cosine. The same
configuration at 8000 steps is still climbing at step 3000, where the rate is still ~78%
of peak. v12 died at step 2366 of a 20000-step cosine, i.e. at ~99% of peak rate, so its
monotone climb is the one measured at nearly constant rate. Compare gradient norms only at
comparable points of the schedule, or with `--lr-scheduler constant`.

## The collapse diagnostics

Logged per eval, on a fixed held-out probe split:

| key | what it says | healthy | collapsed |
|---|---|---|---|
| `eval_ctx_resid_frac` | norm left after removing the dataset mean | 86.6% | 1.9% |
| `eval_ctx_participation_ratio` | directions the surviving variation occupies | 14.5 | 1.20 |
| `eval_ctx_mean_pairwise_cos` | how parallel two random rows are | +0.246 | +0.9998 |
| `eval_ctx_frac_near_mean` | rows within 10% of the mean | ~0 | ~1 |
| `eval_w_ctx_over_w_x` | `mean\|W_ctx\| / mean\|W_x\|` in the couplings | see below | -- |

Reference values from `dump/cnf_sigma_ablation/ctx_stats.py`, which this reproduces
exactly, including that the participation ratio is taken on the **mean-removed** matrix.

Two cautions:

* **PR and `resid_frac` are not redundant.** A constant vector plus isotropic noise has
  `resid_frac` ~ 0 and a full-width participation ratio: collapsed by any useful
  definition, invisible to PR alone. The dead run happened to be collapsed in both senses,
  which is why one number sufficed there and does not in general.
* **`w_ctx_over_w_x` is not a pass/fail number.** FINDINGS is explicit: healthy cold-start
  cells sit at 0.85-0.91 while healthy *warm-started* ones sit at 0.10-0.18 early and
  0.33-0.40 by step 8000, because the warm start loads large trained x-columns. What is
  diagnostic is the ratio failing to grow while the loss falls.

`eval_in_ctx_*` is the same statistics on the *harvested* vector. It is constant by
construction and is the control: it says what a good context looks like on this corpus, so
`eval_ctx_*` can be read against something rather than against intuition.

## What this pipeline cannot tell you

This matters more than anything above, because the pipeline is fast and its numbers are
seductive.

**A head that trains well on frozen context says nothing about whether the trunk will
supply that context under joint training.** The context here was produced by a *different
objective* -- the v9 trunk was trained by flow matching, and the only demonstrably
informative context in the prior work was made by the AR head's cross-entropy. The flow's
own NLL, asked to produce a conditioning vector from scratch, produced a rank-1 vector that
got *more* rank-1 over training. Freezing removes exactly that failure by construction.

Specifically, this pipeline **cannot** tell you:

* **whether the joint optimisation is stable.** A fresh 0.8M-parameter head on cached trunk
  states at 4000 steps is not a projection plus LoRA through a VLM at ~250 chunks/step.
  Different gradient scales, different conditioning, different everything.
* **whether the trunk would remain informative.** The trunk is frozen. The failure mode
  under investigation is the trunk *changing* into something uninformative, and that
  degree of freedom has been removed.
* **whether the head is good.** Held-out NLL on a frozen context is a density-fitting
  score. Closed-loop navigation success is a different measurement; that is what
  `reattach_head.py` exists to make possible, and it has not been run by anything in this
  directory.
* **anything about a different trunk.** Every number is conditional on one checkpoint. The
  hash gate enforces this at reattachment; nothing enforces it on a reader's inferences.

The honest positive claim available from a clean head-only run is narrow and worth stating
precisely: *given a context of this quality, this head and this objective optimise stably
to this NLL.* If the same head degrades end to end, the trunk is implicated. If it degrades
here too, the head or its objective is implicated and the context-collapse diagnosis needs
revisiting.

## Usage

```bash
# 1. Harvest. Four workers on disjoint episode ranges, one GPU each.
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src python data_scripts/harvest_context.py \
    --ckpt dump/pose_injection/run_v9_flow_planar_pose_2p5hz/checkpoint-11400 \
    --dataset /Projects/data/v2_25hz_obs2.5hz/formatted_pose --split train \
    --out dump/head_only/ctx_v9_ck11400/part$i \
    --episode-start $((i*1500)) --episode-stop $(((i+1)*1500)) --obs-hz 2.5 &
done

# 2. Train the head. Minutes.
PYTHONPATH=src python data_scripts/train_head_only.py \
  --store dump/head_only/ctx_v9_ck11400/part* \
  --output-dir dump/head_only/run_cnf_a --max-steps 4000

# 3. Reattach -> an ordinary policy checkpoint.
PYTHONPATH=src python data_scripts/reattach_head.py \
  --head-run dump/head_only/run_cnf_a/final \
  --out dump/head_only/reattached_cnf_a
```

Useful controls:

* `--context-key context` -- freeze the source head too, train only the flow.
* `--init-flow <marginal.pt>` -- the same warm start `train_flow_sft.py` takes.
* `--no-wandb` -- everything still lands in `trainer_state.json`.

## Evaluating a reattached checkpoint

The output is byte-compatible with the ordinary layout (`adapter/`,
`turn_vector_head.pt`, `turn_vector_head_config.json`, `flow_config.json`, processor
files). One caveat, **measured** rather than assumed:

* `VectorRolloutPolicy.from_checkpoint(dir)` **fails** on a CNF checkpoint. It routes
  through `TurnVectorRegressor.from_pretrained`, which puts a `TargetNormalizer` in the
  `normalizer` slot and then strictly loads the flow's ~170 tensors into it.
* `VectorRolloutPolicy(TurnFlowPolicy.from_pretrained(dir, processor), processor, cfg)`
  **works**, and decoding through the `normalizer` slot returns a correctly shaped chunk.

So `flow_head`'s docstring claim -- that `VectorRolloutPolicy.step` works untouched
because `FlowActionDecoder` occupies the `normalizer` slot -- is right about `step` and
wrong about loading. A CNF backend for `objectnav_eval` is a construction change (build the
model with `TurnFlowPolicy.from_pretrained`, then hand it to the existing
`VectorRolloutPolicy`), not a new rollout implementation.

`reattach_head.py` reports both results on every run, so this stays measured rather than
remembered.

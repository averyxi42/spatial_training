# Relaunching `flow_sde_rtc_a09_fullset` after the cycle-142 OOM

Status: **NOT RELAUNCHED.** These are the notes to do it cleanly, so the resumed run
continues the same wandb history rather than starting a second run beside it.

The experiment this run exists to settle is in `FLOW_RL_SAMPLE_EFFICIENCY.md` section 7:
a literal twin of `flow_sde_rtc_a09_held128` differing only in `train_uids` and
`run_name`, to test whether the small-pool doctrine survives a matched-horizon
comparison. That purpose constrains what may be changed on resume (section 4).

---

## 1. What happened

**Rank 0 hit a CUDA OOM on GPU 0** at 13:55:40 on 2026-08-27, inside
`accelerator.backward` -> gradient-checkpoint recompute -> a LoRA layer
(`result = result + lora_B(lora_A(dropout(x))) * scaling`):

    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.04 GiB.
    GPU 0 has a total capacity of 79.14 GiB of which 980.94 MiB is free.
    ... 411.31 MiB is reserved by PyTorch but unallocated.

It died before issuing allreduce **6582**, so ranks 1-3 blocked on that collective --
which is what `last enqueued work: 6604, last completed work: 6581` reports. The NCCL
watchdog fired 600 s later at 14:05:32, rank 3 took the process group down "to avoid
data inconsistency", and the driver logged `FAULT at cycle 142` at 14:06.

**The NCCL timeout is the symptom, not the cause.** Anyone reading only the watchdog
message will diagnose a DDP desync -- unequal collective counts across ranks -- and go
looking for a branch in the training path. There is no such branch: `run_training_epochs`
(`train_loop.py:429`) dispatches exactly `n_rollout / num_vlms` = 16/4 = 4 calls per
worker per epoch, evenly by construction. The missing collective is simply the rank that
died.

Two hypotheses were investigated and **ruled out**, recorded so they are not re-run:

* **Empty micro-batch.** `train_loop.py:435` passes
  `traj_batch[i:i+1, response_mask[i].bool()]` with no emptiness guard, and
  `minibatch_token_scales` clamps lengths to `min=1` -- a tell that someone expected a
  zero. Checked against 2,640 episodes on disk: **minimum length 5 steps, zero episodes
  at 0-3 steps**. The condition never fires.
* **Resource contention from the concurrent tokenizer work.** Those jobs ran with
  `CUDA_VISIBLE_DEVICES=7` throughout; the OOM was on GPU 0. System RAM was 1,481 GB
  available with no OOM-killer activity.

**Why this run and not `held128`.** GPU 0 grew from 44.2 GB (09:02) to 76.6 GB (11:43)
over the session, until a 1.04 GiB request had 981 MiB to land in. Episode length
contributes but weakly -- fullset median 95 steps against held128's 80, both saturating
the 175 cap by p90 -- so the dominant factor is that cards 0-3 carried more per-GPU load
than 4-7, where held128 continued running past cycle 529 unaffected.

## 2. What survived

`dump/flow_rl/flow_sde_rtc_a09_fullset/checkpoints/checkpoint_142_crash/`, written by
`emergency_checkpoint` at 14:07:

| file | contents |
| --- | --- |
| `adapter_model.safetensors` + `adapter_config.json` | the trained RL LoRA delta |
| `optimizer.pt` | AdamW moments |
| `scheduler.pt` | LR schedule position |
| `rl_state.pt` | `{schema: 1, global_cycle: 142, trajectory_list: [...]}` |

`rl_state.pt` is the driver-side state the weights checkpoint cannot hold: the cycle
counter and the **advantage buffer** -- the last `n_adv` episodes the time-kernel
baseline is fitted on. Resuming without it refits the baseline from empty, which is a
real discontinuity in advantage scale exactly at the seam someone will later try to read
across. `load_rl_state` returns `global_cycle + 1`, so the resumed run starts at **cycle
143**, not 142.

GPUs 0-3 are fully released (1 MiB each) -- nothing is wedged, no `ray stop` needed.

## 3. How the seam is made invisible

**wandb continuity is driven entirely by `task.run_name`.** `factories.py:245-274`
queries `api.runs(project, filters={"displayName": run_cfg.run_name})`, takes
`runs[-1]`, and passes that run's `id` into `wandb.init(..., resume="allow")`. So:

* **Keep `task.run_name: "flow_sde_rtc_a09_fullset"` byte-identical.** Any edit --
  a `_r2` suffix, a typo -- creates a second run and the history is split. This is the
  single thing that must not change.
* **Keep `task.logger.project: "cont_dev"`.** The lookup is scoped to the project.
* The same call returns `episodes_to_skip`, built from `history['episode_label']`, so a
  resumed run will not re-log episodes wandb already has. Expect the printed
  `Skipping N episodes already logged in WandB` line; it is the mechanism working.

**Cycle continuity** comes from `training.checkpoint` pointing at the crash directory:
`train_eval_rl.py:110` calls `load_rl_state(tcfg.training.checkpoint)`, which restores
both the counter and the buffer from the same path the weights load from. The two cannot
drift because they live in one directory.

## 4. The relaunch config

Start from `flow_sde_rtc_fullset.yaml` and change **only** the `training` block:

```yaml
training:
  # WAS null (SFT merged into the base, fresh zero-delta LoRA). On resume this points at
  # the RL delta from the crash, which is a DIFFERENT adapter from merge_adapter_dir --
  # so the "applies the delta twice" refusal does not apply. Also the source of
  # global_cycle=142 and the advantage buffer, via load_rl_state.
  checkpoint: /Projects/spatial_training/dump/flow_rl/flow_sde_rtc_a09_fullset/checkpoints/checkpoint_142_crash
  load_optim: true      # WAS false -- AdamW moments, or the seam shows as a scale jump
  load_sched: true      # WAS false -- LR schedule position
```

Everything else stays byte-identical: `train_uids`, `sde_noise_a: 0.9`,
`learning_rate: 2.0e-6`, `action_head_learning_rate: 0.0`, `grad_accum_steps: 2`,
`reject_blind_episodes: true`, `eval_uids_file`, `merge_adapter_dir`, and the RTC delay
law (`exp`, `d_max` 10, base 0.8, seed 0). **The point of this run is that it is a
literal twin of held128** -- any drift here silently converts a controlled comparison
into an uncontrolled one, and the drift will not be visible in the wandb curve.

Same launch shape as before: `+experiment=flow_sde_rtc_fullset +resources=quad`, module
`longnav.scripts.train_eval_rl`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, and
`resources.vlm_conda_env: longnav_vlm` already pinned in the YAML (the documented
landmine that killed two earlier launches).

## 5. The OOM itself -- do not skip this

`expandable_segments` is the mitigation the error message names, and it is worth setting,
but **it would not have saved this allocation**: 411 MiB of fragmentation against a
1.04 GiB request with 981 MiB free.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

The real question is what grew **32 GB over four hours** on GPU 0, because it will do it
again near cycle 142 regardless of allocator flags. Untraced; the suspects are KV-cache
retention across cycles, the driver's rollout buffer, and activation memory scaling with
the longest episodes in a batch. Options, in increasing order of intrusiveness:

1. Resume with `expandable_segments` and nothing else, and watch `nvidia-smi` on GPU 0
   per cycle. Cheapest, and a second failure at a similar cycle count localises it.
2. Trace the growth first. Ten minutes against another four-hour run.
3. Reduce per-GPU load on cards 0-3 (fewer sims sharing a trainer card), which changes
   throughput but not the objective -- so it is safe for the comparison, unlike anything
   touching the reward, the pool or the optimiser.

**A note for whoever resumes.** The run reached cycle 142 against held128's 529. Per the
matched-horizon analysis in `FLOW_RL_SAMPLE_EFFICIENCY.md` section 7, neither arm should
be read before ~300-400 cycles, so this run is not yet near the point where it answers
anything. Losing it and restarting from SFT would cost the whole comparison; resuming
from cycle 143 costs nothing but the wall-clock already spent.

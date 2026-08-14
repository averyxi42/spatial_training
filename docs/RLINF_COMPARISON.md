# RLinf / piRL flow-SDE comparison (2026-08-14)

Source: line-level study of RLinf (github.com/RLinf) openpi flow-SDE + the piRL paper
(arXiv:2510.25889), against our `flow_sde_policy.py`. Repo snapshot + full delta table in
the session records; this file keeps the durable conclusions.

## What their working recipe is (verified in code, not prose)

- N=1 stochastic step per chunk, position drawn once PER BATCH; K=4 denoise steps (pi0),
  ablated {1,2,4,8} -> K=8 was their WORST; noise a=0.5 default (a=0.2 measurably fails:
  73.1 vs 94.5 eval); ignore_last=False; first-step sigma_0 = a*sqrt(K).
- Log-prob: stored chains, mu recomputed at stored index, NO prefix re-integration (the
  same partial-likelihood approximation we use -- it is not our differentiator).
- Credit ONLY the executed slice: [:action_chunk, :action_env_dim].
- PPO + learned critic (GAE 0.99/0.95), advantage broadcast uniform over denoising steps,
  dual clip c=3.0, NO KL-to-reference in any config, entropy hard-zeroed for flow-SDE.
- VLM BACKBONE FROZEN; only the ~300M action expert trains at 5e-6 (critic 1e-4).
- Batch: 2048 chunk-steps global, 64-320 parallel envs, 4 update epochs.
- Sparse success reward (+small shaping), reward_coef ~5. Eval = pure ODE.

## What we adopted (2026-08-14)

- Drift sign corrected (see FLOW_SDE_RL.md post-mortem) -- their algebra, verified
  equivalent to 2.4e-7.
- First-step schedule sigma_0 = a*sqrt(K) -> our `a` is now commensurable with theirs.
- n_exclude_last=0 legal (their default), the singularity rationale refuted by audit.
- Dual clip: already present via hapo (clip_ratio_c=3.0).

## Deliberate deviations that remain

- N=3 in current configs (theirs: 1); sde_position_weight=sigma (no RLinf analogue --
  makes the clipped quantity not a log-ratio; drop when N=1).
- We credit all 20 latent ticks; they credit the executed slice only. NOT TRANSFERABLE,
  REJECTED (2026-08-14): our velocity transformer has full attention across tick rows, so
  each transition is a density over the FULL chunk vector given the previous full vector
  -- executed-row means at step k+1 depend on tail-row values at step k. Dropping tail
  terms from the product does not marginalize the tail out; it scores executed rows
  conditional on a tail path whose probability is unaccounted, and the telescoping that
  makes the ratio a valid importance weight breaks. A true executed-slice marginal means
  integrating tail randomness through the attention -- intractable. RLInf can slice
  because their setups execute essentially the whole emitted chunk (gap = chunk length);
  ours replans at gap < chunk length, which is exactly where the slice stops being a
  marginal. Do not resurrect this without solving the marginalization.
- We recompute old_log_prob at postprocess; they reuse the rollout value
  (recompute_logprobs: False everywhere). Ours adds a numerical seam, theirs adds none.
- LoRA backbone trainable vs their frozen backbone.
- Dense progress reward vs their sparse success.
- Batch ~16 episodes/step x 1 epoch vs their 2048 chunk-steps x 4 epochs.

## The v5 replication spec (proposed, not yet built)

Corrected sign + new schedule, sde_n=1, sde_noise_a calibrated by the init SDE-vs-ODE
gap (not blindly 0.5), n_exclude_last=0, position_weight=none, backbone LoRA FROZEN
(lr 0) with head/velocity-field at ~5e-6, n_rollout=16 (chunk-step parity: 16 eps x
~100 steps ~ 1600 ~ their 2048; the advantage buffer is our baseline smoother),
n_epoch=2-4, dual clip on (already present). No executed-slice credit -- see rejection
above. Run on overfit32 first. K stays 10 until an ODE-eval shows K=4
deploy is competitive for this checkpoint (their K ablation says smaller is better; our
SFT was validated at K=10 -- test before adopting).

## Critic notes (for the shelved PPO path)

Their ablations: 4-layer critic MLP > 1-layer; value averaged over the denoising
trajectory; detach_critic_input=True; pi0.5 puts the value head after the VLM
(`value_after_vlm`, ablated better). Eval DIPS until critic explained-variance rises --
budget for a warmup phase. Matches our critic-is-last-resort plan.

## piRL Appendix F/G insights (read in full, 2026-08-14)

1. **The init SDE-vs-ODE gap is their PRIMARY noise-calibration gauge**: measure the SFT
   checkpoint's stochastic-rollout success against its deterministic eval BEFORE
   training; a large gap means reduce `a` or increase K. This is a principled way to
   pick `a` (rather than importing 0.5 blindly) and is nearly free for us: two
   `run_eval_cycle` passes at init, one `ode=True`, one SDE at candidate `a`. ADOPT as a
   v5 pre-flight step.
2. **Eval oscillating while train climbs -> INCREASE K** (shrinks SDE/ODE divergence).
   Note K trades against their own smaller-K-explores-better ablation: K is a live
   tuning axis in both directions, not a constant.
3. **KL rising steadily -> cosine lr anneal** stabilized their pi0.5 LIBERO-Long runs.
   Ours is a linear schedule; the knob exists if the signfix/v5 runs show KL creep.
4. **VLM fine-tuning gave no evident benefit on LIBERO**; frozen ~= LoRA at conservative
   settings (their LoRA-II: lr 1e-6, 2 epochs). Supports v5's frozen-backbone choice,
   and gives the fallback numbers if we keep LoRA.
5. **Critic warmup dip is expected and diagnosable** (eval dips until explained variance
   rises) -- already folded into the shelved critic plan.
6. **Temporal efficiency emerged from RL without an explicit slack term** (episode
   lengths converged to the expert range) via discounting + their partial-reset
   mechanism rewarding faster completion with more resets per update window. Relevant to
   our slack-penalty debate: their pressure comes from the reset economics, not reward
   shaping.
7. Long-horizon tasks preferred LARGER action chunks (10 vs 5); and GR00T N1.5 gained
   +37.4 avg success from the same PPO+flow-SDE recipe -- the technique transfers across
   architectures when the recipe is right.

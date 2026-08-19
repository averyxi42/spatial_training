# gamma-0.97 horizon falsifier — resolution (2026-08-18)

Run: `dump/flow_rl/flow_sde_gamma097_warm64` (experiment yaml
`flow_sde_gamma097_warm64.yaml`, which carries the full pre-registration).
Warm start ck791 (held128's published peak), treatment = gamma 0.95→0.97 alone,
n_rollout 24, pool 64. Killed at cycle ~1224 after ~430 post-relaunch cycles.

## Verdict: the pre-registered flat-to-worse prediction stands

The eval26 series showed a peak band at cycles 1100–1148 (best 8-cycle window
0.731 oSR / 0.503 oSPL, vs held128's 0.715/0.471 on the same instrument), which
motivated a sample400 capstone on the band's best checkpoint. The capstone
refutes the band:

| sample400 (397 paired episodes, identical protocol) | ck791 (held128) | ck1143 (gamma097) | paired Δ | t |
|---|---|---|---|---|
| success (auto-stop) | 0.589 | 0.574 | −0.015 | −0.59 |
| oracle success      | 0.733 | 0.698 | −0.035 | −1.95 |
| oracle SPL          | 0.483 | 0.477 | −0.006 | −0.35 |
| SPL                 | 0.272 | 0.275 | +0.002 | +0.18 |

Gamma 0.97 with the time-kernel baseline produced **no improvement over the
0.95 peak; oracle success is marginally worse**. The eval26 band was
instrument noise amplified by post-hoc window selection — 26 episodes cannot
resolve deltas of this size, and this is now the second time the sample400
capstone has corrected an eval26 read. Trust the capstone, never the band.

Mechanistically this is the outcome the reward-horizon analysis put on record
before launch: at gamma 0.97 the search-phase credit still inherits the
variance floor (per-step corr(r,A) 0.402→0.314, a 1.64× dilution the larger
rollout only partly buys back). Horizon extension via gamma alone does not
unlock long-horizon learning at this variance level.

## What follows from it

- **The gamma-0.99 escalation is NOT licensed** by the pre-registered rule
  (it required resolved positive movement). Any future horizon push needs the
  variance floor addressed first — which is what the state-probe/value-baseline
  track (cotrain-v4, `docs/STOP_HEAD_PLAN.md`) exists to do. The mini parity
  run's paired counterfactual (`probe/adv_var_probe` vs `adv_var_kernel`)
  is the direct measurement of whether that floor actually moves.
- ck791 remains the published best continuous checkpoint
  (`Aasdfip/...-held128-ck791`). No upload for gamma097.
- Peak-band checkpoints 1111/1119/1127/1143 stay preserved in
  `checkpoints_preserved/` (cheap, and the run's terminal state is dispersed
  across the rolling window). ck1119 was not capstoned: band members 24 cycles
  apart share weights to first order, and ck1143 — the band's single best
  eval26 cycle — already resolved negative.
- Full eval artifacts: `dump/eval_system/s400_gamma097_ck1143/` (+ `.log`),
  launch script `run_gamma097_ck1143.sh`, conversion manifest in
  `checkpoints_preserved/checkpoint_1143_harness/`.

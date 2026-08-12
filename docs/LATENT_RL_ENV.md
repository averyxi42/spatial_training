# RL on the latent intent: what to add, and what not to touch

Status: design, nothing built. Branch `latent_rl`. Companion to `docs/LATENT_RL.md`, which
covers the SFT conversion that produces the policy this document runs.

The constraint that shapes everything below: **the discrete path and the existing continuous
(basic Gaussian head) path must behave exactly as they do today.** Every addition here is a
new class, a new config node, or a hook that is absent-by-default.

## The headline: our policy is already the continuous path

`ContinuousActionHead` returns `{"mu", "log_std"}`. `rollout_core` samples `N(mu, std)`,
clips, computes the diagonal-Gaussian log-prob, sums it over the action dimension, stores
`actions_continuous`, and PPO-ratios it in `train_rl_step`.

`LatentSplit` emits `(mu, log_sigma)`. That is the same contract. So:

* advantage estimators, policy loss, `train_loop`, the critic, the snapshot tests -- **no
  change**. They operate on log-probs and are indifferent to what the action means.
* the sampling, storage and ratio machinery -- **no change**.
* the two-conda-env split is already solved: Ray gives env actors `habitat_conda_env` (`vln`)
  and VLM actors `vlm_conda_env`. That is exactly the simulator/model separation the eval
  harness solves with a unix socket, and here it comes for free.

What is genuinely new is narrow: where `(mu, log_sigma)` comes from, that the action must be
**decoded before it reaches the env**, and that the env is a continuous ObjectNav simulator.

## A. `LatentIntentHead`

New class beside `ContinuousActionHead`; new config node `latent_head` in
`conf/vlm_configs.py`, `type: "continuous"` so the whole existing path applies.

```python
class LatentIntentHead(nn.Module):
    """h -> (mu, log_sigma) from the TRAINED split, not a fresh Linear."""
    def forward(self, hidden_states) -> {"mu": ..., "log_std": ...}
    def decode_action(self, c) -> np.ndarray        # (T, 3) chunk; see B
```

Differences from `ContinuousActionHead` that must be configured, not inherited:

| field | gaussian_head | latent_head | why |
|---|---|---|---|
| `action_space_dim` | 2 | **1024** | `c` is the readout width |
| `continuous_action_clip_low/high` | ±1.0 | **disabled** | clipping `c` is a silent distortion -- and the log-prob is then evaluated at the *clipped* action under the *unclipped* Gaussian |
| `init_log_std` | -0.5 | **from the checkpoint** | `sigma` is trained by the ELBO; re-initialising it discards the measurement the SFT run exists to make |

`sigma` arriving from the checkpoint rather than from `init_log_std` is the whole point. A
head that re-initialises `log_std` to a constant has thrown away the CVAE and is just a
Gaussian head wearing its weights.

**Phase-1 actor scope**: train the readout MLP + split only (~5.2M parameters), backbone and
LoRA frozen. The actor then ships between Ray actors at ~20 MB.

## B. The actuator seam: `c` is not what the env receives

`c` is an intent, 1024 floats. The env needs a `(20, 3)` chunk of poses. The decode is the
flow head -- a torch module holding the trained velocity field -- so it belongs on the
**VLM side**, not in the env actor. Sending `c` over Ray and decoding env-side would put
model weights in the simulator process and couple the two.

`rollout_core` currently does:

```python
action_to_env = action.astype(np.float32).reshape(-1)
```

The minimal additive change is a hook that is absent on every existing head:

```python
action_to_env = (self.policy_head.decode_action(action)
                 if hasattr(self.policy_head, "decode_action") else action)
```

One line; identity for `gaussian_head`; the discrete branch is not reached at all.

**The decode must be deterministic given `c`.** That is what
`FlowActionCodec.pin_flow_noise(seed)` is for: with `z_0` pinned and the ODE step count
fixed, `c -> chunk` is a deterministic, differentiable map. Without it the same `c` produces
a different chunk on every call, and the stored `old_log_prob` no longer describes the action
that was actually executed -- a corruption that shows up as a plausible-but-wrong ratio
rather than an error.

## C. `ContinuousObjectNavEnvActor`

New actor in `longnav/env/`, new `_target_` node in `conf/env_configs.py`, running under
`habitat_conda_env: vln`. It implements the same five methods `DummyEnvActor` does --
`reset`, `step`, `assign_shard`, `flush_logs_to_disk`, `is_exhausted` -- returning
`(rgb, state_dict)` with the established keys (`obs`, `reward`, `done`, `is_exhausted`,
`info`).

`step(chunk)` executes the chunk through the PID tracker for `gap` ticks and returns the
observation at the end of it. One env step is therefore one policy step, at `gap * dt`
seconds of simulated time -- the same currency the eval harness uses.

### Build on `habitat_physical_nav`, and reuse `objectnav_eval`

`vln` already imports `habitat`, `habitat_sim`, `torch`, `continuous_demos` and
`objectnav_eval`, and `continuous_demos` resolves to `/Projects/habitat_physical_nav/src`.
So the task layer is directly usable, and it should be used rather than reimplemented:
episode sources, screening with reason codes, per-dataset navmesh filename resolution, the
live-height geodesic adapter, the geometric budget and the metrics all live there and each
of them exists because getting it wrong produced a plausible wrong number rather than an
error. In particular the reward depends on them: "geodesic progress" is only meaningful when
measured on the right navmesh, against a goal on a reachable island.

Reward: the standard Habitat shaped reward (geodesic progress), **no terminal term**, with
termination by oracle or heuristic. This is settled -- the policy has no stop head, so a
`success` reward would optimise `--auto-stop-delay` rather than navigation.
`HabitatEnvConfig` already carries `explr_bonus` / `collision_penalty` / `fpstop_penalty`, so
the shaping-knob pattern to follow is there.

### What must NOT be built on

`longnav/env/sim.py` and `longnav/env/pure_pursuit.py` are unmaintained copies of
`continuous_demos.sim` (481 lines against 655) and `continuous_demos.pure_pursuit` (276
against 360), imported by nothing. The copy is missing `_build_cfg` / `_reconfigure_scene` --
scene switching, which in the old form failed *silently*, with render, navmesh and spawn snap
all still answering for the previous scene -- plus `get_obs`, `is_stationary`,
`is_tipped_over`. It also carries a bare `from pure_pursuit import ...` that resolves only if
`src/longnav/env/` happens to be on `sys.path`. This is the `continuous_demos` fork incident
in miniature; that one cost a corpus that could not be rebuilt, and is documented at length
in `habitat_physical_nav/CLAUDE.md`.

## D. The 1024-dimensional log-prob

`rollout_core` reduces with `np.sum(log_prob, axis=-1)`. Over 1024 dimensions the PPO ratio
is `exp(sum of 1024 log-ratios)`: variance scales with the dimension and clipping saturates
on nearly every sample. This is the one place the existing math may need extending, and it
should be a config switch on the reduction with `sum` as the default, so no existing run
changes.

Three options, all decidable **after** the effective-dim measurement the SFT run produces,
and therefore not a constraint on anything above:

1. **Mask** the dims with `KL ~ 0` -- they carry no information by construction, so fixing
   them at the prior mean and dropping them from the sum changes the distribution by `eps`.
   Requires the active set to be stable across states, which is measured, not assumed.
2. **Per-dimension ratios** instead of the product -- the same fix sequence-level RLHF
   applies at the token level. A biased surrogate that works in practice.
3. **Ratio-free**: advantage-weighted regression on `c`, or TD3-style with a critic `Q(o, c)`
   exploiting the differentiability that pinning `z_0` buys.

## Preservation

| path | what changes |
|---|---|
| discrete (`lm_head`) | nothing -- every addition is behind `type == "continuous"` or a new class |
| continuous (`gaussian_head`) | nothing -- `decode_action` is absent so the hook is identity; clip and `init_log_std` defaults unchanged; reduction defaults to `sum` |
| existing envs | nothing -- new actor is a new `_target_` node |
| RL math | nothing |

## Two landmines to keep in mind

* **`vlm_conda_env` defaults to `"longnav"`** in `config_schema.py`, but the environment on
  this machine is `longnav_vlm`. Ray actors fail at startup until this is set correctly.
* **`longnav` is a namespace package.** A bare `import longnav` resolves to whichever clone
  is first on `sys.path`; on this machine that is `/Projects/Codex_Projects/spatial_training`.
  `tests/conftest.py` pins it on `sys.path` and exports `PYTHONPATH` so Ray actors -- separate
  processes that re-import from scratch -- inherit the pin. Any new entry point needs the same
  bootstrap.

## Order of work

1. **A + B** -- head and actuator seam. Both are pure logic, testable with no simulator: the
   head against a stub hidden state, the seam by asserting `gaussian_head` is byte-identical
   through it.
2. **C** -- the env actor. Needs a simulator, so it is validated the way everything else in
   that repo is: run it and watch.
3. **D** -- only after the SFT run reports the effective dim.

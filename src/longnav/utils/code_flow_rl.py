"""Path A0: the discrete code as the WHOLE RL action, the flow decode as a fixed table.

docs/CODE_RL_PLAN_V2.md sections 2 and 9. The policy is the tempered categorical
`pi_T(c|h) = softmax(logits(h)/T)`; the executed chunk is a PRECOMPUTED lookup
`decoder(mixer(c, r=0), z0=0)` -- measured the best decode of every z0 regime
(V2 section 9.12) and independent of theta, so the code-only score function is exact
(section 9.1: `r(h)` live would be an uncredited theta-path; zeroed it is environment).

INTEGRATION BY CAPABILITY, deliberately: this head presents the chain head's exact surface
(`sample_chain_np` / `chain_log_prob_batch` / `decode_action`), with the "chain" being the
two code factors as float32. Every existing seam -- rollout storage, collate, the
old-log-prob recompute, the disable_adapter reference pass, rl_loss's chain branch, the
seam-gap gauge -- carries it UNCHANGED, and the discrete / Gaussian / latent / chain paths
are untouched (nothing here edits them; dispatch happens on the head instance alone).
Codes ride exactly in float32 (values < 2^24).

The a09 freeze is assumed for the first runs: `action_head_learning_rate: 0.0` freezes the
readout, the decoder, the mixer AND the code head, so the trainable surface is the LoRA
(`h`) alone -- and `disable_adapter()` therefore remains an EXACT SFT reference with no
frozen-copy machinery (V2 section 2.3 A6 deferred). Training the code head directly needs
that machinery first; `from_policy_head_config` refuses the combination it cannot honour.

RTC (A1, the "completer" actuator) is NOT built: a prefix raises loudly rather than being
ignored. Class merging (V2 section 2.7) is likewise deferred; the hook is `merge_radius`.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from longnav.utils.bin_codec import compose_chunk
from longnav.utils.flow_sde_policy import (
    FlowSDEHead, SDEConfig, _decoder_eval, load_flow_stack,
)


class CodeFlowHead(FlowSDEHead):
    """`hidden_states -> {"h"}`, plus the code sampler/scorer and the table actuator."""

    def __init__(self, readout, codec, gap: int, policy_temperature: float = 0.5,
                 overlay_modes_k: int = 0):
        # The SDEConfig is inert scaffolding (nothing here integrates stochastically);
        # constructed only so the parent's shape bookkeeping and module registration are
        # inherited unchanged.
        super().__init__(readout=readout, codec=codec, gap=gap,
                         sde=SDEConfig(n=1, noise_a=1.0))
        if getattr(codec, "code_head", None) is None or getattr(codec, "code_mixer", None) is None:
            raise ValueError(
                "CodeFlowHead needs a code-conditioned checkpoint: the codec carries no "
                "code_head/code_mixer (is `fm_code` in the checkpoint meta?)"
            )
        if float(policy_temperature) <= 0:
            raise ValueError(f"policy_temperature must be > 0, got {policy_temperature}")
        self.policy_temperature = float(policy_temperature)
        self.n_theta = int(codec.code_head.n_theta)
        self.vocab = int(codec.code_head.vocab)
        self._table: Optional[torch.Tensor] = None      # (V, n_ticks, 3) physical chunks
        #: >0 fills `last_mode_chunks` each sample with the top-K tempered-logit codes'
        #: TABLE ROWS, for the env's video overlay (docs/CODE_RL_PLAN_V2.md section 10).
        #: Free of decoder passes and of RNG -- pure argsort + lookup -- so unlike the
        #: denormalize overlay it cannot perturb the executed trajectory. 0 costs nothing.
        self.overlay_modes_k = int(overlay_modes_k)
        self.last_mode_chunks: Optional[np.ndarray] = None   # (K, n_ticks, 3) physical
        self.last_mode_probs: Optional[np.ndarray] = None    # (K,) under pi_T
        # THE ACTUATOR IS FROZEN AT THE PARAMETER LEVEL, not merely by an lr-0 group.
        # Two reasons, one of them load-bearing: (1) A0's estimator is exact only
        # because the actuator is theta-independent (docs/CODE_RL_PLAN_V2.md 9.1) --
        # requires_grad False makes that structural; (2) DDP raises "Expected to have
        # finished reduction" for REQUIRES-GRAD parameters that never receive grad, and
        # the code-only loss never touches the decoder or the mixer (the chain head's
        # density does, which is why the a09 lr-0 freeze sufficed there). This killed
        # every training step from [8/16] of the first launch's cycle 1. The code head
        # and readout stay requires-grad: the loss flows through them into the LoRA.
        for mod in (codec.decoder, codec.code_mixer):
            for p in mod.parameters():
                p.requires_grad_(False)

    # -- the fixed actuator ------------------------------------------------------------
    @torch.no_grad()
    def _ensure_table(self) -> torch.Tensor:
        """`decoder(mixer(c, r=0), z0=0)` for every code, built once per device.

        r = 0 and z0 = 0 are THE actuator definition (V2 sections 9.1/9.12), not a
        convenience: both make the executed chunk independent of theta, which is what
        keeps the code-only estimator exact. Not a buffer: derived from frozen weights,
        never checkpointed."""
        dev = next(self.parameters()).device
        if self._table is not None and self._table.device == dev:
            return self._table
        mixer = self.codec.code_mixer
        idx = torch.arange(self.vocab, device=dev)
        cx, ct = idx // self.n_theta, idx % self.n_theta
        code = torch.cat([mixer.emb_xy(cx), mixer.emb_theta(ct)], dim=-1)
        ctx = torch.cat([code, code.new_zeros(self.vocab,
                                              mixer.n_reserved * mixer.d_model)], dim=-1)
        with _decoder_eval(self.codec.decoder):
            z0 = torch.zeros(self.vocab, self.n_ticks, self.n_dims, device=dev)
            diffs = self.codec.generate(ctx.float(), noise=z0)
        self._table = compose_chunk(diffs).float()
        return self._table

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        """Tempered logits -- THE policy's, used identically by sampler and scorer."""
        return self.codec.code_head.logits(h.float()) / self.policy_temperature

    # -- rollout: sample ---------------------------------------------------------------
    @torch.no_grad()
    def sample_chain_np(self, h: np.ndarray, prefix: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """One decision: `(code_pair, positions, log pi_T(c|h), executed chunk)`.

        The "chain" is `[c_xy, c_theta]` float32 -- stored for the ratio exactly as the
        chain head stores its latents. `positions` is a single zero, present only so the
        buffer/collate plumbing (which the chain head shaped) carries unchanged; the
        scorer ignores it. `force_ode` (interleaved eval's deterministic switch) maps to
        ARGMAX -- the fixed decode, mirroring the chain head's eval-ODE convention.
        """
        if prefix is not None:
            raise RuntimeError(
                "CodeFlowHead is the A0 table actuator and cannot honour an RTC prefix; "
                "the A1 completer is not built (docs/CODE_RL_PLAN_V2.md section 4)."
            )
        dev = next(self.parameters()).device
        hh = torch.as_tensor(np.asarray(h, np.float32), device=dev).reshape(1, -1)
        logits = self._logits(hh)
        logp = torch.log_softmax(logits, dim=-1)
        if self.force_ode:
            idx = logits.argmax(dim=-1)
        else:
            probs = logp.exp()
            if self._gen is not None and self._gen.device != probs.device:
                # `seed()` runs at build time, BEFORE the worker moves the head to its
                # device (vlm_worker builds then `.to(device)`), so a seeded generator is
                # typically CPU while `probs` is CUDA -- and torch.multinomial requires
                # them to match. Draw on the GENERATOR's device: 1,600 floats, cost nil,
                # and the seeded stream keeps its state instead of being re-created.
                idx = torch.multinomial(probs.to(self._gen.device), 1,
                                        generator=self._gen).squeeze(-1).to(probs.device)
            else:
                idx = torch.multinomial(probs, 1, generator=self._gen).squeeze(-1)
        lp = float(logp[0, idx].item())
        cx, ct = int(idx.item()) // self.n_theta, int(idx.item()) % self.n_theta
        table = self._ensure_table()
        chunk = table[int(idx.item()), : self.gap].cpu().numpy()
        if self.overlay_modes_k > 0:
            # Video-overlay modes. CONTRACT (the env draws by it): row 0 is the SELECTED
            # (executed) code's full table chunk; rows 1.. are the top-K alternatives by
            # pi_T, the selected code excluded so no line is drawn twice. Under sampling
            # the executed code is not the top-probability one, so without row 0 the
            # chosen path would be visually anonymous. The rollout reads-and-clears
            # `last_mode_chunks` and ships it on a dedicated step kwarg; the dynamics
            # and the observation never see it.
            k = min(self.overlay_modes_k, self.vocab)
            top = torch.topk(logp[0], min(k + 1, self.vocab))
            sel = int(idx.item())
            alts = [int(j) for j in top.indices.tolist() if j != sel][:k]
            order = [sel] + alts
            self.last_mode_chunks = table[torch.tensor(order, device=table.device)].cpu().numpy()
            self.last_mode_probs = logp[0, torch.tensor(order, device=logp.device)].exp().cpu().numpy()
        return (np.array([cx, ct], dtype=np.float32),
                np.zeros(1, dtype=np.int64),
                lp,
                chunk)

    # -- the actuator seam -------------------------------------------------------------
    @torch.no_grad()
    def decode_action(self, action: np.ndarray) -> np.ndarray:
        """Stored code pair -> the `(gap, 3)` chunk the environment executes."""
        a = np.asarray(action, np.float32).reshape(-1)
        idx = int(round(float(a[0]))) * self.n_theta + int(round(float(a[1])))
        return self._ensure_table()[idx, : self.gap].cpu().numpy()

    # -- training: score stored codes under current theta ------------------------------
    def chain_log_prob_batch(self, h: torch.Tensor, chains: torch.Tensor,
                             positions: torch.Tensor,
                             prefix_actions: Optional[torch.Tensor] = None,
                             prefix_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`(B, S)` log pi_T at the STORED codes, differentiable through `h`.

        The code head is frozen under the a09 recipe, so the only trainable path is
        `h` <- LoRA -- exactly the freeze-head chain arm's structure. `positions` and the
        RTC kwargs are accepted for seam parity and must be absent/ignored respectively.
        """
        if prefix_actions is not None or prefix_len is not None:
            raise RuntimeError("CodeFlowHead cannot score an RTC prefix (A1 not built)")
        B, S = chains.shape[0], chains.shape[1]
        rows = B * S
        codes = chains.reshape(rows, -1)[:, :2].round().long()
        idx = codes[:, 0] * self.n_theta + codes[:, 1]
        logits = self._logits(h.reshape(rows, -1))
        logp = torch.log_softmax(logits.float(), dim=-1)
        return logp.gather(1, idx[:, None]).squeeze(1).reshape(B, S)

    # -- run-log honesty ---------------------------------------------------------------
    def describe(self) -> str:
        return (f"code-flow head (A0 table): vocab={self.vocab} T={self.policy_temperature} "
                f"gap={self.gap} actuator=decoder(mixer(c, r=0), z0=0) precomputed "
                f"(eval/deploy = argmax)")

    # -- construction ------------------------------------------------------------------
    @classmethod
    def from_policy_head_config(cls, cfg: Dict[str, Any], input_dim: int,
                                dtype: torch.dtype) -> "CodeFlowHead":
        ckpt = cfg.get("checkpoint_dir")
        if not ckpt:
            raise ValueError("code_flow head requires `checkpoint_dir` (a code-conditioned "
                             "SFT checkpoint)")
        readout, codec, _ = load_flow_stack(ckpt, dtype=dtype)
        head = cls(readout=readout, codec=codec, gap=int(cfg["gap"]),
                   policy_temperature=float(cfg.get("policy_temperature", 0.5)),
                   overlay_modes_k=int(cfg.get("overlay_modes_k", 0) or 0))
        seed = cfg.get("code_seed")
        if seed is not None:
            head.seed(int(seed))
        if float(cfg.get("merge_radius", 0.0) or 0.0) > 0.0:
            raise NotImplementedError(
                "class merging (V2 section 2.7) is not built; set merge_radius: 0")
        _ = input_dim
        return head

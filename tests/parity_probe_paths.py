"""Probe readout parity: incremental per-turn forward vs packed post-episode recompute.

The state probe reads one hidden per turn at `worker_value_readout_offset` from the
turn's logit index (both persisted in state_probe_config.json by the value trainer;
the offset absorbs shift_left and the template content length). Two things can go
wrong silently: the index arithmetic reads a neighbouring token, or the sparse remap
diverges between the incremental path (per-turn forward, kv-style accumulation) and
the packed path (one full recompute over saved embeds -- what the RL trainer and the
mini run's value stream actually consume). This script drives BOTH paths over the
same conversation and compares the probe outputs, the same discipline as
`chain/rollout_seam_gap` for h.

What is asserted (not just printed):
  1. every recorded readout position lands on the token id pinned in the checkpoint;
  2. packed-vs-incremental distance expectations agree within SEAM_TOL_M;
  3. packed-vs-incremental value expectations agree within SEAM_TOL_V.

Run on GPU 4 (0-3 carry the cotrain-v4 SFT; 5-7 carry evals):
    /workspace/conda/envs/longnav_vlm/bin/python tests/parity_probe_paths.py \
        --ckpt dump/pose_injection/run_cotrain_v4_probe_mix/checkpoint-200
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys

sys.path.insert(0, "/Projects/spatial_training/src")
sys.path.insert(0, "/Projects/spatial_training")

import json

import numpy as np
import torch
import yaml
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="SFT checkpoint dir with state_probe.pt")
ap.add_argument("--n-steps", type=int, default=10)
ap.add_argument("--seam-tol-m", type=float, default=0.10,
                help="max |packed - incremental| distance expectation, meters")
ap.add_argument("--seam-tol-v", type=float, default=0.10,
                help="max |packed - incremental| value expectation")
args = ap.parse_args()
CKPT = os.path.abspath(args.ckpt)

RUN = yaml.safe_load(open(
    "/Projects/spatial_training/dump/flow_rl/flow_sde_gamma097_warm64/config.yaml"))

# ---------------- conversation material: in-distribution corpus row -------------------
from datasets import load_from_disk

row = load_from_disk(
    "/Projects/data/v2_25hz_obs2.5hz/formatted_nopose_dist097")["train"][0]
goal = row["goal_text"]
imgs = [Image.open(p).convert("RGB") for p in row["images"][: args.n_steps]]
print(f"goal={goal!r}  frames={len(imgs)}", flush=True)

# ---------------- worker built the way the mini run builds it -------------------------
from longnav.utils.rollout_core import RLWorker, substitute_convo_template
from longnav.utils.state_probe import (STATE_PROBE_CONFIG_FILE, load_state_probe)
from longnav.utils.vlm_worker import StateProbeValueAdapter

vlm_kwargs = dict(RUN["vlm"])
vlm_kwargs["policy_head"] = dict(vlm_kwargs["policy_head"]) | {"checkpoint_dir": CKPT}
vlm_kwargs["merge_adapter_dir"] = os.path.join(CKPT, "adapter")
rollout = dict(RUN["rollout"])
w = RLWorker(rollout_config=rollout, **vlm_kwargs)
w._ensure_continuous_action_head()
w.model.eval()
hidden_size = w.language_model.config.hidden_size

probe = load_state_probe(CKPT, input_dim=hidden_size)
probe.eval()
meta = json.loads(open(os.path.join(CKPT, STATE_PROBE_CONFIG_FILE)).read())
w_off = int(meta["worker_value_readout_offset"])
pin = int(meta["probe_token_id"])
w.model.value_head = StateProbeValueAdapter(probe).to(w.model.device)
w.value_readout_offset = w_off
w.probe_expected_token_id = pin
print(f"probe attached: worker offset {w_off}, pinned token {pin}", flush=True)

# ---------------- drive the conversation; capture per-turn hiddens --------------------
turn_hiddens = []       # (kept_len_k, H) float32, one per turn
w.language_model.register_forward_hook(
    lambda m, i, o: turn_hiddens.append(
        o.last_hidden_state.detach().float().cpu().squeeze(0)))

dense_turn_ends = []    # cumulative dense length after each turn
w.reset()
msgs = substitute_convo_template(rollout["convo_start_template"],
                                 {"instr_or_goal": goal} | rollout)
with torch.no_grad():
    for k, img in enumerate(imgs):
        w.infer_probs(images=[img], messages=msgs,
                      temperature=rollout.get("temperature", 1.0),
                      pos_id_kwargs={"mode": "standard"})
        dense_turn_ends.append(int(w.cumulative_inputs["input_ids"].shape[1]))
        msgs = substitute_convo_template(rollout["convo_turn_template"],
                                         {"action": "", "step": k + 1})

dense_ids = w.cumulative_inputs["input_ids"][0].clone()
vli = [int(i) for i in w.value_logit_indices]
assert len(vli) == len(imgs), (len(vli), len(imgs))

# assertion 1: pinned token identity at every recorded position
toks = [int(dense_ids[i]) for i in vli]
assert all(t == pin for t in toks), f"readout token ids {toks} != pinned {pin}"
print(f"token pin OK at {len(vli)} positions", flush=True)

# ---------------- incremental probe series (per-turn hiddens) -------------------------
keep = w.seq_keep_mask.bool()
inc_d, inc_v, inc_p = [], [], []
# separate CPU instance: the hooked per-turn hiddens are on CPU, while `probe`
# was moved to cuda inside the adapter
probe_cpu = load_state_probe(CKPT, input_dim=hidden_size)
probe_cpu.eval()
with torch.no_grad():
    for k, di in enumerate(vli):
        s_dense = 0 if k == 0 else dense_turn_ends[k - 1]
        assert s_dense <= di < dense_turn_ends[k], (k, s_dense, di)
        turn_keep = keep[s_dense:dense_turn_ends[k]]
        assert bool(turn_keep[di - s_dense]), "probe token dropped by sparsifier"
        local = int(turn_keep[: di - s_dense + 1].sum()) - 1
        h = turn_hiddens[k][local].unsqueeze(0)
        dl = probe_cpu.distance_head(h.to(probe_cpu.distance_head.dtype))
        inc_d.append(float(probe_cpu.distance_head.expectation(dl)))
        inc_p.append(float(probe_cpu.distance_head.p_within(dl, 1.0)))
        vl = probe_cpu.value_head(h.to(probe_cpu.value_head.dtype))
        inc_v.append(float(probe_cpu.value_head.expectation(vl)))

# ---------------- packed probe series (the RL trainer's path) -------------------------
embeds = w._pack_embeds()
assert "value_logits_to_keep" in embeds, "packed embeds lack value readout indices"
with torch.no_grad():
    _, values = w._forward_embeds(
        {k: v for k, v in embeds.items()}, True)
    ad = w.model.value_head
    pk_v = ad.value(values).squeeze().float().cpu().numpy().reshape(-1)
    pk_d = ad.last_distance_m.squeeze().numpy().reshape(-1)
    pk_p = ad.last_p_stop.squeeze().numpy().reshape(-1)

# ---------------- seam comparison -----------------------------------------------------
inc_d, inc_v, inc_p = map(np.asarray, (inc_d, inc_v, inc_p))
print("\nstep |  d_inc   d_pack  |dd|   |  v_inc   v_pack  |dv|   | p_inc  p_pack")
for k in range(len(imgs)):
    print(f"{k:4d} | {inc_d[k]:7.3f} {pk_d[k]:7.3f} {abs(inc_d[k]-pk_d[k]):6.3f} | "
          f"{inc_v[k]:7.3f} {pk_v[k]:7.3f} {abs(inc_v[k]-pk_v[k]):6.3f} | "
          f"{inc_p[k]:6.3f} {pk_p[k]:6.3f}")
dd, dv = np.abs(inc_d - pk_d).max(), np.abs(inc_v - pk_v).max()
print(f"\nmax seam: distance {dd:.4f} m (tol {args.seam_tol_m}), "
      f"value {dv:.4f} (tol {args.seam_tol_v})")
assert dd <= args.seam_tol_m, f"distance seam {dd:.4f} m exceeds {args.seam_tol_m}"
assert dv <= args.seam_tol_v, f"value seam {dv:.4f} exceeds {args.seam_tol_v}"
print("PARITY OK")

"""Offline value/distance-head evaluation on a pinned on-policy episode set.

Replaces the mini-RL-run instrument for scoring checkpoints: the critic at
checkpoint N is backbone+heads jointly, i.e. a property of a forward pass, so it
is measured by replaying FIXED conversations (frames + goal from the gamma097
rollout archive, decoded once by mine_rollout_frames --episodes-file) through
the worker's packed forward -- no simulator, no ray, no training. ~4 min per
checkpoint on one GPU, and every checkpoint sees byte-identical inputs, so
numbers are paired across checkpoints by construction.

Reported per checkpoint (dump/eval_system/value_offline/<ckpt>.json):
  corr(v_hat, G), value MSE, Var(G - v_hat)          -- the critic as baseline
  Var(G - kernel(t)) / kernel MSE (sigma 40, 'start') -- the incumbent, same set
  Var(G - mean)                                       -- naive
  dist MAE / bias (m), p_stop near (d<=1m) vs far     -- the stop side

Labels: G_t recomputed from the rollout's own sequence.json distance series via
distance_return_targets (gamma 0.97, clip 0.75) by the frame miner; d_t is the
env's own per-step geodesic. Same units as the RL loop's returns.

Usage (GPU 4; longnav_vlm env):
  CUDA_VISIBLE_DEVICES=4 /workspace/conda/envs/longnav_vlm/bin/python \
      data_scripts/eval_value_heads_offline.py \
      --ckpt dump/pose_injection/run_cotrain_v4_probe_mix/checkpoint-800
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, "/Projects/spatial_training/src")
sys.path.insert(0, "/Projects/spatial_training")

import numpy as np

BASE = "/Projects/spatial_training"
SET_DIR = f"{BASE}/dump/eval_system/value_offline"

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--frames-dir", default=f"{SET_DIR}/frames")
ap.add_argument("--episode-paths", default=f"{SET_DIR}/episode_set64_paths.txt")
ap.add_argument("--kernel-sigma", type=float, default=40.0)
ap.add_argument("--stop-radius", type=float, default=1.0)
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()
CKPT = os.path.abspath(args.ckpt)

import torch
import yaml
from PIL import Image

from longnav.utils.rollout_core import RLWorker, substitute_convo_template
from longnav.utils.state_probe import STATE_PROBE_CONFIG_FILE, load_state_probe
from longnav.utils.vlm_worker import StateProbeValueAdapter

RUN = yaml.safe_load(open(f"{BASE}/dump/flow_rl/flow_sde_gamma097_warm64/config.yaml"))
rollout = dict(RUN["rollout"])

# ---------------- labels: one row per (episode, step) from the miner ------------------
rows_by_ep = collections.defaultdict(dict)
for line in open(f"{args.frames_dir}/labels.jsonl"):
    r = json.loads(line)
    rows_by_ep[r["episode_tag"]][r["step"]] = r

ep_paths = [l.strip() for l in open(args.episode_paths) if l.strip()]
if args.limit:
    ep_paths = ep_paths[: args.limit]

# ---------------- model, built as the RL worker builds it -----------------------------
vlm_kwargs = dict(RUN["vlm"])
vlm_kwargs["policy_head"] = dict(vlm_kwargs["policy_head"]) | {"checkpoint_dir": CKPT}
vlm_kwargs["merge_adapter_dir"] = os.path.join(CKPT, "adapter")
w = RLWorker(rollout_config=rollout, **vlm_kwargs)
w._ensure_continuous_action_head()
w.model.eval()
probe = load_state_probe(CKPT, input_dim=w.language_model.config.hidden_size)
probe.eval()
meta = json.loads(open(os.path.join(CKPT, STATE_PROBE_CONFIG_FILE)).read())
w.model.value_head = StateProbeValueAdapter(probe).to(w.model.device)
w.model.value_head.stop_radius_m = args.stop_radius
w.value_readout_offset = int(meta["worker_value_readout_offset"])
w.probe_expected_token_id = int(meta["probe_token_id"])
print(f"ckpt {CKPT}  offset {w.value_readout_offset}  pin {w.probe_expected_token_id}",
      flush=True)

# ---------------- replay each pinned conversation through the packed forward ----------
V, D, P, G, DT, T = [], [], [], [], [], []   # flat per-step series across episodes
n_done = 0
for ep_path in ep_paths:
    # same sanitization as mine_rollout_frames
    tag = os.path.basename(ep_path).replace("@", "_").replace(":", "_").replace("#", "_")
    srows = rows_by_ep.get(tag)
    if not srows:
        print(f"  SKIP {tag}: no mined rows"); continue
    summary = json.loads(open(os.path.join(ep_path, "summary.json")).read())
    steps = sorted(srows)
    # conversation must be CONTIGUOUS from step 0 -- stop at the first gap
    contig = []
    for k, st in enumerate(steps):
        if st != k: break
        contig.append(st)
    if len(contig) < 10:
        print(f"  SKIP {tag}: only {len(contig)} contiguous steps"); continue
    imgs = [Image.open(srows[k]["image"]).convert("RGB") for k in contig]

    w.reset()
    msgs = substitute_convo_template(rollout["convo_start_template"],
                                     {"instr_or_goal": summary["goal"]} | rollout)
    with torch.no_grad():
        for k, img in enumerate(imgs):
            w.infer_probs(images=[img], messages=msgs,
                          temperature=rollout.get("temperature", 1.0),
                          pos_id_kwargs={"mode": "standard"})
            msgs = substitute_convo_template(rollout["convo_turn_template"],
                                             {"action": "", "step": k + 1})
        embeds = w._pack_embeds()
        _, values = w._forward_embeds(dict(embeds), True)
        ad = w.model.value_head
        v = ad.value(values).squeeze().float().cpu().numpy().reshape(-1)
        d = ad.last_distance_m.squeeze().numpy().reshape(-1)
        p = ad.last_p_stop.squeeze().numpy().reshape(-1)
        ad.last_distance_m = ad.last_p_stop = None
    assert len(v) == len(contig), (len(v), len(contig))
    for k in contig:
        r = srows[k]
        if r["return"] is None:
            continue
        V.append(float(v[k])); D.append(float(d[k])); P.append(float(p[k]))
        G.append(float(r["return"]))
        DT.append(float(r["distance"]) if r["distance"] is not None else np.nan)
        T.append(k)
    n_done += 1
    if n_done % 16 == 0:
        print(f"  {n_done} episodes replayed", flush=True)

V, D, P, G, DT, T = map(np.asarray, (V, D, P, G, DT, T))
print(f"{n_done} episodes, {len(G)} steps", flush=True)

# ---------------- statistics ----------------------------------------------------------
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))

# time-kernel baseline fitted on this same set (sigma 40, 'start' alignment,
# no leave-one-out -- mirrors the training configuration time_loto=false)
tt = T.astype(np.float64)
K = np.exp(-0.5 * ((tt[:, None] - tt[None, :]) / args.kernel_sigma) ** 2)
kern = (K @ G) / K.sum(1)

fin = np.isfinite(DT)
near = fin & (DT <= args.stop_radius)
res = {
    "ckpt": CKPT, "episodes": n_done, "steps": int(len(G)),
    "corr_value_return": corr(V, G),
    "value_mse": float(((V - G) ** 2).mean()),
    "adv_var_critic": float((G - V).var()),
    "adv_var_kernel": float((G - kern).var()),
    "kernel_mse": float(((kern - G) ** 2).mean()),
    "adv_var_naive": float(G.var()),
    "dist_mae_m": float(np.abs(D[fin] - DT[fin]).mean()),
    "dist_bias_m": float((D[fin] - DT[fin]).mean()),
    "dist_corr": corr(D[fin], DT[fin]),
    "p_stop_near": float(P[near].mean()) if near.any() else None,
    "p_stop_far": float(P[fin & ~near].mean()),
    "n_near": int(near.sum()),
}
os.makedirs(SET_DIR, exist_ok=True)
out = os.path.join(SET_DIR, os.path.basename(CKPT.rstrip("/")) + ".json")
json.dump(res, open(out, "w"), indent=2)
print(json.dumps(res, indent=2))
print(f"-> {out}")

"""h/chunk parity: the RL rollout path (VLMWorker+FlowSDEHead) vs the harness path
(VectorRolloutPolicy), same images, same goal.

h is computed BEFORE any sampling and actions never enter the text, so h parity is a
deterministic check with no SDE in it. Chunk-level: decode both h's through one fixed-z0
ODE (isolates the h difference in action units), then sweep (a, N) drawing SDE chains and
comparing each draw against the pure-ODE chunk integrated from ITS OWN z0 (chain block 0)
-- the a->0 limit of that deviation must vanish or the sampler has a bug.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
sys.path.insert(0, "/Projects/spatial_training/src")
sys.path.insert(0, "/Projects/spatial_training")

import numpy as np
import torch
import yaml
from PIL import Image

CKPT = "/Projects/spatial_training/dump/pose_injection/run_cotrain_v3_nopose_mix/checkpoint-12000"
RUN = yaml.safe_load(open("/Projects/spatial_training/dump/flow_rl/flow_sde_train_minimal/config.yaml"))
N_STEPS = 10

# ---------------- images + goal from the SFT corpus (in-distribution, no simulator) ----
from datasets import load_from_disk
row = load_from_disk("/Projects/spatial_training/data/v2_25hz_obs2.5hz/formatted_nopose")["train"][0]
goal = row["goal_text"]
imgs = [Image.open(p).convert("RGB") for p in row["images"][:N_STEPS]]
print(f"goal={goal!r}  frames={len(imgs)}  {imgs[0].size}", flush=True)

# ---------------- Path A: the RL worker, built as the run builds it -------------------
from omegaconf import OmegaConf
from longnav.utils.rollout_core import RLWorker, substitute_convo_template

vlm_kwargs = dict(RUN["vlm"])
rollout = dict(RUN["rollout"])
if os.environ.get("PARITY_ECHO"):
    # The discrete-convention turn shape: a LEADING assistant echo re-supplies the
    # `____**<|im_end|>\n` that infer_step's crop removed from the previous turn -- the
    # multi-prefix crop branch starts at postfix_starts[0]-1, which is the echo's content.
    rollout["convo_turn_template"] = [
        {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Observation $step:"},
            {"type": "image"},
            {"type": "text", "text": "Action:"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]},
    ]
    print("USING LEADING-ECHO TURN TEMPLATE", flush=True)
wA = RLWorker(rollout_config=rollout, **vlm_kwargs)
tcfg = OmegaConf.create(RUN["training"])
wA._ensure_continuous_action_head()
wA._setup_peft(tcfg)
wA.load_checkpoint(RUN["training"]["checkpoint"], False, False)   # eval-branch light path
wA.model.eval()
print("path A loaded:", wA.model.action_head.describe(), flush=True)

hA = []
wA.reset()
msgs = substitute_convo_template(rollout["convo_start_template"], {"instr_or_goal": goal} | rollout)
with torch.no_grad():
    for k, img in enumerate(imgs):
        po, _, _ = wA.infer_probs(images=[img], messages=msgs,
                                  temperature=rollout.get("temperature", 1.0),
                                  pos_id_kwargs={"mode": "standard"})
        hA.append(np.asarray(po["h"], dtype=np.float32).reshape(-1))
        msgs = substitute_convo_template(rollout["convo_turn_template"],
                                         {"action": "", "step": k + 1})
print(f"path A: {len(hA)} h vectors, dim {hA[0].shape}", flush=True)

# ---------------- Path B: the harness policy, built as the eval builds it -------------
from longnav.utils.flow_rollout import load_flow_policy
from longnav.utils.vector_rollout import RolloutConfig

# The 0.663 eval's exact loader: results.json records policy_backend=flow_rollout, whose
# __init__ builds RolloutConfig(device, merge_lora, max_context_tokens=0) and calls this.
polB = load_flow_policy(CKPT, RolloutConfig(device="cuda", merge_lora=True))
hB = []
hook = polB.model.head.register_forward_hook(
    lambda m, i, o: hB.append(o.detach().float().cpu().numpy().reshape(-1)))
chunksB = []
polB.reset(goal_text=goal)
with torch.no_grad():
    for img in imgs:
        chunksB.append(polB.step(img))
hook.remove()
print(f"path B: {len(hB)} h vectors", flush=True)

# ---------------- h parity ------------------------------------------------------------
print("\n=== h parity (per step) ===")
for k, (a, b) in enumerate(zip(hA, hB)):
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    print(f"step {k}: cos={cos:.6f}  relL2={rel:.4f}  |hA|={np.linalg.norm(a):.2f} |hB|={np.linalg.norm(b):.2f}")

# ---------------- chunk-space impact of the h difference (shared fixed-z0 ODE) --------
head = wA.model.action_head
dev = next(head.parameters()).device
K, dt = head.K, -1.0 / head.K

def ode_chunk(h_vec, z0):
    ctx = torch.as_tensor(h_vec, dtype=torch.float32, device=dev).reshape(1, -1)
    x = z0.clone()
    with torch.no_grad():
        for k in range(K):
            t_k = torch.full((1,), 1.0 + k * dt, device=dev)
            x = x + dt * head.codec.decoder(ctx, x, t_k)
    return head._compose(x)          # (gap, 3) real units

g = torch.Generator(device=dev); g.manual_seed(123)
z0 = torch.randn(1, head.n_ticks, head.n_dims, device=dev, dtype=torch.float32, generator=g)
print("\n=== chunk impact of h diff (same z0, pure ODE) ===")
for k in (0, len(hA) // 2, len(hA) - 1):
    ca, cb = ode_chunk(hA[k], z0), ode_chunk(hB[k], z0)
    dxy = np.linalg.norm(ca[:, :2] - cb[:, :2], axis=1)
    dth = np.abs(ca[:, 2] - cb[:, 2])
    print(f"step {k}: final-tick |dxy|={dxy[-1]*100:.2f} cm  |dth|={np.degrees(dth[-1]):.2f} deg  "
          f"(max over ticks {dxy.max()*100:.2f} cm / {np.degrees(dth.max()):.2f} deg)")

# ---------------- (a, N) sweep: SDE draw vs the ODE chunk from its own z0 -------------
from longnav.utils.flow_sde_policy import FlowSDEHead, SDEConfig
print("\n=== SDE deviation vs (a, N): draw vs ODE-from-same-z0, 16 draws on h[0] ===")
for a in (1e-6, 0.05, 0.15):
    for n in (1, 3):
        h2 = FlowSDEHead(head.readout, head.codec, head.gap, SDEConfig(n=n, noise_a=a)).to(dev)
        h2.seed(7)
        dxyf, dthf = [], []
        for _ in range(16):
            chain, pos, lp, chunk = h2.sample_chain_np(hA[0])
            z0_i = torch.as_tensor(chain[: head.block], device=dev).reshape(1, head.n_ticks, head.n_dims)
            ref = ode_chunk(hA[0], z0_i)
            dxyf.append(np.linalg.norm(chunk[-1, :2] - ref[-1, :2]))
            dthf.append(abs(chunk[-1, 2] - ref[-1, 2]))
        print(f"a={a:<6} N={n}: final-tick |dxy| mean={np.mean(dxyf)*100:6.2f} cm "
              f"max={np.max(dxyf)*100:6.2f} cm   |dth| mean={np.degrees(np.mean(dthf)):5.2f} deg")
print("\ndone")

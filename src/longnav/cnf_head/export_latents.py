"""Export the latent dataset the flow stage consumes.

    <longnav_vlm python> -m src.longnav.cnf_head.export_latents --ckpt <run>/ae.pt

------------------------------------------------------------------------------
Format -- one .npz per split, plus a shared manifest.json
------------------------------------------------------------------------------
    z              (n_chunks, latent_dim) float32
                   encoder MEANS, in the decoder's own coordinates. Feed these to
                   `model.decode(z)` unchanged; do NOT standardise them first.
    episode_ptr    (n_episodes + 1,) int64
                   z[ptr[e]:ptr[e+1]] are episode e's chunks, in observation order.
                   So the k-th observation of dataset row `row_index[e]` pairs with
                   z[ptr[e] + k] -- that is the (obs, z) join for the conditional flow.
    row_index      (n_episodes,) int64      index into the HF split
    episode_id     (n_episodes,) str
    recon_mse      (n_chunks,) float32      per-chunk round-trip MSE, raw units.
                   Useful as a sanity filter; nothing is filtered here.

    manifest.json  latent_dim, noise_std, per-dim latent mean/std, channel scale,
                   the checkpoint path, and the gate verdict this export was made
                   under.

------------------------------------------------------------------------------
Three things the flow stage must not get wrong
------------------------------------------------------------------------------
1. **Train on `z + noise_std * eps`, not on `z`.** The encoder is deterministic, so
   the exported means put an ATOM in latent space: every fully-stopped chunk lands
   on essentially the same point. A continuous flow cannot fit an atom -- that is
   the exact problem this design exists to solve, and training on the bare means
   would reintroduce it one level up. The autoencoder was trained with that noise
   injected, the decoder's flat region is sized to swallow a ball of radius
   ~noise_std, and the gate was scored through the noisy round trip. The noise is
   part of the representation, not a training trick.

2. **Latent scale normalisation is already done.** The autoencoder is trained with a
   moment penalty holding the aggregate latent at per-dimension mean 0 / std 1, so
   `z` needs no LDM-style rescaling. `latent_mean`/`latent_std` are in the manifest
   anyway -- as a check that this held, not as something to apply.

3. **`decode` is the actuator, not part of the policy.** It is non-injective by
   design; there is no valid `log p(action | obs)`. RL operates on `z`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import metrics as M
from .models import ActionAutoencoder

OUT = D.REPO / "dump" / "cnf_head" / "autoencoder"


@torch.no_grad()
def encode_split(model, diffs, device="cuda", batch=65536):
    zs, mses = [], []
    for i in range(0, len(diffs), batch):
        x = torch.from_numpy(diffs[i:i + batch]).to(device)
        z = model.encode(x)
        rec = model.decode(z)
        zs.append(z.cpu().numpy().astype(np.float32))
        mses.append(((rec - x) ** 2).mean(dim=(1, 2)).cpu().numpy().astype(np.float32))
    return np.concatenate(zs), np.concatenate(mses)


def verify_alignment(split: str, episode_id: np.ndarray) -> np.ndarray:
    """The cache was built over the whole split in dataset order; confirm that, and
    return the row index for each episode. A silent misalignment here would pair
    every observation with the wrong latent and be near-impossible to debug later."""
    from datasets import load_from_disk

    ds = load_from_disk(str(D.DATASET))[split].select_columns(["episode_id"])
    ids = [str(v) for v in ds["episode_id"]]
    if len(ids) == len(episode_id) and all(
            a == b for a, b in zip(ids, [str(v) for v in episode_id])):
        return np.arange(len(ids), dtype=np.int64)
    lookup = {}
    for i, v in enumerate(ids):
        lookup.setdefault(v, i)
    return np.asarray([lookup[str(v)] for v in episode_id], dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    p.add_argument("--out", type=str, default=str(OUT / "latents"))
    p.add_argument("--device", type=str, default="cuda")
    a = p.parse_args()

    ckpt = Path(a.ckpt)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    model, ck = ActionAutoencoder.from_checkpoint(ckpt, map_location=a.device)
    model = model.to(a.device).eval()

    manifest = {
        "checkpoint": str(ckpt),
        "latent_dim": model.latent_dim,
        "noise_std": model.noise_std,
        "channel_scale": model.scale.cpu().tolist(),
        "channels": list(D.CHANNELS),
        "chunk_len": model.chunk_len,
        "action_space": "per-tick body-frame SE(2) differentials [forward, strafe, "
                        "dtheta] in metres/radians, 10 ticks per chunk at 20 Hz",
        "train_on": "z + noise_std * eps,  eps ~ N(0, I)",
        "decode": "longnav.cnf_head.models.ActionAutoencoder.from_checkpoint(ckpt)"
                  ".decode(z) -> (B, 10, 3) raw differentials",
        "splits": {},
    }
    gate = ckpt.parent / "gate.json"
    if gate.exists():
        g = json.loads(gate.read_text())
        manifest["gate"] = {"noisy": g["noisy"]["verdict"], "clean": g["clean"]["verdict"],
                            "volume": g["volume"], "tau_raw_units": g["tau_raw_units"]}

    for split in a.splits:
        diffs, ptr, eid = D.load_cache(split)
        z, mse = encode_split(model, diffs, device=a.device)
        row = verify_alignment(split, eid)
        path = outdir / f"latents_{split}.npz"
        np.savez(path, z=z, episode_ptr=ptr, row_index=row,
                 episode_id=np.asarray([str(v) for v in eid]), recon_mse=mse)
        manifest["splits"][split] = {
            "path": str(path), "chunks": int(len(z)), "episodes": int(len(eid)),
            "latent_mean": z.mean(0).tolist(), "latent_std": z.std(0).tolist(),
            "recon_mse_mean": float(mse.mean()),
            "recon_rmse_mean_m": float(np.sqrt(mse.mean())),
            "data_atom_rates": M.data_atom_rates(diffs),
        }
        print(f"-> {path}  z{z.shape}  rmse {1000 * np.sqrt(mse.mean()):.2f} mm")

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("->", outdir / "manifest.json")


if __name__ == "__main__":
    main()

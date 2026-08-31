"""Drive MSHabEnvActor directly (no Ray) with random actions; save a frame + timing.

    MS_ASSET_DIR=... /Projects/envs/mshab/bin/python tools/mshab_smoke.py --subtask pick --steps 50

Checks the five-method contract, the uniform info key set, and the rgb shape/dtype.
"""
import argparse
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="tidy_house")
    ap.add_argument("--subtask", default="pick")
    ap.add_argument("--split", default="train")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--sim_backend", default="gpu")
    ap.add_argument("--out", default="dump/mshab_smoke")
    args = ap.parse_args()

    from longnav.env.mshab import MSHabEnvActor

    env = MSHabEnvActor(task=args.task, subtask=args.subtask, split=args.split, gap=args.gap,
                        width=args.size, height=args.size, sim_backend=args.sim_backend,
                        asset_dir=os.environ.get("MS_ASSET_DIR"), seed=0,
                        logging_output_dir=args.out, minimal_logging=True)
    env.assign_shard(None)
    t0 = time.time()
    rgb, st = env.reset()
    print(f"reset {time.time()-t0:.1f}s  rgb {rgb.shape} {rgb.dtype}  instr={st['obs']['instr_or_goal']!r}")
    print("info keys:", sorted(st["info"]))
    print("action dim:", env._act_dim, "action_space:", env._env.action_space)
    keys0 = set(st["info"])
    os.makedirs(args.out, exist_ok=True)
    from PIL import Image
    Image.fromarray(rgb).save(os.path.join(args.out, "reset.png"))
    t0 = time.time(); n = 0; rets = 0.0
    for i in range(args.steps):
        a = np.random.uniform(-1, 1, size=(args.gap, env._act_dim)).astype(np.float32)
        rgb, st = env.step(a)
        n += args.gap; rets += st["reward"]
        assert set(st["info"]) == keys0, (set(st["info"]) ^ keys0)
        assert rgb.dtype == np.uint8 and rgb.shape == (args.size, args.size, 3), rgb.shape
        if st["done"]:
            print(f"done at step {i}: {st['info']}")
            rgb, st = env.reset()
    dt = time.time() - t0
    Image.fromarray(rgb).save(os.path.join(args.out, "last.png"))
    print(f"{n} sim steps in {dt:.1f}s = {n/dt:.1f} steps/s, return {rets:.3f}")
    print("flush ->", env.flush_logs_to_disk())
    print("SMOKE_OK")


if __name__ == "__main__":
    main()

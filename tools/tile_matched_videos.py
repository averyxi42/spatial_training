"""Tile episode-matched habitat (kinematic) vs SAPIEN (matched-mode) videos side by side.

Habitat videos: dump/mshab_deploy/replicacad_habitat_diag_kinematic/rollout/<scene>/<uid>.*/video.mp4
  (uid = v3_sc1_staging_13.scene_instance:<ep_id>)
SAPIEN videos:  dump/mshab_deploy/mshab_objnav_matched/rollout/videos/matched_<ep_id>_<cat>.mp4
  (label = matched:<ep_id>:<cat>, sanitized by the actor's video namer)

Usage: python3 tools/tile_matched_videos.py [ep_id ...]   (default: every pair found)
Output: dump/mshab_deploy/tiled/ep<id>_<cat>_tiled.mp4  (habitat left, SAPIEN right)
"""
import glob
import os
import subprocess
import sys

B = "/Projects/spatial_training_mshab/dump/mshab_deploy"
HAB = f"{B}/replicacad_habitat_diag_kinematic/rollout"
SAP = f"{B}/mshab_objnav_matched/rollout/videos"
OUT = f"{B}/tiled"


def find_hab(ep):
    hits = glob.glob(f"{HAB}/*/v3_sc1_staging_13.scene_instance:{ep}.*/video.mp4")
    return hits[0] if hits else None


def find_sap(ep):
    hits = glob.glob(f"{SAP}/tidy_house_navigate_train_{ep}_*.mp4") + glob.glob(f"{SAP}/*train_{ep}[_:]*.mp4")
    # be permissive about the sanitizer; prefer exact "_<ep>_" token match
    for h in hits:
        base = os.path.basename(h)
        if f"_{ep}_" in base or f":{ep}:" in base:
            return h
    return hits[0] if hits else None


def ok(path):
    return subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                           "-of", "csv=p=0", path], capture_output=True).returncode == 0


def main():
    os.makedirs(OUT, exist_ok=True)
    eps = sys.argv[1:] or [str(i) for i in range(24)]
    made = []
    for ep in eps:
        h, s = find_hab(ep), find_sap(ep)
        if not h or not s or not ok(h) or not ok(s):
            print(f"ep{ep}: missing/incomplete (hab={bool(h)}, sap={bool(s)})")
            continue
        cat = os.path.basename(s).rsplit("_", 1)[-1].split(":")[-1].replace(".mp4", "")
        out = f"{OUT}/ep{ep}_{cat}_tiled.mp4"

        def dur(path):
            r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", path], capture_output=True, text=True)
            return float(r.stdout.strip())

        dh, ds = dur(h), dur(s)
        # pad ONLY the shorter clip (freeze last frame) up to the longer one's length.
        pad_l = max(0.0, ds - dh) + 0.1
        pad_r = max(0.0, dh - ds) + 0.1
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", h, "-i", s,
               "-filter_complex",
               f"[0:v]fps=20,setpts=PTS-STARTPTS,scale=-2:480,tpad=stop_mode=clone:stop_duration={pad_l:.2f}[l];"
               f"[1:v]fps=20,setpts=PTS-STARTPTS,scale=-2:480,tpad=stop_mode=clone:stop_duration={pad_r:.2f}[r];"
               "[l][r]hstack=shortest=1,drawtext=text='habitat (kinematic)':x=10:y=10:fontcolor=white:fontsize=20,"
               "drawtext=text='SAPIEN (matched)':x=w/2+10:y=10:fontcolor=white:fontsize=20",
               "-c:v", "libx264", "-crf", "28", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            made.append(out)
            print(f"ep{ep}: {out}")
        else:
            print(f"ep{ep}: ffmpeg failed: {r.stderr[-200:]}")
    print(f"TILED {len(made)} videos -> {OUT}")


if __name__ == "__main__":
    main()

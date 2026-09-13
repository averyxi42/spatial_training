"""Upload an existing fixed-eval MP4 set to one W&B gallery."""

import argparse
import json
from pathlib import Path

import wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rollout-dir", required=True, type=Path)
    parser.add_argument("--cycle", required=True, type=int)
    parser.add_argument("--project", default="cont_dev")
    parser.add_argument("--entity", default="liandancoin-university-of-michigan")
    args = parser.parse_args()

    entries = []
    for video_path in args.rollout_dir.rglob("video.mp4"):
        summary_path = video_path.with_name("summary.json")
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing summary for {video_path}")
        with summary_path.open() as file:
            episode_id = str(json.load(file)["episode_label"])
        entries.append((episode_id, video_path))
    entries.sort(key=lambda entry: entry[0])
    if not entries:
        raise FileNotFoundError(f"no videos under {args.rollout_dir}")
    if len({episode_id for episode_id, _ in entries}) != len(entries):
        raise ValueError("each gallery upload requires one video per episode ID")

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=args.run_id,
        resume="allow",
        job_type="eval",
    )
    videos = [
        wandb.Video(
            str(video_path),
            format="mp4",
            caption=f"step={args.cycle} episode={episode_id}",
        )
        for episode_id, video_path in entries
    ]
    run.log({"cycle": args.cycle, "eval/video": videos})
    run.finish()
    print(f"uploaded {len(videos)} videos for cycle {args.cycle}", flush=True)


if __name__ == "__main__":
    main()

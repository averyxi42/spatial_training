import random
from collections import defaultdict


class SceneGroupedBatchIterator:
    """Infinite scene-major batches for vectorized NavVerse simulator hosts."""

    def __init__(
        self,
        episode_path: str,
        batch_size: int,
        groups_per_wave: int,
        seed: int = 42,
        excluded_labels=None,
    ):
        with open(episode_path) as episode_file:
            labels = [line.strip() for line in episode_file if line.strip()]
        if not labels:
            raise ValueError(f"Episode set is empty: {episode_path}")
        if len(labels) != len(set(labels)):
            raise ValueError(f"Episode set contains duplicate labels: {episode_path}")
        excluded = set(excluded_labels or ())
        missing = excluded - set(labels)
        if missing:
            raise KeyError(
                f"Excluded eval labels are absent from {episode_path}: "
                f"{sorted(missing)[:3]}"
            )
        labels = [label for label in labels if label not in excluded]
        self.labels_by_scene = defaultdict(list)
        for label in labels:
            self.labels_by_scene[label.rsplit("_", 1)[0]].append(label)
        undersized = {
            scene: len(scene_labels)
            for scene, scene_labels in self.labels_by_scene.items()
            if len(scene_labels) < batch_size
        }
        if undersized:
            raise ValueError(
                f"Every vector scene needs at least {batch_size} episodes: {undersized}"
            )
        self.batch_size = int(batch_size)
        self.groups_per_wave = int(groups_per_wave)
        if self.groups_per_wave > len(self.labels_by_scene):
            raise ValueError(
                f"Cannot draw {self.groups_per_wave} distinct scenes from "
                f"{len(self.labels_by_scene)} scenes"
            )
        self.seed = int(seed)
        self.wave = 0
        self.groups = []
        self.group_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.group_index >= len(self.groups):
            self._start_wave()
        group = self.groups[self.group_index]
        self.group_index += 1
        return group

    def skip_groups(self, count: int) -> None:
        """Advance the deterministic stream without returning skipped groups."""
        count = int(count)
        if count < 0:
            raise ValueError("count must be non-negative")
        for _ in range(count):
            next(self)

    def _start_wave(self):
        rng = random.Random(self.seed + self.wave)
        scenes = rng.sample(sorted(self.labels_by_scene), self.groups_per_wave)
        self.groups = [
            rng.sample(self.labels_by_scene[scene], self.batch_size)
            for scene in scenes
        ]
        rng.shuffle(self.groups)
        self.group_index = 0
        self.wave += 1

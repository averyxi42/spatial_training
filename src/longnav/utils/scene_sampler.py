import hashlib
import random
from collections import defaultdict


def load_episode_labels(path):
    with open(path) as file:
        labels = [line.strip() for line in file if line.strip()]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Episode set contains duplicate labels: {path}")
    return labels


class SceneBalancedShardSampler:
    def __init__(self, labels, episodes_per_scene, scenes_per_update, seed):
        self.episodes_per_scene = int(episodes_per_scene)
        self.scenes_per_update = int(scenes_per_update)
        self.seed = int(seed)
        grouped = defaultdict(list)
        for label in labels:
            grouped[label.rsplit("_", 1)[0]].append(label)
        self.labels_by_scene = {
            scene: sorted(scene_labels) for scene, scene_labels in grouped.items()
        }
        undersized = {
            scene: len(scene_labels)
            for scene, scene_labels in self.labels_by_scene.items()
            if len(scene_labels) < self.episodes_per_scene
        }
        if undersized:
            raise ValueError(
                "Every sampled scene must contain at least "
                f"{self.episodes_per_scene} episodes: {undersized}"
            )
        if self.scenes_per_update > len(self.labels_by_scene):
            raise ValueError(
                f"Requested {self.scenes_per_update} scenes per update from only "
                f"{len(self.labels_by_scene)} available scenes"
            )

    @property
    def scenes(self):
        return set(self.labels_by_scene)

    def sample(self, policy_update):
        rng = random.Random(self.seed + int(policy_update))
        selected_scenes = rng.sample(
            sorted(self.labels_by_scene), self.scenes_per_update
        )
        shards = [
            rng.sample(self.labels_by_scene[scene], self.episodes_per_scene)
            for scene in selected_scenes
        ]
        rng.shuffle(shards)
        labels = [label for shard in shards for label in shard]
        digest = hashlib.sha256(("\n".join(labels) + "\n").encode()).hexdigest()
        return shards, digest

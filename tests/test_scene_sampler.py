from longnav.utils.scene_sampler import SceneBalancedShardSampler


def test_scene_balanced_sampler_is_deterministic_and_changes_by_update():
    labels = [
        f"scene_{scene_index}_{episode_index}"
        for scene_index in range(40)
        for episode_index in range(12)
    ]
    sampler = SceneBalancedShardSampler(
        labels,
        episodes_per_scene=8,
        scenes_per_update=32,
        seed=42,
    )

    shards_1, digest_1 = sampler.sample(1)
    repeated_shards_1, repeated_digest_1 = sampler.sample(1)
    shards_2, digest_2 = sampler.sample(2)

    assert shards_1 == repeated_shards_1
    assert digest_1 == repeated_digest_1
    assert shards_1 != shards_2
    assert digest_1 != digest_2
    assert len(shards_1) == 32
    assert len({shard[0].rsplit("_", 1)[0] for shard in shards_1}) == 32
    assert all(len(shard) == len(set(shard)) == 8 for shard in shards_1)
    assert all(len({label.rsplit("_", 1)[0] for label in shard}) == 1 for shard in shards_1)

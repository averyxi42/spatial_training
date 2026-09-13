from pathlib import Path

import ray

from longnav.utils.train_loop import maybe_checkpoint


class _RemoteSave:
    def remote(self, path):
        target = Path(path)
        target.mkdir(parents=True)
        for name in (
            "adapter_model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
        ):
            (target / name).write_bytes(b"checkpoint")
        return path


class _RemoteRequiredFiles:
    def remote(self):
        return []


class _Trainer:
    save_checkpoint_unsafe = _RemoteSave()
    checkpoint_required_files = _RemoteRequiredFiles()


def test_latest_is_atomic_pointer_and_old_rolling_data_is_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(ray, "get", lambda value: value)
    trainers = [_Trainer()]

    maybe_checkpoint(trainers, 0, 8, str(tmp_path), "run", trajectory_list=[])
    root = tmp_path / "run" / "checkpoints"
    latest = root / "latest"
    first = latest.resolve()
    assert latest.is_symlink()
    assert first.name == ".latest_cycle_0"
    assert (latest / "rl_state.pt").is_file()

    maybe_checkpoint(trainers, 1, 8, str(tmp_path), "run", trajectory_list=[])
    assert latest.resolve().name == ".latest_cycle_1"
    assert not first.exists()

    maybe_checkpoint(trainers, 7, 8, str(tmp_path), "run", trajectory_list=[])
    assert latest.resolve() == root / "checkpoint_7"
    assert not (root / ".latest_cycle_1").exists()

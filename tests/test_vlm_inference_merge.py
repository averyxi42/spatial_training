from unittest.mock import Mock

from longnav.utils.vlm_worker import VLMWorker
from longnav.utils.vector_rollout import RolloutConfig


def _worker(ddp_model):
    worker = VLMWorker.__new__(VLMWorker)
    worker.model = Mock()
    worker.rollout_inference_optimizations = False
    worker._inference_mode_prepared = False
    worker.using_lora = Mock(return_value=True)
    worker.is_merged = Mock(return_value=False)
    worker.merge_adapter = Mock()
    worker.ddp_model = ddp_model
    return worker


def test_train_worker_keeps_lora_unmerged_for_rollout():
    worker = _worker(object())

    worker._prepare_model_for_inference()

    worker.merge_adapter.assert_not_called()


def test_eval_worker_keeps_lora_unmerged_for_rollout_parity():
    worker = _worker(None)

    worker._prepare_model_for_inference()

    worker.merge_adapter.assert_not_called()


def test_standalone_rollout_keeps_lora_unmerged_by_default():
    assert not RolloutConfig().merge_lora

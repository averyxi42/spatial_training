from types import SimpleNamespace

import torch
from accelerate import Accelerator
from transformers import get_scheduler

from longnav.utils.vlm_worker import VLMTrainingMixin


class AccumulationWorker(VLMTrainingMixin):
    def __init__(self):
        self.accelerator = Accelerator(gradient_accumulation_steps=4,
                                       step_scheduler_with_optimizer=False)
        module = torch.nn.Linear(1, 1, bias=False)
        module.weight.data.zero_()
        optimizer = torch.optim.AdamW([{"params": module.parameters(), "name": "action_head"}],
                                      lr=0.01, weight_decay=0)
        scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=1000)
        self.ddp_model, self.optimizer, self.scheduler = self.accelerator.prepare(module, optimizer, scheduler)
        self.device = self.accelerator.device
        self.train_config = SimpleNamespace(separate_gradient_clipping=True, max_grad_norm=1.0)
        self.rl_algo_config = SimpleNamespace(state_probe_balanced_bce=True)
        self.policy_head_config = {"type": "continuous"}
        self.optimizer_step = self.accumulation_microstep = 0

    def _setup_training(self):
        self.ddp_model.train()

    def _training_forward(self, embeds_inputs, *args):
        return {"output": self.ddp_model(embeds_inputs)}, None

    def rl_loss(self, policy_stats, **kwargs):
        return policy_stats["output"].sum(), {}

    def fixed_input_metrics(self):
        return {}


def test_accumulation_cancels_before_clipping_and_advances_scheduler_once():
    worker = AccumulationWorker()
    for index, value in enumerate([100.0, -100.0, 100.0, -100.0]):
        metrics = worker.generic_train_step(torch.tensor([[value]], device=worker.device), ["rl"], {"rl": {}})
        if index < 3:
            assert worker.optimizer_step == 0
            assert worker.scheduler.state_dict()["last_epoch"] == 0
            assert not worker.checkpoint_ready()
    assert metrics["train/grad_norm_action_head"] == 0
    assert worker.ddp_model.weight.item() == 0
    assert worker.optimizer_step == 1
    assert worker.scheduler.state_dict()["last_epoch"] == 1
    assert worker.checkpoint_ready()
    assert next(iter(worker.optimizer.state.values()))["step"] == 1

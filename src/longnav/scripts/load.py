import io

from typing import Optional
from dataclasses import asdict
from longnav.utils.rollout_core import RLWorker
from longnav.config_schema import RolloutConfig,VLMConfig,VLMTrainingConfig
from longnav.utils.factories import resolve_checkpoint_path,get_base_model
from longnav.utils.rollout_core import substitute_convo_template
import os
from pathlib import Path
import numpy as np
rollout_cfg = RolloutConfig()
vlm_cfg = VLMConfig()
vlm_cfg.attn_impl = "sdpa"
training_cfg = VLMTrainingConfig()
# checkpoint_path = "Aasdfip/hm3d_rpp_ke_standard-checkpoint_231"

# checkpoint_path = resolve_checkpoint_path(checkpoint_path)
# base_model_path = get_base_model(checkpoint_path)
# vlm_cfg.model_id = base_model_path


worker = RLWorker(asdict(rollout_cfg),**asdict(vlm_cfg))
worker._setup_peft(training_cfg)
# worker.load_checkpoint(checkpoint_path,False,False)

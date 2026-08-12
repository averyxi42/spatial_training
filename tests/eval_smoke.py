from longnav.config_schema import *
from longnav.conf.env_configs import DummyDiscreteEnvConfig
from longnav.conf.vlm_configs import LMHeadConfig
from longnav.utils.factories import ExpBootstrapper,get_shard_iterator
from longnav.utils.rollout_core import collect_rollouts
import ray

cfg = RLConfig()
cfg.resources.osm_gb=8
cfg.resources.vlm_conda_env=None
cfg.resources.habitat_conda_env=None
cfg.resources.num_sims=2
cfg.sim = DummyDiscreteEnvConfig()
cfg.vlm.policy_head = LMHeadConfig()
cfg.vlm.attn_impl = "sdpa"
cfg.rollout.convo_start_template=[
        {"role": "user", "content": [{"type": "text", "text": "example substitution: $instr_or_goal"}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]}
    ]

bootstrapper = ExpBootstrapper(cfg)
bootstrapper.setup_cluster()

vlms = bootstrapper.bootstrap_vlms_rl(training=False)
sims = bootstrapper.bootstrap_sims()

rollout_list,result_list,log_list = collect_rollouts(sims,vlms,get_shard_iterator(0),16,{"return_inputs":False,"eval":True}) #

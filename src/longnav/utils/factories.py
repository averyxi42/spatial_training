import ray
import os
import time
from omegaconf import OmegaConf
from hydra.utils import get_class
from longnav.config_schema import *

# Metadata keys that exist purely for the factory/registry layer (dispatch, discrimination)
# and have no corresponding constructor parameter on the actor being instantiated. Strip
# these centrally wherever a resolved config dict gets spread into a `.remote()` call.
RESERVED_CONFIG_KEYS = {"_target_", "type", "name"}


def strip_reserved_keys(config_dict: dict) -> dict:
    return {k: v for k, v in config_dict.items() if k not in RESERVED_CONFIG_KEYS}

# Use these imports for type hinting
from typing import List, Iterator, Optional, Union
import logging
import json
thread_cap_env = {
    "env_vars": {
        "OMP_NUM_THREADS": "1", 
        "MKL_NUM_THREADS": "1", 
        "OPENBLAS_NUM_THREADS": "1", 
        "VECLIB_MAXIMUM_THREADS": "1", 
        "NUMEXPR_NUM_THREADS": "1",
        # Optional: Reduce Habitat/Magnum logging spam while we're at it
        "HABITAT_SIM_LOG": "quiet",
        "MAGNUM_LOG": "quiet",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_ENABLE_PARALLEL_LOADING": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    }
}


def _actor_runtime_env(conda_env, extra_pythonpath=None):
    """Build a Ray runtime env without dropping a configured simulator package path."""
    env = {"conda": conda_env} if conda_env else {}
    env |= thread_cap_env
    if extra_pythonpath:
        inherited = os.environ.get("PYTHONPATH", "")
        env["env_vars"] = dict(env["env_vars"])
        env["env_vars"]["PYTHONPATH"] = os.pathsep.join(
            value for value in (extra_pythonpath, inherited) if value)
    return env


def _join_pythonpaths(*paths):
    return os.pathsep.join(path for path in paths if path)

def save_hydra_config(config, save_dir: str, filename: str = "config.yaml"):
    """
    Saves a Hydra/OmegaConf object to a YAML file.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    OmegaConf.save(config=config, f=save_path)
    print(f"📄 Config saved to: {save_path}")

def load_hydra_config(load_dir: str, filename: str = "config.yaml"):
    """
    Loads a Hydra/OmegaConf object from a YAML file.
    Returns None if file is missing.
    """
    load_path = os.path.join(load_dir, filename)
    if not os.path.exists(load_path):
        return None
        
    try:
        conf = OmegaConf.load(load_path)
        return conf
    except Exception as e:
        print(f"⚠️ Failed to load config from {load_path}: {e}")
        return None

def get_base_model(checkpoint):
    config_path = os.path.join(checkpoint, "adapter_config.json")
    saved_base = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            conf = json.load(f)
            saved_base = conf.get("base_model_name_or_path", "")
    return saved_base

def resolve_checkpoint_path(path_or_id):
    """
    Detects if the input is a local path or a HuggingFace Hub ID.
    If Hub ID, downloads the snapshot (adapters + optim states) and returns local cache path.
    If local path, returns as-is.
    """
    import os
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

    # 1. If it exists locally, trust it.
    if os.path.exists(path_or_id):
        return path_or_id

    # 2. Heuristic: Hub IDs usually look like 'user/repo'
    # We attempt to download. If it fails, we assume it was a bad local path.
    print(f"🔍 '{path_or_id}' not found locally. Attempting HF Hub download...")
    
    try:
        # Download everything needed for a resume:
        # - Adapter weights (.bin or .safetensors)
        # - Configs (.json, .yaml)
        # - Optimizer/Scheduler states (.pt) - ONLY if you uploaded them!
        local_dir = snapshot_download(
            repo_id=path_or_id,
            allow_patterns=["*.json", "*.bin", "*.safetensors", "*.pt", "*.yaml"],
            ignore_patterns=["*.msgpack", "*.h5"], # Ignore flax/tf weights if any
            tqdm_class=None # Optional: silence progress bar
        )
        print(f"✅ Downloaded '{path_or_id}' to: {local_dir}")
        return local_dir
        
    except (RepositoryNotFoundError, RevisionNotFoundError):
        print(f"❌ Could not find '{path_or_id}' on HuggingFace Hub or locally.")
        raise FileNotFoundError(f"Checkpoint path not found: {path_or_id}")
    except Exception as e:
        print(f"⚠️ Error during HF download: {e}")
        raise e
    
class InferenceWorkerFactory:
    @staticmethod
    def create(vlm_dict: dict, rollout_dict: dict, res_cfg: ResourceConfig):
        # res_cfg is fine to keep as object for resource logic
        from longnav.utils.rollout_core import InferenceRayWorker
        
        # We use the dicts directly to avoid pickling issues
        RemoteInferenceWorker = ray.remote(InferenceRayWorker).options(
            resources={res_cfg.vlm_resource_tag: 1},
            num_cpus=res_cfg.vlm_cpus,
            num_gpus=res_cfg.vlm_gpu_fraction,
            runtime_env=_actor_runtime_env(res_cfg.vlm_conda_env,
                                           res_cfg.worker_pythonpath),
            max_restarts=0,        # <--- CRITICAL: Do not restart on crash.
            max_task_retries=-1,
        )

        return [
            RemoteInferenceWorker.remote(
                rollout_config=rollout_dict, 
                **vlm_dict
            ) for _ in range(res_cfg.num_vlms)
        ]

class RLWorkerFactory:
    @staticmethod
    def create(
        vlm_dict: dict,
        rollout_dict: dict,
        res_cfg: ResourceConfig,
        scheduling_strategies=None,
    ):
        # res_cfg is fine to keep as object for resource logic
        from longnav.utils.rollout_core import RLActor
        # We use the dicts directly to avoid pickling issues
        actor_options = dict(
            resources={res_cfg.vlm_resource_tag: 1},
            num_cpus=res_cfg.vlm_cpus,
            num_gpus=res_cfg.vlm_gpu_fraction,
            runtime_env=_actor_runtime_env(res_cfg.vlm_conda_env,
                                           res_cfg.worker_pythonpath),
            max_restarts=0,        # <--- CRITICAL: Do not restart on crash.
            max_task_retries=-1,
        )
        strategies = scheduling_strategies or [None] * res_cfg.num_vlms
        if len(strategies) != res_cfg.num_vlms:
            raise ValueError("Need one scheduling strategy per VLM worker")
        workers =  [
            ray.remote(RLActor).options(
                **actor_options,
                **({"scheduling_strategy": strategy} if strategy is not None else {}),
            ).remote(
                rollout_config=rollout_dict, 
                **vlm_dict
            ) for strategy in strategies
        ]
        
        return workers
    def _enable_training(workers,res_cfg:ResourceConfig,train_cfg:VLMTrainingConfig):
        # Auto-detect rendezvous point for the workers
        world_size = len(workers)        
        futures = []
        if res_cfg.master_port is None:
            import socket

            def find_free_port():
                # Create a new socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    # Bind to an empty address/port, letting the OS pick a free port (port 0)
                    s.bind(('', 0))
                    # Return the port number assigned by the OS
                    return s.getsockname()[1]

            # Usage:
            master_port = find_free_port()
        else:
            master_port = res_cfg.master_port
        for rank, w in enumerate(workers):
            futures.append(w.setup_training.remote(
                config=train_cfg,
                rank=rank,
                world_size=world_size,
                master_addr=res_cfg.master_addr,
                master_port=master_port,
            ))
       
        return futures

class SimWorkerFactory:
    @staticmethod
    def create_one(
        sim_dict: dict,
        res_cfg: ResourceConfig,
        task_cfg: RunConfig,
        worker_index: int,
        logger_actor=None,
        scheduling_strategy=None,
    ):
        sim_dict = dict(sim_dict)
        target = sim_dict.pop("_target_")
        config_overrides = sim_dict.pop("per_worker_config_overrides", None)
        env_actor_cls = get_class(target)

        actor_options = dict(
            resources={res_cfg.sim_resource_tag: 1},
            num_cpus=res_cfg.sim_cpus,
            num_gpus=res_cfg.sim_gpu_fraction,
            runtime_env=_actor_runtime_env(
                res_cfg.habitat_conda_env,
                _join_pythonpaths(res_cfg.worker_pythonpath,
                                  res_cfg.habitat_pythonpath),
            ),
            # Actor replacement is supervised explicitly by the rollout driver so the
            # exact episode shard can be retried and the replacement placement verified.
            max_restarts=0,
            max_task_retries=0,
        )

        ctor_kwargs = strip_reserved_keys(sim_dict)
        if config_overrides is not None:
            if worker_index >= len(config_overrides):
                raise ValueError(
                    f"Missing per-worker simulator override for index {worker_index}"
                )
            print(
                f"overriding sim {worker_index} config with: "
                f"{config_overrides[worker_index]}"
            )
            ctor_kwargs["config_path"] = config_overrides[worker_index]
        log_dir = os.path.join(task_cfg.output_dir, task_cfg.run_name, "rollout")
        return ray.remote(env_actor_cls).options(
            **actor_options,
            **(
                {"scheduling_strategy": scheduling_strategy}
                if scheduling_strategy is not None
                else {}
            ),
        ).remote(
            **ctor_kwargs,
            logging_output_dir=log_dir,
            logger_actor=logger_actor,
        )

    @staticmethod
    def create(
        sim_dict: dict,
        res_cfg: ResourceConfig,
        task_cfg: RunConfig,
        logger_actor=None,
        scheduling_strategies=None,
    ):
        strategies = scheduling_strategies or [None] * res_cfg.num_sims
        if len(strategies) != res_cfg.num_sims:
            raise ValueError("Need one scheduling strategy per simulator worker")
        stagger_seconds = float(res_cfg.sim_startup_stagger_s)
        if stagger_seconds < 0:
            raise ValueError("resources.sim_startup_stagger_s must be non-negative")
        workers = []
        for index in range(res_cfg.num_sims):
            worker = SimWorkerFactory.create_one(
                sim_dict=sim_dict,
                res_cfg=res_cfg,
                task_cfg=task_cfg,
                worker_index=index,
                logger_actor=logger_actor,
                scheduling_strategy=strategies[index],
            )
            workers.append(worker)
            if stagger_seconds:
                # Constructor completion is the safety gate: Isaac startup is not safe
                # when all GPU processes initialize concurrently on this host.
                ray.get(worker.worker_placement.remote())
                if index + 1 < res_cfg.num_sims:
                    time.sleep(stagger_seconds)
        return workers

class LoggerFactory:
    @staticmethod
    def create(run_cfg: RunConfig, res_cfg: ResourceConfig, full_dict_cfg: dict):
        '''
        Returns a Ray Actor for logging, and set of episode labels to skip.
        Checks the episode_label column of the existing run (if any) to determine which episodes have already been logged, and returns that as a set to skip.
        '''
        if run_cfg.logger is None:
            return None, set()

        logger_actor_cls = get_class(run_cfg.logger._target_)
        project = run_cfg.logger.project

        RemoteLogger = ray.remote(logger_actor_cls).options(
            num_cpus=0,
            runtime_env=_actor_runtime_env(res_cfg.vlm_conda_env,
                                           res_cfg.worker_pythonpath)
        )
        import wandb
        api = wandb.Api()
        # fetch latest run id matching name
        id = None
        try:
            runs = api.runs(project,filters={"displayName":run_cfg.run_name})
            print(f"Found {len(runs)} existing runs with name '{run_cfg.run_name}' in project '{project}'.")
        except:
            print(f"Could not fetch runs for project '{project}'. Check your WandB connection and project name.")
            runs = []
        episodes_to_skip = set()
        if len(runs) > 0:
            # Sort runs by creation time descending
            latest_run = runs[-1] #defualt wandb behavior oldest to newest. we resume from newest
            id = latest_run.id
            print(f"Resuming WandB run '{run_cfg.run_name}' with ID: {id}")
            history = latest_run.history(samples=50000)
            try:
                episodes_to_skip = set(history['episode_label'])
                print(f"Skipping {len(episodes_to_skip)} episodes already logged in WandB.")
            except Exception as e:
                print(f"Could not determine episodes to skip from WandB history: {e}")
        return RemoteLogger.remote(
            wandb_init_kwargs={
                "project": project,
                "name": run_cfg.run_name,
                "job_type": run_cfg.jobtype,
                "id": id,
                "resume": "allow",
            },
            run_config=full_dict_cfg,
            compact_metrics=bool(getattr(run_cfg.logger, "compact_metrics", False)),
        ),episodes_to_skip


class ExpBootstrapper:
    def __init__(self, cfg: Union[InferenceConfig,RLConfig]):
        # Resolve all interpolations (Stage 1)
        # This turns ${read_text:...} into actual file content
        try:
            self.resolved_dict = OmegaConf.to_container(cfg, resolve=True)
        except Exception as e:
            print(f"⚠️ Failed to resolve config interpolations: {e}")
            try:
                from dataclasses import asdict
                self.resolved_dict = asdict(cfg)
            except Exception as e:
                print(f"⚠️ Failed to convert config to dict: {e}")
                self.resolved_dict = {}
        self.typed_cfg = cfg 

    def setup_cluster(self):
        res = self.typed_cfg.resources
        system_config = {
            # "automatic_object_spilling_enabled": False,
        }
        if res.ray_address == "local":
            ray.init(
                resources={
                    res.vlm_resource_tag: res.num_vlms, 
                    res.sim_resource_tag: res.num_sims,
                },
                ignore_reinit_error=True,
                object_store_memory = res.osm_gb * 1024 * 1024 * 1024,
                object_spilling_directory = res.object_spilling_directory, _system_config=system_config,
            )
        else:
            ray.init(address=res.ray_address, ignore_reinit_error=True,_system_config=system_config)#,object_store_memory=res.osm_gb * 1024 * 1024 * 1024)
    def bootstrap_logger(self):
        save_hydra_config(self.typed_cfg,os.path.join(self.typed_cfg.task.output_dir,self.typed_cfg.task.run_name))
        return LoggerFactory.create(
            self.typed_cfg.task,
            self.typed_cfg.resources,
            self.resolved_dict
        )
    
    def bootstrap_vlms_infer(self):
        return InferenceWorkerFactory.create(
            vlm_dict=self.resolved_dict['vlm'], 
            rollout_dict=self.resolved_dict['rollout'], 
            res_cfg=self.typed_cfg.resources
        )
    
    def bootstrap_vlms_rl(self,training=True, scheduling_strategies=None):
        if self.typed_cfg.training.checkpoint is not None:
            checkpoint_path = self.typed_cfg.training.checkpoint
            checkpoint_path = resolve_checkpoint_path(checkpoint_path)
            base_model_path = get_base_model(checkpoint_path)
            if base_model_path is not None:
                self.resolved_dict['vlm']['model_id'] = base_model_path
                self.typed_cfg.vlm.model_id = base_model_path
            # self.resolved_dict['training']['checkpoint'] = checkpoint_path
            self.typed_cfg.training.checkpoint = checkpoint_path
        elif self.resolved_dict['vlm'].get('merge_adapter_dir'):
            # Merge-based init (no training.checkpoint): the true base model is named by
            # the adapter being merged, exactly as a checkpoint would name it. Without
            # this, model_id silently stays the schema default and the merge applies the
            # adapter to the wrong base.
            merge_path = resolve_checkpoint_path(self.resolved_dict['vlm']['merge_adapter_dir'])
            base_model_path = get_base_model(merge_path)
            if base_model_path is not None:
                self.resolved_dict['vlm']['model_id'] = base_model_path
                self.typed_cfg.vlm.model_id = base_model_path
            self.resolved_dict['vlm']['merge_adapter_dir'] = merge_path
            self.typed_cfg.vlm.merge_adapter_dir = merge_path
        # vlm_dict['save_outputs'] = True # force save outputs for RL
        self.resolved_dict['vlm']['save_outputs'] = True
        workers = RLWorkerFactory.create(
            vlm_dict=self.resolved_dict['vlm'], 
            rollout_dict=self.resolved_dict['rollout'], 
            res_cfg=self.typed_cfg.resources,
            scheduling_strategies=scheduling_strategies,
        )
        if training:
            futures = RLWorkerFactory._enable_training(workers,self.typed_cfg.resources,self.typed_cfg.training)
            ray.get(futures)
        else:
            print("⚠️ Skipping training setup for RL workers.")
            if self.typed_cfg.training.checkpoint is not None:
                print("loading checkpoint for eval")
                for worker in workers:
                    ray.get(worker._setup_peft.remote(self.typed_cfg.training))
                    ray.get(worker.setup_state_probe_for_eval.remote(
                        self.typed_cfg.training))
                    ray.get(worker.load_checkpoint.remote(self.typed_cfg.training.checkpoint,False,False))
        return workers
    
    def bootstrap_sims(self,logger=None, scheduling_strategies=None):
        return SimWorkerFactory.create(
            sim_dict=self.resolved_dict['sim'],
            res_cfg=self.typed_cfg.resources,
            task_cfg=self.typed_cfg.task,
            logger_actor=logger,
            scheduling_strategies=scheduling_strategies,
        )

    def bootstrap_sim(self, worker_index, logger=None, scheduling_strategy=None):
        """Create one simulator actor with the same resolved production config."""
        return SimWorkerFactory.create_one(
            sim_dict=self.resolved_dict['sim'],
            res_cfg=self.typed_cfg.resources,
            task_cfg=self.typed_cfg.task,
            worker_index=worker_index,
            logger_actor=logger,
            scheduling_strategy=scheduling_strategy,
        )

    def create_paired_gpu_placement_groups(self):
        res = self.typed_cfg.resources
        if res.num_vlms != res.num_sims:
            raise ValueError(
                "paired_gpu_workers requires resources.num_vlms == resources.num_sims"
            )
        if res.vlm_gpu_fraction + res.sim_gpu_fraction > 1.0 + 1e-9:
            raise ValueError(
                "Paired VLM/sim GPU fractions must sum to at most one; got "
                f"{res.vlm_gpu_fraction} + {res.sim_gpu_fraction}"
            )
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        groups = []
        strategies = []
        for _ in range(res.num_vlms):
            group = placement_group(
                [
                    {
                        "CPU": res.vlm_cpus + res.sim_cpus,
                        "GPU": 1,
                        res.vlm_resource_tag: 1,
                        res.sim_resource_tag: 1,
                    }
                ],
                strategy="STRICT_PACK",
            )
            groups.append(group)
            strategies.append(
                PlacementGroupSchedulingStrategy(
                    placement_group=group,
                    placement_group_bundle_index=0,
                    placement_group_capture_child_tasks=True,
                )
            )
        ray.get([group.ready() for group in groups])
        return groups, strategies

    @staticmethod
    def verify_paired_gpu_placement(trainers, sims):
        trainer_info = ray.get([worker.worker_placement.remote() for worker in trainers])
        sim_info = ray.get([worker.worker_placement.remote() for worker in sims])
        trainer_gpus = []
        for index, (vlm, sim) in enumerate(zip(trainer_info, sim_info)):
            vlm_ids = tuple(vlm["ray_gpu_ids"])
            sim_ids = tuple(sim["ray_gpu_ids"])
            if len(vlm_ids) != 1 or vlm_ids != sim_ids:
                raise RuntimeError(
                    f"GPU pair {index} is not co-located: vlm={vlm}, sim={sim}"
                )
            trainer_gpus.append(vlm_ids[0])
        if len(set(trainer_gpus)) != len(trainer_gpus):
            raise RuntimeError(
                f"Paired workers do not cover distinct GPUs: {trainer_gpus}"
            )
        print(f"Verified paired GPU placement: {trainer_gpus}")
        return trainer_info, sim_info
    
    def bootstrap_eval(self):
        self.setup_cluster()
        
        # 1. Spawn Logger (Pass the FULL resolved dict for WandB hyperparams)
        logger = self.bootstrap_logger()
        
        # 2. Spawn Inference Workers
        # We pass the resolved dictionaries from our resolved_dict
        vlms = self.bootstrap_vlms_infer()
        
        # 3. Spawn Sim Workers
        sims = self.bootstrap_sims(logger)
        
        return vlms, sims, logger
    
def trivial_shard_iterator(n=256) -> Iterator[None]:
    """Yields the trivial shard (None) once. Habitat handles dataset loading."""
    for i in range(n):
        yield None

def chunk_list(all_episodes: List[str], shard_size: int) -> Iterator[List[str]]:
    """Yields chunks of episodes of a specific size."""
    for i in range(0, len(all_episodes), shard_size):
        yield all_episodes[i : i + shard_size]
        
def get_console_logger(name = "DriverMain"):
    """Sets up a central logger and directory structure."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def get_shard_iterator(
    shard_size: int, 
    subset_label: str = "", 
    episode_json: str = "", logger: Optional[logging.Logger] = None,
    excluded_episodes = None
) -> Iterator[Optional[List[str]]]:
    """
    Orchestrates shard creation based on config.
    Reproduces the original branching logic for trivial vs. explicit shards.
    """
    # Case A: Trivial Shard (Let Habitat handle loading via its own config)
    if shard_size <= 0:
        if logger is not None:
            logger.info("Using trivial shard (full dataset via Habitat config).")
        return trivial_shard_iterator()

    # Case B: Explicit Sharding (We must load the list first)
    all_episodes = []

    if subset_label:
        # Import inside function to avoid circular dependencies or heavy startup
        from longnav.constants import episode_labels_table
        if subset_label in episode_labels_table:
            all_episodes = episode_labels_table[subset_label]
            if logger is not None:
                logger.info(
                    f"Loaded {len(all_episodes)} episodes from subset: {subset_label}"
                )
        else:
            raise ValueError(f"Subset label '{subset_label}' not found in constants.")

    elif episode_json:
        with open(episode_json, 'r') as f:
            all_episodes = json.load(f)
        if logger is not None:
            logger.info(f"Loaded {len(all_episodes)} episodes from JSON: {episode_json}")

    else:
        raise ValueError("Shard size > 0 but no episode source (subset_label or episode_json) provided.")

    if not all_episodes:
        raise ValueError("The resolved episode list is empty.")
    if excluded_episodes is not None:
        excluded_episodes = set(excluded_episodes)
        all_episodes = [episode for episode in all_episodes if episode not in excluded_episodes]
        if logger is not None:
            logger.info(
                f"After exclusion, {len(all_episodes)} episodes remain for sharding."
            )
        
    return chunk_list(all_episodes, shard_size)

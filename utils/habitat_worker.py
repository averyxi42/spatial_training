from collections import defaultdict
import numpy as np
import regex as re
import os
import ray

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

import os
from habitat.utils.visualizations import utils as vut
import time

def apply_schema(data, schema):
    """
    Recursively filters 'data' to only include keys present in 'schema'.
    Structure matches schema; values in schema are ignored (placeholders).
    """
    if schema is None:
        return data
    result = {}
    for key, sub_schema in schema.items():
        if key in data:
            # If both are dictionaries, recurse to allow fine-grained filtering
            if isinstance(sub_schema, dict) and isinstance(data[key], dict):
                result[key] = apply_schema(data[key], sub_schema)
            else:
                # Leaf node: Return the data found at this key
                result[key] = data[key]
    return result


def save_video_task(steps_data, filename, output_dir, fps=4, quality=3,return_thumbnail=True):
    """
    Stateless Ray task to render video.
    """
    from habitat.utils.visualizations import utils as vut

    if not steps_data or 'obs' not in steps_data:
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    action_list = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]
    
    # Process images
    # Note: We use the helper 'apply_schema' instead of self._apply_schema
    images = [
        vut.observations_to_image(apply_schema(obs, {"rgb": True}), info) 
        for obs, info in zip(steps_data['obs'], steps_data['info'])
    ]
    
    # Handle the appended action 0 logic from your original code
    # (Ensure we don't mutate the passed read-only object in a way that breaks things)
    actions = steps_data['action'] + [0]
    
    texts = [
        [
            f"episode: {steps_data['info'][0]['episode_label']} step: {idx}",
            action_list[int(action)],
            f"distance_to_goal: {info['distance_to_goal']}",
            f"distance_reward: {info['distance_to_goal_reward']}",
            f"goal: {obs['goal_name']}",
            f"spl: {steps_data['info'][-1]['spl']}",
        ]
        for idx, (action, obs, info) in enumerate(zip(actions, steps_data['obs'], steps_data['info']))
    ]
    
    images = [vut.overlay_text_to_image(image, text) for image, text in zip(images, texts)]

    vut.images_to_video(
        images=images,
        output_dir=output_dir,
        video_name=filename,
        fps=fps,
        quality=quality,
        verbose=False
    )
    if return_thumbnail:
        return os.path.join(output_dir, filename + ".mp4"),images[-1]

    return os.path.join(output_dir, filename + ".mp4")

summary_schema = {
    "episode_label":True, 
    "scene_id":True, #redundant, but convenient for scene based analysis
    "spl":True,
    "distance_to_goal":True,
    "soft_spl":True,
    "success":True,
}
@ray.remote
def default_habitat_log_task(actor_handle,episode_data,output_dir,**video_kwargs):
    result_row = apply_schema(episode_data['info'][-1],summary_schema) 
    #TODO: check safety. does habitat change the internal episode the moment step(stop) is called? if so, we are in trouble
    # safety overrides for now, using first step guarantees correctness
    result_row['episode_label'] = episode_data['info'][0]['episode_label']
    result_row['scene_id'] = episode_data['info'][0]['scene_id']
    result_row['goal_name'] = episode_data['obs'][0]['goal_name'] #need this for obvious reasons

    result_row['steps'] = len(episode_data['actions'])
    result_row['duration'] = episode_data['timestamp'][-1]-episode_data['timestamp'][0]
    result_row['throughput'] = result_row['steps']/max(result_row['duration'],1e-5)

    result_row['fpg_trigger_count'] = np.sum(np.array(episode_data['fp_stop'])) #works thanks to defaultdict shenanigans
    result_row['fng_trigger_count'] = np.sum(np.array(episode_data['fn_stop'])) #works thanks to defaultdict shenanigans

    # sequence level logs
    result_row['raw/actions'] = episode_data['actions']
    result_row['raw/positions'] = [info['pos_rots'][:3] for info in episode_data['info'][:-1]] #final position is redundant due to stop action
    result_row['raw/quaterions'] = [info['pos_rots'][3:] for info in episode_data['info'][:-1]] #doubt this is meaningful in wandb but it can't cost that much storage right?
    result_row['raw/collisions'] = episode_data.get('stuck',[]) #this is already aligned to actions, no need to slice. 

    # auxiliary performance metrics
    result_row['collision_rate'] = np.sum(np.array(result_row['raw/collisions']))/result_row['steps']
    result_row['mean_distance_reward'] = np.mean(np.array(episode_data['reward'][1:-1])) # slice to exclude the terminal reward
    # save the video.
    result_row['video_path'],result_row['thumbnail'] = save_video_task(episode_data,output_dir,result_row['episode_label'],return_thumbnail=True, **video_kwargs) 
    
    actor_handle.log_row.remote(result_row) # generic table logging interface. 
    # and we're done?

# # from collections import deque
# # --- Helper Function ---
# def get_scene_id(scene_path):
#     scene_id = scene_path.split("/")[-1]
#     scene_id = re.sub(r'\.basis\.glb$', '', scene_id)
#     return scene_id
# robust version
def get_scene_id(scene_path):

    # os.path.basename handles both "/" and "\" correctly
    filename = os.path.basename(scene_path)
    # Optional: Make regex handle both .basis.glb and standard .glb
    scene_id = re.sub(r'(\.basis)?\.glb$', '', filename)
    return scene_id

class HabitatWorker:
    def __init__(self, assigned_episode_labels=None,workspace='/Projects/SG_VLN_HumanData/SG-VLN', config_path="configs/objectnav_hm3d_rgbd_semantic.yaml", enable_caching=True,dataset_path = None, scenes_dir=None,split="val",postprocess= True,output_schema=None):
        from habitat.config.default import get_config
        from habitat.config import read_write
        from habitat.config.default_structured_configs import (
            TopDownMapMeasurementConfig,
            FogOfWarConfig,
        )
        from habitat import Env, make_dataset
        """
        assigned_episode_labels: List of strings ['scene_id_episode_id', ...] specific to this worker.
        enable_caching: If True, stores observations for video generation.
        """
        if workspace is not None:
            # nuclear option to ensure proper habitat loading
            import os
            os.chdir(workspace)

        self.postprocess = postprocess
        self.enable_caching = enable_caching
        if enable_caching:
            self.steps = defaultdict(list)
        self.output_schema = output_schema
        self.logging_actor = None
        # --- Initialize Config & Env ---
        self.config_env = get_config(config_path)
        with read_write(self.config_env):
        # --- OVERRIDE PATHS IF PROVIDED ---
            if dataset_path:
                # 1. The Episode Dataset (The JSON file)
                # Example: "/mnt/data/habitat/datasets/objectnav/val.json.gz"
                self.config_env.habitat.dataset.data_path = dataset_path
            else:
                self.config_env.habitat.dataset.split = split

            if scenes_dir:
                # 2. The Scene Assets (The folder containing .glb files)
                # Example: "/mnt/data/habitat/scene_datasets/"
                self.config_env.habitat.dataset.scenes_dir = scenes_dir
            # Add TopDownMap for visualization
            self.config_env.habitat.task.measurements.top_down_map = (
                TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=512,
                    draw_goal_positions=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_border=True,
                    fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5, fov=72),
                )
            )

        # Create Dataset
        self.full_dataset = make_dataset(
            self.config_env.habitat.dataset.type, config=self.config_env.habitat.dataset
        )
        self.assign_shard(assigned_episode_labels)

    def enable_logging(self,logging_actor,
        logging_task_handle=default_habitat_log_task, # The raw @ray.remote function
        logging_task_config=None):

        self.logging_actor = logging_actor
        self.logging_task_handle = logging_task_handle
        # Default to empty dict if None to prevent TypeError in ** unpacking
        self.task_config = logging_task_config if logging_task_config else {}

    def assign_shard(self,assigned_episode_labels = None):
        from habitat.core.dataset import EpisodeIterator
        from habitat.gym import make_gym_from_config
        
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
        self.steps = defaultdict(list)
        self.last_step = None
        self.assigned_labels = set(assigned_episode_labels) if assigned_episode_labels is not None else None

        if self.assigned_labels is not None:
            # Filter Episodes
            def filter_fn(eps):
                scene_id = get_scene_id(eps.scene_id)
                episode_label = f'{scene_id}_{eps.episode_id}'
                return episode_label in self.assigned_labels

            dataset = self.full_dataset.filter_episodes(filter_fn)
        else:
            dataset = self.full_dataset
        try:
            self.shard_length = dataset.num_episodes
        except:
            self.shard_length = len(dataset)

        if self.shard_length == 0:
            print("WARNING: Assigned shard is empty!")
        self.episode_counter = 0 #distinguish from "steps" concept per episode

        # Initialize Env
        # self.env = Env(self.config_env, dataset)
        self.env = make_gym_from_config(self.config_env,dataset)
        # Setup Iterator 
        self.env.habitat_env.episode_iterator = EpisodeIterator(
            dataset.episodes,
            cycle=True,
            shuffle=True,
            group_by_scene=False,
            seed=17,
            # max_scene_repeat_episodes=10
        )
        print(f"Actor assigned with shard of {len(dataset.episodes)} episodes.")


    def step(self, action:int, output_schema=None, fp_guard = False, fn_guard = False, supplementary_logs=None):
        """
        Standard step. 
        Args:
            action: The action id to take.
            output_schema: sets the output schema
            fp_guard: guard against false positive "stop“ calls with oracle
            fn_guard: guard against false negative "stop" calls with oracle
            supplementary logs: dict of extra info to log for this step. useful for recording action logprobs, agent inference latency, etc
                - WARNING: if you provide this, you MUST provide it every step, with the same keys. otherwise the data won't align properly!
        """
        last_distance = self.last_step['info']['distance_to_goal']
        extras = {'fp_stop':-1*int(not fp_guard),'fn_stop':-1*int(not fn_guard)} # -1 for not enabled, 0 for enabled but not triggerd.
        # oracle stop guards useful for reducin eval noise. #TODO: use habitat config instead of magic number
        if fp_guard and action==0 and last_distance > 0.1:
            action = np.random.choice([1,2,3]) #chose random non stop action
            extras['fp_stop'] = 1 #record the false positive incident
        if fn_guard and last_distance<0.1 and action!=0:
            action = 0
            extras['fn_stop'] = 1

        obs, reward, done, info = self.env.step(action) 
        
        if output_schema is not None:
            self.output_schema = output_schema
        
        step_dict = {
        "obs":obs,
        "reward": reward,
        "done": done,
        "info": info
        }

        if self.postprocess:
            step_dict = self._postprocess_step(step_dict)
            extras['stuck'] = np.linalg.norm(np.array(self.last_step['info']['pos_rots'])-np.array(step_dict['info']['pos_rots']))<1e-5 # record collisions

        if self.enable_caching:
            extras["timestamp"] = time.time()
            if supplementary_logs is None:
                    supplementary_logs = {}
            self._cache_step(step_dict | supplementary_logs | extras)
            self.steps['action'].append(action)
        self.last_step = step_dict
        return self._apply_schema(step_dict,self.output_schema)

    def get_last_step(self,output_schema=None):
        """Returns the current observation without stepping the environment."""
        return self._apply_schema(self.last_step,output_schema)
    
    def get_episodes(self):
        return self.env.episodes
    
    def reset(self, episode_id=None,output_schema=None):
        """
        Args:
            episode_id: please don't try this.
            output_schema: use and set the step output schema
        """
        # Access the habitat core environment
        self.invoke_logging()
        self.steps = defaultdict(list)  # Clear video cache for new episode

        if episode_id is not None:
            # Find specific episode
            episodes = self.env.habitat_env._dataset.episodes
            episode = next((e for e in episodes if e.episode_id == str(episode_id)), None)
            self.env.habitat_env._current_episode = episode

        if output_schema is not None:
            self.output_schema = output_schema
        # Force the iterator to point here
        # Note: Habitat iterators are complex; the safest way is to force the current episode
        
        obs,info = self.env.reset(return_info=True)
        self.episode_counter+=1
        # Calling reset() now loads 'current_episode'
        step_dict = {
        "obs":obs,
        "reward": 0,
        "done": False,
        "info": info
        }
        if self.postprocess:
            step_dict = self._postprocess_step(step_dict)
        if self.enable_caching:
            self._cache_step(step_dict | {"timestamp": time.time()})
        self.last_step = step_dict
        return self._apply_schema(step_dict,self.output_schema)

    def save_video(self, output_dir, prefix="ep",fps=4,quality = 3):
        from habitat.utils.visualizations import utils as vut

        """
        Renders the cached video to disk using Habitat's utils.
        Returns the path to the saved file.
        - quality: 0-10
        -
        """
        if not self.steps:
            return None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Construct filename
        current_ep_id = self.env.current_episode().episode_id
        scene_id = get_scene_id(self.env.current_episode().scene_id)
        filename = f"{prefix}_{scene_id}_{current_ep_id}"
        obs_schema = {"rgb":True}
        action_list = ["STOP","MOVE_FORWARD","TURN_LEFT","TURN_RIGHT","LOOK_UP","LOOK_DOWN"]
        images = [vut.observations_to_image(self._apply_schema(obs,obs_schema),info) for obs,info in zip(self.steps['obs'],self.steps['info'])]
        self.steps['action'].append(0)
        texts = [   [
                    f"episode: {self.steps['info'][0]['episode_label']} step: {idx}",
                    action_list[int(action)],
                    f"distance_to_goal: {info['distance_to_goal']}",
                    f"distance_reward: {info['distance_to_goal_reward']}",
                    f"goal: {obs['goal_name']}",
                    f"spl: {self.steps['info'][-1]['spl']}",
                    ]
                for idx, (action,obs,info) in enumerate(zip(self.steps['action'],self.steps['obs'],self.steps['info']))]
        images = [vut.overlay_text_to_image(image,text) for image,text in zip(images,texts)]
        # Use Habitat's video utility
        
        vut.images_to_video(
            images=images,
            # primary_image_key="rgb",
            output_dir=output_dir,
            video_name=filename,
            fps=fps,
            quality = quality,
            verbose=False
        )
        # self.steps = defaultdict(list) # clear buffer after save
        return os.path.join(output_dir, filename + ".mp4")
        
    def invoke_logging(self):
        """
        Invokes the logging task with strict Node Affinity to prevent data transfer costs.
        """
        if not self.logging_actor or not self.logging_task_handle or not self.steps:
            return
        # 1. Capture and Clear (Earliest Safe Time)
        raw_episode_data = self.steps
        self.steps = defaultdict(list)
        # 2. Get Current Node ID for Affinity
        # We must ensure the processing task runs on THIS node to process the 
        # large 'raw_episode_data' locally before sending the summary to the actor.
        current_node_id = ray.get_runtime_context().get_node_id()

        # We unroll 'self.task_config' here. This satisfies the "baking" requirement
        # while keeping the handle raw so we can access .options().
        self.logging_task_handle.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=current_node_id, 
                soft=False # Force strict locality
            )
        ).remote(
            actor_handle=self.logging_actor,
            episode_data=raw_episode_data,
            **self.task_config # e.g. contains output_dir="/tmp/logs"
        )

    def _cache_step(self, step):
        """Internal helper to format observations for make_video."""
        # Habitat's make_video expects a specific structure or just a list of dicts/arrays
        # We store the raw obs; make_video handles extracting 'rgb'
        for key, value in step.items():
            self.steps[key].append(value)

    def getattr(self, attr_name):
        """Proxy to access attributes of the underlying Env."""
        return getattr(self.env, attr_name)
    
    def get_metrics(self):
        """Get episode metrics"""
        if hasattr(self.env, 'get_metrics'):
            return self.env.get_metrics()
        return {}

    def _get_goal_name(self, goal_index):
        """
        Decodes the integer objectgoal from the observation into a string.
        """
        # In Habitat ObjectNav, the goal index maps to a list of categories 
        # defined in the dataset config/mapping.
        # We can access the mapping via the task's dataset.
        
        # Access the underlying dataset attribute mapping
        # Note: 'category_to_task_category_id' maps Name -> ID. We need the reverse.
        if not hasattr(self, '_id_to_category_name'):
            # Build the cache once
            dataset = self.env.habitat_env._dataset
            mapping = dataset.category_to_task_category_id
            self._id_to_category_name = {v: k for k, v in mapping.items()}
        
        # Handle scalar or single-element array input
        if hasattr(goal_index, 'item'):
            goal_index = goal_index.item()
            
        return self._id_to_category_name.get(goal_index, f"Unknown({goal_index})")

    def is_exhausted(self):
        return self.episode_counter >= self.shard_length

    def _postprocess_step(self,step_dict):
        obs = step_dict['obs']
        info = step_dict['info']
        if 'objectgoal' in obs:
            obs['goal_name'] = self._get_goal_name(obs['objectgoal'])
        
        if self.env.habitat_env.sim:
            state = self.env.habitat_env.sim.get_agent_state()
            pos = state.position  # Vector3
            rot = state.rotation  # Quaternion

            info['pos_rots'] = (
                float(pos[0]), float(pos[1]), float(pos[2]),      # x, y, z
                float(rot.x), float(rot.y), float(rot.z), # qx, qy, qz
                float(rot.w)                                 # w
            )
        info['scene_id']=get_scene_id(self.env.current_episode().scene_id)
        info['episode_id']=self.env.current_episode().episode_id
        info['episode_label']=f"{info['scene_id']}_{info['episode_id']}"
        curr_idx = self.episode_counter-1
        info['epoch']= curr_idx // self.shard_length
        info['step_in_epoch']=curr_idx % self.shard_length
        # 2. Provide Semantic Mapping (ID -> Label), too expensive, should just request once instead of always send.
        # The driver can now map any pixel in obs['semantic'] to a string.
        # if 'semantic' in obs:
        #     info['semantic_mapping'] = self.get_semantic_label_mapping()
        return self._sanitize(step_dict)
    
    def get_semantic_label_mapping(self):
        """
        Returns a dictionary mapping the Integer ID found in the semantic sensor 
        to the String Category Name.
        
        Format: {104: "chair", 105: "chair", 106: "table", ...}
        """
        # Return cached mapping if the scene hasn't changed
        current_scene = self.env.current_episode().scene_id
        if getattr(self, '_cached_scene_id', None) == current_scene and \
        hasattr(self, '_cached_semantic_mapping'):
            return self._cached_semantic_mapping

        # Double check sim exists
        if not self.env.habitat_env.sim:
            return {}

        # Access the Scene Graph
        semantic_scene = self.env.habitat_env.sim.semantic_scene
        objects = semantic_scene.objects
        
        # Build the mapping: Instance ID (Index) -> Category Name
        # Note: Habitat Instance IDs correspond to the index in the objects list.
        mapping = {}
        for i, obj in enumerate(objects):
            if obj and obj.category:
                mapping[i] = obj.category.name()
            else:
                mapping[i] = "unknown"

        # Cache it to avoid rebuilding this large dict every step
        self._cached_scene_id = current_scene
        self._cached_semantic_mapping = mapping
        
        return mapping

    def _apply_schema(self,data,schema):
        return apply_schema(data,schema)
    
    def close(self):
        """Close environment"""
        if self.env is not None:
            self.env.close()
            self.env = None

    def _sanitize(self, data):
        """
        Recursively converts Habitat/Magnum types into standard Python/Numpy types
        to ensure the Driver (which lacks Habitat) can unpickle the result.
        """
        # 1. Handle Dictionaries (and things that act like them, e.g. Habitat Observations)
        if isinstance(data, dict) or hasattr(data, "keys"):
            # Force conversion to standard dict to strip custom class types
            return {k: self._sanitize(v) for k, v in dict(data).items()}
        
        # 2. Handle Lists/Tuples
        elif isinstance(data, (list, tuple)):
            return [self._sanitize(v) for v in data]
        
        # 3. Handle Numpy (Safe to pass as-is, assuming Driver has Numpy)
        elif isinstance(data, (np.ndarray, np.generic)):
            return data
        
        # 4. Handle Magnum Vectors/Quaternions (Common in Habitat Info)
        # We detect them by checking if the module name starts with 'magnum'
        elif hasattr(type(data), "__module__") and type(data).__module__.startswith("magnum"):
            # Magnum types are usually iterable (Vector3 -> [x,y,z])
            return np.array(list(data))
        # 5. Handle Basic Primitives (int, float, str, bool, None)
        return data
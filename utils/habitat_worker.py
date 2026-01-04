from collections import defaultdict
import numpy as np
import regex as re
import os
from habitat.config.default import get_config
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    TopDownMapMeasurementConfig,
    FogOfWarConfig,
)
from habitat import Env, make_dataset
from habitat.core.dataset import EpisodeIterator
from habitat.utils.visualizations import utils as vut
from habitat.gym import make_gym_from_config

# from collections import deque
# --- Helper Function ---
def get_scene_id(scene_path):
    scene_id = scene_path.split("/")[-1]
    scene_id = re.sub(r'\.basis\.glb$', '', scene_id)
    return scene_id

class HabitatWorker:
    def __init__(self, assigned_episode_labels=None,workspace='/Projects/SG_VLN_HumanData/SG-VLN', config_path="configs/objectnav_hm3d_rgbd_semantic.yaml", enable_caching=True,dataset_path = None, scenes_dir=None,split="val",postprocess= True,output_schema=None):
        """
        assigned_episode_labels: List of strings ['scene_id_episode_id', ...] specific to this worker.
        enable_caching: If True, stores observations for video generation.
        """
        if workspace is not None:
            # nuclear option to unsure proper habitat loading
            import os
            os.chdir(workspace)

        self.steps = defaultdict(list)
        self.postprocess = postprocess
        self.last_obs = None
        self.enable_caching = enable_caching
        self.assigned_labels = set(assigned_episode_labels) if assigned_episode_labels is not None else None
        self.output_schema = output_schema
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
        dataset = make_dataset(
            self.config_env.habitat.dataset.type, config=self.config_env.habitat.dataset
        )
        if self.assigned_labels is not None:
            # Filter Episodes
            def filter_fn(eps):
                scene_id = get_scene_id(eps.scene_id)
                episode_label = f'{scene_id}_{eps.episode_id}'
                return episode_label in self.assigned_labels

            dataset = dataset.filter_episodes(filter_fn)
        
        # Initialize Env
        # self.env = Env(self.config_env, dataset)
        self.env = make_gym_from_config(self.config_env,dataset)
        # Setup Iterator (Cycling enabled as per your boilerplate)
        self.env.unwrapped.episode_iterator = EpisodeIterator(
            dataset.episodes,
            cycle=True,
            shuffle=True,
            group_by_scene=False,
            seed=17
        )
        print(f"Actor initialized with {len(dataset.episodes)} episodes.")

    def step(self, action,output_schema=None):
        """
        Standard step. 
        Args:
            action: The action to take.
            drop_data (bool): Optional flag to return None instead of heavy obs 
                              if bandwidth is tight (default: False).
        """
        obs, reward, done, info = self.env.step(action)
        
        self.last_obs = obs
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
        if self.enable_caching:
            self._cache_step(step_dict)
            self.steps['action'].append(action)
        return self._apply_schema(step_dict,self.output_schema)

    def render(self):
        """Returns the current observation without stepping the environment."""
        return self.last_obs
    
    def get_episodes(self):
        return self.env.episodes
    
    def reset(self, episode_id=None,output_schema=None):
        """
        Forces the environment to reset to a specific episode ID.
        Note: This is an expensive O(N) operation in the standard iterator.
        """
        # Access the habitat core environment
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
            self._cache_step(step_dict)
            self.last_obs = obs
        return self._apply_schema(step_dict,self.output_schema)

    def save_video(self, output_dir, prefix="ep"):
        """
        Renders the cached video to disk using Habitat's utils.
        Returns the path to the saved file.
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
        images = [vut.overlay_text_to_image(image,[action_list[int(action)],f"distance_to_goal: {info['distance_to_goal']}",f"distance_reward: {info['distance_to_goal_reward']}"]) for image,action,info in zip(images,self.steps['action'],self.steps['info'])]
        # Use Habitat's video utility
        
        vut.images_to_video(
            images=images,
            # primary_image_key="rgb",
            output_dir=output_dir,
            video_name=filename,
            fps=4
        )
        # self.steps = defaultdict(list) # clear buffer after save
        return os.path.join(output_dir, filename + ".mp4")
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
            dataset = self.env.unwrapped.unwrapped.habitat_env._dataset
            mapping = dataset.category_to_task_category_id
            self._id_to_category_name = {v: k for k, v in mapping.items()}
        
        # Handle scalar or single-element array input
        if hasattr(goal_index, 'item'):
            goal_index = goal_index.item()
            
        return self._id_to_category_name.get(goal_index, f"Unknown({goal_index})")
    
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
        # 2. Provide Semantic Mapping (ID -> Label), too expensive, should just request once instead of always send.
        # The driver can now map any pixel in obs['semantic'] to a string.
        # if 'semantic' in obs:
        #     info['semantic_mapping'] = self.get_semantic_label_mapping()
        return step_dict
    
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

    def _apply_schema(self, data, schema):
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
                    result[key] = self._apply_schema(data[key], sub_schema)
                else:
                    # Leaf node: Return the data found at this key
                    result[key] = data[key]
        return result
    
    def close(self):
        """Close environment"""
        if self.env is not None:
            self.env.close()
            self.env = None
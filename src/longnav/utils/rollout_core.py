import ray
import os
from collections import deque
from typing import List, Dict, Any, Iterator
from string import Template
from PIL import Image
from longnav.utils.vlm_worker import VLMWorker,VLMTrainingMixin
import numpy as np
from longnav.utils.tensor_utils import TensorPacker
import time

def substitute_convo_template(conversation_template: List[Dict], substitutions: Dict[str, Any]) -> List[Dict]:
    """
    Traverses the conversation template and substitutes any string.Template 
    objects found in 'text' fields using values from the 'obs' dictionary.
    
    Args:
        conversation_template: List of message dicts (role, content).
        substitutions: Dictionary containing substitution keys (e.g., 'instr_or_goal').
        
    Returns:
        A new conversation list with strings substituted.
    """
    new_conversation = []
    
    for message in conversation_template:
        # Shallow copy the message container
        new_message = message.copy()
        new_content = []
        
        # Iterate over the content list (e.g., [{"type": "image"}, {"type": "text", ...}])
        for item in message.get("content", []):
            new_item = item.copy()
            
            # Check if this item is a text component
            if "text" in new_item:
                text_obj = new_item["text"]
                
                # CASE A: It's a Template object (from the config)
                if "$" in text_obj:
                    try:
                        text_template = Template(text_obj)
                        # Perform the substitution
                        new_item["text"] = text_template.substitute(substitutions)
                    except KeyError:
                        raise
                        # Fallback to safe_substitute to prevent crashing on missing keys,
                        # but log it so we know something is wrong.
                        # print(f"Warning: Missing substitution key {e} in template.")
                        # new_item["text"] = text_template.safe_substitute(substitutions)
                        
                # CASE B: It's already a str (static text)
                elif isinstance(text_obj, str):
                    pass # Keep as is
                    
            new_content.append(new_item)
        new_message["content"] = new_content
        new_conversation.append(new_message)
        
    return new_conversation

class EpisodeRolloutMixin:
    def _pack_trajectory(self, buffer: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Converts list of dicts to a dict of numpy arrays (Columnar format).
        This format allows Ray to zero-copy transfer individual columns.
        """
        if not buffer:
            return {}
        
        # Fast dictionary of lists to list of dictionaries inversion
        keys = buffer[0].keys()
        stacked = {k: np.array([d[k] for d in buffer]) for k in keys}
        
        # Optimization: Cast probabilities to float32 to save 50% bandwidth
        if "probs" in stacked:
            stacked["action_probs"] = stacked["action_probs"].astype(np.float32)
        if "rewards" in stacked:
             stacked["rewards"] = stacked["rewards"].astype(np.float32)    
        return stacked
    
    def run_episode(self,env_handle, initial_state_ref,collect_trajectory=False,compute_value=False):
        """
        Returns:
        - is_exhausted: Whether the sim ran out of episodes
        - final_info: The final info dict from the sim (contains success, spl, etc.)
        - trajectory: If collect_trajectory is True, returns the collected trajectory in columnar format (dict of numpy arrays).
        """
        import time
        self.reset() #we reset at the start to ensure clean state. not resetting at the end preserves state for downstream.
        try:
            # 1. Resolve the initial state (Blocking wait for reset to finish)
            # Ray automatically waits for initial_state_ref to be ready before starting this task,
            # but we call ray.get to access the data.
            pos_id_kwargs={
                "mode": "standard"
            }
            if len(initial_state_ref)==2:
                rgb,state_dict = initial_state_ref
                print("using normal mode")
            elif len(initial_state_ref)==3:
                print("using bev mode")
                rgb,patch_coords,state_dict = initial_state_ref
                pos_id_kwargs['patch_coords'] = patch_coords
                pos_id_kwargs['mode'] = "bev"
            step_count = 0
            done = False
            messages = substitute_convo_template(self.rollout_config['convo_start_template'],state_dict['obs'] | self.rollout_config)
            # 2. The Interaction Loop
            vlm_logs={}
            # Trajectory Buffer (List is fine here!)
            trajectory_buffer = []
            instr_or_goal = state_dict['obs']['instr_or_goal']
            # episode_label = state_dict['info']['episode_label']
            while not done and step_count < self.rollout_config['max_steps']:
                # A. Prepare VLM Input
                rgb_numpy = rgb
                rgb_pil = Image.fromarray(rgb_numpy)
                # B. Call VLM (Blocking)
                # We must wait for the answer to decide the next step
                # print("inferring VLM with messages:")
                # print(messages)
                t0 = time.time()
                action_probs,action_logprobs,outputs = self.infer_probs(images=[rgb_pil],messages=messages,temperature = self.rollout_config['temperature'],pos_id_kwargs=pos_id_kwargs)
                
                vlm_logs |= {
                    'mean/vlm_latency': time.time() - t0,
                    'min/vlm_latency': time.time() - t0,
                    'max/vlm_latency': time.time() - t0,
                    'sum/spguard_trigger_count': 0,
                }
                try:
                    import torch
                    vlm_logs |= {"vlm_mem_GB":torch.cuda.memory_allocated()/(1024**3)}
                except:
                    print("warning: could not get vlm mem")
                # print(f"vlm step{step_count}")
                # print("done")
                #except for the first turn, all messages follow the exact same template.
                action_id = int(np.random.choice(len(action_probs), p=action_probs)) # sampling
                if action_id ==0 and self.rollout_config['stop_prob_threshold'] is not None:
                    if action_probs[0] >= self.rollout_config['stop_prob_threshold']:
                        action_id = 0
                    else:
                        vlm_logs['sum/spguard_trigger_count']=1
                        action_id = int(np.random.choice(len(action_probs)-1,p=action_probs[1:]/np.sum(action_probs[1:]))+1)
                    
                entropy = -np.sum(action_probs * np.log(action_probs + 1e-9))
                vlm_logs |= {'mean/entropy':float(entropy),'mean/action_prob':float(action_probs[action_id]),"action_probs":action_probs.tolist()}
                # D. Store Transition
               

                # D. Step Simulator (Blocking) ---------------------------RAY----------------------------- 
                t0 = time.time()
                # del rgb,state_dict
                state_ref = ray.get(env_handle.step.remote(action_id,supplementary_logs=vlm_logs))
                if len(state_ref)==2:
                    rgb,state_dict = state_ref
                elif len(state_ref)==3:
                    rgb,patch_coords,state_dict = state_ref
                    pos_id_kwargs['patch_coords'] = patch_coords
                    pos_id_kwargs['mode'] = "bev"
                vlm_logs = {'mean/sim_latency':time.time()-t0,'min/sim_latency':time.time()-t0,'max/sim_latency':time.time()-t0}
                if collect_trajectory:
                    # Append dict to list - fast and simple
                    trajectory_dict = {
                        "actions": action_id,
                        "rollout_logprobs": action_logprobs,
                        "rollout_probs": action_probs,
                        "rewards": state_dict.get("reward", 0.0),
                        "dones": state_dict['done'],
                        # "spl": state_dict['info']['spl'],
                        # "success": state_dict['info']['success'],
                        # "distance_to_goal": state_dict['info']['distance_to_goal'],
                        **state_dict['info'],
                    }
                    if compute_value:
                        import torch
                        # Compute value estimate for the current state
                        with torch.no_grad():
                            trajectory_dict["values"] = self._compute_value(outputs).cpu().numpy()
                    trajectory_buffer.append(trajectory_dict)
                messages = substitute_convo_template(self.rollout_config['convo_turn_template'],{"action":self.rollout_config['action_space'][action_id]})
                # print(f"sim step{step_count}")
                done = state_dict['done']
                step_count += 1
                # Convert list of dicts -> Dict of Numpy Arrays (Zero-Copy Friendly)
            final_trajectory = self._pack_trajectory(trajectory_buffer) if collect_trajectory else None
            final_info = state_dict['info'] | {"steps":step_count, "instr_or_goal":instr_or_goal}
            # Return Clean Tuple (No Actor Handles here)
            return state_dict['is_exhausted'], final_info, final_trajectory
        
        except Exception as e:
            print(f"Episode failed: {e}")
            import traceback
            traceback.print_exc()
            raise

class RolloutWorker(VLMWorker, EpisodeRolloutMixin):
    def __init__(self, rollout_config: Dict[str, Any], **vlm_kwargs):
        """
        Explicitly handles argument separation to avoid MRO issues.
        
        Args:
            rollout_config: Arguments intended for the EpisodeRolloutMixin.
            **vlm_kwargs: All other arguments (model_id, dtype, etc.) passed to VLMWorker.
        """
        # 1. Initialize the VLM Worker (The Heavy Lifter)
        # We pass only the relevant VLM args to avoid 'unexpected keyword argument' errors.
        VLMWorker.__init__(self, **vlm_kwargs)
        import os
        np.random.seed(os.getpid())
        # 2. Initialize the Mixin State
        # Since the Mixin's __init__ was just setting this variable, we can do it here directly
        # effectively bypassing the need for cooperative inheritance in the parents.
        self.rollout_config = rollout_config

class RLWorker(RolloutWorker,VLMTrainingMixin):
    def __init__(self, rollout_config: Dict[str, Any], **vlm_kwargs):
        """
        Combines VLM inference, RL Data Collection, and Training capabilities.
        """
        # 1. Initialize VLM (Heavy weights)
        VLMWorker.__init__(self, **vlm_kwargs)
        
        # 2. Initialize Rollout Config
        self.rollout_config = rollout_config
        
    def run_episode(self,env_handle,initial_state_ref,rtn_inputs = False):
        '''
        Returns:
        - is_exhausted: Whether the sim ran out of episodes
        - final_info: The final info dict from the sim (contains success, spl, etc.)
        - trajectory: If collect_trajectory is True, returns the collected trajectory in columnar format (dict of numpy arrays).    
        - inputs: optional, full input for model forward pass.
        
        '''
        self.rl_seq_inputs = None
        self.rl_embeds_inputs = None
        self.rl_trajectory = None
        if rtn_inputs:
            self.save_pixels = True #need pixels to reconstruct sequence inputs.
        else:
            self.save_pixels = False
        is_exhausted,result,trajectory = super().run_episode(env_handle, initial_state_ref,collect_trajectory=True,compute_value=False)
        self.rl_trajectory = trajectory
        inputs = None

        if rtn_inputs:
            inputs = self._pack_inputs()
            self.rl_seq_inputs = inputs
        return is_exhausted,result,trajectory,inputs

    @staticmethod
    def _write_eval_video(frames, output_path, fps):
        import imageio.v2 as imageio

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimwrite(
            output_path,
            frames,
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=8,
        )

    @staticmethod
    def _annotate_eval_frame(
        rgb,
        *,
        width,
        lines,
    ):
        from PIL import ImageDraw

        image = Image.fromarray(rgb).convert("RGB")
        if width and image.width != width:
            height = max(1, round(image.height * width / image.width))
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(image)
        line_height = 15
        panel_height = line_height * len(lines) + 8
        draw.rectangle((0, 0, image.width, panel_height), fill=(0, 0, 0))
        for index, line in enumerate(lines):
            draw.text((5, 4 + index * line_height), str(line), fill=(255, 255, 255))
        return np.asarray(image)

    def run_episode_batch(
        self,
        env_handle,
        initial_state_ref,
        eval_config=None,
    ):
        import torch

        batch_started = time.perf_counter()
        eval_config = dict(eval_config or {})
        eval_mode = bool(eval_config.get("enabled", False))
        capture_video = bool(eval_config.get("capture_video", False)) and (
            eval_config.get("video_style", "simple") == "simple"
        )
        video_width = int(eval_config.get("video_width", 480))
        video_fps = int(eval_config.get("video_fps", 4))
        video_output_dir = eval_config.get("video_output_dir")
        policy_update = eval_config.get("policy_update")
        previous_save_outputs = self.save_outputs
        if eval_mode:
            self.save_outputs = False

        rgb_batch, state_dicts = initial_state_ref
        batch_size = len(state_dicts)
        self.save_pixels = False
        self.rl_batch_sequence_states = [self.new_sequence_state() for _ in range(batch_size)]
        self.rl_batch_trajectories = [[] for _ in range(batch_size)]
        messages = [
            substitute_convo_template(
                self.rollout_config["convo_start_template"],
                state["obs"] | self.rollout_config,
            )
            for state in state_dicts
        ]
        instructions = [state["obs"]["instr_or_goal"] for state in state_dicts]
        step_counts = np.zeros(batch_size, dtype=np.int32)
        done = np.asarray([state["done"] for state in state_dicts], dtype=bool)
        video_frames = [[] for _ in range(batch_size)] if capture_video else None
        vlm_inference_seconds = 0.0
        environment_step_seconds = 0.0
        env_step_timeout_seconds = float(
            os.environ.get("LONGNAV_ENV_STEP_TIMEOUT_SECONDS", "300")
        )
        if env_step_timeout_seconds <= 0:
            raise ValueError("LONGNAV_ENV_STEP_TIMEOUT_SECONDS must be positive")

        try:
            while np.any(~done & (step_counts < self.rollout_config["max_steps"])):
                actions = np.zeros(batch_size, dtype=np.int64)
                decision_data = [None] * batch_size
                for index in range(batch_size):
                    if done[index] or step_counts[index] >= self.rollout_config["max_steps"]:
                        continue
                    self.restore_sequence_state(self.rl_batch_sequence_states[index])
                    rgb_pil = Image.fromarray(rgb_batch[index])
                    started = time.perf_counter()
                    action_probs, action_logprobs, outputs = self.infer_probs(
                        images=[rgb_pil],
                        messages=messages[index],
                        temperature=self.rollout_config["temperature"],
                        pos_id_kwargs={"mode": "standard"},
                    )
                    latency = time.perf_counter() - started
                    vlm_inference_seconds += latency
                    action_id = int(np.random.choice(len(action_probs), p=action_probs))
                    stop_threshold = self.rollout_config["stop_prob_threshold"]
                    spguard_triggered = 0
                    if (
                        action_id == 0
                        and stop_threshold is not None
                        and action_probs[0] < stop_threshold
                    ):
                        spguard_triggered = 1
                        action_id = int(
                            np.random.choice(
                                len(action_probs) - 1,
                                p=action_probs[1:] / np.sum(action_probs[1:]),
                            )
                            + 1
                        )
                    actions[index] = action_id
                    decision_data[index] = (
                        action_probs,
                        action_logprobs,
                        outputs,
                        latency,
                        spguard_triggered,
                    )
                    self.rl_batch_sequence_states[index] = self.capture_sequence_state()
                    if capture_video:
                        info = state_dicts[index].get("info", {})
                        video_frames[index].append(
                            self._annotate_eval_frame(
                                rgb_batch[index],
                                width=video_width,
                                lines=[
                                    f"policy update: {policy_update}",
                                    f"episode: {info.get('episode_label', 'unknown')}",
                                    f"step: {step_counts[index]}  goal: {instructions[index]}",
                                    (
                                        f"action: {self.rollout_config['action_space'][action_id]}  "
                                        f"p={float(action_probs[action_id]):.3f}"
                                    ),
                                    (
                                        "probs [stop/fwd/left/right]: "
                                        + "/".join(f"{float(value):.2f}" for value in action_probs)
                                    ),
                                    f"distance: {float(info.get('distance_to_goal', float('nan'))):.2f}",
                                ],
                            )
                        )

                env_started = time.perf_counter()
                step_ref = env_handle.step_batch.remote(
                    actions.tolist(), supplementary_logs=None
                )
                try:
                    rgb_batch, next_states = ray.get(
                        step_ref,
                        timeout=env_step_timeout_seconds,
                    )
                except ray.exceptions.GetTimeoutError as exc:
                    active_episodes = [
                        state.get("info", {}).get("episode_label", "unknown")
                        for index, state in enumerate(state_dicts)
                        if not done[index]
                    ]
                    raise RuntimeError(
                        "NavVerse vector environment step exceeded "
                        f"{env_step_timeout_seconds:.0f}s; "
                        f"active_episodes={active_episodes}"
                    ) from exc
                environment_step_seconds += time.perf_counter() - env_started
                for index, next_state in enumerate(next_states):
                    if decision_data[index] is None:
                        continue
                    (
                        action_probs,
                        action_logprobs,
                        outputs,
                        latency,
                        spguard_triggered,
                    ) = decision_data[index]
                    entropy = -np.sum(action_probs * np.log(action_probs + 1e-9))
                    trajectory_dict = {
                        "actions": int(actions[index]),
                        "rollout_logprobs": action_logprobs,
                        "rollout_probs": action_probs,
                        "rewards": next_state.get("reward", 0.0),
                        "dones": next_state["done"],
                        **next_state["info"],
                        "mean/vlm_latency": latency,
                        "mean/entropy": float(entropy),
                        "mean/action_prob": float(action_probs[actions[index]]),
                        "sum/spguard_trigger_count": spguard_triggered,
                        "vlm_mem_GB": torch.cuda.memory_allocated() / (1024**3),
                    }
                    self.rl_batch_trajectories[index].append(trajectory_dict)
                    messages[index] = substitute_convo_template(
                        self.rollout_config["convo_turn_template"],
                        {"action": self.rollout_config["action_space"][actions[index]]},
                    )
                    step_counts[index] += 1
                    done[index] = next_state["done"]
                state_dicts = next_states
        finally:
            self.save_outputs = previous_save_outputs

        self.rl_batch_trajectories = [
            self._pack_trajectory(buffer) for buffer in self.rl_batch_trajectories
        ]
        results = [
            state["info"]
            | {
                "steps": int(step_counts[index]),
                "instr_or_goal": instructions[index],
            }
            for index, state in enumerate(state_dicts)
        ]
        video_encode_started = time.perf_counter()
        if capture_video:
            if not video_output_dir:
                raise ValueError("video_output_dir is required when capture_video is enabled")
            for index, result in enumerate(results):
                video_frames[index].append(
                    self._annotate_eval_frame(
                        rgb_batch[index],
                        width=video_width,
                        lines=[
                            f"policy update: {policy_update}",
                            f"episode: {result.get('episode_label', 'unknown')}",
                            f"steps: {result['steps']}  goal: {instructions[index]}",
                            (
                                f"success: {float(result.get('success', 0.0)):.0f}  "
                                f"spl: {float(result.get('spl', 0.0)):.3f}"
                            ),
                            (
                                f"final distance: "
                                f"{float(result.get('distance_to_goal', float('nan'))):.2f}"
                            ),
                            f"termination: {result.get('termination_reason') or 'max_steps'}",
                        ],
                    )
                )
                label = result.get("episode_label", f"episode_{index}")
                output_path = os.path.join(video_output_dir, f"{label}.mp4")
                self._write_eval_video(video_frames[index], output_path, video_fps)
                result["video_path"] = output_path
        video_encode_seconds = time.perf_counter() - video_encode_started
        environment_profile = ray.get(
            env_handle.get_vector_profile.remote(reset=True)
        )
        worker_runtime = {
            "wall_seconds": time.perf_counter() - batch_started,
            "vlm_inference_seconds": vlm_inference_seconds,
            "environment_step_seconds": environment_step_seconds,
            "video_encode_seconds": video_encode_seconds,
            "episode_steps_total": int(step_counts.sum()),
            "episode_steps_max": int(step_counts.max(initial=0)),
            "environment_profile": environment_profile,
        }
        return bool(state_dicts[0]["is_exhausted"]), results, worker_runtime

    def postprocess_episode(self,eval=False):
        '''
        clears the internal state and returns the processed trajectory and model inputs.
        - trajectory includes: 
            - rollout logprobs (for rollout correction)
            - old logprobs (calculated from same weights as rollout model but with full forward pass instead of kv cache)
        If eval is True, skips forward passes and just returns raw trajectory
        '''
        model_inputs = None

        if not eval: # skip logprobs calculation during eval for speed.
            embeds = self._pack_embeds()
            self.rl_embeds_inputs = embeds
            values = None
            reuse_rollout_logprobs = os.environ.get(
                "LONGNAV_REUSE_ROLLOUT_LOGPROBS",
                "0",
            ).lower() in {"1", "true", "yes"}
            if reuse_rollout_logprobs:
                if self.rl_algo_config.use_value or self.rl_algo_config.use_ref:
                    raise ValueError(
                        "LONGNAV_REUSE_ROLLOUT_LOGPROBS requires use_value=false "
                        "and use_ref=false"
                    )
                if self.rl_embeds_inputs is None:
                    raise ValueError("No stored model inputs found for postprocessing.")
                model_inputs = self.rl_embeds_inputs
                self.rl_trajectory["old_logprobs"] = np.asarray(
                    self.rl_trajectory["rollout_logprobs"],
                    dtype=np.float32,
                )
            else:
                import torch

                with torch.no_grad():
                    if self.rl_embeds_inputs is not None:
                        logits,values = self._forward_embeds(self.rl_embeds_inputs,self.rl_algo_config.use_value)
                        model_inputs = self.rl_embeds_inputs
                    elif self.rl_seq_inputs is not None:
                        logits,values = self._forward_seq(self.rl_seq_inputs,self.rl_algo_config.use_value)
                        model_inputs = self.rl_seq_inputs
                    else:
                        raise ValueError("No stored model inputs found for postprocessing.")
                    logprobs = self._calculate_action_logprobs(logits).squeeze().float().cpu()
                    if logprobs.dim() == 1:
                        logprobs = logprobs.unsqueeze(0) # ensure batch dim
                    self.rl_trajectory['old_logprobs'] = logprobs.numpy()
                    if values is not None:
                        self.rl_trajectory['values'] = values.squeeze().float().cpu().numpy()

            if self.rl_algo_config.use_ref:
                import torch

                with torch.no_grad():
                    self.unmerge_adapter()
                    with self.model.disable_adapter():
                        if self.rl_embeds_inputs is not None:
                            logits,values = self._forward_embeds(self.rl_embeds_inputs,False)
                        elif self.rl_seq_inputs is not None:
                            logits,values = self._forward_seq(self.rl_seq_inputs)
                        ref_logprobs = self._calculate_action_logprobs(logits).squeeze().float().cpu()
                        if ref_logprobs.dim() == 1:
                            ref_logprobs = ref_logprobs.unsqueeze(0) # ensure batch dim
                        self.rl_trajectory['ref_logprobs'] = ref_logprobs.numpy()

        return self.rl_trajectory,model_inputs    

    def postprocess_batch(self, eval=False):
        processed = []
        for sequence_state, trajectory in zip(
            self.rl_batch_sequence_states, self.rl_batch_trajectories
        ):
            self.restore_sequence_state(sequence_state)
            self.rl_trajectory = trajectory
            self.rl_seq_inputs = None
            self.rl_embeds_inputs = None
            processed.append(RLWorker.postprocess_episode(self, eval=eval))
        self.reset(clear_cuda_cache=False)
        return processed
    
class RLActor(RLWorker):
    def run_episode(self,env_handle,initial_state_ref):
        is_exhausted,state_dict,_,_ = super().run_episode(env_handle, initial_state_ref)
        return is_exhausted,state_dict
    
    def postprocess_episode(self,return_inputs = True,eval=False):
        trajectory,model_inputs = super().postprocess_episode(eval=eval)
        if return_inputs:
            inputs_tensors,inputs_metadata = TensorPacker.pack(model_inputs)
            return trajectory,inputs_tensors,inputs_metadata
        return trajectory, None, None

    def postprocess_batch(self, return_inputs=True, eval=False):
        processed = RLWorker.postprocess_batch(self, eval=eval)
        results = []
        for trajectory, model_inputs in processed:
            if return_inputs:
                input_tensors, input_metadata = TensorPacker.pack(model_inputs)
                results.append((trajectory, input_tensors, input_metadata))
            else:
                results.append((trajectory, None, None))
        return results

    def train_rl_step(self, embeds_inputs_np, embeds_inputs_meta,traj_batch):
        embeds_inputs = TensorPacker.unpack(embeds_inputs_np,embeds_inputs_meta,device=self.accelerator.device)
        actions = traj_batch['actions']
        old_log_prob = traj_batch['old_log_prob']
        advantages = traj_batch['advantages']
        returns = traj_batch['returns']
        old_values = traj_batch.get('values',None)
        rollout_log_probs = traj_batch.get('rollout_logprobs',None)
        ref_logprobs = traj_batch.get('ref_logprobs',None)
    
        return super().train_rl_step(embeds_inputs, actions, old_log_prob, advantages, returns, old_values, rollout_log_probs,ref_logprobs)
    
    def train_dagger_step(self, embeds_inputs_np, embeds_inputs_meta, traj_batch):
        """
        Pure DAgger (Behavior Cloning) Step.
        """
        embeds_inputs = TensorPacker.unpack(embeds_inputs_np, embeds_inputs_meta, device=self.accelerator.device)
        
        # 1. Prepare Data
        # Ensure we have the mask (default to all valid tokens if not provided)
        dagger_mask = traj_batch.get('dagger_mask', traj_batch.get('response_mask', None))
        if dagger_mask is None:
            import torch
             # Fallback: Create a ones mask matching action shape
            dagger_mask = torch.ones_like(traj_batch['actions'], dtype=torch.bool)

        expert_actions = traj_batch['oracle_actions']
        # Robustness: Handle One-Hot vs Indices
        # If input is (B, S, A) probabilities, convert to (B, S) indices
        if expert_actions.dim() > 2:
            expert_actions = expert_actions.argmax(dim=-1)

        # 2. Dispatch
        return super().generic_train_step(
            embeds_inputs=embeds_inputs,
            loss_fn_names=['bc'],
            loss_kwargs_list={
                'bc': {
                    'expert_actions': expert_actions,
                    'dagger_mask': dagger_mask
                }
            }
        )

    def train_dagrl_step(self, embeds_inputs_np, embeds_inputs_meta, traj_batch,rl_weight=1.0,bc_weight=0.1):
        """
        Hybrid Step: PPO + DAgger.
        Assumes the driver has populated 'response_mask' (for PPO) and 
        'dagger_mask' (for BC) to separate the data.
        """
        embeds_inputs = TensorPacker.unpack(embeds_inputs_np, embeds_inputs_meta, device=self.accelerator.device)
        
        # 1. Unpack RL Args
        actions = traj_batch['actions']
        old_log_prob = traj_batch['old_log_prob']
        advantages = traj_batch['advantages']
        returns = traj_batch['returns']
        old_values = traj_batch.get('values', None)
        rollout_log_probs = traj_batch.get('rollout_logprobs', None)
        ref_logprobs = traj_batch.get('ref_logprobs', None)
        
        # The driver MUST provide this to avoid double-counting gradients
        # (e.g. response_mask should be FALSE for DAgger samples)
        rl_mask = traj_batch.get('response_mask') 
        if rl_mask is None:
            import torch
             # Fallback (dangerous for hybrid): Assume all valid tokens are RL
            rl_mask = torch.ones_like(actions, dtype=torch.bool)

        # 2. Unpack DAgger Args
        expert_actions = traj_batch['oracle_actions']
        if expert_actions.dim() > 2:
            expert_actions = expert_actions.argmax(dim=-1)
            
        dagger_mask = traj_batch.get('dagger_mask')
        if dagger_mask is None:
            import torch
            # If missing in hybrid, assume NO DAgger (safety default)
            dagger_mask = torch.zeros_like(actions, dtype=torch.bool)

        # 3. Dispatch
        return super().generic_train_step(
            embeds_inputs=embeds_inputs,
            loss_fn_names=['rl', 'bc'],
            loss_kwargs_list={
                'rl': {
                    'actions': actions,
                    'old_log_prob': old_log_prob,
                    'advantages': advantages,
                    'returns': returns,
                    'old_values': old_values,
                    'rollout_log_probs': rollout_log_probs,
                    'ref_log_probs': ref_logprobs,
                    'response_mask': rl_mask
                },
                'bc': {
                    'expert_actions': expert_actions,
                    'dagger_mask': dagger_mask
                }
            },
            loss_weights=[rl_weight, bc_weight]
        )
        
def collect_rollouts(
    env_handles: list,
    vlm_handles: list,
    shard_iterator: Iterator[list[str]],
    target_episodes: int = float('inf'),
    postprocess_kwargs = {"return_inputs":True, "eval":False},
    wandb_logger = None
) -> tuple[list,list,list]:
    """
    Orchestrates the RL collection pipeline.

    returns: trajectory buffer, result list, log list, indexed by dispatch_id
    """

    # --- 1. Initialize Pools ---
    idle_vlms = deque(vlm_handles)
    ready_sims = deque()

    # --- 2. Tracking Futures ---
    pending_resets = {}   # reset_ref -> sim_handle
    active_episodes = {}  # ep_ref -> dispatch_id

    # VLM post-processing
    pending_postproc = {} # pp_ref -> vlm_handle, dispatch_id 
    # Sim logging
    pending_logs = {} # log_ref -> sim_handle, dispatch_id

    trajectory_buffer = []
    trajectory_ids = []

    result_dict = {}
    log_dict = {}
    last_dispatch_time = time.time()
    # --- 3. Bootstrap: Initial Sharding & Resets ---
    for env_handle in env_handles:
        try:
            if ray.get(env_handle.is_exhausted.remote()):
                initial_shard = next(shard_iterator)
                env_handle.assign_shard.remote(initial_shard)
            reset_ref = env_handle.reset.remote()
            pending_resets[reset_ref] = env_handle
        except StopIteration:
            print("Warning: Not enough shards for all workers during bootstrap.")
    print(f"Bootstrapping: Initializing {len(env_handles)} environments...")
    initial_live_sims = len(pending_resets)
    # Helper to check if we should keep the loop alive
    def has_work():

        # 1. Are tasks currently running?
        is_active = len(active_episodes) > 0 or len(pending_postproc) > 0#or len(pending_logs) > 0
        # 2. Do we still want to launch new tasks (now or in the future)? (Resources available AND Target not met)
        potential = len(trajectory_buffer) + len(active_episodes) + len(pending_postproc)
        want_launch = (potential < target_episodes) and initial_live_sims > 0
        return is_active or want_launch

    dispatch_counter = 0
    # --- Event Loop ---
    while has_work():
        # A. Dispatch (IDENTICAL)
        total_potential = len(trajectory_buffer) + len(active_episodes) + len(pending_postproc)

        while (idle_vlms and ready_sims and total_potential < target_episodes):
            vlm = idle_vlms.popleft()
            sim, init_state_ref = ready_sims.popleft()
            ep_ref = vlm.run_episode.remote(sim, init_state_ref)
            active_episodes[ep_ref] = dispatch_counter,vlm,sim
            dispatch_counter +=1
            total_potential +=1


        # B. Wait for Events
        all_watch_refs = list(pending_resets.keys()) + \
                         list(active_episodes.keys()) + \
                         list(pending_postproc.keys()) + \
                         list(pending_logs.keys())

        if not all_watch_refs:
            break

        ready_refs, _ = ray.wait(all_watch_refs, num_returns=1,timeout=15.0)
        if not ready_refs:
            # If we get here, the orchestrator is "stuck" waiting.
            # We can use this moment to diagnose.

            # Simple deadlock detector:
            current_time = time.time()
            if current_time - last_dispatch_time > 360: # 6 minutes
                print(f"DEBUG: System frozen for >6m. Active: {len(active_episodes)}, PostProc: {len(pending_postproc)}")
                if wandb_logger is not None:
                    ray.get(wandb_logger.alert.remote(title="Rollout Collection Frozen",text=f"Active Episodes: {len(active_episodes)}, Pending PostProc: {len(pending_postproc)}",level="ERROR"))
                # Check 1: Are we waiting on a specific ref forever?
                # Dump the first few active refs to inspect
                import ipdb; ipdb.set_trace()

            continue # Jump back to start of loop (and potentially dispatch more if resources freed up)

        # CHANGE 3: Update timestamp when we actually get a result
        last_dispatch_time = time.time()
        for ref in ready_refs:

            # --- CASE 1: Reset Finished ---
            if ref in pending_resets:
                env_handle = pending_resets.pop(ref)
                ready_sims.append((env_handle, ref))

            # --- CASE 2: Episode Finished ---
            elif ref in active_episodes:
                # print("handling finished episode")
                dispatch_id, vlm, sim =  active_episodes.pop(ref)
                # Unpack results
                is_exhausted, result = ray.get(ref)
                result_dict[dispatch_id] = result

                # send vlm and sim to post episode processing
                pp_ref = vlm.postprocess_episode.remote(**postprocess_kwargs)
                pending_postproc[pp_ref] = vlm,dispatch_id

                log_ref = sim.flush_logs_to_disk.remote()
                pending_logs[log_ref] = sim,dispatch_id,is_exhausted

            # --- CASE 3: VLM Post-Processing Finished ---
            elif ref in pending_postproc:
                vlm,dispatch_id = pending_postproc.pop(ref)
                trajectory_buffer.append(ref)
                trajectory_ids.append(dispatch_id)
                idle_vlms.append(vlm)
                print(f"Collected episode {len(trajectory_buffer)}")

            # --- CASE 4: Sim Log Flush Finished
            elif ref in pending_logs:
                sim,dispatch_id,is_exhausted = pending_logs.pop(ref)
                log_dict[dispatch_id] = ref # save the path to the log
                # send the sim to reset/reshard so it can start working again asap
                try:
                    # print("logging done",end="")
                    if is_exhausted:
                        # print("assigning shard")
                        new_shard = next(shard_iterator)
                        sim.assign_shard.remote(new_shard)
                    # print("resetting sim")
                    new_reset_ref = sim.reset.remote()
                    pending_resets[new_reset_ref] = sim
                except StopIteration:
                    # No more work. Retire the Habitat worker.
                    # iterator_exhausted = True
                    pass
    rollouts = [t for _, t in sorted(zip(trajectory_ids, trajectory_buffer))]
    log_dict |={v[1]:k for k,v in pending_logs.items()}
    num_rollouts = len(rollouts)
    result_list = [result_dict[i] for i in range(num_rollouts)]
    log_list = [log_dict[i] for i in range(num_rollouts)]
    return ray.get(rollouts), result_list, log_list


def collect_vector_rollouts(
    env_handles,
    vlm_handles,
    shard_iterator: Iterator[list[str]],
    target_episodes: int,
    episodes_per_worker: int,
    postprocess_kwargs={"return_inputs": True, "eval": False},
    eval_config=None,
    return_timings=False,
):
    collection_started = time.perf_counter()
    if len(env_handles) != len(vlm_handles):
        raise ValueError(
            f"Vector rollout requires one VLM per simulator; got "
            f"{len(vlm_handles)} VLMs and {len(env_handles)} simulators"
        )
    episodes_per_wave = len(env_handles) * episodes_per_worker
    if target_episodes <= 0 or target_episodes % episodes_per_wave:
        raise ValueError(
            "Vector rollout target must be a positive multiple of workers * "
            f"episodes_per_worker; got {target_episodes} and wave size "
            f"{len(env_handles)} * {episodes_per_worker} = {episodes_per_wave}"
        )
    wave_count = target_episodes // episodes_per_wave
    benchmark_video = bool(
        eval_config
        and eval_config.get("capture_video", False)
        and eval_config.get("video_style") == "benchmark"
    )
    assign_seconds = 0.0
    episode_seconds = 0.0
    postprocess_seconds = 0.0
    benchmark_video_futures = []
    rollouts = []
    results = []
    logs = []
    worker_runtimes = []
    wave_runtimes = []
    wave_timeout_seconds = float(
        os.environ.get("LONGNAV_VECTOR_WAVE_TIMEOUT_SECONDS", "1800")
    )
    if wave_timeout_seconds <= 0:
        raise ValueError("LONGNAV_VECTOR_WAVE_TIMEOUT_SECONDS must be positive")
    for wave_index in range(wave_count):
        worker_labels = [next(shard_iterator) for _ in env_handles]
        if any(len(labels) != episodes_per_worker for labels in worker_labels):
            raise ValueError(
                "Vector rollout shards must contain exactly "
                f"{episodes_per_worker} labels"
            )
        print(
            "Vector rollout wave start "
            f"wave={wave_index + 1}/{wave_count} "
            f"episodes={sum(len(labels) for labels in worker_labels)} "
            f"scenes={[labels[0].rsplit('_', 1)[0] for labels in worker_labels]}"
        )

        assign_started = time.perf_counter()
        ray.get(
            [
                env_handle.assign_shard.remote(labels)
                for env_handle, labels in zip(env_handles, worker_labels)
            ]
        )
        wave_assign_seconds = time.perf_counter() - assign_started
        assign_seconds += wave_assign_seconds
        ray.get(
            [
                env_handle.configure_benchmark_video_capture.remote(
                    eval_config if benchmark_video else None
                )
                for env_handle in env_handles
            ]
        )

        episode_started = time.perf_counter()
        initial_states = [
            env_handle.reset_batch.remote() for env_handle in env_handles
        ]
        episode_futures = [
            vlm_handle.run_episode_batch.remote(
                env_handle,
                initial_state,
                eval_config=eval_config,
            )
            for env_handle, vlm_handle, initial_state in zip(
                env_handles, vlm_handles, initial_states
            )
        ]
        try:
            episode_outputs = ray.get(
                episode_futures,
                timeout=wave_timeout_seconds,
            )
        except ray.exceptions.GetTimeoutError as exc:
            ready, _ = ray.wait(
                episode_futures,
                num_returns=len(episode_futures),
                timeout=0,
            )
            raise RuntimeError(
                "Vector rollout wave exceeded "
                f"{wave_timeout_seconds:.0f}s: wave={wave_index + 1}/{wave_count}, "
                f"completed_workers={len(ready)}/{len(episode_futures)}, "
                f"worker_labels={worker_labels}"
            ) from exc
        wave_episode_seconds = time.perf_counter() - episode_started
        episode_seconds += wave_episode_seconds
        print(
            "Vector rollout wave complete "
            f"wave={wave_index + 1}/{wave_count} "
            f"seconds={wave_episode_seconds:.1f}"
        )

        if benchmark_video:
            from longnav.utils.benchmark_video_worker import (
                render_benchmark_video_batch,
            )

            video_runtime_env = {
                "conda": os.environ.get("LONGNAV_HABITAT_CONDA_ENV", "navverse"),
                "env_vars": {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
            }
            render_video = ray.remote(render_benchmark_video_batch).options(
                num_cpus=2,
                num_gpus=0,
                runtime_env=video_runtime_env,
            )
            captures = [
                env_handle.export_benchmark_video_capture.remote()
                for env_handle in env_handles
            ]
            benchmark_video_futures.extend(
                render_video.remote(capture) for capture in captures
            )

        postprocess_started = time.perf_counter()
        rollout_batches = ray.get(
            [
                vlm_handle.postprocess_batch.remote(**postprocess_kwargs)
                for vlm_handle in vlm_handles
            ]
        )
        wave_postprocess_seconds = time.perf_counter() - postprocess_started
        postprocess_seconds += wave_postprocess_seconds
        log_refs = [
            env_handle.flush_logs_to_disk.remote() for env_handle in env_handles
        ]
        wave_workers = []
        for worker_rollouts, episode_output, log_ref in zip(
            rollout_batches,
            episode_outputs,
            log_refs,
        ):
            if len(episode_output) == 3:
                _, worker_results, worker_runtime = episode_output
            else:
                _, worker_results = episode_output
                worker_runtime = {}
            if len(worker_rollouts) != episodes_per_worker:
                raise RuntimeError(
                    f"Vector VLM returned {len(worker_rollouts)} rollouts; "
                    f"expected {episodes_per_worker}"
                )
            if len(worker_results) != episodes_per_worker:
                raise RuntimeError(
                    f"Vector simulator returned {len(worker_results)} results; "
                    f"expected {episodes_per_worker}"
                )
            if benchmark_video:
                for result in worker_results:
                    label = result.get("episode_label")
                    result["video_path"] = os.path.join(
                        eval_config["video_output_dir"],
                        f"{label}_obs_action_LongNav.mp4",
                    )
            rollouts.extend(worker_rollouts)
            results.extend(worker_results)
            logs.extend([log_ref] * episodes_per_worker)
            worker_runtimes.append(worker_runtime)
            wave_workers.append(worker_runtime)
        wave_runtimes.append(
            {
                "wave_index": wave_index,
                "assign_seconds": wave_assign_seconds,
                "episode_seconds": wave_episode_seconds,
                "postprocess_seconds": wave_postprocess_seconds,
                "worker_runtimes": wave_workers,
            }
        )
    timings = {
        "total_seconds": time.perf_counter() - collection_started,
        "assign_seconds": assign_seconds,
        "episode_seconds": episode_seconds,
        "postprocess_seconds": postprocess_seconds,
        "worker_runtimes": worker_runtimes,
        "wave_runtimes": wave_runtimes,
        "video_futures": benchmark_video_futures,
    }
    if return_timings:
        return rollouts, results, logs, timings
    return rollouts, results, logs

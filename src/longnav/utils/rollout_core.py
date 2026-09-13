import ray
import os
from collections import deque
from typing import List, Dict, Any, Iterator, Callable, Optional
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


def reduce_gaussian_logprob(log_prob: np.ndarray, mode: str = "sum") -> np.ndarray:
    """Collapse a diagonal Gaussian's per-dimension log-prob to one number per step.

    `sum` is the joint log-density and is the DEFAULT, so every existing continuous run is
    byte-identical. It is also the only correct joint likelihood; the alternative exists for
    a specific, measured reason:

    a PPO ratio is `exp(sum of D log-ratios)`, whose variance grows with `D`. At the
    Gaussian head's `D = 2` that is a non-issue. At the latent head's `D = 1024` the ratio
    saturates the clip on nearly every sample and the gradient is whatever survives. `mean`
    is the dimension-level analogue of the token-level ratios sequence-level RLHF uses: a
    biased surrogate, chosen deliberately, and never silently -- it must be asked for in the
    policy-head config.

    Masking the inactive subspace instead is the third option and is NOT implemented here:
    it needs the per-dimension KL from the SFT run to decide which dims carry information,
    and inventing that set before it is measured would be guessing dressed as a feature.
    """
    if mode == "sum":
        return np.sum(log_prob, axis=-1).astype(np.float32)
    if mode == "mean":
        return np.mean(log_prob, axis=-1).astype(np.float32)
    raise ValueError(
        f"logprob_reduction must be 'sum' or 'mean', got {mode!r}"
    )


def _align_readouts_to_executed_actions(
    trajectory, logit_indices, value_logit_indices, policy_head_type
):
    action_key = (
        "actions_continuous" if policy_head_type == "continuous" else "actions"
    )
    action_count = len(trajectory[action_key])
    readout_count = len(logit_indices)
    if readout_count < action_count:
        raise RuntimeError(
            f"Trajectory has {action_count} actions but only "
            f"{readout_count} policy readouts"
        )
    if readout_count == action_count:
        return 0

    reasons = np.asarray(trajectory.get("termination_reason", [])).reshape(-1)
    watchdog_stop = bool(
        len(reasons) and str(reasons[-1]) == "wall_timeout_soft_stop"
    )
    if not watchdog_stop:
        raise RuntimeError(
            f"Trajectory has {action_count} actions but {readout_count} "
            "policy readouts without a watchdog stop"
        )

    dropped = readout_count - action_count
    del logit_indices[action_count:]
    del value_logit_indices[action_count:]
    print(
        f"[rollout] dropping {dropped} readout(s) for watchdog-stopped "
        "actions that were never executed",
        flush=True,
    )
    return dropped

class EpisodeRolloutMixin:
    def _seed_ode_episode(self, state_dict):
        """Give each ODE evaluation episode a stable latent trajectory and stop RNG."""
        from hashlib import blake2b
        from longnav.utils.flow_sde_policy import FlowSDEHead

        label = str(state_dict.get("info", {}).get("episode_label", ""))
        if not label:
            return
        seed = int.from_bytes(blake2b(label.encode(), digest_size=8).digest(), "little")
        self._policy_stop_rng = np.random.default_rng(seed)
        model = getattr(self, "model", None)
        if model is None:
            return
        for module in model.modules():
            if isinstance(module, FlowSDEHead) and module.force_ode:
                module.seed(seed)

    def _stop_target(self, state_dict):
        distance = state_dict.get("info", {}).get("distance_to_goal")
        if distance is None or not np.isfinite(distance):
            return None
        return float(distance <= self.rollout_config.get("stop_head_radius_m", 1.0))

    def _shadow_stop_decision(self, state_dict, probability=None):
        """Sample an auxiliary STOP action without changing simulator control."""
        target = self._stop_target(state_dict)
        probability = getattr(self, "last_stop_probability", None) if probability is None else probability
        if (self.rollout_config.get("stop_execution_mode") != "shadow"
                or probability is None or target is None):
            return {"stop_target": target, "shadow_stop_action": np.nan,
                    "shadow_stop_reward": np.nan, "probe_p_stop": probability}
        temperature = float(getattr(
            getattr(self, "rl_algo_config", None), "state_probe_stop_temperature", 1.0
        ))
        if temperature <= 0.0:
            raise ValueError("state_probe_stop_temperature must be positive")
        clipped = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
        tempered = 1.0 / (1.0 + np.exp(-np.log(clipped / (1.0 - clipped)) / temperature))
        action = float(np.random.random() < tempered)
        if action:
            reward = (float(self.rollout_config.get("stop_shadow_correct_reward", 1.0))
                      if target > 0.5 else
                      -float(self.rollout_config.get("stop_shadow_false_penalty", 1.0)))
        elif target > 0.5:
            reward = -float(self.rollout_config.get("stop_shadow_miss_penalty", 1.0))
        else:
            reward = 0.0
        return {
            "stop_target": target,
            "shadow_stop_action": action,
            "shadow_stop_reward": reward,
            "probe_p_stop": probability,
        }

    @staticmethod
    def _trajectory_path_length_m(action_to_env) -> float:
        """Measure the decoded cumulative XY path that the controller would execute."""
        action = np.asarray(action_to_env, dtype=np.float64)
        if action.ndim == 1:
            if action.size >= 3 and action.size % 3 == 0:
                xy = action.reshape(-1, 3)[:, :2]
            elif action.size >= 2:
                xy = action[:2].reshape(1, 2)
            else:
                return float("nan")
        elif action.shape[-1] >= 2:
            xy = action.reshape(-1, action.shape[-1])[:, :2]
        else:
            return float("nan")
        if not np.isfinite(xy).all():
            return float("nan")
        anchored = np.vstack((np.zeros((1, 2), dtype=np.float64), xy))
        return float(np.linalg.norm(np.diff(anchored, axis=0), axis=1).sum())

    def _policy_stop_decision(self, action_to_env, stop_probability):
        """Resolve the configured stop actuator from the same decoded action as control."""
        mode = str(self.rollout_config.get("stop_execution_mode", "physical"))
        if self.policy_head_config["type"] != "continuous":
            return False, mode, float("nan")
        path_length = self._trajectory_path_length_m(action_to_env)
        if mode == "shadow":
            return False, mode, path_length
        if mode == "physical":
            threshold = self.rollout_config.get("stop_prob_threshold")
            return (
                threshold is not None
                and stop_probability is not None
                and stop_probability >= float(threshold),
                mode,
                path_length,
            )
        if mode == "sampled":
            if stop_probability is None:
                return False, mode, path_length
            temperature = float(self.rollout_config.get("stop_sample_temperature", 1.0))
            if temperature <= 0.0:
                raise ValueError("stop_sample_temperature must be positive")
            probability = float(np.clip(stop_probability, 1e-6, 1.0 - 1e-6))
            logit = np.log(probability / (1.0 - probability)) / temperature
            probability = 1.0 / (1.0 + np.exp(-logit))
            rng = getattr(self, "_policy_stop_rng", None)
            if rng is None:
                rng = np.random.default_rng()
                self._policy_stop_rng = rng
            return bool(rng.random() < probability), mode, path_length
        if mode == "trajectory_length":
            threshold = self.rollout_config.get("trajectory_stop_threshold_m")
            if threshold is None or float(threshold) < 0.0:
                raise ValueError(
                    "trajectory_length stop requires a non-negative "
                    "trajectory_stop_threshold_m"
                )
            return path_length <= float(threshold), mode, path_length
        raise ValueError(
            "stop_execution_mode must be one of shadow, physical, sampled, "
            "trajectory_length; "
            f"got {mode!r}"
        )

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

    def _sample_chain_action(self, policy_out, logs, sample_chain):
        if self.rollout_config.get("use_oracle_action"):
            raise RuntimeError(
                "use_oracle_action is incompatible with a chain-action head: "
                "an oracle chunk does not determine a denoising chain"
            )
        chain, positions, logprob, action_to_env = sample_chain(
            np.asarray(policy_out["h"], dtype=np.float32).reshape(-1)
        )
        logs.update(
            {
                "mean/action_l2": float(np.linalg.norm(action_to_env)),
                "mean/action_abs_mean": float(np.mean(np.abs(action_to_env))),
                "mean/chain_logprob": float(logprob),
            }
        )
        return chain, action_to_env, np.float32(logprob), positions

    def _sample_continuous_action(self, policy_out, state_dict, logs, head):
        mu = policy_out["mu"]
        log_std = policy_out["log_std"]
        std = np.exp(log_std)
        use_oracle = self.rollout_config.get("use_oracle_action")
        if getattr(head, "force_mean", False) and not use_oracle:
            action = np.asarray(mu, dtype=np.float32)
        elif use_oracle:
            if "oracle_action" not in state_dict["info"]:
                raise RuntimeError(
                    "use_oracle_action is enabled but state_dict['info'] has no "
                    "oracle_action"
                )
            action = np.asarray(state_dict["info"]["oracle_action"], dtype=np.float32)
        else:
            action = np.clip(
                np.random.normal(mu, std),
                self.policy_head_config["continuous_action_clip_low"],
                self.policy_head_config["continuous_action_clip_high"],
            )
        log_prob = -0.5 * (
            ((action - mu) / std) ** 2 + 2.0 * log_std + np.log(2.0 * np.pi)
        )
        credited_action = action.astype(np.float32).reshape(-1)
        action_to_env = credited_action
        decode_action = getattr(head, "decode_action", None)
        if decode_action is not None:
            action_to_env = decode_action(action_to_env)
        logs.update(
            {
                "mean/action_l2": float(np.linalg.norm(action_to_env)),
                "mean/action_abs_mean": float(np.mean(np.abs(action_to_env))),
            }
        )
        logprob = reduce_gaussian_logprob(
            log_prob, self.policy_head_config.get("logprob_reduction", "sum")
        )
        return credited_action, action_to_env, logprob, None

    def _sample_discrete_action(self, action_probs, state_dict, logs):
        if self.rollout_config.get("use_oracle_action"):
            if "oracle_action" not in state_dict["info"]:
                raise RuntimeError(
                    "use_oracle_action is enabled but state_dict['info'] has no "
                    "oracle_action"
                )
            action_id = int(state_dict["info"]["oracle_action"])
        else:
            action_id = int(np.random.choice(len(action_probs), p=action_probs))
            threshold = self.rollout_config["stop_prob_threshold"]
            if action_id == 0 and threshold is not None and action_probs[0] < threshold:
                logs["sum/spguard_trigger_count"] = 1
                action_id = int(
                    np.random.choice(
                        len(action_probs) - 1,
                        p=action_probs[1:] / np.sum(action_probs[1:]),
                    )
                    + 1
                )
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-9))
        logs.update(
            {
                "mean/entropy": float(entropy),
                "mean/action_prob": float(action_probs[action_id]),
                "action_probs": action_probs.tolist(),
            }
        )
        return action_id, action_id, None

    def _sample_action_for_state(self, policy_out, action_logprobs, state_dict, logs):
        """Sample one policy action and keep the credited action separate from actuation."""
        chain_head = getattr(getattr(self, "model", None), "action_head", None)
        sample_chain = getattr(chain_head, "sample_chain_np", None)
        uses_chain_action = (
            self.policy_head_config["type"] == "continuous"
            and sample_chain is not None
        )
        if uses_chain_action:
            action_id, action_to_env, action_logprobs, positions = (
                self._sample_chain_action(policy_out, logs, sample_chain)
            )
        elif self.policy_head_config["type"] == "continuous":
            action_id, action_to_env, action_logprobs, positions = (
                self._sample_continuous_action(policy_out, state_dict, logs, chain_head)
            )
        else:
            action_id, action_to_env, positions = self._sample_discrete_action(
                policy_out, state_dict, logs
            )

        if self.policy_head_config["type"] == "continuous":
            action_for_context = action_to_env if uses_chain_action else action_id
            logs["action_path_length_m"] = self._trajectory_path_length_m(action_to_env)
            action_text = ",".join(
                f"{value:.3f}" for value in np.asarray(action_for_context).reshape(-1)
            )
        else:
            action_text = self.rollout_config["action_space"][action_id]
        return action_id, action_to_env, action_logprobs, positions, action_text

    @staticmethod
    def _unpack_env_state(payload, pos_id_kwargs):
        if len(payload) == 2:
            rgb, state_dict = payload
        elif len(payload) == 3:
            rgb, patch_coords, state_dict = payload
            pos_id_kwargs["patch_coords"] = patch_coords
            pos_id_kwargs["mode"] = "bev"
        else:
            raise ValueError(f"Unsupported environment state tuple length: {len(payload)}")
        return rgb, state_dict

    def _build_transition(
        self,
        action_id,
        action_logprobs,
        sde_positions,
        state_dict,
        outputs,
        compute_value,
        decision_logs,
        stop_target=None,
        shadow_stop_action=np.nan,
        shadow_stop_reward=np.nan,
        policy_action_mask=True,
        reward_override=None,
        done_override=None,
    ):
        trajectory = {
            "rollout_logprobs": action_logprobs,
            "rewards": (state_dict.get("reward", 0.0)
                        if reward_override is None else float(reward_override)),
            "dones": state_dict["done"] if done_override is None else bool(done_override),
            **state_dict["info"],
            "stop_target": (np.nan if stop_target is None else float(stop_target)),
            "shadow_stop_action": float(shadow_stop_action),
            "shadow_stop_reward": float(shadow_stop_reward),
            "policy_action_mask": bool(policy_action_mask),
            "action_path_length_m": float(
                decision_logs.get("action_path_length_m", np.nan)
            ),
            "policy_stop_mode": decision_logs.get("policy_stop_mode"),
            "decision_distance_to_goal_m": float(
                decision_logs.get("decision_distance_to_goal_m", np.nan)
            ),
            "history_turns": int(decision_logs.get("history_turns", 0)),
            "probe_p_stop": (
                np.nan
                if getattr(self, "last_stop_probability", None) is None
                else float(self.last_stop_probability)
            ),
        }
        if self.policy_head_config["type"] == "continuous":
            trajectory["actions_continuous"] = np.asarray(action_id, dtype=np.float32)
            if sde_positions is not None:
                trajectory["sde_positions"] = np.asarray(sde_positions, dtype=np.int64)
            elif callable(getattr(
                getattr(getattr(self, "model", None), "action_head", None),
                "chain_log_prob_batch", None,
            )):
                # STOP turns carry an uncredited sampled chain; retain a valid dummy
                # position vector so postprocessing can score the masked transition.
                n_sde = int(getattr(self.model.action_head.sde, "n", 0))
                trajectory["sde_positions"] = np.zeros(n_sde, dtype=np.int64)
        else:
            trajectory["actions"] = action_id
            trajectory["rollout_probs"] = np.asarray(decision_logs["action_probs"])
        if compute_value:
            if not hasattr(self, "_compute_value"):
                raise NotImplementedError(
                    "per-step rollout values need a _compute_value implementation"
                )
            import torch

            with torch.no_grad():
                trajectory["values"] = self._compute_value(outputs).cpu().numpy()
        return trajectory
    
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
            rgb, state_dict = self._unpack_env_state(initial_state_ref, pos_id_kwargs)
            if state_dict.get('exhausted_sentinel'):
                # The env's shard ran dry during reset (see objectnav_continuous.reset):
                # report exhausted with no episode; the collector retires this sim.
                return True, None, None
            self._seed_ode_episode(state_dict)
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
                policy_out,action_logprobs,outputs = self.infer_probs(images=[rgb_pil],messages=messages,temperature = self.rollout_config['temperature'],pos_id_kwargs=pos_id_kwargs)
                
                vlm_logs |= {'mean/vlm_latency':time.time()-t0,'min/vlm_latency':time.time()-t0,'max/vlm_latency':time.time()-t0,'sum/spguard_trigger_count':0}
                try:
                    import torch
                    vlm_logs |= {"vlm_mem_GB":torch.cuda.memory_allocated()/(1024**3)}
                except Exception:
                    print("warning: could not get vlm mem")
                # print(f"vlm step{step_count}")
                # print("done")
                sampled = self._sample_action_for_state(
                    policy_out, action_logprobs, state_dict, vlm_logs
                )
                action_id, action_to_env, action_logprobs, sde_positions, action_text = sampled
                decision_logs = vlm_logs
                shadow_stop = self._shadow_stop_decision(state_dict)
                stop_target = shadow_stop["stop_target"]
                post_goal_action = bool(
                    state_dict.get("info", {}).get("post_goal_active", False)
                )
                stop_probability = getattr(self, "last_stop_probability", None)
                policy_stop, policy_stop_mode, action_path_length_m = (
                    self._policy_stop_decision(action_to_env, stop_probability)
                )
                decision_logs["action_path_length_m"] = action_path_length_m
                decision_logs["policy_stop_mode"] = policy_stop_mode
                decision_logs["decision_distance_to_goal_m"] = float(
                    state_dict.get("info", {}).get("distance_to_goal", np.nan)
                )
                decision_logs["history_turns"] = step_count + 1
                decision_logs["mean/probe_p_stop"] = (
                    np.nan if stop_probability is None else float(stop_probability))
                # D. Store Transition
               

                # D. Step Simulator (Blocking) ---------------------------RAY----------------------------- 
                t0 = time.time()
                # del rgb,state_dict
                state_ref = ray.get(
                    (env_handle.policy_stop.remote(supplementary_logs=vlm_logs)
                     if policy_stop else
                     env_handle.step.remote(action_to_env, supplementary_logs=vlm_logs))
                )
                rgb, state_dict = self._unpack_env_state(state_ref, pos_id_kwargs)
                vlm_logs = {'mean/sim_latency':time.time()-t0,'min/sim_latency':time.time()-t0,'max/sim_latency':time.time()-t0}
                if collect_trajectory:
                    trajectory_buffer.append(
                        self._build_transition(
                            action_id,
                            action_logprobs,
                            sde_positions,
                            state_dict,
                            outputs,
                            compute_value,
                            decision_logs,
                            stop_target=stop_target,
                            shadow_stop_action=shadow_stop["shadow_stop_action"],
                            shadow_stop_reward=shadow_stop["shadow_stop_reward"],
                            policy_action_mask=(
                                (not post_goal_action
                                 or bool(self.rollout_config.get(
                                     "learn_post_goal_actions", False)))
                                and (not policy_stop
                                     or policy_stop_mode == "trajectory_length")
                            ),
                            reward_override=(
                                0.0 if post_goal_action and not bool(
                                    self.rollout_config.get(
                                        "learn_post_goal_actions", False))
                                else None
                            ),
                            done_override=(
                                bool(state_dict["done"])
                                or (
                                    bool(state_dict.get("info", {}).get("just_reached"))
                                    and not bool(self.rollout_config.get(
                                        "learn_post_goal_actions", False))
                                )
                            ),
                        )
                    )
                # `step` is the index of the observation the appended turn introduces:
                # "Observation 0:" lives in the start template, so after acting on
                # observation k this turn carries observation k+1. Templates that need SFT
                # turn numbering ("Observation $step:") read it; older templates ignore it.
                messages = substitute_convo_template(self.rollout_config['convo_turn_template'],{"action":action_text,"step":step_count+1})
                if (collect_trajectory and state_dict["done"]
                        and self._stop_target(state_dict) == 1.0
                        and self.rollout_config.get("stop_execution_mode") == "shadow"):
                    terminal_policy, terminal_logprobs, terminal_outputs = self.infer_probs(
                        images=[Image.fromarray(rgb)], messages=messages,
                        temperature=self.rollout_config["temperature"],
                        pos_id_kwargs=pos_id_kwargs)
                    terminal_action, _, terminal_logprobs, terminal_positions, _ = \
                        self._sample_action_for_state(
                            terminal_policy, terminal_logprobs, state_dict, {}
                        )
                    terminal_shadow = self._shadow_stop_decision(state_dict)
                    trajectory_buffer.append(self._build_transition(
                        terminal_action, terminal_logprobs, terminal_positions, state_dict,
                        terminal_outputs, compute_value, {},
                        stop_target=terminal_shadow["stop_target"],
                        shadow_stop_action=terminal_shadow["shadow_stop_action"],
                        shadow_stop_reward=terminal_shadow["shadow_stop_reward"],
                        policy_action_mask=False, reward_override=0.0,
                    ))
                # print(f"sim step{step_count}")
                done = state_dict['done']
                step_count += 1
                # Convert list of dicts -> Dict of Numpy Arrays (Zero-Copy Friendly)
            final_trajectory = self._pack_trajectory(trajectory_buffer) if collect_trajectory else None
            stop_counterfactual = {}
            if final_trajectory is not None:
                action_lengths = np.asarray(
                    final_trajectory.get("action_path_length_m", []), dtype=np.float64
                )
                finite_lengths = action_lengths[np.isfinite(action_lengths)]
                if len(finite_lengths):
                    final_info_action_length = float(finite_lengths.mean())
                else:
                    final_info_action_length = float("nan")
                probability = np.asarray(final_trajectory.get("probe_p_stop", []), dtype=np.float64)
                target = np.asarray(final_trajectory.get("stop_target", []), dtype=np.float64)
                valid = np.isfinite(probability) & np.isfinite(target)
                if bool(valid.any()):
                    threshold = self.state_probe_stop_threshold
                    if threshold is None:
                        threshold = self.rollout_config.get("stop_prob_threshold")
                    if threshold is not None:
                        predicted = probability[valid] >= float(threshold)
                        positive = target[valid] > 0.5
                        stop_counterfactual = {
                            "stop_counterfactual_tp": int(np.sum(predicted & positive)),
                            "stop_counterfactual_fp": int(np.sum(predicted & ~positive)),
                            "stop_counterfactual_fn": int(np.sum(~predicted & positive)),
                            "stop_counterfactual_tn": int(np.sum(~predicted & ~positive)),
                            "stop_counterfactual_threshold": float(threshold),
                        }
            final_info = state_dict['info'] | {
                "steps": step_count,
                "instr_or_goal": instr_or_goal,
                "mean_action_path_length_m": (
                    final_info_action_length if final_trajectory is not None else float("nan")
                ),
                "policy_stop_probability": (
                    np.nan
                    if getattr(self, "last_stop_probability", None) is None
                    else float(self.last_stop_probability)
                ),
            } | stop_counterfactual
            # Return Clean Tuple (No Actor Handles here)
            return state_dict['is_exhausted'], final_info, final_trajectory
        
        except Exception as e:
            print(f"Episode failed: {e}")
            import traceback
            traceback.print_exc()
            # Return handles anyway so we don't leak resources (or handle crash logic)
            return False, None,None

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

    def _infer_batch_slot(self, index, rgb, state_dict, messages):
        import torch

        self.restore_sequence_state(self.rl_batch_sequence_states[index])
        logs = {"sum/spguard_trigger_count": 0}
        started = time.perf_counter()
        policy_out, action_logprobs, _ = self.infer_probs(
            images=[Image.fromarray(rgb)],
            messages=messages,
            temperature=self.rollout_config["temperature"],
            pos_id_kwargs={"mode": "standard"},
        )
        latency = time.perf_counter() - started
        logs.update(
            {
                "mean/vlm_latency": latency,
                "min/vlm_latency": latency,
                "max/vlm_latency": latency,
                "vlm_mem_GB": torch.cuda.memory_allocated() / (1024**3),
            }
        )
        sampled = self._sample_action_for_state(
            policy_out, action_logprobs, state_dict, logs
        )
        logs.update(self._shadow_stop_decision(state_dict))
        logs["post_goal_action"] = bool(
            state_dict.get("info", {}).get("post_goal_active", False)
        )
        self.rl_batch_sequence_states[index] = self.capture_sequence_state()
        return (*sampled, logs, latency)

    def _infer_slots_with_batched_vision(
        self,
        indexes,
        rgb_batch,
        state_dicts,
        messages,
    ):
        import torch

        if self.policy_head_config["type"] != "continuous":
            raise ValueError("Batched rollout vision currently requires a continuous policy")
        self._prepare_model_for_inference()
        started = time.perf_counter()
        prepared_inputs = []
        prepared_states = []
        for index in indexes:
            self.restore_sequence_state(self.rl_batch_sequence_states[index])
            prepared_inputs.append(
                self._prepare_infer_inputs(
                    messages=messages[index],
                    images=[Image.fromarray(rgb_batch[index])],
                    pos_id_kwargs={"mode": "standard"},
                )
            )
            prepared_states.append(self.capture_sequence_state())

        image_features = self.batch_image_features(prepared_inputs)
        policy_outputs = []
        for index, inputs, sequence_state, features in zip(
            indexes, prepared_inputs, prepared_states, image_features
        ):
            self.restore_sequence_state(sequence_state)
            policy_out, _ = self._forward_infer_inputs(
                inputs,
                temperature=self.rollout_config["temperature"],
                precomputed_image_features=features,
            )
            policy_outputs.append((policy_out, getattr(self, "last_stop_probability", None)))
            self.rl_batch_sequence_states[index] = self.capture_sequence_state()

        latency = time.perf_counter() - started
        latency_per_slot = latency / len(indexes)
        decisions = {}
        for index, (policy_out, stop_probability) in zip(indexes, policy_outputs):
            logs = {
                "sum/spguard_trigger_count": 0,
                "mean/vlm_latency": latency_per_slot,
                "min/vlm_latency": latency_per_slot,
                "max/vlm_latency": latency_per_slot,
                "vlm_mem_GB": torch.cuda.memory_allocated() / (1024**3),
            }
            sampled = self._sample_action_for_state(
                policy_out, None, state_dicts[index], logs
            )
            logs.update(self._shadow_stop_decision(state_dicts[index], stop_probability))
            logs["post_goal_action"] = bool(
                state_dicts[index].get("info", {}).get("post_goal_active", False)
            )
            decisions[index] = (*sampled, logs, latency_per_slot)
        return decisions

    def _record_batch_transition(self, index, decision, next_state, step_count,
                                 virtual_stop=False):
        action_id, _, action_logprobs, sde_positions, action_text, logs, _ = decision
        trajectory = {
            "rollout_logprobs": action_logprobs,
            "rewards": (0.0 if virtual_stop or logs.get("post_goal_action")
                        else next_state.get("reward", 0.0)),
            "dones": bool(next_state["done"] or next_state.get("info", {}).get("just_reached")),
            **next_state["info"],
            **logs,
            "stop_target": logs.get("stop_target", np.nan),
            "shadow_stop_action": logs.get("shadow_stop_action", np.nan),
            "shadow_stop_reward": logs.get("shadow_stop_reward", np.nan),
            "probe_p_stop": logs.get("probe_p_stop", np.nan),
            "policy_action_mask": not virtual_stop and not logs.get("post_goal_action"),
        }
        if self.policy_head_config["type"] == "continuous":
            trajectory["actions_continuous"] = np.asarray(action_id, dtype=np.float32)
            if sde_positions is not None:
                trajectory["sde_positions"] = np.asarray(
                    sde_positions, dtype=np.int64
                )
        else:
            trajectory["actions"] = int(action_id)
            trajectory["rollout_probs"] = np.asarray(
                logs["action_probs"]
            )
        self.rl_batch_trajectories[index].append(trajectory)
        next_messages = substitute_convo_template(
            self.rollout_config["convo_turn_template"],
            {"action": action_text, "step": int(step_count) + 1},
        )
        return next_messages

    def run_episode_batch(self, env_handle, initial_state_ref):
        """Run one fixed simulator/VLM pair over all vector slots in lockstep."""
        rgb_batch, state_dicts = initial_state_ref
        batch_size = len(state_dicts)
        if len(rgb_batch) != batch_size:
            raise ValueError(
                f"Vector reset returned {len(rgb_batch)} images for {batch_size} states"
            )
        self.save_pixels = False
        self.rl_batch_sequence_states = [
            self.new_sequence_state() for _ in range(batch_size)
        ]
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
        inference_seconds = 0.0
        environment_seconds = 0.0
        transition_seconds = 0.0
        batch_started = time.perf_counter()
        timeout_seconds = float(os.environ.get("LONGNAV_ENV_STEP_TIMEOUT_SECONDS", "300"))
        if timeout_seconds <= 0:
            raise ValueError("LONGNAV_ENV_STEP_TIMEOUT_SECONDS must be positive")
        soft_timeout_seconds = float(
            self.rollout_config.get("episode_soft_timeout_seconds", 600.0)
        )
        if soft_timeout_seconds <= 0:
            raise ValueError("episode_soft_timeout_seconds must be positive")

        def mark_external_stop(index, terminal_state):
            if not self.rl_batch_trajectories[index]:
                raise RuntimeError(
                    "Episode soft timeout occurred before its first policy transition; "
                    f"slot={int(index)}"
                )
            trajectory = self.rl_batch_trajectories[index][-1]
            trajectory["dones"] = True
            trajectory.update(terminal_state.get("info", {}))
            done[index] = True

        def soft_stop_unfinished():
            nonlocal rgb_batch, state_dicts, done
            active = np.flatnonzero(~done)
            if not len(active):
                return
            rgb_batch, terminal_states = ray.get(
                env_handle.force_stop_vector.remote("wall_timeout_soft_stop"),
                timeout=timeout_seconds,
            )
            for index in active:
                terminal_state = terminal_states[index]
                mark_external_stop(index, terminal_state)
            state_dicts = terminal_states

        while np.any(~done & (step_counts < self.rollout_config["max_steps"])):
            if time.perf_counter() - batch_started >= soft_timeout_seconds:
                soft_stop_unfinished()
                break
            actions = [None] * batch_size
            decisions = [None] * batch_size
            active_indexes = [
                index
                for index in range(batch_size)
                if not done[index]
                and step_counts[index] < self.rollout_config["max_steps"]
            ]
            if getattr(self, "batch_vision_inference", False) and len(active_indexes) > 1:
                batch_decisions = self._infer_slots_with_batched_vision(
                    active_indexes,
                    rgb_batch,
                    state_dicts,
                    messages,
                )
            else:
                batch_decisions = {
                    index: self._infer_batch_slot(
                        index, rgb_batch[index], state_dicts[index], messages[index]
                    )
                    for index in active_indexes
                }
            for index, decision in batch_decisions.items():
                actions[index] = decision[1]
                decisions[index] = decision
                inference_seconds += decision[-1]

            env_started = time.perf_counter()
            step_ref = env_handle.step_vector.remote(actions)
            try:
                rgb_batch, next_states = ray.get(step_ref, timeout=timeout_seconds)
            except ray.exceptions.GetTimeoutError as exc:
                active_labels = [
                    state.get("info", {}).get("episode_label", "unknown")
                    for index, state in enumerate(state_dicts)
                    if not done[index]
                ]
                raise RuntimeError(
                    f"NavVerse vector step exceeded {timeout_seconds:.0f}s; "
                    f"active_episodes={active_labels}"
                ) from exc
            environment_seconds += time.perf_counter() - env_started

            for index, next_state in enumerate(next_states):
                transition_started = time.perf_counter()
                if decisions[index] is None:
                    continue
                if next_state.get("info", {}).get("external_timeout_stop"):
                    mark_external_stop(index, next_state)
                    continue
                messages[index] = self._record_batch_transition(
                    index, decisions[index], next_state, step_counts[index]
                )
                step_counts[index] += 1
                done[index] = bool(next_state["done"])
                if (done[index] and bool(next_state.get("info", {}).get("success"))
                        and self.rollout_config.get("stop_execution_mode") == "shadow"):
                    terminal_decision = self._infer_batch_slot(
                        index, rgb_batch[index], next_state, messages[index]
                    )
                    inference_seconds += terminal_decision[-1]
                    self._record_batch_transition(
                        index, terminal_decision, next_state, step_counts[index],
                        virtual_stop=True,
                    )
                transition_seconds += time.perf_counter() - transition_started
            state_dicts = next_states
            if (
                np.any(~done)
                and time.perf_counter() - batch_started >= soft_timeout_seconds
            ):
                soft_stop_unfinished()
                break

        unfinished = np.flatnonzero(~done)
        if len(unfinished):
            raise RuntimeError(
                "rollout.max_steps ended before the simulator marked every vector slot "
                f"done; unfinished_slots={unfinished.tolist()}"
            )
        self.rl_batch_trajectories = [
            self._pack_trajectory(buffer) for buffer in self.rl_batch_trajectories
        ]
        results = [
            state["info"]
            | {"steps": int(step_counts[index]), "instr_or_goal": instructions[index]}
            for index, state in enumerate(state_dicts)
        ]
        runtime = {
            "wall_seconds": time.perf_counter() - batch_started,
            "vlm_inference_seconds": inference_seconds,
            "environment_step_seconds": environment_seconds,
            "transition_seconds": transition_seconds,
            "episode_steps_total": int(step_counts.sum()),
            "episode_steps_max": int(step_counts.max(initial=0)),
        }
        return False, results, runtime

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
            # Score stored chains in the same unmerged adapter parameterization as PPO.
            if self.is_merged():
                self.unmerge_adapter()
            _align_readouts_to_executed_actions(
                self.rl_trajectory,
                self.logit_indices,
                self.value_logit_indices,
                self.policy_head_config["type"],
            )
            embeds = self._pack_embeds()
            self.rl_embeds_inputs = embeds
            values = None
            import torch
            with torch.no_grad():
                _want_values = (self.rl_algo_config.value_head is not None
                                or (bool(getattr(self.rl_algo_config, "state_probe", None))
                                    and not self.state_probe_trainable))
                if self.rl_embeds_inputs is not None:
                    policy_stats,values = self._forward_embeds(self.rl_embeds_inputs,_want_values)
                    model_inputs = self.rl_embeds_inputs
                elif self.rl_seq_inputs is not None:
                    policy_stats,values = self._forward_seq(self.rl_seq_inputs,_want_values)
                    model_inputs = self.rl_seq_inputs
                else:
                    raise ValueError("No stored model inputs found for postprocessing.")
                if self.policy_head_config['type'] == "continuous":
                    # Chain head first: `policy_stats` is `{"h"}` here, so both the dtype
                    # anchor (`policy_stats['mu'].dtype`) and `_continuous_log_prob` below
                    # would KeyError -- same ordering constraint as the sampling branch.
                    _head = getattr(getattr(self, "model", None), "action_head", None)
                    _chain_lp = getattr(_head, "chain_log_prob_batch", None)
                    if _chain_lp is not None:
                        hh = policy_stats['h']
                        actions_continuous = torch.as_tensor(
                            self.rl_trajectory['actions_continuous'],
                            dtype=torch.float32, device=hh.device)
                        if actions_continuous.dim() == 2:
                            actions_continuous = actions_continuous.unsqueeze(0)
                        sde_positions = torch.as_tensor(
                            np.asarray(self.rl_trajectory['sde_positions']),
                            device=hh.device).reshape(1, actions_continuous.shape[1], -1)
                        old_log_prob = _chain_lp(hh, actions_continuous,
                                                 sde_positions).squeeze(0).float().cpu()
                    else:
                        actions_continuous = torch.as_tensor(self.rl_trajectory['actions_continuous'], dtype=policy_stats['mu'].dtype, device=policy_stats['mu'].device)
                        if actions_continuous.dim() == 2:
                            actions_continuous = actions_continuous.unsqueeze(0)
                        old_log_prob = self._continuous_log_prob(actions_continuous, policy_stats['mu'], policy_stats['log_std']).squeeze(0).float().cpu()
                    self.rl_trajectory['old_log_prob'] = old_log_prob.numpy()
                    if getattr(self.rl_algo_config, 'ref_kl', False):
                        # THE h-SPACE TETHER, measure-first: log pi_ref at the STORED
                        # actions, via the DEFAULT ref mechanism (the same
                        # disable_adapter the discrete path uses at the bottom of this
                        # function). CORRECT ONLY WHEN THE SFT POLICY WAS MERGED INTO
                        # THE BASE (vlm.merge_adapter_dir) so the trainable LoRA is a
                        # fresh zero-delta on top -- setup_training enforces this
                        # pairing. disable_adapter also switches modules_to_save back
                        # to their original (init) copies, so the reference is the init
                        # POLICY, head included, even in arms where the head trains.
                        # rl_loss turns this into ref/kl_k1 and ref/kl_k3 every step;
                        # only a nonzero ref_kl_coeff makes it a constraint.
                        self.unmerge_adapter()
                        with self.model.disable_adapter():
                            if self.rl_embeds_inputs is not None:
                                ref_stats, _ = self._forward_embeds(self.rl_embeds_inputs, False)
                            else:
                                ref_stats, _ = self._forward_seq(self.rl_seq_inputs, False)
                            _ref_head = getattr(getattr(self, "model", None), "action_head", None)
                            _ref_chain_lp = getattr(_ref_head, "chain_log_prob_batch", None)
                            if _ref_chain_lp is not None:
                                ref_lp = _ref_chain_lp(ref_stats['h'], actions_continuous,
                                                       sde_positions).squeeze(0)
                            else:
                                ref_lp = self._continuous_log_prob(
                                    actions_continuous, ref_stats['mu'],
                                    ref_stats['log_std']).squeeze(0)
                        self.rl_trajectory['ref_logprobs'] = ref_lp.float().cpu().numpy()
                else:
                    logits = policy_stats['logits']
                    logprobs = self._calculate_action_logprobs(logits).squeeze().float().cpu()
                    if logprobs.dim() == 1:
                        logprobs = logprobs.unsqueeze(0) # ensure batch dim
                    self.rl_trajectory['old_logprobs'] = logprobs.numpy()
                if values is not None:
                    _vh = getattr(getattr(self, "model", None), "value_head", None)
                    if getattr(_vh, "is_distributional", False):
                        values = _vh.value(values)   # logits -> scalar; GAE consumes scalars
                    self.rl_trajectory['values'] = values.squeeze().float().cpu().numpy()
                    # StateProbeValueAdapter caches the distance head's outputs from
                    # the SAME forward; drain immediately (single-producer/consumer
                    # within this episode -- see the adapter's docstring).
                    if getattr(_vh, "last_distance_m", None) is not None:
                        self.rl_trajectory['probe_distance_m'] = \
                            _vh.last_distance_m.squeeze().numpy()
                        self.rl_trajectory['probe_p_stop'] = \
                            _vh.last_p_stop.squeeze().numpy()
                        _vh.last_distance_m = None
                        _vh.last_p_stop = None

            if self.rl_algo_config.use_ref and self.policy_head_config['type'] != "continuous":
                with torch.no_grad():
                    self.unmerge_adapter()
                    with self.model.disable_adapter():
                        if self.rl_embeds_inputs is not None:
                            policy_stats,values = self._forward_embeds(self.rl_embeds_inputs,False)
                            logits = policy_stats['logits']
                        elif self.rl_seq_inputs is not None:
                            policy_stats,values = self._forward_seq(self.rl_seq_inputs)
                            logits = policy_stats['logits']
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
    def worker_placement(self):
        return {
            "pid": os.getpid(),
            "ray_gpu_ids": ray.get_gpu_ids(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def set_ode_sampling(self, flag: bool) -> bool:
        """Toggle deterministic acting for interleaved eval cycles.

        One switch, two heads: the chain head's pure-ODE sampler (`force_ode`) and the
        latent head's act-at-the-prior-mean mode (`force_mean`). Both mean the same thing
        -- "evaluate the policy, not the exploration noise" -- so `task.eval_ode` drives
        them through one call and a config never has to know which head it runs.

        Set on every head INSTANCE (the peft ModulesToSaveWrapper forwards attribute
        READS to the active module, but a write would land on the wrapper).
        No-op (returns False) for heads without either flag."""
        from longnav.utils.flow_sde_policy import FlowSDEHead
        from longnav.utils.latent_policy import LatentIntentHead
        hit = False
        for m in getattr(self, "model", None).modules() if getattr(self, "model", None) else []:
            if isinstance(m, FlowSDEHead):
                m.force_ode = bool(flag)
                hit = True
            elif isinstance(m, LatentIntentHead):
                m.force_mean = bool(flag)
                hit = True
        return hit

    def set_sde_noise_a(self, a: float) -> bool:
        """Set the chain head's SDE exploration scale (the H2 credited-channel probe
        sweeps `noise_a` per arm).

        SDEConfig is a frozen dataclass, so the config is REPLACED, not mutated. As with
        `set_ode_sampling`, the walk is by isinstance so the write lands on every
        FlowSDEHead INSTANCE (the peft ModulesToSaveWrapper forwards attribute READS to
        the active module, but a write would land on the wrapper).
        Returns False when the model carries no chain head."""
        import dataclasses

        from longnav.utils.flow_sde_policy import FlowSDEHead
        hit = False
        for m in getattr(self, "model", None).modules() if getattr(self, "model", None) else []:
            if isinstance(m, FlowSDEHead):
                m.sde = dataclasses.replace(m.sde, noise_a=float(a))
                hit = True
        return hit

    def run_episode(self,env_handle,initial_state_ref):
        is_exhausted,state_dict,_,_ = super().run_episode(env_handle, initial_state_ref)
        return is_exhausted,state_dict
    
    def postprocess_episode(self,return_inputs = True,eval=False):
        trajectory,model_inputs = super().postprocess_episode(eval=eval)
        if return_inputs:
            inputs_tensors,inputs_metadata = TensorPacker.pack(model_inputs)
            return trajectory,inputs_tensors,inputs_metadata
        else:
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

    def discard_episode_batch(self):
        """Clear a failed vector batch without postprocessing partial trajectories."""
        batch_size = len(getattr(self, "rl_batch_trajectories", ()))
        self.rl_batch_sequence_states = []
        self.rl_batch_trajectories = []
        self.rl_trajectory = None
        self.rl_seq_inputs = None
        self.rl_embeds_inputs = None
        self.reset(clear_cuda_cache=False)
        return [(None, None, None) for _ in range(batch_size)]

    def train_rl_step(self, embeds_inputs_np, embeds_inputs_meta,traj_batch, loss_scale=1.0):
        embeds_inputs = TensorPacker.unpack(embeds_inputs_np,embeds_inputs_meta,device=self.accelerator.device)
        actions = traj_batch.get('actions', None)
        actions_continuous = traj_batch.get('actions_continuous', None)
        old_log_prob = traj_batch['old_log_prob']
        advantages = traj_batch['advantages']
        returns = traj_batch['returns']
        old_values = traj_batch.get('values',None)
        rollout_log_probs = traj_batch.get('rollout_logprobs',None)
        ref_logprobs = traj_batch.get('ref_logprobs',None)
        sde_positions = traj_batch.get('sde_positions', None)
        stop_targets = traj_batch.get('stop_target', None)
        shadow_stop_action = traj_batch.get('shadow_stop_action', None)
        shadow_stop_reward = traj_batch.get('shadow_stop_reward', None)
        policy_action_mask = traj_batch.get('policy_action_mask', None)

        # NOTE on the chain head's `old_log_prob`: the postprocess value is passed through
        # UNCHANGED -- there is no re-anchoring (an earlier anchor was removed; it papered
        # over an sdpa-only numeric split, see the chain branch in `rl_loss`). Under
        # flash_attention_2 the postprocess forward and the training forward agree to
        # ~0.01 nats, verified per-cycle by the first on-policy minibatches of every cycle
        # sitting at seam level. If actor/ppo_kl is large on minibatches trained BEFORE any
        # optimizer step of the cycle, suspect attn_impl or a trajectory/model_inputs
        # misalignment (see the alignment invariant in train_rl.py), never the seam.
        return super().train_rl_step(embeds_inputs, actions=actions, old_log_prob=old_log_prob, advantages=advantages, returns=returns, old_values=old_values, rollout_log_probs=rollout_log_probs, ref_log_probs=ref_logprobs, actions_continuous=actions_continuous, sde_positions=sde_positions, stop_targets=stop_targets, shadow_stop_action=shadow_stop_action, shadow_stop_reward=shadow_stop_reward, policy_action_mask=policy_action_mask, loss_scale=loss_scale)
    
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
        
def collect_vector_rollouts(
    env_handles,
    vlm_handles,
    shard_iterator,
    target_episodes,
    episodes_per_worker,
    postprocess_kwargs=None,
    episode_soft_timeout_seconds=600.0,
    episode_hard_timeout_seconds=900.0,
    sim_restart_limit=1,
    sim_rebuilder: Optional[Callable[[int], Any]] = None,
    sim_rebuild_validator: Optional[Callable[[int, Any], None]] = None,
    timing_out: Optional[Dict[str, float]] = None,
):
    """Collect complete host-sized waves with one fixed VLM per NavVerse simulator."""
    postprocess_kwargs = postprocess_kwargs or {"return_inputs": True, "eval": False}
    if len(env_handles) != len(vlm_handles):
        raise ValueError(
            f"Vector rollout needs one VLM per simulator; got {len(vlm_handles)} and "
            f"{len(env_handles)}"
        )
    episodes_per_wave = len(env_handles) * episodes_per_worker
    if target_episodes <= 0 or target_episodes % episodes_per_wave:
        raise ValueError(
            f"target_episodes must be a positive multiple of {episodes_per_wave}; "
            f"got {target_episodes}"
        )
    soft_timeout = float(episode_soft_timeout_seconds)
    hard_timeout = float(episode_hard_timeout_seconds)
    restart_limit = int(sim_restart_limit)
    if soft_timeout <= 0:
        raise ValueError("episode_soft_timeout_seconds must be positive")
    if hard_timeout <= soft_timeout:
        raise ValueError(
            "episode_hard_timeout_seconds must be greater than "
            "episode_soft_timeout_seconds"
        )
    if restart_limit < 0:
        raise ValueError("sim_restart_limit must be non-negative")

    rollouts = []
    results = []
    logs = []
    collection_wall_seconds = 0.0
    postprocess_wall_seconds = 0.0
    vlm_inference_device_seconds = 0.0
    environment_step_device_seconds = 0.0
    transition_device_seconds = 0.0
    worker_wall_seconds = []
    reset_wall_seconds = []
    episode_steps_total = 0
    abandoned_timeout_shards = 0
    abandoned_timeout_episodes = 0
    rollout_wall_started = time.perf_counter()

    def is_vector_step_timeout(exc):
        try:
            cause = exc.as_instanceof_cause()
        except (AttributeError, TypeError):
            cause = exc
        return isinstance(cause, RuntimeError) and (
            "NavVerse vector step exceeded " in str(cause)
        )

    for wave_index in range(target_episodes // episodes_per_wave):
        wave_collection_started = time.perf_counter()
        worker_labels = [next(shard_iterator) for _ in env_handles]
        if any(len(labels) != episodes_per_worker for labels in worker_labels):
            raise ValueError(
                f"Every vector shard must contain {episodes_per_worker} episode labels"
            )
        worker_states = {}
        episode_outputs = [None] * len(env_handles)
        abandoned_workers = set()

        def submit_attempt(worker_index, restart_count):
            env = env_handles[worker_index]
            env.assign_shard.remote(worker_labels[worker_index])
            worker_states[worker_index] = {
                "phase": "reset",
                "ref": env.reset_vector.remote(),
                "started": time.monotonic(),
                "reset_started": time.monotonic(),
                "soft_sent": False,
                "restart_count": restart_count,
            }

        def replace_sim(worker_index, reason):
            state = worker_states[worker_index]
            restart_count = state["restart_count"]
            if sim_rebuilder is None:
                raise RuntimeError(
                    f"Worker {worker_index} requires simulator replacement ({reason}), "
                    "but no sim_rebuilder was configured"
                )
            if restart_count >= restart_limit:
                raise RuntimeError(
                    f"Worker {worker_index} exceeded sim_restart_limit={restart_limit} "
                    f"while retrying labels={worker_labels[worker_index]} ({reason})"
                )
            old_env = env_handles[worker_index]
            old_ref = state["ref"]
            print(
                f"[vector_watchdog] replacing worker={worker_index} "
                f"phase={state['phase']} labels={worker_labels[worker_index]}; "
                f"reason={reason}; rebuilding sim "
                f"({restart_count + 1}/{restart_limit})",
                flush=True,
            )
            ray.kill(old_env, no_restart=True)
            if state["phase"] == "episode":
                try:
                    ray.get(old_ref, timeout=30.0)
                except ray.exceptions.GetTimeoutError as exc:
                    raise RuntimeError(
                        f"Simulator {worker_index} was killed but its VLM task did not "
                        "unwind within 30s; refusing to queue a second episode on that VLM"
                    ) from exc
                except Exception:
                    pass
            new_env = sim_rebuilder(worker_index)
            if new_env is None:
                raise RuntimeError(
                    f"sim_rebuilder returned no actor for worker {worker_index}"
                )
            env_handles[worker_index] = new_env
            submit_attempt(worker_index, restart_count + 1)

        def abandon_repeated_vector_timeout(worker_index, reason):
            nonlocal abandoned_timeout_shards, abandoned_timeout_episodes
            state = worker_states[worker_index]
            if sim_rebuilder is None:
                raise RuntimeError(
                    f"Worker {worker_index} requires simulator replacement ({reason}), "
                    "but no sim_rebuilder was configured"
                )
            print(
                f"[vector_watchdog] abandoning timed-out shard worker={worker_index} "
                f"labels={worker_labels[worker_index]}; reason={reason}; rebuilding sim "
                "for the next cycle",
                flush=True,
            )
            ray.kill(env_handles[worker_index], no_restart=True)
            new_env = sim_rebuilder(worker_index)
            if new_env is None:
                raise RuntimeError(
                    f"sim_rebuilder returned no actor for worker {worker_index}"
                )
            env_handles[worker_index] = new_env
            if sim_rebuild_validator is not None:
                sim_rebuild_validator(worker_index, new_env)
            failure_results = [
                {
                    "episode_label": label,
                    "steps": 0,
                    "instr_or_goal": label,
                    "success": False,
                    "oracle_success": False,
                    "truncated": True,
                    "termination_reason": "sim_vector_step_timeout",
                }
                for label in worker_labels[worker_index]
            ]
            episode_outputs[worker_index] = (
                False,
                failure_results,
                {
                    "wall_seconds": time.monotonic() - state["started"],
                    "vlm_inference_seconds": 0.0,
                    "environment_step_seconds": 0.0,
                    "transition_seconds": 0.0,
                    "episode_steps_total": 0,
                },
            )
            state["phase"] = "abandoned"
            abandoned_workers.add(worker_index)
            abandoned_timeout_shards += 1
            abandoned_timeout_episodes += len(worker_labels[worker_index])

        for worker_index in range(len(env_handles)):
            submit_attempt(worker_index, 0)

        while any(output is None for output in episode_outputs):
            active = {
                state["ref"]: worker_index
                for worker_index, state in worker_states.items()
                if state["phase"] in {"reset", "episode"}
                and episode_outputs[worker_index] is None
            }
            if active:
                now = time.monotonic()
                next_deadline = min(
                    state["started"]
                    + (hard_timeout if state["soft_sent"] else soft_timeout)
                    for index, state in worker_states.items()
                    if episode_outputs[index] is None
                )
                wait_seconds = max(0.0, min(1.0, next_deadline - now))
                ready, _ = ray.wait(
                    list(active), num_returns=len(active), timeout=wait_seconds
                )
                for ref in ready:
                    worker_index = active[ref]
                    state = worker_states[worker_index]
                    try:
                        ready_value = ray.get(ref)
                    except ray.exceptions.RayActorError as exc:
                        replace_sim(
                            worker_index,
                            f"actor died during {state['phase']}: {exc}",
                        )
                        continue
                    except ray.exceptions.RayTaskError as exc:
                        if not is_vector_step_timeout(exc):
                            raise
                        reason = (
                            f"simulator vector step timed out during {state['phase']}: "
                            f"{exc.as_instanceof_cause()}"
                        )
                        if state["restart_count"] >= restart_limit:
                            abandon_repeated_vector_timeout(worker_index, reason)
                        else:
                            replace_sim(worker_index, reason)
                        continue
                    if state["phase"] == "reset":
                        reset_wall_seconds.append(
                            time.monotonic() - state["reset_started"]
                        )
                        if state["soft_sent"]:
                            state["phase"] = "soft_wait"
                        else:
                            if (
                                state["restart_count"] > 0
                                and sim_rebuild_validator is not None
                            ):
                                sim_rebuild_validator(
                                    worker_index, env_handles[worker_index]
                                )
                            state["phase"] = "episode"
                            state["ref"] = vlm_handles[
                                worker_index
                            ].run_episode_batch.remote(
                                env_handles[worker_index], ready_value
                            )
                    else:
                        episode_outputs[worker_index] = ready_value
                        state["phase"] = "complete"
            else:
                time.sleep(0.1)

            now = time.monotonic()
            for worker_index, state in list(worker_states.items()):
                if episode_outputs[worker_index] is not None:
                    continue
                elapsed = now - state["started"]
                if elapsed >= hard_timeout:
                    replace_sim(
                        worker_index,
                        f"attempt exceeded {hard_timeout:.1f}s",
                    )
                    continue
                if elapsed >= soft_timeout and not state["soft_sent"]:
                    print(
                        f"[vector_watchdog] SOFT timeout worker={worker_index} "
                        f"phase={state['phase']} labels={worker_labels[worker_index]}; "
                        "requesting external STOP",
                        flush=True,
                    )
                    env_handles[worker_index].force_stop_vector.remote(
                        "wall_timeout_soft_stop"
                    )
                    state["soft_sent"] = True
        collection_wall_seconds += time.perf_counter() - wave_collection_started
        for _, _, worker_runtime in episode_outputs:
            worker_wall_seconds.append(float(worker_runtime.get("wall_seconds", 0.0)))
            vlm_inference_device_seconds += float(
                worker_runtime.get("vlm_inference_seconds", 0.0)
            )
            environment_step_device_seconds += float(
                worker_runtime.get("environment_step_seconds", 0.0)
            )
            transition_device_seconds += float(
                worker_runtime.get("transition_seconds", 0.0)
            )
            episode_steps_total += int(worker_runtime.get("episode_steps_total", 0))
        postprocess_started = time.perf_counter()
        rollout_batches = ray.get(
            [
                (
                    vlm.discard_episode_batch.remote()
                    if worker_index in abandoned_workers
                    else vlm.postprocess_batch.remote(**postprocess_kwargs)
                )
                for worker_index, vlm in enumerate(vlm_handles)
            ]
        )
        postprocess_wall_seconds += time.perf_counter() - postprocess_started
        for worker_rollouts, (_, worker_results, _) in zip(
            rollout_batches, episode_outputs
        ):
            if len(worker_rollouts) != episodes_per_worker:
                raise RuntimeError(
                    f"VLM returned {len(worker_rollouts)} rollouts; expected "
                    f"{episodes_per_worker}"
                )
            if len(worker_results) != episodes_per_worker:
                raise RuntimeError(
                    f"Simulator returned {len(worker_results)} results; expected "
                    f"{episodes_per_worker}"
                )
            rollouts.extend(worker_rollouts)
            results.extend(worker_results)
        for worker_index, env in enumerate(env_handles):
            logs.extend(
                ray.put(None)
                if worker_index in abandoned_workers
                else env.flush_logs_to_disk_slot.remote(slot)
                for slot in range(episodes_per_worker)
            )
    if timing_out is not None:
        timing_out.update(
            {
                "runtime/rollout_wall_seconds": time.perf_counter()
                - rollout_wall_started,
                "runtime/collection_wall_seconds": collection_wall_seconds,
                "runtime/postprocess_wall_seconds": postprocess_wall_seconds,
                "runtime/vlm_inference_device_seconds": vlm_inference_device_seconds,
                "runtime/environment_step_device_seconds": environment_step_device_seconds,
                "runtime/transition_device_seconds": transition_device_seconds,
                "runtime/worker_wall_max_seconds": max(worker_wall_seconds, default=0.0),
                "runtime/worker_wall_mean_seconds": float(
                    np.mean(worker_wall_seconds) if worker_wall_seconds else 0.0
                ),
                "runtime/reset_wall_max_seconds": max(reset_wall_seconds, default=0.0),
                "runtime/reset_wall_mean_seconds": float(
                    np.mean(reset_wall_seconds) if reset_wall_seconds else 0.0
                ),
                "runtime/policy_steps": float(episode_steps_total),
                "runtime/sim_timeout_abandoned_shards": float(
                    abandoned_timeout_shards
                ),
                "runtime/sim_timeout_abandoned_episodes": float(
                    abandoned_timeout_episodes
                ),
                "runtime/vlm_seconds_per_policy_step": (
                    vlm_inference_device_seconds / max(episode_steps_total, 1)
                ),
                "runtime/environment_seconds_per_policy_step": (
                    environment_step_device_seconds / max(episode_steps_total, 1)
                ),
            }
        )
    return rollouts, results, logs


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
            pass
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
                try:
                    import importlib
                    importlib.import_module("ipdb").set_trace()
                except Exception:
                    print("ipdb is not installed; skipping interactive deadlock breakpoint.")

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

import ray
from collections import deque
from typing import List, Dict, Any, Iterator
from string import Template
from PIL import Image
# from utils.habitat_worker import HabitatWorker
# from utils.vlm_worker import VLMWorker
import numpy as np

def substitute_convo_template(conversation_template: List[Dict], obs: Dict[str, Any]) -> List[Dict]:
    """
    Traverses the conversation template and substitutes any string.Template 
    objects found in 'text' fields using values from the 'obs' dictionary.
    
    Args:
        conversation_template: List of message dicts (role, content).
        obs: Dictionary containing substitution keys (e.g., 'goal_name').
        
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
                if isinstance(text_obj, Template):
                    try:
                        # Perform the substitution
                        new_item["text"] = text_obj.substitute(obs)
                    except KeyError as e:
                        # Fallback to safe_substitute to prevent crashing on missing keys,
                        # but log it so we know something is wrong.
                        print(f"Warning: Missing substitution key {e} in template.")
                        new_item["text"] = text_obj.safe_substitute(obs)
                        
                # CASE B: It's already a str (static text)
                elif isinstance(text_obj, str):
                    pass # Keep as is
                    
            new_content.append(new_item)
        new_message["content"] = new_content
        new_conversation.append(new_message)
        
    return new_conversation

@ray.remote(num_cpus=2) # Very lightweight
def run_episode_supervisor(vlm_handle, habitat_handle, initial_state_ref, rollout_config):
    """
    Manages one episode loop between a paired VLM and Habitat worker.
    """
    import time
    # DEBUG: Check what we actually received
    try:
        # 1. Resolve the initial state (Blocking wait for reset to finish)
        # Ray automatically waits for initial_state_ref to be ready before starting this task,
        # but we call ray.get to access the data.
        try:
            curr_state = ray.get(initial_state_ref)
        except:
            curr_state = initial_state_ref

        step_count = 0
        done = False
        messages = substitute_convo_template(rollout_config['convo_start_template'],curr_state['obs'])
        # 2. The Interaction Loop
        while not done and step_count < rollout_config['max_steps']:
            # A. Prepare VLM Input
            # (Assuming your formatting logic is here)
            rgb_numpy = curr_state['obs']['rgb']
            rgb_pil = Image.fromarray(rgb_numpy)
            # B. Call VLM (Blocking)
            # We must wait for the answer to decide the next step
            # print("inferring VLM with messages:")
            # print(messages)
            t0 = time.time()
            action_probs = ray.get(vlm_handle.infer_probs.remote(images=[rgb_pil],messages=messages,temperature = rollout_config['temperature']))
            vlm_logs = {'mean/vlm_latency':time.time()-t0}
            print(f"vlm step{step_count}")

            # print("done")
            #except for the first turn, all messages follow the exact same template.
            messages = substitute_convo_template(rollout_config['convo_turn_template'],curr_state['obs'])
            action_id = np.random.choice(len(action_probs),p=action_probs) # sampling
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-9))
            vlm_logs |= {'mean/entropy':entropy,'mean/action_prob':action_probs[action_id]} 
            # D. Step Simulator (Blocking)
            curr_state = ray.get(habitat_handle.step.remote(action_id))
            print(f"sim step{step_count}")
            done = curr_state['done']
            step_count += 1
        vlm_handle.reset.remote() #reset the kv cache to prepare for next sequence.
        needs_reshard = ray.get(habitat_handle.is_exhausted.remote())
        return vlm_handle, habitat_handle, needs_reshard, curr_state['info']

    except Exception as e:
        print(f"Episode failed: {e}")
        # Return handles anyway so we don't leak resources (or handle crash logic)
        return vlm_handle, habitat_handle, False, None

def run_inference_driver(
    sim_handles: List[Any],
    vlm_handles: List[Any],
    rollout_config: Dict[str, Any],
    shard_iterator: Iterator[List[str]]
) -> List[Dict]:
    """
    Orchestrates the evaluation pipeline.

    Args:
        sim_handles: List of Ray actor handles for Habitat workers.
        vlm_handles: List of Ray actor handles for VLM workers.
        config: Configuration dict passed to the supervisor.
        shard_iterator: An iterator yielding lists of episode IDs (shards).
    """

    # --- 1. Initialize Pools ---
    # Idle VLMs: Ready to be assigned immediately
    idle_vlms = deque(vlm_handles)
    
    # Ready Habitats: Tuple of (actor_handle, initial_state_ref)
    # These are workers that have finished resetting and are waiting for a VLM.
    ready_habitats = deque()
    # --- 2. Tracking Futures (The State Machine) ---
    # Map: reset_future -> habitat_handle
    # Tracks workers currently resetting (loading scene or moving to next episode).
    pending_resets = {} 
    # Map: supervisor_future -> "metadata"
    # Tracks active episodes running in the background.
    active_episodes = {}
    # Collection of all results
    results = []
    # --- 3. Bootstrap: Initial Sharding & Resets ---
    print(f"Bootstrapping: Initializing {len(sim_handles)} environments...")
    
    # We must assign an initial shard to every habitat worker before they can reset.
    for sim_handle in sim_handles:
        try:
            # Assign first shard
            initial_shard = next(shard_iterator)
            sim_handle.assign_shard.remote(initial_shard)
            
            # Trigger first reset
            # The worker will load the first episode in the shard.
            reset_ref = sim_handle.reset.remote()
            pending_resets[reset_ref] = sim_handle   
        except StopIteration:
            print("Warning: Not enough shards for all workers during bootstrap.")
            # Worker is retired immediately if no work exists
            pass
    # --- 4. The Event Loop ---
    # We run as long as there is active work (resets or episodes) or potential work.
    
    while active_episodes or pending_resets or (ready_habitats and idle_vlms):
        
        # A. Check for "Ready to Pair" Condition
        # If we have an idle VLM and a ready Habitat, launch a task immediately.
        while idle_vlms and ready_habitats:
            vlm = idle_vlms.popleft()
            hab, init_state_ref = ready_habitats.popleft()
            
            # LAUNCH SUPERVISOR
            # The supervisor coordinates the interaction between VLM and Habitat for ONE episode.
            print("dispatching episode supervisor!")
            sup_ref = run_episode_supervisor.remote(
                vlm, hab, init_state_ref, rollout_config=rollout_config
            )
            print("episode done!")
            
            active_episodes[sup_ref] = "running"

        # B. Wait for SOMETHING to happen
        # We listen to both pool (resets) and active tasks (episodes).
        all_watch_refs = list(pending_resets.keys()) + list(active_episodes.keys())
        
        if not all_watch_refs:
            # Only happens if iterator exhausted and all workers are idle (shutdown)
            break

        # Blocking wait for the FIRST completed future to maximize responsiveness
        ready_refs, _ = ray.wait(all_watch_refs, num_returns=1)
        
        for ref in ready_refs:
            
            # --- CASE 1: A Habitat Finished Resetting ---
            if ref in pending_resets:
                sim_handle = pending_resets.pop(ref)
                
                # The worker is now ready for a VLM.
                # We store the ref (initial observation) to pass to the supervisor.
                ready_habitats.append((sim_handle, ref))
            
            # --- CASE 2: An Episode Finished ---
            elif ref in active_episodes:
                del active_episodes[ref]
                
                # Retrieve the recycled actors and signals
                # needs_reshard: bool indicating if the worker finished its assigned shard
                vlm, hab, needs_reshard, stats = ray.get(ref)
                results.append(stats)
                # 1. Recycle VLM (It becomes idle immediately)
                idle_vlms.append(vlm)
                
                # 2. Recycle Habitat
                try:
                    if needs_reshard:
                        # Pull new work from the iterator
                        new_shard = next(shard_iterator)
                        hab.assign_shard.remote(new_shard)
                    # Whether we got a new shard or are continuing the old one,
                    # we must reset to prepare the next episode.
                    new_reset_ref = hab.reset.remote()
                    pending_resets[new_reset_ref] = hab
                except StopIteration:
                    # No more work available. Retire the Habitat worker.
                    print(f"Worker finished and no shards remain. Retiring.")
                    pass

    print(f"Inference complete. Processed {len(results)} episodes.")
    return results
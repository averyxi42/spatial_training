import json
import os
import time

from longnav.env.navverse import NavverseEnv


EPISODES = [
    "vcp_stanford_10",
    "vcp_stanford_26",
    "vcp_stanford_27",
    "vcp_stanford_7",
    "vcp_stanford_0",
    "vcp_stanford_29",
    "vcp_stanford_13",
    "vcp_stanford_12",
]
ACTION_SEQUENCE = [
    [1, 1, 2, 3, 1, 2, 3, 1],
    [1, 2, 1, 3, 1, 3, 2, 1],
    [2, 1, 3, 1, 2, 1, 1, 3],
    [1, 3, 1, 2, 3, 1, 2, 1],
] * 3


def main():
    env = NavverseEnv()
    try:
        env.assign_shard(EPISODES)
        reset_started = time.perf_counter()
        env.reset_batch()
        reset_seconds = time.perf_counter() - reset_started
        env.get_vector_profile(reset=True)
        rollout_started = time.perf_counter()
        final_states = None
        for actions in ACTION_SEQUENCE:
            _, final_states = env.step_batch(actions)
        rollout_seconds = time.perf_counter() - rollout_started
        result = {
            "episodes": EPISODES,
            "action_sequence": ACTION_SEQUENCE,
            "reset_seconds": reset_seconds,
            "rollout_seconds": rollout_seconds,
            "profile": env.get_vector_profile(reset=True),
            "final_states": [
                {
                    "episode_label": state["info"]["episode_label"],
                    "distance_to_goal": state["info"]["distance_to_goal"],
                    "path_length": state["info"]["path_length"],
                    "collision": state["info"]["collision"],
                    "done": state["done"],
                    "termination_reason": state["info"]["termination_reason"],
                }
                for state in final_states
            ],
        }
        output_path = os.environ.get("NAVVERSE_PROFILE_OUTPUT")
        if output_path:
            with open(output_path, "w") as output_file:
                json.dump(result, output_file, indent=2)
        print(json.dumps(result, indent=2))
    finally:
        env.vln_sim.close()


if __name__ == "__main__":
    main()

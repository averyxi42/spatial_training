import json
import os

from longnav.env.navverse import NavverseEnv


EPISODES = [
    "vcp_columbus_7",
    "vcp_columbus_1",
    "vcp_columbus_16",
    "vcp_columbus_26",
    "vcp_columbus_10",
    "vcp_columbus_19",
    "vcp_columbus_18",
    "vcp_columbus_21",
]


def main():
    env = NavverseEnv()
    try:
        env.assign_shard(EPISODES)
        rgb, states = env.reset_batch()
        reset_summary = {
            "rgb_shape": list(rgb.shape),
            "episodes": [state["info"]["episode_label"] for state in states],
            "distances": [state["info"]["distance_to_goal"] for state in states],
        }
        actions = [1, 2, 3, 1, 2, 3, 1, 0]
        rgb, states = env.step_batch(actions)
        step_summary = {
            "rgb_shape": list(rgb.shape),
            "actions": actions,
            "states": [
                {
                    "episode_label": state["info"]["episode_label"],
                    "distance_to_goal": state["info"]["distance_to_goal"],
                    "distance_progress": state["info"]["distance_progress"],
                    "collision": state["info"]["collision"],
                    "done": state["done"],
                    "termination_reason": state["info"]["termination_reason"],
                }
                for state in states
            ],
        }
        output_path = os.environ.get("NAVVERSE_VECTOR_SMOKE_OUTPUT")
        result = {"reset": reset_summary, "step": step_summary}
        print(json.dumps(result, indent=2))
        if output_path:
            with open(output_path, "w") as output_file:
                json.dump(result, output_file, indent=2)
    finally:
        env.vln_sim.close()


if __name__ == "__main__":
    main()

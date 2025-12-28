import argparse
import os
import json
import bisect
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from collections import Counter
from datasets import load_dataset, DatasetDict
from PIL import Image

# ------------------------------------------------------------------------------
# 1. ABSTRACT STRATEGY (The Formatter Interface)
# ------------------------------------------------------------------------------

class BaseFormatter(ABC):
    @abstractmethod
    def format(self, example: dict) -> dict:
        """
        Transforms raw data into VLM-ready conversation format.
        Must return the example with 'messages' key.
        """
        pass

# ------------------------------------------------------------------------------
# 2. CONCRETE STRATEGY (Scene Graph Injection Logic)
# ------------------------------------------------------------------------------

class SceneGraphFormatter(BaseFormatter):
    """
    Injects Scene Graph topology into System Prompt and 
    current region tags into User Observations.
    """
    def format(self, example: dict) -> dict:
        sg = example.get('scene_graph', None)
        
        # --- A. Parse Scene Graph & Build Mappings ---
        topology_text = ""
        frame_to_region_tag = {}
        frame_to_region_id = []
        
        # Default fallback if SG is missing
        total_frames = len(example['action_sequence'])
        
        if sg and 'rooms' in sg:
            # 1. Flatten all regions from all rooms
            all_regions = []
            for room in sg['rooms']:
                if 'regions' in room:
                    all_regions.extend(room['regions'])
            
            # 2. Create ID -> Unique Caption Mapping (e.g. "0.0" -> "kitchen (0.0)")
            id_to_unique_name = {r['id']: f"{r['caption']} ({r['id']})" for r in all_regions}

            # 3. Build Topology Text (For System Prompt)
            topo_lines = []
            for region in all_regions:
                curr_name = id_to_unique_name.get(region['id'], region['id'])
                neighbor_ids = region.get('neighbors', [])
                neighbor_names = [id_to_unique_name.get(nid, nid) for nid in neighbor_ids]
                
                if neighbor_names:
                    topo_lines.append(f"- {curr_name} connects to: {', '.join(neighbor_names)}")
            
            if topo_lines:
                topology_text = "\n\n**Known Environment Topology:**\n" + "\n".join(topo_lines)

            # 4. Map Frames to Regions (For User Observation)
            # Use bisect to handle sparse SG updates (Forward Fill)
            valid_regions = [r for r in all_regions if 'image_idx' in r]
            sorted_regions = sorted(valid_regions, key=lambda x: x['image_idx'])
            key_frames = [r['image_idx'] for r in sorted_regions]
            
            for i in range(total_frames):
                idx = bisect.bisect_right(key_frames, i) - 1
                if idx >= 0:
                    current_r = sorted_regions[idx]
                    unique_name = id_to_unique_name[current_r['id']]
                    # Tag for Prompt: "(Location: kitchen (0.0))"
                    frame_to_region_tag[i] = f" (Location: {unique_name})"
                    # ID for Embedding: "0.0"
                    frame_to_region_id.append(current_r['id'])
                else:
                    frame_to_region_tag[i] = ""
                    frame_to_region_id.append("unknown")
        else:
            # Fallback for data without SG
            frame_to_region_id = ["unknown"] * total_frames

        # --- B. Construct Prompt ---
        system_prompt = textwrap.dedent(f"""\
        You are a visual navigation agent tasked with finding "{example.get('goal_text', 'target')}" in an unknown environment.
        You will receive a sequence of observations showing your movement history up to the current moment.
        {topology_text}

        **Action Space:**
        [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN]

        **Mission:**
        1. Analyze history to understand location.
        2. Select next discrete action.

        **Constraints:**
        * Detect collisions (no visual change after move).
        * STOP only when target is plainly in view within 1m.

        **Output Format:**
        Respond with action inside double asterisks.
        """)

        convo = [{"role": "user", "content": [{"type": "text", "text": system_prompt}]}]

        for i, action in enumerate(example['action_sequence']):
            region_tag = frame_to_region_tag.get(i, "")
            convo += [
                {"role": "user", "content": [
                    {"type": "text", "text": f"Observation {i}{region_tag}:"},
                    {"type": "image", "text": None}, # Image loaded by collator
                    {"type": "text", "text": f"Action {i}:"}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": f"**{action}**"}
                ]}
            ]

        # --- C. Output ---
        example['messages'] = convo
        # Save region IDs for training-side embedding fusion
        example['region_ids_sequence'] = frame_to_region_id
        return example

# ------------------------------------------------------------------------------
# 3. DATA PIPELINE (Orchestrator)
# ------------------------------------------------------------------------------

class DatasetPipeline:
    def __init__(self, args, formatter: BaseFormatter):
        self.args = args
        self.formatter = formatter

    def _make_absolute_paths(self, example):
        example["images"] = [os.path.join(self.args.image_root, p) for p in example["images"]]
        return example

    def _temporal_shift(self, example):
        """
        Aligns Input(t) -> Label(t).
        Standard Logic: Image[t] predicts Action[t+1].
        """
        if len(example["images"]) < 2:
            example["images"] = []
            example["action_sequence"] = []
            return example

        # Drop last image (result of last action)
        example["images"] = example["images"][:-1]
        # Drop first action (cause of first image)
        example["action_sequence"] = example["action_sequence"][1:]
        
        # Sync auxiliary fields
        if "pos_rots" in example:
            example["pos_rots"] = example["pos_rots"][:-1]
        
        return example

    def run(self):
        print(f"--- Starting Pipeline with {self.formatter.__class__.__name__} ---")
        
        # 1. Load
        data_files = sorted([str(p) for p in Path(self.args.input_dir).glob("*.jsonl")])
        ds = load_dataset("json", data_files=data_files, split="train")
        
        # 2. Rename & Clean
        if "rgb_paths" in ds.column_names: ds = ds.rename_column("rgb_paths", "images")
        if "actions" in ds.column_names: ds = ds.rename_column("actions", "action_sequence")

        # 3. Fix Paths
        ds = ds.map(self._make_absolute_paths, num_proc=self.args.num_proc, desc="Fixing Paths")

        # 4. Temporal Shift
        ds = ds.map(self._temporal_shift, num_proc=self.args.num_proc, desc="Temporal Shift")
        ds = ds.filter(lambda x: len(x["images"]) > 0, num_proc=self.args.num_proc)

        # 5. Length Filter
        ds = ds.filter(lambda x: len(x["action_sequence"]) <= self.args.max_length, num_proc=self.args.num_proc)

        # 6. Split Train/Val (Based on Scene Rarity)
        print("Splitting by Scene...")
        scene_counts = Counter(ds["scene_id"])
        # Sort by count ascending (rarest scenes first)
        sorted_scenes = sorted(scene_counts.items(), key=lambda x: x[1])
        val_scenes = set([s for s, _ in sorted_scenes[:self.args.val_scene_count]])
        
        train_ds = ds.filter(lambda x: x["scene_id"] not in val_scenes, num_proc=self.args.num_proc)
        val_ds = ds.filter(lambda x: x["scene_id"] in val_scenes, num_proc=self.args.num_proc)

        # 7. Apply Formatting (Injecting SG)
        print(f"Applying VLM formatting...")
        train_ds = train_ds.map(self.formatter.format, num_proc=self.args.num_proc, desc="Format Train")
        val_ds = val_ds.map(self.formatter.format, num_proc=self.args.num_proc, desc="Format Val")

        # 8. Save
        final = DatasetDict({"train": train_ds, "validation": val_ds})
        final.save_to_disk(self.args.output_dir)
        print(f"Saved to {self.args.output_dir}")

# ------------------------------------------------------------------------------
# 4. ENTRY POINT
# ------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with .jsonl files")
    parser.add_argument("--image_root", type=str, required=True, help="Root for relative image paths")
    parser.add_argument("--output_dir", type=str, required=True, help="HuggingFace dataset output path")
    parser.add_argument("--val_scene_count", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=300)
    parser.add_argument("--num_proc", type=int, default=16)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Strategy Pattern: Easily swap formatters here
    formatter = SceneGraphFormatter()
    
    pipeline = DatasetPipeline(args, formatter)
    pipeline.run()
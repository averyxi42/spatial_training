from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import torch
import datasets
from torch.utils.data import DataLoader
from functools import partial
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from transformers import AutoConfig, Trainer
from utils.modeling import Qwen3VLSparseForConditionalGeneration
import os
import sys
import argparse
import json
from pathlib import Path
from PIL import Image as PILImage
def validate_episode_images(example):
    """
    Checks if ALL images in the episode's sequence can be opened.
    Returns False if even one image is broken or missing.
    """
    # We access 'images' because we rename 'rgb_paths' -> 'images' earlier in the pipeline
    image_paths = example.get("images", [])
    
    if not image_paths:
        return False 
    
    for path in image_paths:
        try:
            with PILImage.open(path) as img:
                img.verify() 
        except:
            return False
    return True


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.modules["vllm"] = None

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTTrainer, SFTConfig
from utils.trainers import PrunedSFTTrainer
from peft import LoraConfig

# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# --- CONFIGURATION ---
MODEL_ID = "Aasdfip/qwen3_webnav_0.1" #,"Qwen/Qwen3-VL-2B-Instruct" # Or your specific VLM backbone
TARGET_SEQ_LEN = 1024                  # The 'L' we are testing
BATCH_SIZE = 1                         # The 'B' we are testing
GRADIENT_CHECKPOINTING = True          # Standard for VLM/LLM training
USE_FLASH_ATTN = True                  # Highly recommended for A100


TRAIN_DATASET_DIR = "tccoin/navverse-benchmark"
EVAL_DATASET_DIR = None
OUTPUT_DIR = "./dump/navverse_sft"
EVAL_MAX_SAMPLES = 40

def get_peak_memory_gb():
    """Helper to get peak GPU memory in GB"""
    return torch.cuda.max_memory_allocated() / (1024 ** 3)

def preprocess_logits_for_metrics(logits, labels):
    """
    Reduces logits to argmax predictions on GPU to save memory.
    """
    if isinstance(logits, tuple):
        # Depending on model/config, logits might be (logits, loss) or similar
        logits = logits[0]
    
    # Argmax on GPU, return integer IDs
    return logits.argmax(dim=-1)


from utils.training_utils import SpatialFeatureExtractor, get_image_token_indices
from utils.pose import SfMPoseLoss
class PoseTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hook is ephemeral, so we keep it here. 
        # Layer -6 is a good choice (High semantics, but before final reasoning).
        self.spatial_extractor = SpatialFeatureExtractor(self.model, layer_index=-12)
        self.pose_loss = SfMPoseLoss()
        self.loss_weights = {
        "loss_scale": 0.015,  # Down-weight the noisy scale loss
        "loss_grav": 1.0,    # Gravity is a regularization, keep it small
        "loss_rot": 1.0,     # Explicitly stating 1.0 is fine for clarity
        "loss_trans": 0.7
    }

    def create_optimizer(self):
        """
        Setup the optimizer with two parameter groups:
        1. Spatial Head: High Learning Rate (e.g., 1e-3)
        2. Backbone (LoRA): Low Learning Rate (e.g., 3e-5)
        """
        if self.optimizer is None:
            decay_parameters = []
            no_decay_parameters = []
            spatial_head_params = []
            
            # 1. Separate Parameters
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Check if it belongs to our new head
                if "spatial_head" in name:
                    spatial_head_params.append(param)
                else:
                    # Standard Weight Decay logic for Backbone
                    if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                        no_decay_parameters.append(param)
                    else:
                        decay_parameters.append(param)

            optimizer_grouped_parameters = [
                {
                    "params": spatial_head_params,
                    "weight_decay": self.args.weight_decay,
                    "lr": 1e-3,  # <--- CRITICAL: High LR for the scratch head
                },
                {
                    "params": decay_parameters,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate, # The global args.learning_rate (3e-5)
                },
                {
                    "params": no_decay_parameters,
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]
            # optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            
        return self.optimizer
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        # 1. Pop custom args
        gt_t = inputs.pop('gt_t')
        gt_q = inputs.pop('gt_q')
        if 'batch_image_counts' in inputs:
            batch_image_counts = inputs.pop('batch_image_counts').tolist()
        else:
            batch_image_counts = []
        # 2. Standard LM Loss
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        
        # 3. Extract Hidden State
        base_model = model.module if hasattr(model, 'module') else model
        intermediate_hidden = self.spatial_extractor.get_and_clear()
        
        # 4. Spatial Forward
        batch_image_indices, _ = get_image_token_indices(inputs['input_ids'], self.processing_class)
        pred_t, pred_q = base_model.spatial_head(intermediate_hidden, batch_image_indices)
        pred_t = pred_t.float()
        pred_q = pred_q.float()
        # 5. Compute Spatial Loss Components
        loss_components_dict = self.pose_loss.forward(pred_t, pred_q, gt_t, gt_q, batch_image_counts)
        
        # Sum components
        spatial_loss_scalar = sum(
            val * self.loss_weights.get(key, 1.0) 
            for key, val in loss_components_dict.items()
        )        
        total_loss = loss + spatial_loss_scalar

        # 6. Detailed Metric Logging (Train AND Eval)
        mode = "train" if self.model.training else "eval"
        
        # Log total spatial loss
        if "spatial loss" not in self._metrics[mode]:
            self._metrics[mode]["spatial loss"] = []
        self._metrics[mode]["spatial loss"].append(spatial_loss_scalar.item())
        
        # Log individual components
        for k, v in loss_components_dict.items():
            if k not in self._metrics[mode]:
                self._metrics[mode][k] = []
            self._metrics[mode][k].append(v.item())

        return (total_loss, outputs) if return_outputs else total_loss

from utils.collators import ActionMaskingVLMCollator
from utils.data_misc import make_dynamic_resize_transform
SYSTEM_TOKENS = 190
TURN_TOKENS = 33
ORIG_H = 480
ORIG_W = 640
TOTAL_BUDGET = 39000#34000
dynamic_resize_transform = make_dynamic_resize_transform(SYSTEM_TOKENS,TURN_TOKENS,ORIG_H,ORIG_W,TOTAL_BUDGET-600)


















NAVVERSE_ACTION_TEXT = {
    "STOP": "stop",
    "MOVE_FORWARD": "forward",
    "TURN_LEFT": "left",
    "TURN_RIGHT": "right",
}
NAVVERSE_NORMALIZE_ACTIONS = False
NAVVERSE_WINDOW_IMAGES = 0


def _normalize_navverse_text(text):
    if text is None:
        return text
    text = text.replace("[STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]", "[stop, forward, left, right]")
    for src, dst in NAVVERSE_ACTION_TEXT.items():
        text = text.replace(f"**{src}**", f"**{dst}**")
    return text


def normalize_navverse_messages(example):
    messages = example.get("messages")
    if messages is None:
        return example
    return {"messages": _normalize_navverse_message_list(messages)}


def _normalize_navverse_message_list(messages):
    new_messages = []
    for message in messages:
        new_message = dict(message)
        new_content = []
        for item in message.get("content", []):
            new_item = dict(item)
            if "text" in new_item:
                new_item["text"] = _normalize_navverse_text(new_item["text"])
            new_content.append(new_item)
        new_message["content"] = new_content
        new_messages.append(new_message)
    return new_messages


def normalize_navverse_batch(batch):
    batch = _maybe_window_sample_navverse_batch(batch)
    messages = batch.get("messages")
    if NAVVERSE_NORMALIZE_ACTIONS and messages is not None:
        if messages and isinstance(messages[0], list):
            batch["messages"] = [_normalize_navverse_message_list(item) for item in messages]
        else:
            batch["messages"] = _normalize_navverse_message_list(messages)
    images = batch.get("images")
    if images is not None:
        if images and isinstance(images[0], list):
            batch["images"] = [_decode_navverse_image_sequence(item) for item in images]
        else:
            batch["images"] = _decode_navverse_image_sequence(images)
    return batch


def _maybe_window_sample_navverse_batch(batch):
    if NAVVERSE_WINDOW_IMAGES <= 0:
        return batch
    images = batch.get("images")
    if images is None:
        return batch
    if images and isinstance(images[0], list):
        sampled = {key: [] for key in batch.keys()}
        for idx in range(len(images)):
            item = {key: value[idx] for key, value in batch.items()}
            item = _window_sample_navverse_item(item)
            for key in batch.keys():
                sampled[key].append(item.get(key))
        return sampled
    return _window_sample_navverse_item(dict(batch))


def _window_sample_navverse_item(example):
    import random

    images = example.get("images")
    if not images:
        return example
    num_images = len(images)
    window = min(NAVVERSE_WINDOW_IMAGES, num_images)
    if window <= 0 or num_images <= window:
        return example

    start = random.randint(0, num_images - window)
    end = start + window
    example["images"] = images[start:end]

    messages = example.get("messages")
    if messages is not None:
        if len(messages) == 1 + 2 * num_images:
            example["messages"] = [messages[0]] + messages[1 + 2 * start : 1 + 2 * end]
        elif len(messages) == 2 * num_images:
            example["messages"] = messages[2 * start : 2 * end]

    for key in ("action_sequence", "pos_rots", "poses", "rgb_sequence"):
        value = example.get(key)
        if isinstance(value, list) and len(value) == num_images:
            example[key] = value[start:end]

    action_ids = example.get("action_ids")
    if isinstance(action_ids, list):
        if len(action_ids) == num_images:
            example["action_ids"] = action_ids[start:end]
        elif len(action_ids) == num_images + 1:
            example["action_ids"] = action_ids[start : end + 1]

    return example


def _decode_navverse_image_sequence(images):
    import io
    from PIL import Image as PILImage

    decoded = []
    for image in images:
        if hasattr(image, "size"):
            decoded.append(image)
        elif isinstance(image, dict):
            if image.get("bytes") is not None:
                with PILImage.open(io.BytesIO(image["bytes"])) as img:
                    decoded.append(img.convert("RGB"))
            elif image.get("path"):
                with PILImage.open(os.path.expanduser(image["path"])) as img:
                    decoded.append(img.convert("RGB"))
            else:
                raise ValueError("image dict has neither bytes nor path")
        else:
            decoded.append(image)
    return decoded


def _dataset_paths(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _parquet_files(path):
    files = []
    for raw_path in _dataset_paths(path):
        files.extend(_parquet_files_for_path(raw_path))
    if files:
        return sorted(dict.fromkeys(files))
    return []


def _parquet_files_for_path(raw_path):
    files = []
    if not raw_path:
        return files
    expanded = os.path.expanduser(raw_path)
    if os.path.isdir(expanded):
        root = Path(expanded)
        files.extend(sorted(str(p) for p in root.glob("*.parquet")))
        data_dir = root / "data"
        if data_dir.is_dir():
            files.extend(sorted(str(p) for p in data_dir.glob("*.parquet")))
    elif os.path.isfile(expanded) and expanded.endswith(".parquet"):
        files.append(expanded)
    return sorted(dict.fromkeys(files))


def _parquet_file_groups(path):
    groups = []
    for raw_path in _dataset_paths(path):
        files = _parquet_files_for_path(raw_path)
        if files:
            groups.append((raw_path, files))
    return groups


def _apply_navverse_transform(dataset, args, decode_images=False):
    global NAVVERSE_NORMALIZE_ACTIONS, NAVVERSE_WINDOW_IMAGES
    NAVVERSE_NORMALIZE_ACTIONS = bool(getattr(args, "normalize_navverse_actions", False))
    NAVVERSE_WINDOW_IMAGES = int(getattr(args, "window_sample_images", 0) or 0)
    if not NAVVERSE_NORMALIZE_ACTIONS and not decode_images and NAVVERSE_WINDOW_IMAGES <= 0:
        return dataset
    if NAVVERSE_NORMALIZE_ACTIONS:
        print("Normalizing NavVerse action text: STOP/MOVE_FORWARD/TURN_LEFT/TURN_RIGHT -> stop/forward/left/right")
    if NAVVERSE_WINDOW_IMAGES > 0:
        print(f"Window sampling Train: random contiguous windows of <= {NAVVERSE_WINDOW_IMAGES} images")
    if hasattr(dataset, "with_transform"):
        return dataset.with_transform(normalize_navverse_batch)
    return dataset.map(normalize_navverse_messages)


def _maybe_filter_navverse_max_images(dataset, args):
    max_images = int(getattr(args, "max_train_images", 0) or 0)
    if max_images <= 0:
        return dataset
    print(f"Filtering Train: keeping episodes with <= {max_images} images")
    try:
        return dataset.filter(
            lambda example: len(example.get("images") or []) <= max_images,
            desc=f"Max train images <= {max_images}",
        )
    except TypeError:
        return dataset.filter(lambda example: len(example.get("images") or []) <= max_images)


def _filter_by_episode_ids(dataset, episode_ids, keep=True):
    if "episode_id" not in dataset.column_names:
        raise ValueError("Holdout eval manifest requires an episode_id column")
    episode_id_set = {str(episode_id) for episode_id in episode_ids}
    return dataset.filter(
        lambda episode_id: (str(episode_id) in episode_id_set) == keep,
        input_columns=["episode_id"],
    )


def _holdout_eval_manifest_path(args):
    manifest_path = str(getattr(args, "holdout_eval_manifest", "") or "").strip()
    if manifest_path:
        return os.path.expanduser(manifest_path)
    ratio = float(getattr(args, "holdout_eval_ratio", 0.0) or 0.0)
    if ratio > 0.0:
        return os.path.join(args.output_dir, "holdout_eval_manifest.json")
    return ""


def _load_holdout_eval_manifest(args):
    import json

    manifest_path = _holdout_eval_manifest_path(args)
    if not manifest_path or not os.path.exists(manifest_path):
        return {"version": 1, "sources": {}}
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    manifest.setdefault("version", 1)
    manifest.setdefault("sources", {})
    print(f"Loading holdout eval manifest: {manifest_path}")
    return manifest


def _save_holdout_eval_manifest(args, manifest):
    import json

    manifest_path = _holdout_eval_manifest_path(args)
    if not manifest_path:
        return
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved holdout eval manifest: {manifest_path}")


def _split_train_eval_dataset(dataset, args, source_label, manifest):
    source_manifests = manifest.setdefault("sources", {})
    source_manifest = source_manifests.get(source_label)
    if source_manifest and source_manifest.get("eval_episode_ids"):
        eval_episode_ids = source_manifest["eval_episode_ids"]
        eval_dataset = _filter_by_episode_ids(dataset, eval_episode_ids, keep=True)
        train_dataset = _filter_by_episode_ids(dataset, eval_episode_ids, keep=False)
        print(
            f"Loaded holdout split for {source_label}: train={len(train_dataset)}, "
            f"eval={len(eval_dataset)}, manifest_ids={len(eval_episode_ids)}"
        )
        return train_dataset, eval_dataset

    ratio = float(getattr(args, "holdout_eval_ratio", 0.0) or 0.0)
    if ratio <= 0.0:
        return dataset, None
    if ratio >= 1.0:
        raise ValueError("--holdout_eval_ratio must be smaller than 1.0")
    if len(dataset) < 2:
        print(f"Skipping holdout split for {source_label}: dataset has {len(dataset)} row(s)")
        return dataset, None

    seed = int(getattr(args, "holdout_eval_seed", 42) or 42)
    split = dataset.train_test_split(test_size=ratio, seed=seed, shuffle=True)
    eval_episode_ids = [str(episode_id) for episode_id in split["test"]["episode_id"]]
    source_manifests[source_label] = {
        "source": source_label,
        "ratio": ratio,
        "seed": seed,
        "train_count": len(split["train"]),
        "eval_count": len(split["test"]),
        "eval_episode_ids": eval_episode_ids,
    }
    print(
        f"Holdout split for {source_label}: train={len(split['train'])}, "
        f"eval={len(split['test'])}, ratio={ratio:g}, seed={seed}"
    )
    return split["train"], split["test"]


def _concat_datasets(datasets):
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    from datasets import concatenate_datasets

    return concatenate_datasets(datasets)


def load_train_eval_datasets_for_navverse(args, use_streaming):
    from datasets import load_from_disk, load_dataset, Sequence, Image
    from utils.data_misc import decode_image_sequence

    parquet_groups = _parquet_file_groups(args.train_dataset_dir)
    if parquet_groups:
        train_parts = []
        eval_parts = []
        holdout_manifest = _load_holdout_eval_manifest(args)
        total_shards = sum(len(files) for _, files in parquet_groups)
        print(f"Loading Train (Parquet): {total_shards} shard(s) from {args.train_dataset_dir}")
        for source_label, parquet_files in parquet_groups:
            print(f"Loading Train source (Parquet): {len(parquet_files)} shard(s) from {source_label}")
            dataset = load_dataset("parquet", data_files=parquet_files, split="train")
            dataset = _maybe_filter_navverse_max_images(dataset, args)
            train_part, eval_part = _split_train_eval_dataset(dataset, args, source_label, holdout_manifest)
            train_parts.append(train_part)
            if eval_part is not None:
                eval_parts.append(eval_part)
        if eval_parts:
            _save_holdout_eval_manifest(args, holdout_manifest)
        train_dataset = _concat_datasets(train_parts)
        eval_dataset = _concat_datasets(eval_parts)
        train_dataset = _apply_navverse_transform(train_dataset, args, decode_images=True)
        if eval_dataset is not None:
            eval_dataset = _apply_navverse_transform(eval_dataset, args, decode_images=True)
        return train_dataset, eval_dataset

    if not os.path.exists(os.path.expanduser(args.train_dataset_dir)) and "/" in args.train_dataset_dir:
        print(f"Loading Train (Streaming): {args.train_dataset_dir}")
        train_dataset = load_dataset(args.train_dataset_dir, split="train", streaming=use_streaming)
        if use_streaming:
            train_dataset = train_dataset.map(decode_image_sequence)
        else:
            train_dataset = train_dataset.shuffle(seed=42)
            train_dataset = train_dataset.cast_column("images", Sequence(Image(decode=True)))
        train_dataset = _maybe_filter_navverse_max_images(train_dataset, args)
        return _apply_navverse_transform(train_dataset, args), None

    print(f"Loading Train (Disk): {args.train_dataset_dir}")
    train_dataset = load_from_disk(args.train_dataset_dir)
    train_dataset = _maybe_filter_navverse_max_images(train_dataset, args)
    return _apply_navverse_transform(train_dataset, args, decode_images=True), None


def load_train_dataset_for_navverse(args, use_streaming):
    train_dataset, _ = load_train_eval_datasets_for_navverse(args, use_streaming)
    return train_dataset


def parse_args():
    p = argparse.ArgumentParser(description="SFT training (minimal CLI: only overrides hardcoded paths)")
    p.add_argument("--model_id", type=str, default=MODEL_ID, help="HF model id or local path")
    p.add_argument("--train_dataset_dir", type=str, default=TRAIN_DATASET_DIR, help="load_from_disk() dir")
    p.add_argument(
        "--eval_dataset_dir",
        type=str,
        default=EVAL_DATASET_DIR,
        help="load_from_disk() dir (use empty string to disable eval)",
    )
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Trainer output_dir")
    p.add_argument("--eval_max_samples", type=int, default=EVAL_MAX_SAMPLES, help="Eval subset size")
    p.add_argument("--print_config", action="store_true", help="Print config then exit")
    p.add_argument("--resume_path", type=str, default="",help = "checkpoint to resume")
    p.add_argument("--max_steps", type=int, default=1000, help="Trainer max_steps")
    p.add_argument("--save_steps", type=int, default=77, help="Checkpoint save interval")
    p.add_argument("--eval_steps", type=int, default=77, help="Eval interval")
    p.add_argument("--logging_steps", type=int, default=1, help="Logging interval")
    p.add_argument("--learning_rate", type=float, default=3e-5, help="Trainer learning rate")
    p.add_argument("--optim", type=str, default="adamw_torch_fused", help="Trainer optimizer name")
    p.add_argument("--warmup_steps", type=int, default=0, help="Trainer warmup steps")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Trainer gradient accumulation steps")
    p.add_argument("--dataloader_num_workers", type=int, default=2, help="Trainer dataloader workers")
    p.add_argument("--report_to", type=str, default="none", help="Trainer report_to target; use none to disable")
    p.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"], help="Attention backend")
    p.add_argument("--lora_r", type=int, default=128, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=256, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--collator_dropout", type=float, default=0.3, help="Action masking collator dropout")
    p.add_argument("--normalize_navverse_actions", action="store_true", help="Map NavVerse action labels to stop/forward/left/right text")
    p.add_argument("--max_train_images", type=int, default=0, help="Drop train episodes longer than this many images; 0 disables filtering")
    p.add_argument("--window_sample_images", type=int, default=0, help="Randomly crop each train episode to this many contiguous images; 0 disables window sampling")
    p.add_argument("--holdout_eval_ratio", type=float, default=0.0, help="Per-train-source eval holdout ratio; 0 disables automatic holdout")
    p.add_argument("--holdout_eval_seed", type=int, default=42, help="Seed for per-source holdout eval split")
    p.add_argument("--holdout_eval_manifest", type=str, default="", help="JSON file of per-source holdout eval episode ids to create or reuse")
    p.add_argument("--ddp_find_unused_parameters", type=str, default="false", choices=["true", "false"], help="DDP find_unused_parameters setting")
    return p.parse_args()


def main():
    args = parse_args()

    if args.print_config:
        print(vars(args))
        return

    print(f"--- Starting Memory Profile ---")
    print(f"Model: {args.model_id}")
    print(f"Batch Size: {BATCH_SIZE}")
    
    # 1. Load Model in bfloat16 (No quantization)
    print("Loading model...")
    # model = AutoModelForImageTextToText.from_pretrained(
    #     args.model_id,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2", # Changed from flash_attention_2 to sdpa to avoid extra dependencies
    #     # device_map="auto",
    #     # Force single-GPU placement
    #     device_map={"": 0} if torch.cuda.is_available() else None,
    #     # use_cache=False # Important for training with gradient checkpointing
    # )

    
    # 2. Instantiate our Custom Sparse Class
    # This creates the model with random weights but the correct sparse architecture
    # model = Qwen3VLSparseForConditionalGeneration(config)
    
    # 3. Load Pretrained Weights
    # We use from_pretrained on our class, pointing to the original model directory.
    # Because our attribute names (self.model, self.text_model) match the original,
    # the weights map 1:1.
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    model = Qwen3VLSparseForConditionalGeneration.from_pretrained(
        args.model_id, 
        config=config,
        # device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_impl,
    )
    model.enable_input_require_grads()
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Setup LoRA
    print("Applying LoRA...")

    peft_config = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                # modules_to_save=["multi_modal_projector"],
                modules_to_save=["spatial_head"], 
    )

    # Load data from HF
    from datasets import load_from_disk, load_dataset, Sequence, Image,Value
    from utils.data_misc import decode_image_sequence
    use_streaming = True
    train_dataset, holdout_eval_dataset = load_train_eval_datasets_for_navverse(args, use_streaming)
    if holdout_eval_dataset is not None:
        eval_dataset = holdout_eval_dataset
    elif args.eval_dataset_dir:
        if not os.path.exists(os.path.expanduser(args.eval_dataset_dir)) and "/" in args.eval_dataset_dir:
            print(f"Loading Eval (Streaming): {args.eval_dataset_dir}")
            # Try to load 'validation' split, or fallback to 'train' if needed (user provided specific val repo)
            try:
                eval_dataset = load_dataset(args.eval_dataset_dir, split="validation", streaming=use_streaming)
            except:
                eval_dataset = load_dataset(args.eval_dataset_dir, split="train", streaming=use_streaming)
            
            if use_streaming:
                eval_dataset = eval_dataset.map(decode_image_sequence)
                print(f"length of images in sample before dynamic resize: {len(next(iter(eval_dataset))['images'])}")
                eval_dataset = eval_dataset.map(dynamic_resize_transform, batched=True,batch_size=1)
                print(f"length of images in sample: {len(next(iter(eval_dataset))['images'])}")
            else:
                pass
                # eval_dataset.set_transform(dynamic_resize_transform)
            # For eval, we need a finite number of samples
            if args.eval_max_samples:
                eval_dataset = eval_dataset.take(args.eval_max_samples)
            
        else:
            print(f"Loading Eval (Disk): {args.eval_dataset_dir}")
            eval_dataset = load_from_disk(args.eval_dataset_dir)
            eval_dataset = eval_dataset.cast_column('images',Sequence(Value(dtype='string')))

            if args.eval_max_samples is not None and args.eval_max_samples > 0:
                eval_dataset = eval_dataset.select(range(min(args.eval_max_samples, len(eval_dataset))))
            eval_dataset = eval_dataset.filter(validate_episode_images, num_proc=32, desc="Img Verify",batch_size=10)

            eval_dataset = eval_dataset.cast_column("images", Sequence(Image()))
            # eval_dataset.set_transform(dynamic_resize_transform)
    else:
        eval_dataset = None
    # eval_dataset = None

    # 4. Training Arguments
    training_args = SFTConfig(
        accelerator_config={
            "dispatch_batches":False
        },
        output_dir=args.output_dir,
        # run_name="qwen-vln-action-dropout",
        save_strategy="steps",        # Save checkpoints frequently
        save_steps=args.save_steps,
                  # Save every 500 steps
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=1,
        
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        max_length=None,#TARGET_SEQ_LEN,
        packing=False, # FALSE is critical to strictly enforce batch_size x seq_len shape
        bf16=True,     # Use bfloat16 for A100
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        report_to=[] if args.report_to.lower() in {"none", "disabled", "disable"} else args.report_to,
        # dataset_text_field="text",

        resume_from_checkpoint=args.resume_path or None,
        assistant_only_loss=False,
        optim=args.optim,
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters.lower() == "true",

        remove_unused_columns=False,
        # resume_from_checkpoint='/Projects/SG_VLN_HumanData/contrastive_training_5view_mlp/checkpoint-4050'
        # processing_class = processor
    )

    # 5. Initialize Trainer
    trainer = PrunedSFTTrainer(
        model=model,
        data_collator=ActionMaskingVLMCollator(
            processor=processor,
            length_warning = TOTAL_BUDGET,
            # max_length=TOTAL_BUDGET,
            dropout=args.collator_dropout,
        ),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processor,
        # callbacks=[LRSanityCheckCallback(),GradientDebugCallback()],
        # compute_metrics=compute_metrics_wrapper,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # tokenizer=tokenizer,
    )
    if hasattr(trainer.model, "base_model"):
        if hasattr(trainer.model.base_model, "spatial_head"):
            for param in trainer.model.base_model.spatial_head.parameters():
                param.requires_grad = True
    # 6. Run Training & Measure
    torch.cuda.reset_peak_memory_stats()
    print("Starting training loop...")
    trainer.train(resume_from_checkpoint=args.resume_path or None)
    # try:
        
    # except Exception as e:
    #     print(f"\n[!] Training interrupted (likely OOM or interrupt): {e}")
    
    peak_mem = get_peak_memory_gb()
    print(f"\n" + "="*30)
    print(f"RESULTS for B={BATCH_SIZE}, L={TARGET_SEQ_LEN}")
    print(f"Peak VRAM Used: {peak_mem:.2f} GB")
    print(f"="*30)

if __name__ == "__main__":
    main()

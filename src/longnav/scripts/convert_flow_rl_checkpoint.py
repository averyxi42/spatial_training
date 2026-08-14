"""Convert an RL flow-SDE checkpoint back to the SFT/harness layout.

The RL trainer saves one peft adapter dir (``save_checkpoint_unsafe``): 392 backbone LoRA
tensors plus the whole ``FlowSDEHead`` flattened under
``base_model.model.action_head.{readout,codec}.*`` (peft ``modules_to_save``). The eval
harness's ``flow_rollout`` backend instead loads the SFT layout:

    <out>/turn_vector_head_config.json     (unchanged -- RL trains weights, not config)
    <out>/turn_vector_head.pt              {"head": readout, "normalizer": codec, "modality": ...}
    <out>/adapter/{adapter_config.json, adapter_model.safetensors}
    <out>/tokenizer + preprocessor files   (so AutoProcessor.from_pretrained(out) works)

The two layouts name the LoRA tensors IDENTICALLY (verified 392/392 exact overlap), so the
conversion is key surgery, not remapping. The head blob's key sets are asserted equal to
the source checkpoint's -- a missing or extra key is a refusal, never a partial write.

Usage:
    python -m longnav.scripts.convert_flow_rl_checkpoint \
        --rl-checkpoint dump/flow_rl/<run>/checkpoints/checkpoint_15 \
        --sft-checkpoint dump/pose_injection/run_cotrain_v3_nopose_mix/checkpoint-12000 \
        --out dump/flow_rl/<run>/checkpoints/checkpoint_15_harness

Verify with tests/parity_rollout_paths.py-style h parity before trusting an eval number.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

RL_HEAD_PREFIX = "base_model.model.action_head."
PROCESSOR_FILES = (
    "added_tokens.json", "chat_template.jinja", "merges.txt", "preprocessor_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    "video_preprocessor_config.json", "vocab.json",
)


def convert(rl_checkpoint: Path, sft_checkpoint: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    rl_weights = load_file(str(rl_checkpoint / "adapter_model.safetensors"))

    # --- head blob: swap head/normalizer states, keep everything else (e.g. modality) ---
    blob = torch.load(sft_checkpoint / "turn_vector_head.pt",
                      map_location="cpu", weights_only=False)
    for blob_key, rl_sub in (("head", "readout."), ("normalizer", "codec.")):
        pref = RL_HEAD_PREFIX + rl_sub
        state = {k[len(pref):]: v.float().cpu() for k, v in rl_weights.items()
                 if k.startswith(pref)}
        if set(state) != set(blob[blob_key]):
            missing = set(blob[blob_key]) - set(state)
            extra = set(state) - set(blob[blob_key])
            raise RuntimeError(
                f"{blob_key} key set mismatch: missing {sorted(missing)[:5]}, "
                f"extra {sorted(extra)[:5]} -- refusing a partial conversion")
        blob[blob_key] = state
    torch.save(blob, out / "turn_vector_head.pt")

    # --- adapter dir: SFT config (no modules_to_save), RL LoRA tensors, names as-is ----
    sft_lora = load_file(str(sft_checkpoint / "adapter" / "adapter_model.safetensors"))
    lora = {k: v for k, v in rl_weights.items() if ".lora_" in k}
    if set(lora) != set(sft_lora):
        raise RuntimeError(
            f"LoRA key sets differ ({len(lora)} vs {len(sft_lora)}); the two checkpoints "
            "do not share a backbone/peft structure -- refusing to convert")
    (out / "adapter").mkdir(exist_ok=True)
    save_file(lora, str(out / "adapter" / "adapter_model.safetensors"))
    shutil.copy2(sft_checkpoint / "adapter" / "adapter_config.json",
                 out / "adapter" / "adapter_config.json")

    # --- config + processor files ------------------------------------------------------
    shutil.copy2(sft_checkpoint / "turn_vector_head_config.json",
                 out / "turn_vector_head_config.json")
    for name in PROCESSOR_FILES:
        src = sft_checkpoint / name
        if src.exists():
            shutil.copy2(src, out / name)

    manifest = {
        "converted_from": str(rl_checkpoint),
        "sft_layout_source": str(sft_checkpoint),
        "lora_tensors": len(lora),
        "head_tensors": len(blob["head"]),
        "normalizer_tensors": len(blob["normalizer"]),
    }
    (out / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"converted -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rl-checkpoint", type=Path, required=True)
    p.add_argument("--sft-checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    convert(a.rl_checkpoint, a.sft_checkpoint, a.out)

"""Export an RL flow-SDE checkpoint as a canonical fresh-reload bundle.

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

The bundle deliberately keeps the frozen SFT adapter separate.  The rollout loader
moves the bf16 base model to its target GPU before merging that adapter, matching
the RL evaluator's arithmetic.  Do not pre-merge it during export.
"""
import argparse
import hashlib
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

    # --- adapter dirs: native eval merges SFT before applying the RL LoRA --------------
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
    shutil.copytree(sft_checkpoint / "adapter", out / "sft_adapter")

    # --- config + processor files ------------------------------------------------------
    shutil.copy2(sft_checkpoint / "turn_vector_head_config.json",
                 out / "turn_vector_head_config.json")
    for name in PROCESSOR_FILES:
        src = sft_checkpoint / name
        if src.exists():
            shutil.copy2(src, out / name)

    # RL checkpoints train this head under a distinct resume-only filename;
    # the external rollout harness expects the SFT layout filename.
    state_probe = rl_checkpoint / "state_probe_rl.pt"
    if not state_probe.exists():
        state_probe = sft_checkpoint / "state_probe.pt"
    if state_probe.exists():
        shutil.copy2(state_probe, out / "state_probe.pt")
        shutil.copy2(sft_checkpoint / "state_probe_config.json",
                     out / "state_probe_config.json")

    manifest = {
        "bundle_format": "longnav.flow_rl.fresh_reload.v1",
        "converted_from": str(rl_checkpoint),
        "sft_layout_source": str(sft_checkpoint),
        "lora_tensors": len(lora),
        "head_tensors": len(blob["head"]),
        "normalizer_tensors": len(blob["normalizer"]),
        "state_probe_source": str(state_probe) if state_probe.exists() else None,
        "sft_adapter_source": str(sft_checkpoint / "adapter"),
        "sft_adapter_sha256": hashlib.sha256(
            (out / "sft_adapter" / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "rl_adapter_sha256": hashlib.sha256(
            (out / "adapter" / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "sft_adapter_merge": "target_gpu_at_load",
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

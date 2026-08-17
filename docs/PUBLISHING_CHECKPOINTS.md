# Publishing a checkpoint

Written 2026-08-15 by reading back what the already-published models actually do, so the
next upload matches them instead of inventing a second convention.

Published models live under the **`Aasdfip/`** HF namespace and are **public**. The
existing family:

| repo | what |
|---|---|
| `longnav-objectnav-flow-nopose-cotrain-2p5hz` | cotrain-v3 ck12000 — the SFT policy every RL run initialises from |
| `longnav-objectnav-flow-pose-2p5hz` | pose-injected sibling |
| `longnav-objectnav-flow-pose-1hz` | 1 Hz decision-rate sibling |
| `longnav-objectnav-flow-nopose-cotrain-2p5hz-merged` | SFT LoRA folded into weights — the base every RL adapter applies to |
| `longnav-objectnav-flow-nopose-cotrain-2p5hz-rl-a09-ck303` | RL delta, 24-episode run (its card's table came from the rank-stacked path) |
| `longnav-objectnav-flow-nopose-cotrain-2p5hz-rl-a09-held128-ck791` | RL delta, held128 run — sample400 0.733 oracle / 0.483 oSPL, measured on the shipped merged+adapter composition |

RL releases carry the cycle number in the name (unlike SFT repos): an RL checkpoint is a
frozen artifact of a specific cycle, selected by an eval series, and the run it came from
does not "continue" the way an SFT run does.

Naming: `longnav-<task>-<head>-<pose variant>-<observation rate>`, all lowercase, hyphens.
Append `-merged` for a weights-merged variant (below). Do NOT put the step number in the
name — `trainer_state.json` carries it, and a name that says `ckpt12000` goes stale the
moment the run continues.

## What goes in the repo

The **adapter layout** — what the eval harness loads directly:

| file | why |
|---|---|
| `adapter/` | LoRA (`adapter_config.json` + `adapter_model.safetensors`) |
| `turn_vector_head.pt` | action head + pose encoder |
| `turn_vector_head_config.json` | head config, including the `<pose>` modality spec |
| `trainer_state.json` | the whole training curve — this is the provenance, and it is how a reader identifies which local checkpoint the repo is |
| tokenizer / preprocessor files | `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, `merges.txt`, `vocab.json`, `preprocessor_config.json`, `video_preprocessor_config.json`, `chat_template.jinja` |

**Optimizer and RNG state are deliberately excluded**: these are inference checkpoints, not
resume points. Uploading them quadruples the size and invites someone to resume a run whose
data pipeline no longer exists.

## The model card

Frontmatter, verbatim shape:

```yaml
---
license: apache-2.0
base_model: Qwen/Qwen3-VL-2B-Instruct
library_name: peft          # omit for a merged model; it is no longer a peft repo
tags: [robotics, navigation, objectnav, pointnav, vision-language-action, habitat, flow-matching]
---
```

Then, in order, and each section earns its place:

1. **One paragraph**: what the policy is, its observation rate, its training mixture in one
   line, and the headline result.
2. **What this is**: the architecture in physical terms — chunk of 20 relative planar poses
   at 0.04 s spacing, PID-tracked on a holonomic base — because "ObjectNav policy" otherwise
   reads as a discrete-action agent, which this is not.
3. **The result table, paired.** State the arms and the episode set (`sample101`, n=101).
4. **Training mixture table**: component, ratio, rows, what each is. Note realised split, and
   whether components share episodes.
5. **Files table**, as above.
6. **What the checkpoint depends on that is NOT public yet.** The cotrain card names
   `--no-pose-injection` and the mixed-modality collator fix, and says plainly that the
   commands below will not run without them. A checkpoint whose documented invocation does
   not run is worse than an undocumented one.
7. **Running / evaluating it**: the FULL eval command, every flag. This is not a courtesy —
   the command IS the record of the conditions the reported numbers were measured under.
   Flags that silently change what a number means (`--no-pose-injection`, `--dt`, `--gap`,
   `--max-steps`, `--navmesh`, `--success-distance`) must appear explicitly even where they
   match a default. Mention the two-environment split (`--policy-python`,
   `--policy-sys-path`) — the simulator and the model cannot share an interpreter.

## Merged variants (`-merged`)

A merged repo is the base model with the SFT LoRA **folded into the weights**: full
`model.safetensors`, no `adapter/`.

Publish one when RL is trained on a merged base. Those runs' saved adapters contain the RL
delta ALONE (the SFT policy lives in the base weights), so evaluating them from the plain
SFT repo requires composing two adapters — rank-stacking `A` and `B` into an r=256/alpha=512
adapter — which is easy to get silently wrong: apply the RL delta to the raw base and you
evaluate the pretrained VLM plus a 1%-magnitude delta, which scores near zero and errors
nowhere. With a merged repo published, an RL adapter applies directly on top and nobody
has to reproduce that arithmetic.

Building one:

```python
base = AutoModelForImageTextToText.from_pretrained(base_id, dtype=torch.bfloat16, device_map="cpu")
w_base = base.state_dict()[probe_key].clone().float()          # BEFORE merging
model = PeftModel.from_pretrained(base, sft/"adapter", is_trainable=False).merge_and_unload()
# VERIFY, do not assume: merged == base + (alpha/r) * B @ A on one probe module.
# bf16 rounding puts the residual around 1e-3; anything larger is a real mismatch.
model.save_pretrained(out, safe_serialization=True)
AutoProcessor.from_pretrained(base_id).save_pretrained(out)
# then copy turn_vector_head.pt, turn_vector_head_config.json, trainer_state.json
```

Do it on CPU: it is a weight operation, and the GPUs are usually busy with the run that
motivated it. ~4 GB on disk and to upload.

The merged card must additionally say: which local checkpoint it corresponds to (match
`trainer_state.json` `global_step` against the adapter repo), that it is numerically the
same policy as the adapter repo, and that a merged model cannot be diffed against the base
to recover the LoRA — so the adapter repo stays the canonical artifact.

## Is `merged + RL adapter` the same as `raw base + stacked adapter`?

**In exact arithmetic yes; in stored bf16 no, and the gap is larger than the RL update.**
Measured 2026-08-15 on `layers.0.self_attn.q_proj` and five siblings:

| quantity | Frobenius norm | as % of `||W||` |
|---|---|---|
| weight `||W||` | 67.44 | -- |
| SFT delta | 1.043 | 1.55% |
| **bf16 rounding of the merged store** | **0.112** | 0.166% |
| RL delta (a09 ck303) | 0.023 | 0.03% |

Algebraically the two paths are identical: stacking gives
`W + (512/256)([B_s B_r][A_s; A_r]) = W + 2 B_s A_s + 2 B_r A_r`, and merged-plus-RL gives
`(W + 2 B_s A_s) + 2 B_r A_r`. What differs is that the merged repo **stores** the first
sum in bf16, so the two differ by that rounding -- ULP-scale per element (max 1.5e-3,
100% of elements touched) but **~5x the RL delta in norm** (4.1x-7.6x across probes).

Two things follow, and the second is the one that matters:

* The rounding is **unstructured** where the RL delta is a learned direction, so equal norm
  does not mean equal behavioural effect -- a bf16 store is exactly the perturbation the
  model already tolerates by construction. Expect the metric difference to be small. It is
  nonetheless **not measured**, so do not claim the paths are interchangeable; claim they
  are algebraically equal and numerically within bf16 store noise.
* **The merged path IS the training path -- verified bit-identical, not merely argued.**
  RL runs with `vlm.merge_adapter_dir` build their base by
  `PeftModel.from_pretrained(base, sft_adapter).merge_and_unload()` on the bf16 model at
  load time (`vlm_worker.load_model`). Reproducing that on GPU (`device_map="cuda"`, as
  training does) and comparing against the published CPU merge gives `torch.equal` **True**
  and max |diff| **0.0** on `layers.0.{q_proj,o_proj,down_proj}` and `layers.15.q_proj` --
  so device does not perturb the rounding, and the published repo is the base the policy
  trained against. Stacking the two LoRAs onto the RAW base is a different base by exactly
  the rounding above, and **that is the path our own sample400 numbers came from**. A
  third-party running merged-repo + RL adapter is therefore closer to training conditions
  than our reported numbers are. If a reproduction disagrees slightly, check this first.

To settle it empirically rather than by argument, run the same episode subset both ways and
compare paired outcomes; nobody has done that yet.

**FOR FUTURE CORRECTION.** The eval harness should load the merged repo as the base and
apply the RL adapter directly, instead of rank-stacking two LoRAs over the raw base. The
stacking exists only because the RL run merges its base in memory and saves a delta-only
adapter; it buys nothing and costs faithfulness. This is the concrete form of the
long-standing objection to the two-adapter design.

## Uploading

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
```

Before uploading, confirm the local directory identifies itself: `trainer_state.json`'s
`global_step` should match the checkpoint you think you are publishing. That one check is
what lets a reader six months later map the repo back to a run directory.

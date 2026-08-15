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

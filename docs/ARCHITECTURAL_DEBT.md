# Architectural debt in the turn-vector → flow head stack

Status: **diagnosis and a possible plan.** Nothing here is scheduled. `run_code_v4_mlp_warm`
is deliberately being left to run to completion with all of this in place; the decision was
that a restart is not worth 23 h of GPU on an unmeasured hunch.

Everything below was read out of the code and the shipped `turn_vector_head_config.json`,
not inferred from memory. Line references are to `src/longnav/utils/`.

---

## 0. What the conditioning path actually is

```
Qwen3-VL-2B                       text hidden 2048
  │
  │  find_turn_spans(ACTION_PREFIX, ACTION_POSTFIX, shift_left=True)
  ▼
ONE TOKEN per turn                the `**` that opens the assistant turn
  │                               (pool over a width-1 span)
  ▼
TurnVectorHead                    LayerNorm → 2048→1024→1024→1024
  │
  ▼
context, 1024                     `fm_context_dim`
  │
  ├─────────────► CodePolicyHead.logits  →  p(c | h)      1600-way
  │
  ▼
CodeContextMixer                  reshape to 8 tokens × 128
  ├ tokens 0-1 ← emb_xy(c_xy)
  ├ tokens 2-3 ← emb_theta(c_theta)
  └ tokens 4-7 ← r(h)             ONLY 4×128 = 512
  │
  ▼
FlowActionDecoder                 d_model 128, n_layers 4, n_context_tokens 8
```

Shipped values (`run_code_v4_mlp_warm/checkpoint-*/turn_vector_head_config.json`):

| field | value |
|---|---|
| `model_id` | `Qwen/Qwen3-VL-2B-Instruct` |
| `pool_mode` | `mean` |
| `shift_left` | `true` |
| `head_hidden_dims` | `[1024, 1024]` |
| `standardize_head_inputs` | `false` |
| `fm_context_dim` | `1024` |
| `fm_decoder_kwargs.d_model` / `n_context_tokens` | `128` / `8` |
| `fm_code.tok_xy` / `tok_theta` | `2` / `2` |
| `fm_code.r_hidden` | `1024` |

---

## 1. `fm_context_dim = 1024` is `8 × 128`, not a representational choice

`FlowActionDecoder.__init__` (`flow_matching_head.py:678`) refuses anything else:

```python
want = int(n_context_tokens) * d_model
if context_dim != want:
    raise ValueError(
        "this head has no context projection: the readout MLP must emit exactly "
        f"n_context_tokens * d_model ({n_context_tokens} * {d_model} = {want}), "
        f"got context_dim={context_dim}")
```

The decoder has no input projection, so the readout MLP is forced to emit exactly the
decoder's flattened token buffer. 1024 is that buffer's size and nothing else.

`context_dim: int = 1024` at `flow_matching_head.py:1218` is a **default that no launch
script has ever overridden** — the same category of unexamined inheritance as the r=64 LoRA
default that turned out to confound every pre-2026-08-04 run.

The consequence that matters now: when the code head was added, it inherited that width as
its input. `CodePolicyHead` reads the 1024-d context, so **`p(c|h)` is being predicted from
a vector whose width was chosen to match a transformer's token layout.** There is no reason
for the discrete head to live in the decoder's coordinate system.

## 2. The 2048 → 1024 squeeze happens at the readout's first layer

`head_hidden_dims = [1024, 1024]` means the readout is `2048 → 1024 → 1024 → 1024`. The
backbone's width is halved immediately and never recovered, so all three readout layers
work in the decoder's coordinate system rather than the backbone's.

## 3. The mean pool is inert

With `ACTION_PREFIX`/`ACTION_POSTFIX` and `shift_left=True`, the pooled span is a **single
token** — the `**` opening each assistant turn (`vector_sft.py:38-51`, and the
`find_turn_spans` docstring at `turn_vectors.py:144-150` says the same). At width 1,
`mean`, `last` and `attn` are all the identity (`attn`'s softmax over one element is 1).

`pool_mode = 'mean'` therefore reads as a design decision and does nothing. Harmless, but
it is a knob that will mislead the next person who tries to tune it — and it means the
projection in §2 is the *only* place capacity is lost between backbone and code head.

## 4. Adding codes silently halved `r(h)`'s bandwidth

`CodeContextMixer` (`code_conditioned_head.py:77`) computes
`n_reserved = n_tokens - tok_xy - tok_theta = 8 - 2 - 2 = 4`, and `r` emits
`n_reserved * d_model = 512`.

Before codes, `h` filled all 8 context tokens (1024) of decoder conditioning. It now fills
4 (512). **The continuous conditioning was cut by 2× when the discrete head landed, with no
compensating widening**, and nothing in the config surfaces this — `r_hidden = 1024` refers
to `r`'s *hidden* layer, not its output.

Whether 512 is enough is unmeasured. The mixer's own comment argues the opposite direction
(if `r(h)`'s block grows much larger than the code block, the decoder routes around `c` and
the codes become decorative — the latent programme's collapse), so this is a real trade-off
and not simply a bug. But the current split was inherited from `n_context_tokens = 8`, not
chosen against that trade-off.

## 5. `standardize_head_inputs = False` on a raw residual stream

`TurnVectorHead.fit_input_stats`' own docstring warns that the residual stream carries
outlier dimensions orders of magnitude larger than the rest, and that a freshly initialised
head on a frozen-ish backbone is badly conditioned without centring/scaling. The shipped
config leaves it off. Relevant to any head moved upstream of the readout (§6), which would
read that raw 2048 directly and needs a LayerNorm on its input.

---

## Possible plan

Ordered by cost. **(A) is cheap and isolated; (B) is surgery.**

### A. Predict `c` from the pooled 2048, not the projected 1024

`CodePolicyHead` (and only it) reads `TurnVectorHead.pooled_context(...)` — with a
LayerNorm on its input, per §5 — instead of the projected context. The flow path is
untouched.

Two independent arguments:

* **Width.** The code head stops inheriting a width that exists to satisfy `8 × 128`.
* **Gradient factorisation — the stronger one.** Today the code CE flows back through the
  readout MLP, which is *the same MLP that produces the flow decoder's conditioning*. The
  code objective and the flow objective compete to shape the same 1024 coordinates. Moved
  upstream, code CE touches only the backbone LoRA and the code head, and `c` becomes a
  *description* of `h` rather than something that deforms `h`'s projection. `p(c|h)` names a
  mode the frozen tokenizer already defined; it has no business renegotiating the flow's
  conditioning geometry.

Honest counter-argument: this also removes a path by which the code objective could improve
the shared representation. The backbone still receives code gradients through the pooled
vector, which is the larger lever, so this is judged acceptable — but it is a real change in
what the code loss shapes, not merely a width change.

**MEASURED, 2026-08-28 — the width argument is dead.** `probe_code_from_h.py` on
`checkpoint-800`, ObjectNav validation, 32,312 turns from 695 episodes, split BY ROW,
LayerNorm+linear on both sides of the projection from the same forward pass:

| probe | val CE (1600-way) |
|---|---|
| marginal (bias only) | 5.3169 |
| in-run code head | 4.5780 |
| linear on ctx (1024) | **4.2539** |
| MLP on ctx (1024) | 4.2866 |
| LN+linear on ctx (1024) | 4.3038 |
| LN+linear on **pooled (2048)** | **4.3114** |

The projection costs **−0.008 nats** — the 2048 is fractionally *worse*, i.e. the two are
identical within noise. **The 2048 → 1024 readout is not discarding code information**, so
§1/§2 give no reason to move the code head, and no reason to restart the run.

Caveat on scope: this measures the projection *as currently trained*, with the code head
already reading its output and therefore already pressured to preserve code-relevant
directions. A readout trained by flow matching alone might discard more. It does not
generalise to a from-scratch run with a different factorisation.

What survives: the **gradient-factorisation** argument above, which was never about
information content — it is about which objective shapes the shared MLP. That remains
untested, and is not worth a restart on its own.

The same probe answers the original question that motivated it. Against the in-run head's
4.578, the best probe reaches 4.254 — so `h` carries **at least 0.32 nats** more about `c`
than the live head extracts, and the ObjectNav code CE is *not* sitting at `H(c|h)`. Read
the direction carefully: the probe saw 32k turns and had no backbone, so beating the live
head is a genuine lower bound, whereas losing would have said nothing. Note also that an
earlier version of this probe split train/val by TURN rather than by row and reported a
0.73-nat gap; episode leakage accounted for more than half of it.

### B. Give `r(h)` its 8 tokens back

Raise `n_context_tokens` 8 → 12 so the codes take 4 tokens and `r(h)` keeps 8 (1024).

This changes `context_dim` to 1536, which changes the readout's output width, which
**breaks warm-start from both the c-only prototype (`code_flow_40x40_cycle2`) and from v3**.
The decoder's positional/token layout changes with it. This is not a config tweak; it is a
retrain from a colder start.

Do not attempt without first measuring whether 512 is actually binding — e.g. by ablating
`tok_xy`/`tok_theta` down to 1 each (giving `r(h)` 6 tokens = 768) on a short run, which
needs no width change anywhere and no new warm-start.

### C. Cosmetic, do opportunistically

* Make `pool_mode` raise or warn when the affixes give a width-1 span, so the knob stops
  advertising a choice it cannot make (§3).
* Record `r`'s **output** width (`n_reserved * d_model`) in the head config, so §4 is
  visible in the checkpoint rather than derivable only by arithmetic.

---

## What is NOT claimed here

None of this is established as the cause of the slow ObjectNav code-accuracy climb. The
competing explanations are still open and some are more likely:

* `H(c|h)` on ObjectNav is genuinely high — search is multi-modal in a way point-to-point
  navigation is not, which is the obvious reading of the PointNav/ObjectNav CE split
  (2.73 vs 4.58 against a corpus marginal of 4.838).
* The single-token readout (§0) funnels a long context through one position. For PointNav
  the decisive evidence (the goal vector) is local and always present; for ObjectNav it is
  spread across the whole history. Untested.
* The eval condition itself was mismatched: ObjectNav in this project is the **nopose**
  format and `run_code_v4_mlp_warm` trains `objectnav_nopose`, but the first qualitative
  rollout ran with pose injection ON, feeding markers the model never saw on an ObjectNav
  conversation. Reruns use `--no-pose-injection`
  (`habitat_physical_nav/scripts/run_code_modes_viz.sh`).
* **The eval cadence was mismatched too, and this one changed the conclusion.** Every
  `live*` rollout ran at `--gap 5 --dt 0.05` against a corpus stride of 10 at dt 0.04; at
  the trained cadence the checkpoint's motion statistics match the expert. Do not read any
  behavioural claim from a rollout whose `episodes.partial.jsonl` shows `ticks/steps != 10`.
  See `docs/LIVE_ROLLOUT_CADENCE.md`.

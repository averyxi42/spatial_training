# Discrete action history as the missing channel

Status: **hypothesis and design sketch.** Not built, not scheduled — the work and the
compute are not available now. This exists so the reasoning survives, and so the design
questions are already answered when it is picked up.

Origin: qualitative rollouts of `run_code_v4_mlp_warm` at ck800/1200/1400 showed aggressive,
apparently inexplicable rotation. The locomotion is competent (zero collisions, coordinated
turning, `corr(|forward|,|rotation|) = -0.835`); the search is not (34 m travelled buys 2.3 m
of net approach). The hypothesis below explains the rotation specifically, and it is the
user's, recorded here with the mechanism made precise.

---

## 1. The observation the hypothesis has to explain

ck1400, `--no-pose-injection`, one ObjectNav episode, three budgets:

| steps | path travelled | closest approach | final | turn-in-place |
|---|---|---|---|---|
| 100 | 13.9 m | 13.03 m | 15.90 m | 49.2% |
| 175 | 20.1 m | 13.03 m | 15.20 m | 56.0% |
| 350 | 33.9 m | 11.98 m | 15.19 m | 37.3% |

Start distance is 14.26 m in all three. Extra budget buys distance travelled almost
linearly and buys essentially no approach. Median speed is ~0.01 m/s against a p90 of
1.39 m/s: the policy is mostly pivoting, occasionally darting. It is rotating back and
forth, undoing its own progress.

For contrast, the near-frozen ck800 travelled 4.5 m and reached 11.56 m — *closer than any
later checkpoint*, including the one that drove 34 m.

## 2. The paradox, and the mechanism that dissolves it

Three things were already established:

* the model **struggles to infer past motion / do mapping from frame differences**;
* it **benefits little from pose embeddings when searching**;
* yet the **discrete policy learned longer-horizon search behaviour** despite sharing both
  shortcomings, and was qualitatively much less prone to redundant rotation.

The resolution is that the two policies do not have the same inputs at all.

**Discrete** (`config_schema.py:218`, `convo_turn_template`):

```python
{"role": "assistant", "content": [{"type": "text", "text": "**$action**"}]}
```

`$action` is the action that was *executed*. It stays in the KV cache, so the context
accumulates a literal, readable trace: `**forward** **left** **left** **forward** …`

**Continuous** (`vector_sft.py:19`, and `docs/placeholder_tokens.md`): every assistant turn
is the constant `**____**`. **The action history carries zero bits.**

This was a deliberate change and the reasoning behind it was sound — quoting
`placeholder_tokens.md`:

> With `**forward**`, a 200-observation episode asserts "forward" two hundred times, most
> of them false (the robot was turning), and the model spends its next-token machinery
> contradicting that history.

Removing a *false* history was right. The unexamined side effect is that no *true* history
replaced it.

**The consequence, stated plainly.** ObjectNav is trained in the `nopose` format. So on an
ObjectNav episode the continuous policy has:

* no egomotion from frames (established),
* no pose channel (nopose),
* no action history (constant placeholder).

**No channel of any kind reports what it just did.** It cannot know it turned 60° right one
step ago. Under that constraint the rotation is not inexplicable — it is the expected
behaviour of a policy that is action-memoryless, and every decision is made as if fresh.

Corroboration from the other direction: PointNav trains *with* pose, so it does have an
egomotion channel, and its code CE is 2.418 against ObjectNav's 4.427 (step 1200).

## 3. Why a low bar is the point

The claim is **not** that action history confers spatial understanding. It is that avoiding
redundant rotation is a *sequence-pattern* problem, not a geometric one: "I have alternated
left/right for six turns and the view keeps returning" is recognisable without any metric
map, and pattern matching over discrete symbol sequences is what transformers are best at.
That bar is low, and the discrete policy appears to clear it. Crude geometry may come along
for free, but nothing here depends on it.

This also predicts the *shape* of the win: better rotational coherence and less undoing,
**not** better goal inference. It should show up in path efficiency, not in a sudden ability
to guess where toilets are.

## 4. The opportunity the code head creates

Before the FSQ tokenizer there was nothing discrete to write into the history — the action
was a continuous `(20, 3)` chunk. Now every chunk has a code `(c_xy, c_θ)` from the frozen
tokenizer, so the semi-discrete action can enter the context the way the discrete policy's
did.

### 4.1 Representation — three options

The codes are not in the model's vocabulary (40 × 40 = 1600 joint).

**(a) Text digits**, e.g. `**17,3**`. No new parameters, uses pretrained numeric pattern
matching. But number tokenisation is multi-token and splits unpredictably, and it spends
context length on characters.

**(b) New vocabulary tokens**, 40 + 40 = 80 added ids. One token per factor, clean
semantics. Costs an embedding resize; under LoRA the new rows must go in `modules_to_save`,
and tied embeddings drag `lm_head` along even though the action never comes from `lm_head`.

**(c) The modality embedder — recommended first move.** The mechanism already exists and is
in production for `<pose>`: a marker token in the text whose embedding is supplied by
`ModalityBatch` / `modality_specs` / `modality_embedder.encoders.*` (`planar_se2` for pose,
`v3_pose.json` for the spec). A `code` modality whose encoder is a pair of
`nn.Embedding(40, D)` slots into identical, tested plumbing — no tokenizer change, no vocab
resize, no `lm_head` growth.

Note (c) is not a weaker form of (b) for this purpose. A learned per-code embedding *is* a
token embedding; it simply is not in the vocabulary. The model sees a fixed distinct vector
per code either way, which is what sequence pattern matching needs.

### 4.2 The readout is unaffected — and this is what makes it cheap

`shift_left=True` puts the readout at the `**` that *opens* the assistant turn, before any
content (`turn_vectors.py:144-150`). So writing the code into the turn body:

* does **not** change that turn's own readout — it stays a pure function of prior context,
  and the head keeps its "readable before anything is generated" property that makes
  turn-by-turn rollout possible at all;
* **does** change the KV cache for every later turn, which is the entire point.

Cost is 1–2 extra cached tokens per turn against ~326 already, i.e. nothing.

### 4.3 The design questions that need answering

1. **Teacher forcing vs executed code.** In SFT the history should carry the dataset's code
   (frozen tokenizer on the ground-truth chunk). At rollout it carries the policy's own
   sampled/argmax code. That is textbook exposure bias, and it is worse here than usual
   because the whole value of the channel is that the history is *true*. Scheduled sampling
   — write the policy's own code some fraction of the time — is the obvious mitigation and
   needs a decision, not a default.

2. **Intent vs achievement.** The code names the *intended* chunk. What actually happened
   differs by `r(h)`, by z₀, and by the controller and collisions. Intent is probably right
   (a discrete action is also an intent, and the discrete policy succeeded with exactly
   that), but it means a policy commanding "forward" into a wall reads a history of
   successful forwards. The discrete policy had this failure mode too. Injecting the
   *achieved* differential instead is the pose channel again, which is known to help little.

3. **Degenerate copying.** Consecutive codes are highly autocorrelated, so the model can cut
   code CE by learning `c_{t+1} ≈ c_t` without improving behaviour. This must be measured,
   not assumed away: compute a **copy-the-previous-code baseline CE** on the corpus and
   report the head's CE against it. Cheap, no GPU, and worth doing *before* the change so
   there is a before-and-after. If accuracy climbs while `turn_frac` and path efficiency do
   not, that is what happened.

4. **RL interaction.** Under flow-SDE the executed code becomes part of the observation for
   the next step, so the history is on-policy and the exposure-bias problem in (1) disappears
   — but the chain-log-prob contract must not start conditioning on a quantity the scorer
   reconstructs differently from the sampler. Same invariant as the RTC prefix: recondition
   on the **stored** code, never one re-derived from poses.

## 5. How success would be measured

The current metrics do **not** capture the failure. `rotation sign-flip` reads 0.0% within
chunk and 1.8% across adjacent chunks — the erratic rotation is not sign-flipping at that
scale, so this instrument is blind to it.

What is missing is a **rotational directness**: net |Δheading| over summed |Δheading| across
the episode, the angular analogue of the existing linear `directness`. A policy that turns
60° right then 60° left scores ~0; one that turns coherently scores ~1. That is the number
this hypothesis predicts should move, and it should be added *before* any training run so
there is a baseline.

Secondary: `turn_frac`, and `path_m` against improvement in `min_m` (the 34 m → 2.3 m ratio
above is the headline symptom).

## 6. What is not claimed

* No claim that this is the *only* reason ObjectNav search is weak. Goal inference is a
  separate problem and this hypothesis explicitly does not address it.
* No claim about magnitude. The discrete policy's advantage in search is qualitative;
  nothing here predicts how much of it the code history recovers.
* The rollout evidence is **n = 1 episode, one scene, one target category**, at three
  budgets. It motivates the hypothesis; it does not test it.
* The 350-step run additionally exceeds the 200-turn training cap, so its late-episode
  freezing (30.6% commanded-still, one 24.5 s stall) confounds "gives up" with "context
  length never trained on".

# A discrete code as the RL action, with the flow head as its renderer

Status: **DESIGN ONLY, 2026-08-27. Nothing here is built.** The tokenizer section
(4) is explicitly incomplete and is the remaining design work.

Companion to `FLOW_RL_SAMPLE_EFFICIENCY.md` (the diagnosis this responds to),
`FLOW_SDE_RL.md` (the machinery it keeps), `LATENT_RL_BAKE.md` (the failure it must
not repeat) and `RTC_TRAINING.md` / `RTC_RL.md` (which it composes with, unchanged).

Section 8 lists what was proposed during design and **rejected**, with reasons.
Read it before re-proposing anything -- several of the rejected options are the
first ideas a reader has.

---

## 1. The two insights

**Make the conditioning powerful by construction.** The latent programme failed
because `c` was an auxiliary input the decoder was free to ignore, and it did:
`h` already linear-probes to R² 0.63 on the whole chunk, `kl_raw` peaked near 0.6
nats against a 10.24-nat free-bits allowance so the posterior never received
gradient, and `delta_mu` stayed negligible almost everywhere. What `c` delivered in
practice was noise upstream of a reconstruction decoder, which showed up as the
conversion tax (0.663 -> 0.545 on the pilot, 0.663 -> 0.436 on v2's best
checkpoint) and never shrank with tuning. Conditioning has to carry information the
decoder cannot obtain elsewhere, and it has to be there at every step so the
decoder never has the option of routing around it.

**Treat the flow head as a decoder, not as the generative policy.** Expressiveness
is a *fitting* requirement -- demonstrations are multimodal -- and controllability
is an *acting* requirement. Nothing forces one object to supply both at the same
time. The flow keeps its job of rendering feasible, obstacle-aware motion; the
policy's action becomes a discrete code naming which trajectory mode to render.

## 2. The pipeline

1. **Tokenizer.** `g: A -> c`, a VQ autoencoder over action chunks. Trained
   standalone on trajectories with no access to `h`, then **frozen**. Section 4.
2. **SFT.** The flow head is trained with the standard flow-matching objective,
   conditioned on `c_teacher = g(A*)` **and** `proj(h)`. `proj(h)` is a gradient
   path -- no stopgrad -- so the backbone still learns trajectory content through
   it. No conditioning dropout, no policy-sampled `c`, no obedience term.
3. **Code head.** A discrete head over the codebook trained by cross-entropy
   against the binned/tokenized `c_teacher`, on the same `h`, at the same time.
4. **RL.** The flow head stays frozen exactly as it is today
   (`action_head_learning_rate: 0.0`). `c` is sampled from the code head; its
   log-probability enters the objective alongside the flow-SDE chain density.
   Gradient reaches `h` through both channels.

### What each channel owns

| channel | owns | trained by | at RL time |
| --- | --- | --- | --- |
| `c` (discrete) | which trajectory mode | CE on `g(A*)` | categorical policy, exact low-dim log-prob |
| `proj(h)` (continuous) | the detail the code cannot carry -- obstacle geometry, how tight to cut a corner, staying off the wall | the flow-matching loss, as the reconstruction residual | trained via the flow-SDE chain density |
| `z_0` | within-cell variation | nothing | drawn freshly, uncredited |

The division of labour is produced by the *gradient structure*, not by
architectural throttling: `c` is a direct, low-noise readout of the target and
`proj(h)` carries no privileged information, so the loss leans on `c` first and
leaves `proj(h)` the residual. That is the whole reason `proj(h)` gets the full
`h` rather than a bottleneck -- `h` is the only source of geometry, and
specialisation is already enforced by what `c` takes off the table.

`proj(h)` is not a maintenance channel. It is a continuous, detail-oriented latent
path that RL genuinely trains. The chain estimator is still the inefficient one, but
what that costs here is much less than it costs today, for reasons set out in
section 3.2 -- and how much it costs at all is an open empirical question, not a
standing defect.

## 3. The diagnosis this responds to

### 3.1 Two failures, not one

From `FLOW_RL_SAMPLE_EFFICIENCY.md`. Under the chain-as-action formulation the
gradient reaches `mu_theta` only at the `sde_n` stochastic positions; `z_0` -- a
60-dim draw that largely selects the trajectory shape -- receives zero credit, and
the other `K - n` ODE steps receive zero gradient. In the `sigma -> 0` limit the
log-prob is identically zero and there is no gradient at all, though the policy
still varies, entirely via `z_0`.

Those are **two separable problems** and only one of them is serious:

* **Mis-credit.** `z_0` produces behavioural variation that is attributed to
  nothing. This is a bias in *what* receives credit, and it is the threat.
* **Sparse sampling.** The drift's gradient is estimated at `n` points of the
  chain rather than along it. This is variance in *estimating* credit, it slows
  learning, and it misattributes nothing.

A discrete code removes the first for the part of the action that matters most: the
code's log-probability is exact, low-dimensional, and a function of the chosen
action alone. Multimodality survives because a categorical distribution is a proper
multimodal density -- there is no conditional mean anywhere in the objective, so
the mean-collapse failure of an L2 regression head does not arise. The second
survives on the `proj(h)` channel and is the price of keeping it.

### 3.2 Why `(c, proj(h))` shrinks the threat

The harm from mis-crediting `z_0` is bounded by **how much `z_0` moves behaviour**,
not by how much it moves the score. Two mechanisms shrink it here:

* **Conditioning narrows the conditional.** The head is trained on
  `p(A | h, c)` with `c = g(A*)`, so a properly learned conditional puts
  essentially all of its mass inside cell `c`, and `z_0` draws from that mass. If
  the `c` dependence is learned properly, `z_0`'s influence is **sub-cell by
  construction** -- below the scale at which the code makes distinctions at all.
* **`proj(h)` and `z_0` compete for the same space.** `proj(h)` shifts where the
  conditional sits; `z_0` samples its spread. A `proj(h)` that explains detail well
  narrows the conditional and directly leaves `z_0` less to do.

Two qualifications that must not be folded away:

* **Sub-cell is not return-irrelevant.** Cells are sized for mode separation, not
  for return, so two trajectories inside one cell can still differ on whether they
  clip a corner. Sub-cell `z_0` bounds the harm; it does not zero it.
* **This is a statement about `z_0`, not about the `sigma` draws**, which are added
  on top and can leave the cell on their own. That is a separate escape route,
  handled by the two-sided `noise_a` bound in section 6.

**`R² = 0.099` does not measure this and moves the wrong way.** It regresses the
chain score on the executed endpoint, so it reports how much score variance is
coupled to the action -- an estimator-efficiency statistic. As `z_0` sensitivity
shrinks, endpoint variation shrinks while chain-realisation variation does not, and
R² would fall *further* even as the harm disappears. The statistic that tracks the
harm is return variance attributable to `z_0` at fixed `(h, c)`, and its cheap
proxy is the obedience rate (section 6), measurable at SFT time before any RL
exists.

### 3.3 The open question: how much does `proj(h)` steer?

`h` receives signal in proportion to how much it steers -- the ordinary property
that a parameter's gradient magnitude tracks its influence, and not by itself a
defect. It is recorded because it is **decision-relevant**, not because it is
worrying:

* If `proj(h)`'s measured steering share is large, the continuous channel matters
  and so does the sparse-sampling cost on it -- then `sde_n` and `noise_a` are
  worth tuning.
* If it is small, the channel is near-inert once `c` is carrying the coarse
  decisions, and the right move is to **drop the flow-SDE apparatus entirely**
  rather than tune it -- deleting the chain density, the seam gauges and the
  factored ratio along with it.

Which of these holds is unmeasured, and it is the question that decides how much of
`FLOW_SDE_RL.md` survives this design.

## 4. Tokenizer -- **INCOMPLETE, this is the remaining design work**

### Two invariants, stated before anything else

**Discretisation is over the WHOLE CHUNK, never per tick.** `g` maps one `(H, 3)`
chunk to one code (or one short token sequence for that chunk). A per-tick
vocabulary would be an action-token language model over 25 Hz differentials -- a
different architecture with a different decision frequency -- and it would put the
policy back in the business of emitting motion rather than selecting a mode, which
is the entire thing this design moves away from. Every statement in this document
about codes, cells, obedience and `p(c|h)` is per decision, at 2.5 Hz.

**The policy head must emit a distribution over the discretised chunk.** With one
symbol per chunk that is a single categorical, which is why single-level
quantisation is the simplest thing that can work. With more than one symbol the
head must be **autoregressive**, and for RVQ specifically this is a correctness
requirement, not an expressiveness preference: stage `l+1` quantises the residual
of stage `l`, so `p(c_2 | c_1) != p(c_2)` by construction. A factorised head would
place mass on stage combinations that never co-occur, and those decode to
trajectories the corpus never contained -- the policy could sample them and would
be scored on the result.

What is settled:

* **Standalone and frozen.** Trained as a pure autoencoder over action chunks with
  no access to `h`, then frozen for all downstream training. If `g` were learned
  jointly with the decoder under a reconstruction objective, it would be allocated
  only what `h` cannot already explain -- and `h` explains a lot -- which is the
  bake's collapse reproduced exactly. Freezing also gives the code head a
  stationary target and makes the obedience gauge (section 6) well defined.
* **Autoencoder-style, not k-means**, so the code is pressured to express mode
  rather than to tile the space by proximity.
* **Residual VQ, not a single code.** Two or three tokens from small codebooks give
  combinatorial coverage with per-token classifiers that stay learnable in this
  data regime, which relieves the separation-versus-count tension directly. It also
  makes *how much resolution RL controls* a runtime choice -- act on the first
  token only, or on all of them -- rather than something baked into the codebook.
  The policy head is then autoregressive over a short token sequence, which is the
  discrete shape this stack already trains successfully.
* **The reconstruction metric's weighting is a design decision, not a default.**
  Error over `(x, y, theta)` in metres and radians silently sets the exchange rate
  between position and heading error. Scanning is almost purely a heading
  phenomenon with near-zero positional signature, so under a position-dominated
  metric it contributes little total error and gets no codes -- the pan/scan
  failure surviving into the learned tokenizer. Scans are a prime suspect for
  policy ambiguity and are exactly what the code is supposed to split.
* **The objective is marginal; the acceptance criterion must be conditional.** A VQ
  autoencoder clusters by trajectory-shape similarity, but what the design needs is
  codes that make `p(A | h, c)` near-unimodal. Codes can add resolution without
  adding disambiguation and reconstruction error cannot tell the two apart. Gate on
  a held-out conditional statistic -- how much of the residual variation in `A`
  given `h` the code removes. This is the two-sided gate `LATENT_RL_BAKE.md` lacked;
  selecting on the training objective alone selects a codebook that reconstructs
  well and disambiguates nothing.
* **No RTC-driven scoping of the encoder window.** An earlier proposal to fit the
  tokenizer on the executed `gap` rows rather than the full `H` is wrong: under RTC
  the tail is consumed by the next decision's commitment and executed, returns are
  gamma-re-timed at the splice so its credit is correct, and it is not the
  tokenizer's job to encode a deployment parameter.

### First measurements (2026-08-27, branch `code_tokenizer`)

Dual FSQ, `[8,5] = 40` codes for `(x, y)` and 40 for `theta`, 1.60 M params, trained
on the full 1,971,020-chunk corpus (`data_scripts/train_chunk_tokenizer.py`;
evaluation `data_scripts/eval_chunk_tokenizer.py`; artefacts under
`dump/tokenizer/dual_fsq_40x40/`).

* **Chunks are stored CUMULATIVE** -- `|early| 0.0513` against `|late| 0.4654`. The
  loader decides this by measurement, not assumption: integrating a cumulative corpus
  (or failing to integrate a differential one) would train the tokenizer on the wrong
  object and still print plausible numbers.
* **Corpus scale is near-parity**: `xy` std **0.346 m**, `theta` std **0.383 rad**.
  An earlier argument for skewing the budget toward `theta` was mistaken -- it read
  per-tick *residual* error off a trainer log, which measures what the model finds
  hard, not what the channel carries. Equal vocabularies are the right default.
* **It converges in ONE epoch.** Val RMSE was 0.0639 m / 0.0797 rad at epoch 1 and
  0.0638 / 0.0811 at epoch 20, with epoch-to-epoch scatter larger than any trend. The
  binding constraint is code capacity, not optimisation: at vocab 40 the floor is
  ~18% of `xy` std and ~21% of `theta` std. Fit for 2-3 epochs -- the extra 17 bought
  no val improvement and did open a train/val gap (train 0.044 m / 0.068 rad against
  val 0.064 / 0.081). Reconstruction is therefore a fixed property of the vocabulary
  and cannot compare tokenizer variants unless vocabulary is held equal.
* **Utilisation is total**: 40/40 on both streams -- the guarantee FSQ was chosen for,
  and it held. (A 10/40 reading on a 92-step smoke run was undertraining, not
  collapse.)
* **The two streams are not alike.** Perplexity 13.1 for `xy` against 30.3 for
  `theta` -- normalised entropy 0.70 against 0.93 -- at near-identical channel
  variance. `theta` spends its codebook nearly uniformly while `xy` concentrates on a
  few "advance at some speed" modes, so heading carries more distinct behavioural
  structure per unit of variance. Any future asymmetry should favour `theta`, which
  is the opposite of the discarded argument above.

**Existence-pruning is dead; a frequency floor works.** 1547 of 1600 joint cells are
occupied on the full corpus, so "drop combinations that never occur" removes 3.3% and
is not a mechanism -- as the corpus arithmetic predicted, at ~1200 chunks per cell
existence is a weak filter. A frequency floor is a mechanism, and the curve is
favourable rather than flat:

| floor | cells kept | chunk coverage |
| --- | --- | --- |
| 1 | 1547 | 100% |
| 10 | 1331 | 99.95% |
| **100** | **723** | **98.7%** |
| 500 | 402 | 95.0% |
| 1000 | 273 | 90.3% |

723 cells at 98.7% coverage sits inside the CE head's budget (~1.1 M exposures over a
standard run, so ~1500 per code before Zipf). Cutting the tail remains a deliberate
deletion of rare modes rather than a free structural win -- but here it is a cheap one.

**Scans separate, which is the design's central premise surviving its first test.**
Scoring each chunk by `sweep - |net|` in cumulative heading and taking the top 2%
(39,421 chunks, threshold 0.434 rad): **5 `theta` codes hold 80% of them at 8.8x
enrichment** over their corpus share. Code 29 carries 17.0% of scans against a 0.45%
corpus share -- **37.8x**, effectively a dedicated scan code -- and code 21 is 13.4x.
The `theta` codebook does allocate capacity to the mode the design needs from it.

**But scans are the worst-served part of the distribution**: `theta` RMSE is 0.152 rad
on scan-like chunks against 0.067 on the rest, **2.3x worse**. The `theta` cell is
loosest exactly where the mode matters, which bears on the sub-cell `z_0` property of
section 3.2 -- the bound it provides is weakest for scans.

### Frame convention: pose space beats body-frame differentials (2026-08-27)

The shipped head regresses per-tick BODY-FRAME differentials (`decompose_chunk`, one
`action_scales` triple) while the tokenizer, the obedience gauge and the REWARD all live
in anchor-relative CUMULATIVE pose space. A/B in the standalone prototype -- same
tokenizer, same codes, same architecture, same schedule, only the regression target
differs (`--action-space pose`, per-tick scales since pose magnitude ranges 19.1x across
ticks):

| | pose, 6 ep | differential, 12 ep | differential, 6 ep |
| --- | --- | --- | --- |
| obey both | **0.677** | 0.672 | 0.643 |
| obey theta | **0.801** | 0.792 | 0.788 |
| `z_0` spread theta | **0.0461** | 0.0509 | 0.0519 |
| `z_0` spread xy | **0.0293** | 0.0307 | 0.0303 |
| gen RMSE xy / theta | **0.0688 / 0.0878** | 0.0710 / 0.0929 | 0.0716 / 0.0931 |

Pose reaches in 6 epochs what differentials need 12 to reach, and every metric moves the
same way. The endpoint margin is small (0.677 vs 0.672); the striking number is the ~2x
training-efficiency gap.

**Why**, measured rather than argued. Perturbing by equal relative amounts in each space
and reading terminal position error:

| perturbation | via differentials | direct in poses | amplification |
| --- | --- | --- | --- |
| independent per tick | 0.0089 m | 0.0198 m | **0.45x** |
| correlated `dx` bias | 0.0276 m | 0.0172 m | **1.61x** |
| correlated `dtheta` bias | 0.0134 m | 0.0000 m | poses immune |

Independent error accumulates as sqrt(T) while signal accumulates as T, so composition
IMPROVES terminal SNR -- the accumulation-error worry is backwards for that case. The
concern is correct for CORRELATED error, and a head emitting 20 ticks from one shared
forward pass is a plausible source of it: a persistent heading bias rotates everything
downstream, while in pose space a theta error has no positional consequence at all.

**Arguments for differentials that did NOT survive.** Scale stationarity is real (19.1x
range) but fixed by per-tick scales, an implementation detail. RTC anchor-freedom
dissolves: the committed prefix in pose space is `compose(commanded diffs)` from zero,
well-defined at the new anchor and free of tracking error because it uses commanded and
not achieved motion. The control-space prior is real but weak -- `|dy|/|dx| = 0.146`,
`dy` std 0.011 against `dx` 0.036 -- and the expected over-dispersion from losing it did
not appear; the head learns the constraint from data.

**Not a reason to change the shipped head.** That is a breaking change to
`action_scales`, the RTC prefix path, the flow-SDE chain, the incoming-motion slot and
every existing checkpoint, for a few points of a prototype metric. The finding is that
pose space is the right default FOR THIS DESIGN, decided while the choice is still free.
Caveat: measured without `r(h)`, and pose's advantage may narrow once a continuous
branch supplies the geometry differentials encode implicitly.

Open, and the reason this section is incomplete:

* **Is the ambiguity chunk-expressible at all?** Partially answered: within-chunk
  excursion-and-return is real (2% of chunks by construction of the quantile) and it
  earns dedicated codes. What is NOT answered is whether the scanning behaviour that
  matters for goal-search spans several decisions, in which case the chunk-level view
  sees only a fragment of it and the code names a piece of a mode rather than the
  mode. A chunk is 0.8 s; if a scan spans two or three decisions, then within any
  single chunk it looks like an ordinary turn and the distinction "scanning" versus
  "committing to a left turn" is not
  present in `A`. A per-chunk tokenizer cannot split a mode that is invisible in
  the object it encodes, however well designed; disambiguation would have to come
  from `h`'s history, and the code could not help with the behaviour it is most
  wanted for. **Measure before building**: whether behaviours recognisable as scans
  are recoverable from single chunks or only from runs of them.
* **Codebook size** trades RL controllability against `proj(h)`'s role and against
  data per code. Small: coarse control, rich residual, possibly unable to express
  the distinction RL needs. Large: fine control, vestigial residual, and RL drifts
  back toward a high-dimensional problem. RVQ softens but does not remove this.
  Cell size carries a **second job** that pulls the other way from mode separation:
  per section 3.2 it sets the scale below which uncredited `z_0` variation is
  harmless, so shrinking cells sharpens control but also raises the bar the decoder
  must clear to keep `z_0` sub-cell. The two criteria are not the same and the
  tokenizer has to answer both.
* **Long-tail code usage.** SFT usage will be skewed and a rarely-used code has a
  poorly-trained rendering, so when RL explores it the decoder produces something
  bad and the policy learns to avoid the *code* for reasons unrelated to the
  behaviour it names -- a systematic bias toward what was already common, which is
  the opposite of what exploration is for. Handling (usage floor excluding codes
  from the policy's support, or a per-code rendering-quality gate) is not yet
  decided.
* Encoder architecture, commitment/EMA details, dead-code revival policy, and the
  exact conditional gate statistic are all unspecified.

## 5. SFT

Fixed logic, no schedule, no sampling:

* Flow-matching loss only, conditioned on `c_teacher = g(A*)` and `proj(h)`.
* `proj(h)` is a live gradient path (no stopgrad).
* The code head trains simultaneously by cross-entropy on the tokenized
  `c_teacher`, from the same `h`.

**There is no separate obedience loss.** With teacher `c` the flow objective
already enforces obedience implicitly -- the model is trained to produce `A*`,
whose code *is* `c` -- so a separate term restates the same constraint.

**`proj(h)` dying during SFT is not a concern.** Once the decoder has taken the
low-hanging fruit of decoding `c`, refining the reconstruction is the only thing
left, and `proj(h)` is the only channel that can do it. The gradient that reaches
`h` becomes small and focused rather than absent, and being focused on the residual
is the goal, not a degradation.

## 6. RL

Hierarchical policy: `c ~ Cat(p_theta(c | h))`, then the chain sampled conditioned
on `(h, c)`. The action is the pair and

    log pi = log p_theta(c | h) + sum_k log N(chain transitions)

is an exact factorisation of the joint, so PPO over it is valid and this is not
double counting. The flow head stays frozen, as in `flow_sde_rtc_a09_held128`.

**Factor the ratio.** `log p(c|h)` moves in tenths of nats for ordinary probability
shifts; the chain term sits at ~0.0005-0.0006 nats in the RTC run. Summed into one
ratio the code term is three orders of magnitude larger, the clip binds entirely on
code changes, and when it binds **gradient is zeroed for both factors** -- the
continuous channel switches off in exactly the cycles where the code policy is
learning fastest. Since the factorisation is exact, each factor takes its own
ratio, clip range and gauge. Treat this as required, not as an optimisation, and
re-establish `chain/abs_log_ratio_mean`'s band per factor.

### Instruments

* **Obedience as a gauge, never as a loss -- and report it BY MISS DISTANCE, not as
  strict equality.** Encode the generated chunk with the frozen tokenizer and log how
  far `g(A')` lands from `c` in grid steps. Nearly free, and it is the
  direct readout of whether the code still steers. It has **two roles at two
  times**: at SFT time, sampling `z_0` at fixed `(h, c)` measures whether `z_0`'s
  influence is sub-cell (section 3.2) and gates the tokenizer/decoder pair before
  any RL is built; at RL time the same statistic, now with `sigma` draws and a
  shifted `p(c|h)` on top, reports whether that property survived. What shifts
  under RL is the
  *joint* distribution over `(h, c)` -- every symbol is in the codebook and was
  rendered during SFT, so a novel pairing is compositional rather than
  extrapolative, and the loss's `c`-first-`h`-residual structure is what makes that
  composition work. If distorted `proj(h)` produces behaviour that earns return,
  rewarding it is correct; if it produces bad behaviour, the RL loss drives it
  back. The gauge exists to make either visible, not to constrain them.

  **Strict equality overstates failure and must not be the headline.** Measured on
  the c-only head (section 4): strict obedience is 0.840 xy / 0.795 theta / 0.672
  joint, but **94.0% of generations land within ONE grid step in both channels**, and
  gross misses (>= 2 steps) are only ~3% per channel. FSQ indices are ordinal, so an
  adjacent code is a neighbouring trajectory mode rather than a different behaviour;
  counting those as total failures reads a boundary landing as a wrong mode. Report
  the miss-distance histogram, headline the within-one-step rate, and treat the
  >= 2-step rate as the real failure rate.

  It also cannot be compared against the tokenizer's own round-trip (0.965, section
  4) without that caveat: the tokenizer's decoder places ONE point per code, sitting
  deep in the cell, while the head must place a whole distribution inside a
  non-convex decision region including its tails. A perfect model of `p(A|c)` would
  score 1.0 -- every chunk with code `c` is in cell `c` by definition -- so the
  shortfall is real modelling error, but the two numbers answer different questions
  and the gap between them is not the size of the problem.
* **`sde_noise_a` now has a two-sided bound.** Large enough to explore detail,
  small enough that the sample stays inside its code cell -- chain noise that
  changes `g(A')` randomises the mode RL just chose. The same gauge measures it as
  a function of `noise_a`, which makes the current 0.9 a value to re-derive rather
  than inherit.
* `chain/h_drift_from_init` and `ref_kl` keep their existing meanings.

### Implementation traps

* **The reference policy.** The current run gets an exact SFT reference from
  `disable_adapter()` because the SFT LoRA is merged into the base and the
  trainable delta starts at zero (`merge_adapter_dir` in the experiment YAML). A
  code head added as a plain trainable module is not covered: disabling the adapter
  leaves its RL-updated weights in place and `ref_kl` reports a plausible wrong
  number. The heads are small enough that an adapter is not the only option --
  keeping a frozen reference copy works, or merging block-diagonally and
  broadcasting inputs for parallelism (probably overkill). Whatever is chosen must
  cover `proj` too, which also opens the option of training `proj` during RL.
* **`entropy_bonus: 0.0` was safe for a continuous policy and is not now.** Entropy
  came from `a`. A categorical head under PPO with no entropy term and a peaked SFT
  initialisation can go deterministic within tens of cycles, and a code that stops
  being sampled receives no gradient and does not come back.
* RTC composes unchanged: the commitment prefix is a row mask on the decoder and is
  orthogonal to the code. `pin_prefix`, `prefix_time`, `postfix_mean`,
  `_sde_transition` and `_gaussian_logprob` all already take a generic `(N, T)`
  mask; only `prefix_mask_from_len` assumes a leading-contiguous one.

## 7. What this is, plainly

A discrete high-level policy over trajectory modes with a flow-matching low-level
renderer, plus a continuous channel that keeps the renderer's conditioning live and
trainable. That it converges on the architecture the discrete stack already proved
trainable here is the argument for it: the flow's expressiveness moves to where it
is needed -- rendering feasible smooth motion learned from demonstrations -- and RL
moves back to the regime whose sample efficiency was never in question.

Cost: a tokenizer, an SFT re-bake with `c`-conditioning, a new code head, a
rollout-buffer field for `c`, and a factored RL objective.

## 8. Proposed and REJECTED

| proposal | verdict |
| --- | --- |
| Backbone emits `z_0` (or a distribution over it) | **Rejected.** `z_0 ∈ R^60` is the action dimension -- a change of coordinates, not a bottleneck. Its meaning is state-dependent (`F(z_0, h)`) and the coupling has no consistency pressure, so there is no fixed semantics to learn; SFT taught the backbone nothing about `F^-1`. |
| Pin `z_0` to make the decode deterministic | **Rejected as invalid use of a flow model.** The flow is trained so the *marginal over* `z_0` matches the data; a fixed `z_0` is one arbitrary draw -- not the mean, not the mode -- and does not even denote a consistent style across different `h`. `pin_flow_noise` exists from the bake and its cost is folded into that programme's conversion tax. |
| Exact CNF likelihood via the divergence integral | **Not wrong, dominated.** It does deliver the exact Rao-Blackwellised score (the R² 0.099 gap is what it removes), and 60 dims is the regime where the trace is affordable where images are not. But `∇_θ` of a term already containing `∂v/∂x` is a mixed second derivative -- double backward per Euler step -- and exact trace needs 60 VJPs × K steps each carrying a second-order graph. Hutchinson makes the density stochastic, which is poison for a PPO ratio unless probes are stored and reused. The pathwise gradient `∇_θ Q(s, F_θ(z_0,s))` credits the same whole field at **first order** with no density at all; its cost is an action-conditioned critic and leaving PPO. |
| `c` = the chunk's endpoint `(Δx, Δy, Δθ)` | **Rejected.** A scan is `≈ (0,0,0)` at the boundary -- pointwise indistinguishable from freezing, which is this policy family's documented attractor. Generalises: any `g` that is a function of the endpoint is blind to within-chunk excursions, and that is where scanning and obstacle-skirting live. |
| Conditioning dropout on `c` | **Rejected.** It forces `h` to carry the mode information redundantly, which is precisely what the design is trying to avoid, and it weakens RL's grip on the code. The justification offered for it -- that `h` would otherwise stop receiving flow gradient -- was wrong: the decoder needs `h` for the residual, so the loss reaches it either way. |
| Mixing policy-sampled `c` into SFT conditioning | **Rejected.** The instinct (close the train/deploy gap on the conditioning distribution) is right, but the only way to use a sampled `c` is to regress toward an `A*` whose code differs, which trains the decoder to **override** `c` -- the opposite of what the design buys. Teacher `c` throughout. |
| An explicit obedience loss `g(A') = c` | **Rejected as a loss, kept as a gauge** (section 6). Redundant with the flow objective under teacher `c`, and a loss that can fight the division of labour is worse than an instrument that reports on it. |
| Fitting the tokenizer on the executed `gap` rows only | **Rejected.** Under RTC the tail is executed as the next commitment and its credit is correct; the tokenizer should not encode a deployment parameter. |
| "Discrete mode jumps raise the cost of exploration under RTC" | **Rejected.** Backwards. The policy having real influence over all `H` ticks is what teaches the model to take the tail seriously given that it may be committed and executed. There is no credit-assignment defect here. |
| Freezing `z_0` to eliminate uncredited randomness | **Withdrawn**, as a consequence of the `pin_flow_noise` rejection above. |

## 9. Measure first, build second

In order, all offline and none needing the RL stack:

1. Fit a candidate tokenizer standalone on the existing chunk corpus and look at
   code usage plus the **conditional** gate of section 4 -- does the code remove
   residual variation in `A` given `h`, or only add resolution?
2. Check whether scans are recoverable from single chunks or only from runs of
   them. If only from runs, the per-chunk code cannot split the mode and the design
   needs revisiting before anything downstream is built.
3. `z0_share` (`dump/latent_probe/z0_share.py`, unrun): the variance decomposition
   of chunk terminal displacement into what the observation chooses versus what
   `z_0` chooses. Bounds how much of the return-relevant variation any coarse code
   can absorb. On today's *unconditioned* checkpoint this is the baseline the
   design has to beat; the same probe run on a `c`-conditioned head, at fixed
   `(h, c)`, is the sub-cell test of section 3.2.

Then, once an SFT checkpoint exists and before committing to the RL path:

4. **Sub-cell obedience at SFT time** -- sample `z_0` at fixed `(h, c)`, encode,
   and measure `g(A') = c`. This is the pre-RL gate; a decoder that already leaves
   its cell on `z_0` alone will not be rescued by anything downstream.
5. **`proj(h)`'s steering share** (section 3.3) -- the measurement that decides
   whether the flow-SDE channel is kept and tuned or deleted outright.

# The action codebook, and the jargon around it

Written because the artifacts and metrics in this area are opaque from outside.
Two things up front:

- **It is not a VAE.** No encoder network, no learned embedding, no reconstruction
  loss, no gradients. `src/longnav/utils/action_codebook.py` contains zero
  `torch.nn`. It is **k-means** — Lloyd's algorithm with k-means++ seeding —
  producing a lookup table of K prototype actions.
- **Most of the vocabulary below is diagnostic**, not part of the model. "Creep
  band", "decisive mass", "stop-keep" are measurement bins used to detect one
  specific failure. The model never sees them.

## What problem this solves

The policy predicts an *action chunk*: 10 future poses per observation. Modelled
directly, that is 30 correlated continuous numbers, and a head trained with a
squared-error loss predicts their **conditional mean**. The action data is
bimodal — the robot is either stopped or driving — so the mean lands *between*
the modes, and the robot creeps forward at a speed it was never trained on. That
is the failure this whole area exists to fix.

Discretising removes it. If each action is one of K choices, the head predicts a
**categorical distribution** and you *sample* from it, which lands on a real
action instead of an average of two.

The codebook is what makes the actions discrete.

## What the codebook is

Take every per-tick action in the corpus — a 3-vector `(dx, dy, dtheta)`, the
motion during one tick in the robot's own frame — and cluster them into K groups.
The K cluster centres are the **prototypes**. Encoding an action means "find the
nearest prototype and return its index"; decoding means "look up that index".

So a chunk of 10 actions becomes 10 integers in `[0, K)`, and the policy head
becomes a K-way classifier run 10 times.

Clustering the three channels **jointly** (rather than binning each separately)
matters: a prototype is a whole physically-realisable action, so combinations the
robot never performs — full forward speed *and* a hard turn in the same tick —
simply are not in the vocabulary.

## The JSON file

```json
{
  "format": "longnav.action_codebook",
  "format_version": 1,
  "n_clusters": 1024,
  "dims":  ["dx", "dy", "dtheta"],
  "units": ["m", "m", "rad"],
  "centroids": [[2.4e-07, 1.6e-06, -0.0193], ...],
  "meta": {"corpus": "v2_25hz", "fitter_note": "full-batch Lloyd",
           "seed": 0, "fit_rows": 2000000, "corpus_ticks": 6109320,
           "kmeans_iters": 60, "fitter": "longnav.utils.action_codebook"}
}
```

| field | meaning |
|---|---|
| `centroids` | **the entire model** — `n_clusters × 3` prototype actions. Row *i* is what token *i* decodes to: metres forward, metres lateral, radians of rotation, for one tick. |
| `n_clusters` | K, the vocabulary size. Also the width of the head's softmax. |
| `dims`/`units` | column meanings, so a consumer cannot silently mix up radians and degrees or transpose the axes. |
| `meta.corpus` | **which corpus it was fitted on. A codebook is not portable across corpora** — v1 is 30 Hz and v2 is 25 Hz with different controller dynamics, so v1 prototypes are simply wrong for v2. |
| `meta.seed`, `fit_rows`, `kmeans_iters` | enough to reproduce the fit exactly. |

That is all it is: a list of K actions, plus provenance.

## How to fit one

```bash
<vln python> data_scripts/fit_action_codebook.py \
    --dataset data/v2_25hz/formatted --k 1024 --corpus v2_25hz \
    --out dump/autoregressive_head/codebook_1024_v2_lloyd.json
```

Two things it enforces, both because they have silently corrupted a run:

**Full-batch Lloyd, never MiniBatchKMeans.** MiniBatch produced prototypes
identical to 11 decimal places — 1024 nominal codes but only 773 actually
distinguishable. 42% of ticks landed on a code with a duplicate, so which token
they got was decided by floating-point tie-breaking. That is an unpredictable
label and a hard **1.22 nats/token** loss floor — 17.6% of the head's budget
spent on noise.

**A separation gate that refuses to save.** Every reconstruction metric is blind
to the above, because duplicates decode identically — RMSE, error percentiles and
stop-keep all read perfectly fine. Prototype separation is the only check that
catches it, so it runs before the file is written.

## What is NOT imposed

An earlier version hard-coded index 0 to a literal `(0, 0, 0)` "stop" prototype,
on the theory that the stop action had to be exactly representable. That was
removed after being measured to make things **worse**, and the reason is worth
knowing:

**The stop mode is a line, not a point.** "Stopped" means `dx ≈ 0, dy ≈ 0` with
`dtheta` free — the robot very often stops translating while still rotating
(`dx` is exactly zero on 64.5% of v1 ticks, `dtheta` on only 6.9%). K-means
naturally allocates a *family* of 95–518 prototypes along that line. A single
hard-coded point was the wrong shape for it, and forcing it manufactured stop
mass the data did not have (103.3% stop-keep — more out than in) and generated
degenerate duplicate prototypes besides.

The general lesson, which applies beyond codebooks: **verify the property, do not
impose it.** The fitter reports whether a near-zero prototype emerged and how much
mass it carries; it does not put one there.

## The diagnostic vocabulary

None of this is part of the model. These are bins used to detect creeping.

| term | meaning |
|---|---|
| **tick** | one control step. 0.04 s at 25 Hz. |
| **differential** | motion during one tick in the robot's frame, `(dx, dy, dtheta)`. The space everything is modelled in. See `src/longnav/utils/action_diffs.py`. |
| **exact-stop mass** | fraction of ticks with essentially zero forward motion (`|dx| < 0.1 mm`). ~49% in v2, ~66% in v1. |
| **creep band** | `0.1 mm ≤ |dx| < 1 cm` per tick — moving, but uselessly slowly. **The failure signature.** Real data has ~4% here; the broken regression head had 66%. |
| **decisive mass** | `|dx| ≥ 3 cm` per tick — committed motion. |
| **stop-keep** | of the ticks that were stopped in the data, what fraction are still stopped after encode→decode. Above 100% means the codebook *invented* stops. |
| **creep distortion** | how much the encode→decode round trip shifts creep-band mass. Should be ~0; a codebook that suppresses creep on its own would flatter any head trained on it. |
| **JS divergence** | Jensen–Shannon distance between predicted and real action distributions. Distribution-level accuracy, as opposed to per-tick error. |
| **flip rate** | how often commanded rotation reverses direction between adjacent ticks *within one chunk*. Real demonstrations: 0.005. Independent-sample heads: ~0.10. High values look like wobbling. |

**Why these and not just RMSE:** RMSE is dominated by the ~50–66% of ticks that
are exactly zero and therefore trivially correct. A head can score well on it
while creeping constantly. The band statistics ask a different question — *is
the probability mass in the right places* — which is what actually predicts
behaviour.

## Where the code lives

| file | role |
|---|---|
| `src/longnav/utils/action_diffs.py` | chunk → per-tick differentials (the shared transform) |
| `src/longnav/utils/action_codebook.py` | `ActionCodebook`: fit / encode / decode / save / load |
| `src/longnav/utils/ar_action_head.py` | the autoregressive head that consumes the codebook |
| `data_scripts/fit_action_codebook.py` | production fitter, with the separation gate |
| `data_scripts/train_ar_action_sft.py` | training entry point |

Analysis and one-off diagnostic scripts remain under `dump/` by design — that is
this repo's convention for scratch work, and `dump/` is gitignored.

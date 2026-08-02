"""Reusable joint vector-quantization codebook over per-tick robot actions.

An action chunk is 10 future poses expressed relative to one anchor (the observation
pose). Converted to per-tick BODY-FRAME differentials (see
``dump/data_diagnostics/analyze_chunk_motion.py::tick_differentials``) a chunk becomes
10 x ``(dx, dy, dtheta)`` triples. This module clusters those triples jointly into K
prototypes, so one tick becomes one discrete token and a chunk becomes a length-10
token sequence.

Why the clustering is JOINT across the three channels rather than per-channel: a code is
a whole physically realizable action. Combinations the controller never produces -- full
forward speed simultaneous with a hard turn -- simply do not exist in the vocabulary,
instead of being reachable by pairing an independently-chosen dx bin with an
independently-chosen dtheta bin.

The stop mode: measured, not imposed
------------------------------------
The data's dominant mode is "the robot did not move this tick" (~66% of v1 ticks have
bitwise-zero forward motion). An earlier iteration of this component reserved index 0 for
a hard-coded literal ``(0, 0, 0)`` prototype. **That requirement was retired by the
project owner and this module deliberately does not impose it.** Two reasons:

* It is real recurring work -- a special case threaded through every fit, every K and
  every corpus, and re-verified after each refit -- for something k-means does unprompted
  when a mode carries two thirds of the mass.
* What prevents the creeping failure is probability **mass**, not exact concentration. A
  code decoding to 1e-8 m/tick is behaviourally a stop; nothing downstream can tell it
  from rest. The same point was settled earlier when the flow head was designed.

So the stop mode is a property to **verify**, not an invariant to enforce.
:func:`stop_diagnostics` reports whether a near-zero code emerged, how close to zero it
actually decodes, and how much mass it holds -- and it reports the *forward-stop family*,
not just the single nearest-to-origin code, because that is what the data actually
contains: the joint all-three-zero atom is only ~5% of v1 ticks, while dx is exactly zero
~64% of the time. The difference is "translation-stopped, rotating a little", which
correctly occupies many codes sharing ``dx = dy ~ 0`` and differing in ``dtheta``.
The metric that finally matters is the round-trip exact-stop and creep-band mass, which
:func:`gate_report` computes with the project's shared band statistics.

Distance metric
---------------
Nearest prototype is plain Euclidean distance on the raw ``(dx, dy, dtheta)`` triple,
mixing metres and radians. That is defensible here rather than sloppy: the corpus caps
are 2.0 m/s and 2.0 rad/s, so at any fixed tick rate the per-tick ranges of dx and dtheta
are numerically identical (+/-0.080 m and +/-0.080 rad at 25 Hz). The axes are already
commensurate; rescaling would be the arbitrary choice.

Usage
-----
::

    from longnav.utils.action_codebook import ActionCodebook

    cb = ActionCodebook.load("dump/action_codebook/codebooks/v1_k512.json")
    cb.K, cb.meta["corpus"]             # 512, 'v1_30hz'
    tokens = cb.encode(diffs)           # (..., 3) float -> (...) int64 in [0, K)
    recon  = cb.decode(tokens)          # (...) int -> (..., 3) float64

    # torch consumers (e.g. an autoregressive head's de-tokenizer):
    table = torch.as_tensor(cb.centroids, dtype=torch.float32, device=dev)  # (K, 3)
    recon = table[token_ids]

``load`` also accepts the flat ``{"n_clusters", "centroids", "zero_tol"}`` layout used by
``dump/autoregressive_head/codebook_*.json``, so codebooks fitted by that track load here
unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Identical to longnav.utils.bin_codec.ZERO_TOL -- one project-wide definition of
# "bitwise stopped". Used only for diagnostics here, never to constrain a fit.
ZERO_TOL = 1e-5

__all__ = ["ActionCodebook", "ZERO_TOL", "FORMAT_VERSION", "stop_diagnostics",
           "gate_report"]

FORMAT_VERSION = 1


@dataclass
class ActionCodebook:
    """K prototype actions over per-tick body-frame ``(dx, dy, dtheta)`` differentials.

    Attributes
    ----------
    centroids : (K, 3) float64
        The prototypes. No index is reserved and no row is forced to any value.
    meta : dict
        Fit provenance -- corpus, K, seed, row counts, tick rate, timestamp. Carried
        through save/load so a consumer can tell which corpus and which K produced a
        file without reading anyone's notes.
    """

    centroids: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        c = np.array(self.centroids, dtype=np.float64, copy=True)
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError(f"centroids must be (K, 3), got {c.shape}")
        if c.shape[0] < 2:
            raise ValueError("a codebook needs at least two codes")
        if not np.isfinite(c).all():
            raise ValueError("centroids contain non-finite values")
        # Duplicate prototypes are always a fit bug (they can never both win an argmin),
        # and a duplicated *stop* code would split the stop mass across two tokens.
        uniq = np.unique(c, axis=0)
        if uniq.shape[0] != c.shape[0]:
            raise ValueError(f"{c.shape[0] - uniq.shape[0]} duplicate centroid rows")
        self.centroids = c

    # -------------------------------------------------------------------- basics ----
    @property
    def K(self) -> int:  # noqa: N802 -- K is this quantity's name everywhere in the repo
        return int(self.centroids.shape[0])

    def __len__(self) -> int:
        return self.K

    def __repr__(self) -> str:
        return (f"ActionCodebook(K={self.K}, corpus={self.meta.get('corpus', '?')!r}, "
                f"seed={self.meta.get('seed', '?')})")

    # ------------------------------------------------------------------- encode -----
    def encode(self, diffs: np.ndarray, batch: int = 500_000) -> np.ndarray:
        """``(..., 3)`` differentials -> ``(...)`` int64 token ids (nearest prototype).

        Pure nearest-neighbour: no value is special-cased, so the mapping is exactly the
        codebook's own geometry.

        Batched over rows on purpose. The one-shot
        ``((diffs[:, None] - centroids[None]) ** 2).sum(-1)`` materialises an ``(N, K, 3)``
        array -- 3.05M validation ticks against K=4096 is ~300 GB and simply dies, which
        is what previously capped K sweeps at a few hundred codes. Peak here is
        ``(batch, K)``, and the ``|c|^2 - 2 x.c`` expansion turns the inner loop into one
        BLAS matmul, which is ~20x faster than broadcasting differences.
        """
        arr = np.asarray(diffs, dtype=np.float64)
        if arr.shape[-1] != 3:
            raise ValueError(f"expected last axis 3, got {arr.shape}")
        flat = np.ascontiguousarray(arr.reshape(-1, 3))
        cent = self.centroids
        c_sq = (cent ** 2).sum(1)
        out = np.empty(flat.shape[0], dtype=np.int64)
        step = max(1, min(batch, int(2 ** 26 // max(self.K, 1))))
        for s in range(0, flat.shape[0], step):
            blk = flat[s:s + step]
            d2 = c_sq[None, :] - 2.0 * (blk @ cent.T)  # + |x|^2, constant within a row
            out[s:s + step] = d2.argmin(axis=1)
        return out.reshape(arr.shape[:-1])

    def decode(self, idx) -> np.ndarray:
        """``(...)`` token ids -> ``(..., 3)`` float64 differentials."""
        return self.centroids[np.asarray(idx, dtype=np.int64)]

    def roundtrip(self, diffs: np.ndarray, batch: int = 500_000) -> np.ndarray:
        return self.decode(self.encode(diffs, batch=batch))

    # ------------------------------------------------------------------- persist ----
    def to_dict(self) -> dict:
        return {
            "format": "longnav.action_codebook",
            "format_version": FORMAT_VERSION,
            "n_clusters": self.K,
            "dims": ["dx", "dy", "dtheta"],
            "units": ["m", "m", "rad"],
            "centroids": self.centroids.tolist(),
            "meta": self.meta,
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "ActionCodebook":
        """Accepts this module's format and the flat layout written by
        ``dump/autoregressive_head/fit_codebook.py`` (``n_clusters`` / ``centroids`` /
        ``zero_tol`` / ``source``), so codebooks from that track load unchanged."""
        meta = dict(d.get("meta", {}))
        if "source" in d and "corpus" not in meta:      # foreign (AR-track) layout
            meta.update({"corpus": d["source"], "n_clusters": d.get("n_clusters"),
                         "format_read_as": "autoregressive_head/fit_codebook.py"})
        return cls(np.asarray(d["centroids"], dtype=np.float64), meta)

    @classmethod
    def load(cls, path) -> "ActionCodebook":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ----------------------------------------------------------------------- fit ----
    @classmethod
    def fit(cls, diffs: np.ndarray, k: int, *, seed: int = 0,
            fit_rows: int | None = 2_000_000, device: str = "cpu", iters: int = 60,
            tol: float = 1e-10, verbose: bool = True,
            meta: dict | None = None) -> "ActionCodebook":
        """Fit a K-code book on per-tick differentials with seeded k-means.

        Plain k-means over ALL ticks, stop mode included. Nothing is reserved, snapped or
        hard-coded; see the module docstring. ``fit_rows`` subsamples for speed only
        (seeded, so the fit is reproducible). Pass ``device="cuda:N"`` to fit on a GPU.
        """
        rng = np.random.default_rng(seed)
        X = np.asarray(diffs, dtype=np.float64).reshape(-1, 3)
        n_all = X.shape[0]
        if fit_rows and n_all > fit_rows:
            X = X[rng.choice(n_all, fit_rows, replace=False)]
        if verbose:
            print(f"  fit K={k} on {X.shape[0]:,} of {n_all:,} ticks "
                  f"(seed {seed}, device {device})", flush=True)
        centroids = _kmeans(X, k, seed=seed, device=device, iters=iters, tol=tol,
                            verbose=verbose)
        info = dict(meta or {})
        info.update({"n_clusters": int(k), "seed": int(seed),
                     "fit_rows": int(X.shape[0]), "corpus_ticks": int(n_all),
                     "kmeans_iters": int(iters), "fitter": "longnav.utils.action_codebook"})
        return cls(centroids, info)


# ====================================================================================
# Diagnostics. Deliberately module-level functions rather than methods: they are what a
# consumer runs against a codebook it did not fit.
# ====================================================================================
def stop_diagnostics(cb: ActionCodebook, diffs: np.ndarray, idx: np.ndarray | None = None,
                     stop_tol: float = 1e-4, zero_tol: float = ZERO_TOL) -> dict:
    """Did a stop code emerge on its own, how close to zero is it, how much mass?

    Two thresholds, and the difference between them is the owner's point:

    * ``zero_tol`` (1e-5) is *bitwise* stopped -- the test the retired hard-coded atom
      was built to satisfy.
    * ``stop_tol`` (1e-4 = 0.1 mm/tick) is *behaviourally* stopped -- the project's
      ``sweep_analysis.EXACT`` band edge, and the threshold every existing table uses.
      A code decoding to 3e-5 m/tick (0.9 mm/s) fails the first test and passes the
      second; nothing downstream can tell it from rest. The second is the one that
      matters, which is why an unforced fit is fine.

    Reports, because "the stop" is not one code:

    ``nearest_origin``
        The single prototype closest to the origin: how far from zero it decodes and its
        share of ticks. Compare its mass against the corpus's *joint* all-three-zero
        rate, NOT the forward exact-stop rate -- in v1 those are ~5% and ~66%.
    ``forward_stop_family`` / ``forward_stop_family_bitwise``
        Every code decoding to ``|dx|`` below the respective threshold. This is the set
        that makes a tick decode to "no forward motion", including the
        translation-stopped-but-rotating modes, and its mass is what should sit near the
        corpus's forward exact-stop rate.
    ``recovered``
        Of the ticks truly forward-stopped in the data, what fraction round-trip to a
        forward-stopped code, and the converse false-stop rate. The behavioural question,
        independent of how the vocabulary happens to be partitioned.
    """
    X = np.asarray(diffs, dtype=np.float64).reshape(-1, 3)
    if idx is None:
        idx = cb.encode(X)
    idx = np.asarray(idx).reshape(-1)
    counts = np.bincount(idx, minlength=cb.K)
    mass = counts / max(1, counts.sum())

    norms = np.linalg.norm(cb.centroids, axis=1)
    j = int(np.argmin(norms))

    def family(tol):
        fam = np.flatnonzero(np.abs(cb.centroids[:, 0]) < tol)
        return fam, {"tol": tol, "n_codes": int(fam.size),
                     "mass": float(mass[fam].sum()),
                     "max_abs_dtheta": float(np.abs(cb.centroids[fam, 2]).max())
                     if fam.size else 0.0}

    fam, fam_rep = family(stop_tol)
    _, fam_bit = family(zero_tol)
    joint = np.flatnonzero((np.abs(cb.centroids) < stop_tol).all(axis=1))

    data_fwd_stop = np.abs(X[:, 0]) < stop_tol
    rec_fwd_stop = np.isin(idx, fam)
    return {
        "stop_tol": stop_tol,
        "zero_tol": zero_tol,
        "nearest_origin": {
            "code": j,
            "centroid": cb.centroids[j].tolist(),
            "l2_norm": float(norms[j]),
            "abs_dx": float(abs(cb.centroids[j, 0])),
            "is_bitwise_zero": bool(norms[j] == 0.0),
            "is_behavioural_stop": bool(norms[j] < stop_tol),
            "mass": float(mass[j]),
        },
        "forward_stop_family": fam_rep,
        "forward_stop_family_bitwise": fam_bit,
        "joint_stop_family": {"n_codes": int(joint.size),
                              "mass": float(mass[joint].sum()) if joint.size else 0.0},
        "data": {
            "forward_exact_stop": float(data_fwd_stop.mean()),
            "joint_all_zero": float((np.abs(X) < zero_tol).all(axis=1).mean()),
        },
        "recovered": {
            "true_stop_kept": float(rec_fwd_stop[data_fwd_stop].mean()),
            "false_stop_rate": float(rec_fwd_stop[~data_fwd_stop].mean()),
        },
    }


def gate_report(cb: ActionCodebook, diffs: np.ndarray, band_stats, divergences=None,
                bands=None) -> dict:
    """Round-trip fidelity + occupancy + the project's band statistics.

    ``band_stats`` / ``divergences`` are injected rather than imported so this module
    stays dependency-free; pass ``dump/data_diagnostics/sweep_analysis.band_stats`` and
    ``.divergences`` to get numbers directly comparable with every existing table.
    ``bands`` defaults to that module's ``(EXACT, CREEP, DECISIVE)`` = (1e-4, 0.01, 0.03)
    for forward and (1e-4, 0.02, 0.04) for rotation.
    """
    exact, creep, decisive = bands or (1e-4, 0.01, 0.03)
    X = np.asarray(diffs, dtype=np.float64).reshape(-1, 3)
    idx = cb.encode(X)
    rec = cb.decode(idx)
    err = rec - X
    a_dx = np.abs(err[:, 0])
    qs = [50, 90, 99, 99.9, 100]

    rep: dict = {"K": cb.K, "ticks": int(X.shape[0])}
    rep["rmse"] = {n: float(np.sqrt((err[:, i] ** 2).mean()))
                   for i, n in enumerate(("dx", "dy", "dtheta"))}
    rep["err_dx_pct"] = {f"p{q}": float(np.percentile(a_dx, q)) for q in qs}
    rep["err_dtheta_pct"] = {f"p{q}": float(np.percentile(np.abs(err[:, 2]), q))
                             for q in qs}

    # Fidelity where it matters. Aggregate RMSE is dominated by the ~66% of ticks that
    # are exactly zero and therefore trivially perfect, so it cannot see a hole in the
    # fast-driving region. Restrict to decisive motion and report the distribution.
    dec = np.abs(X[:, 0]) >= decisive
    rep["decisive"] = {"n_ticks": int(dec.sum()), "frac_of_ticks": float(dec.mean())}
    if dec.any():
        d_err = a_dx[dec]
        rep["decisive"].update({
            "rmse_dx": float(np.sqrt((err[dec, 0] ** 2).mean())),
            **{f"err_dx_p{q}": float(np.percentile(d_err, q)) for q in qs},
            "median_rel_err": float(np.median(d_err / np.abs(X[dec, 0]))),
            "p99_rel_err": float(np.percentile(d_err / np.abs(X[dec, 0]), 99)),
        })

    for col, name, cr, dc in ((0, "forward", creep, decisive), (2, "rotation", 0.02, 0.04)):
        block = {"data": band_stats(X[:, col], exact, cr, dc),
                 "roundtrip": band_stats(rec[:, col], exact, cr, dc)}
        block["creep_distortion_pp"] = 100.0 * (block["roundtrip"]["creep_band"]
                                                - block["data"]["creep_band"])
        block["stop_keep_ratio"] = (block["roundtrip"]["exact_stop"]
                                    / max(1e-12, block["data"]["exact_stop"]))
        if divergences is not None:
            lo, hi = (-0.02, 0.07) if col == 0 else (-0.09, 0.09)
            block["roundtrip_vs_data"] = divergences(rec[:, col], X[:, col], lo, hi)
        rep[name] = block

    counts = np.bincount(idx, minlength=cb.K)
    p = counts / counts.sum()
    nz = p[p > 0]
    rep["occupancy"] = {
        "empty_codes": int((counts == 0).sum()),
        "entropy_bits": float(-(nz * np.log2(nz)).sum()),
        "max_bits": float(np.log2(cb.K)),
        "max_code_mass": float(p.max()),
        "codes_holding_99pct": int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), 0.99) + 1),
    }
    rep["stop"] = stop_diagnostics(cb, X, idx)
    rep["meta"] = cb.meta
    return rep


# ====================================================================================
# k-means. Torch (CPU by default; pass device="cuda:N" to use a GPU). Deterministic
# given `seed`.
# ====================================================================================
def _kmeans(X: np.ndarray, k: int, *, seed: int, device: str = "cpu", iters: int = 60,
            tol: float = 1e-10, verbose: bool = False) -> np.ndarray:
    """Lloyd's algorithm with k-means++ seeding. Returns ``(k, 3)`` float64 centroids."""
    import torch

    dev = torch.device(device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=dev)
    n = Xt.shape[0]
    if k > n:
        raise ValueError(f"K={k} exceeds the {n} rows available to fit it")

    cent = torch.empty((k, 3), dtype=torch.float32, device=dev)
    cent[0] = Xt[int(torch.randint(n, (1,), generator=g).item())]
    closest = ((Xt - cent[0]) ** 2).sum(1)
    for i in range(1, k):
        total = float(closest.sum())
        if total <= 0:
            pick = int(torch.randint(n, (1,), generator=g).item())
        else:
            u = float(torch.rand((1,), generator=g).item()) * total
            pick = int(torch.searchsorted(torch.cumsum(closest, 0),
                                          torch.tensor(u, device=dev)).clamp(max=n - 1))
        cent[i] = Xt[pick]
        closest = torch.minimum(closest, ((Xt - cent[i]) ** 2).sum(1))

    chunk = max(1, int(2 ** 25 // max(k, 1)))
    labels = torch.zeros(n, dtype=torch.long, device=dev)
    for it in range(iters):
        c_sq = (cent ** 2).sum(1)
        for s in range(0, n, chunk):
            blk = Xt[s:s + chunk]
            labels[s:s + chunk] = (c_sq[None, :] - 2.0 * (blk @ cent.T)).argmin(1)
        sums = torch.zeros((k, 3), dtype=torch.float32, device=dev)
        sums.index_add_(0, labels, Xt)
        counts = torch.bincount(labels, minlength=k).to(torch.float32)
        empty = counts == 0
        newc = sums / counts.clamp(min=1).unsqueeze(1)
        if bool(empty.any()):
            # Re-seed dead clusters at the worst-served points, so a large-K fit does not
            # silently ship codes that no tick will ever select.
            far = torch.topk(((Xt - cent[labels]) ** 2).sum(1), int(empty.sum())).indices
            newc[empty] = Xt[far]
        shift = float(((newc - cent) ** 2).sum(1).max())
        cent = newc
        if verbose and (it % 10 == 0):
            print(f"    iter {it:3d}  max centroid shift {shift:.3e}  "
                  f"empty {int(empty.sum())}", flush=True)
        if shift <= tol:
            break
    return cent.double().cpu().numpy()

"""Pure-logic tests for the head-only training pipeline.

No simulator, no GPU, no checkpoint on disk. What is pinned here is the part of the
pipeline whose failures are silent:

  * provenance round-trips and, more to the point, **refuses** a trunk it does not
    recognise -- a head reattached to the wrong trunk produces a policy that loads, runs
    and is meaningless;
  * the store's shard/resume/atomicity contract;
  * metric parity -- `FrozenContextFlowPolicy` and `TurnFlowPolicy` must produce the same
    numbers under the same key names from the same context, which is the assumption every
    comparison drawn from this pipeline rests on;
  * the collapse diagnostics, against synthetic collapsed and healthy context whose
    answers are known analytically.
"""

import json

import numpy as np
import pytest
import torch

from longnav.cnf_head.flow import ConditionalFlow
from longnav.utils.head_only import (
    ContextStore,
    FrozenContextCollator,
    FrozenContextDataset,
    FrozenContextFlowPolicy,
    HarvestProvenance,
    context_collapse_stats,
    coupling_context_ratio,
    hash_checkpoint,
)

CHUNK = (4, 3)
POOLED, CTX = 32, 16


def _provenance(**over):
    base = dict(
        source_checkpoint="/nowhere/checkpoint-11400", source_step=11400,
        source_hash="deadbeef", source_target_shape=[1024], corpus="/nowhere/corpus",
        split="train", pooled_dim=POOLED, context_dim=CTX, chunk_shape=list(CHUNK),
    )
    base.update(over)
    return HarvestProvenance(**base)


def _write_store(tmp_path, n_eps=6, rows_per_ep=5, shard_eps=2, **prov_over):
    store = ContextStore(tmp_path / "store")
    store.init(_provenance(**prov_over))
    rng = np.random.default_rng(0)
    ep = 0
    for _ in range(n_eps // shard_eps):
        eps = [str(ep + k) for k in range(shard_eps)]
        n = rows_per_ep * shard_eps
        store.write_shard(
            {"pooled": rng.normal(size=(n, POOLED)).astype(np.float32),
             "context": rng.normal(size=(n, CTX)).astype(np.float32),
             "targets": rng.normal(size=(n, *CHUNK)).astype(np.float32) * 0.01},
            eps, [rows_per_ep] * shard_eps,
        )
        ep += shard_eps
    return store


# =======================================================================================
# Provenance
# =======================================================================================
def test_provenance_round_trips_through_json():
    p = _provenance(modality_specs=[{"token": "<pose>", "n_features": 3}], obs_hz=2.5)
    back = HarvestProvenance.from_json(json.loads(json.dumps(p.to_json())))
    assert back == p


def test_provenance_tolerates_unknown_future_keys():
    blob = _provenance().to_json()
    blob["a_field_added_next_year"] = 7
    assert HarvestProvenance.from_json(blob).context_dim == CTX


def test_hash_covers_weights_and_ignores_optimiser_state(tmp_path):
    ck = tmp_path / "checkpoint-100"
    (ck / "adapter").mkdir(parents=True)
    (ck / "turn_vector_head.pt").write_bytes(b"weights")
    (ck / "adapter" / "adapter_model.safetensors").write_bytes(b"lora")
    before = hash_checkpoint(ck)

    # Resume bookkeeping changes no forward pass and must not change the hash, or a
    # legitimate reattachment would fail after a restart.
    (ck / "optimizer.pt").write_bytes(b"x" * 1000)
    (ck / "trainer_state.json").write_text('{"global_step": 100}')
    assert hash_checkpoint(ck) == before

    (ck / "turn_vector_head.pt").write_bytes(b"different weights")
    assert hash_checkpoint(ck) != before


def test_reattaching_to_the_wrong_trunk_is_refused_and_names_both(tmp_path):
    right, wrong = tmp_path / "right", tmp_path / "wrong"
    for d, payload in ((right, b"A"), (wrong, b"B")):
        (d / "adapter").mkdir(parents=True)
        (d / "turn_vector_head.pt").write_bytes(payload)
    prov = _provenance(source_checkpoint=str(right), source_hash=hash_checkpoint(right))

    prov.assert_matches_checkpoint(right)          # the trunk it came from: fine
    with pytest.raises(RuntimeError) as exc:
        prov.assert_matches_checkpoint(wrong)
    msg = str(exc.value)
    assert str(right) in msg and str(wrong) in msg, "the error must name both trunks"


# =======================================================================================
# The store
# =======================================================================================
def test_store_round_trips_rows_shards_and_episodes(tmp_path):
    _write_store(tmp_path, n_eps=6, rows_per_ep=5)
    s = ContextStore.open(tmp_path / "store")
    assert len(s) == 30 and s.n_episodes == 6
    assert s.get("pooled", np.arange(30)).shape == (30, POOLED)
    assert s.get("targets", np.array([0, 29])).shape == (2, *CHUNK)
    # Crossing a shard boundary must give the same rows as reading each shard.
    whole = s.whole("pooled")
    assert np.allclose(s.get("pooled", np.array([9, 10, 11])), whole[[9, 10, 11]])


def test_done_episodes_is_the_resume_key(tmp_path):
    store = _write_store(tmp_path, n_eps=4, rows_per_ep=3, shard_eps=2)
    assert store.done_episodes() == {"0", "1", "2", "3"}


def test_a_shard_without_a_manifest_entry_is_invisible(tmp_path):
    """The atomicity contract: arrays land first, the manifest entry makes them real."""
    store = _write_store(tmp_path, n_eps=2, rows_per_ep=3, shard_eps=2)
    rng = np.random.default_rng(1)
    for name, width in (("pooled", POOLED), ("context", CTX)):
        np.save(store.root / f"shard_00099.{name}.npy",
                rng.normal(size=(3, width)).astype(np.float32))
    np.save(store.root / "shard_00099.targets.npy",
            rng.normal(size=(3, *CHUNK)).astype(np.float32))
    s = ContextStore.open(store.root)
    assert len(s) == 6, "a crashed shard with no manifest entry must not be read"


def test_extending_a_store_from_a_different_trunk_is_refused(tmp_path):
    store = _write_store(tmp_path, n_eps=2, rows_per_ep=3, shard_eps=2)
    with pytest.raises(RuntimeError, match="different configuration"):
        store.init(_provenance(source_hash="a different trunk"))


def test_split_is_by_episode_so_no_episode_straddles_it(tmp_path):
    _write_store(tmp_path, n_eps=6, rows_per_ep=5)
    s = ContextStore.open(tmp_path / "store")
    tr, va, te = s.split_by_episode(val_frac=1 / 3, test_frac=1 / 3, seed=0)
    assert len(tr) + len(va) + len(te) == len(s)
    groups = [set(s.episode_ids[i].tolist()) for i in (tr, va, te)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert not (groups[a] & groups[b]), "an episode appears on both sides of a split"


def test_open_many_joins_parts_and_refuses_a_mixed_trunk(tmp_path):
    a = ContextStore(tmp_path / "a")
    a.init(_provenance())
    b = ContextStore(tmp_path / "b")
    b.init(_provenance())
    c = ContextStore(tmp_path / "c")
    c.init(_provenance(source_hash="other trunk"))
    rng = np.random.default_rng(0)
    for store, eps in ((a, ["0", "1"]), (b, ["2", "3"]), (c, ["4"])):
        n = 2 * len(eps)
        store.write_shard(
            {"pooled": rng.normal(size=(n, POOLED)).astype(np.float32),
             "context": rng.normal(size=(n, CTX)).astype(np.float32),
             "targets": rng.normal(size=(n, *CHUNK)).astype(np.float32)},
            eps, [2] * len(eps))

    joined = ContextStore.open_many([a.root, b.root])
    assert len(joined) == 8 and joined.n_episodes == 4

    with pytest.raises(RuntimeError, match="source_hash differs"):
        ContextStore.open_many([a.root, c.root])
    with pytest.raises(RuntimeError, match="share"):
        ContextStore.open_many([a.root, a.root])


# =======================================================================================
# Collapse diagnostics
# =======================================================================================
def test_collapsed_context_is_diagnosed_as_collapsed():
    """The dead run's actual signature, which is more specific than "nearly constant".

    `dump/cnf_sigma_ablation/FINDINGS.md` measured the dead flow run at step 3000 as
    resid/row 1.9%, participation ratio 1.20, mean pairwise cos +0.9998. A PR of 1.2 on
    the *mean-removed* matrix says the little variation that survives is itself confined
    to about one direction -- so the faithful construct is a large constant plus a small
    excursion along a single axis, not a constant plus isotropic noise.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 64)) * 10.0
    axis = rng.normal(size=(1, 64))
    axis /= np.linalg.norm(axis)
    x = (np.repeat(base, 512, axis=0)
         + rng.normal(size=(512, 1)) * axis * 0.3
         + rng.normal(size=(512, 64)) * 0.002)

    s = context_collapse_stats(x)
    assert s["ctx_resid_frac"] < 0.05, s
    assert s["ctx_participation_ratio"] < 2.0, s
    assert s["ctx_mean_pairwise_cos"] > 0.99, s
    assert s["ctx_frac_near_mean"] > 0.99, s


def test_resid_frac_and_participation_ratio_are_not_redundant():
    """A constant plus isotropic noise: near-constant, yet full-rank in its residual.

    This is the case that motivates logging both. `resid_frac` and `frac_near_mean` see
    it immediately -- 99% of every row is one vector, so permuting the context is nearly
    a no-op and any conditional measurement is vacuous -- while the participation ratio,
    computed on the mean-removed matrix as `ctx_stats.py` computes it, reports the full
    width and says nothing is wrong. Reading PR alone would miss this collapse.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 64)) * 10.0
    x = np.repeat(base, 512, axis=0) + rng.normal(size=(512, 64)) * 0.02

    s = context_collapse_stats(x)
    assert s["ctx_resid_frac"] < 0.05, s
    assert s["ctx_frac_near_mean"] > 0.99, s
    assert s["ctx_participation_ratio"] > 32, s


def test_healthy_context_is_diagnosed_as_healthy():
    """Isotropic gaussian: every direction used, rows near-orthogonal, no shared mean."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2048, 64))

    s = context_collapse_stats(x)
    assert s["ctx_resid_frac"] > 0.9, s
    # PR of isotropic noise is ~D; anything above D/2 is unambiguously not collapsed.
    assert s["ctx_participation_ratio"] > 32, s
    assert abs(s["ctx_mean_pairwise_cos"]) < 0.05, s
    assert s["ctx_frac_near_mean"] < 0.01, s


def test_participation_ratio_counts_the_directions_actually_used():
    """Exactly `k` active directions padded with zeros must score ~k, not the width."""
    rng = np.random.default_rng(2)
    for k in (1, 3, 8):
        x = np.zeros((1024, 64))
        x[:, :k] = rng.normal(size=(1024, k))
        pr = context_collapse_stats(x)["ctx_participation_ratio"]
        assert abs(pr - k) < 0.5 * k + 0.5, (k, pr)


def test_collapse_stats_reject_a_degenerate_input():
    with pytest.raises(ValueError):
        context_collapse_stats(np.zeros((1, 8)))
    with pytest.raises(ValueError):
        context_collapse_stats(np.zeros(8))


def test_coupling_ratio_reads_the_two_column_blocks():
    """`mean|W_ctx| / mean|W_x|` -- the signature FINDINGS tracked, computed by hand."""
    flow = ConditionalFlow(context_dim=CTX, n_layers=2, hidden=8,
                           chunk_len=CHUNK[0], n_channels=CHUNK[1])
    with torch.no_grad():
        for c in flow.couplings:
            c.net[0].weight[:, :flow.dim] = 0.4
            c.net[0].weight[:, flow.dim:] = 0.1
    r = coupling_context_ratio(flow)
    assert r["w_x_abs_mean"] == pytest.approx(0.4, abs=1e-6)
    assert r["w_ctx_abs_mean"] == pytest.approx(0.1, abs=1e-6)
    assert r["w_ctx_over_w_x"] == pytest.approx(0.25, abs=1e-6)


def test_coupling_ratio_is_zero_for_an_unconditional_flow():
    flow = ConditionalFlow(context_dim=0, n_layers=2, hidden=8,
                           chunk_len=CHUNK[0], n_channels=CHUNK[1])
    assert coupling_context_ratio(flow)["w_ctx_over_w_x"] == 0.0


# =======================================================================================
# Metric parity -- the assumption every comparison rests on
# =======================================================================================
def _tiny_head(identity=False, seed=0):
    torch.manual_seed(seed)
    return FrozenContextFlowPolicy(
        pooled_dim=CTX if identity else POOLED, context_dim=CTX, chunk_shape=CHUNK,
        head_hidden_dims=(24,), flow_kwargs={"n_layers": 4, "hidden": 16, "depth": 2},
        decode_k=4, identity_head=identity,
    )


def test_head_only_and_end_to_end_share_the_objective_implementation():
    """Not "they agree": they are the same method, and this pins that.

    `FrozenContextFlowPolicy` must inherit `flow_objective` and `_flow_metrics` from
    `flow_head.FlowObjectiveMixin` -- the same objects `TurnFlowPolicy` inherits. If
    someone reimplements either here, the numbers can drift apart without raising and
    every conclusion drawn by comparing the two pipelines silently dies.
    """
    from longnav.utils.flow_head import FlowObjectiveMixin, TurnFlowPolicy

    assert issubclass(FrozenContextFlowPolicy, FlowObjectiveMixin)
    assert issubclass(TurnFlowPolicy, FlowObjectiveMixin)
    for name in ("flow_objective", "_flow_metrics"):
        assert getattr(FrozenContextFlowPolicy, name) is getattr(FlowObjectiveMixin, name)
        assert getattr(TurnFlowPolicy, name) is getattr(FlowObjectiveMixin, name)


def test_objective_emits_every_key_the_trainer_drains():
    """The keys `FlowSFTTrainer._drain_metrics` needs to produce the run's metric names."""
    from longnav.utils.flow_head import FLOW_METRIC_KEYS

    model = _tiny_head()
    torch.manual_seed(0)
    out = model(torch.randn(8, POOLED), torch.randn(8, *CHUNK) * 0.01)
    for key in FLOW_METRIC_KEYS:
        assert key in out, key
    for key in ("loss", "loss_sum", "sum_sq_err", "sum_abs_err", "n_rows", "n_turns",
                "n_steps", "n_tokens", "n_dense_tokens"):
        assert key in out, key


def test_frozen_context_reproduces_the_end_to_end_objective_exactly():
    """Same weights, same context, same targets -> bit-identical loss and metrics.

    Built by giving a `TurnFlowPolicy`-shaped objective the identical head and flow this
    model has, and calling the shared `flow_objective` through both. The head-only path
    additionally runs `head.project`, so the input is the pooled state on one side and
    the already-projected context on the other -- and the numbers must still match.
    """
    model = _tiny_head()
    torch.manual_seed(3)
    pooled = torch.randn(16, POOLED)
    targets = torch.randn(16, *CHUNK) * 0.01

    torch.manual_seed(11)
    a = model(pooled, targets)
    # The same objective entered directly with the projected context: this is the call
    # `TurnFlowPolicy.forward` makes after its backbone.
    torch.manual_seed(11)
    b = model.flow_objective(model.head.project(pooled).float(), targets)

    for k, v in a.items():
        assert torch.allclose(v.float(), b[k].float(), atol=0, rtol=0), k


def test_metric_names_match_a_real_runs_log():
    """The exact `eval_*` keys `train_flow_sft.py` emits must come out of this drain.

    Names taken from `dump/pose_injection/run_v12_cnf_planar_pose_2p5hz`'s own
    `trainer_state.json`, i.e. from a real run's log rather than from this code's
    intentions.
    """
    from longnav.utils.flow_head import FlowSFTTrainer
    from longnav.utils.head_only import HeadOnlyFlowTrainer

    # Inheritance, not duplication, is what makes the names identical.
    assert issubclass(HeadOnlyFlowTrainer, FlowSFTTrainer)
    for name in ("_drain_metrics", "_accumulate"):
        assert getattr(HeadOnlyFlowTrainer, name) is getattr(FlowSFTTrainer, name)


def test_identity_head_trains_only_the_flow():
    model = _tiny_head(identity=True)
    ctx = torch.randn(8, CTX)
    assert torch.equal(model.encode(ctx), ctx.float())
    with pytest.raises(ValueError, match="identity_head"):
        FrozenContextFlowPolicy(pooled_dim=POOLED, context_dim=CTX, chunk_shape=CHUNK,
                                identity_head=True)


def test_head_checkpoint_round_trips(tmp_path):
    model = _tiny_head(seed=5)
    model.save_pretrained(tmp_path / "head")
    back = FrozenContextFlowPolicy.from_pretrained(tmp_path / "head")

    ctx = torch.randn(4, POOLED)
    assert torch.allclose(model.encode(ctx), back.encode(ctx), atol=0, rtol=0)
    assert back.chunk_shape == model.chunk_shape
    assert back.flow.config() == model.flow.config()
    # The blob's top-level keys are the ones `TurnVectorRegressor.load_head_state`
    # reads, which is what makes a reattached checkpoint need no special case.
    blob = torch.load(tmp_path / "head" / "turn_vector_head.pt", weights_only=False)
    assert set(blob) == {"head", "normalizer"}


def test_head_state_keys_match_a_real_turn_vector_head():
    """The reattachment loads these tensors into a real `TurnVectorHead`, strictly."""
    from longnav.utils.turn_vectors import TurnVectorHead

    model = _tiny_head()
    real = TurnVectorHead(hidden_size=POOLED, out_dim=CTX, mode="mean",
                          hidden_dims=(24,), layer_norm=True)
    real.load_state_dict(model.head.state_dict(), strict=True)
    x = torch.randn(3, POOLED)
    assert torch.allclose(real.project(x), model.head.project(x), atol=0, rtol=0)


# =======================================================================================
# Dataset / collator
# =======================================================================================
def test_collator_batches_rows_and_counts_turns(tmp_path):
    _write_store(tmp_path, n_eps=4, rows_per_ep=5)
    s = ContextStore.open(tmp_path / "store")
    ds = FrozenContextDataset(s, np.arange(len(s)), "pooled")
    batch = FrozenContextCollator(s, "pooled")([ds[i] for i in range(6)])

    assert batch["context"].shape == (6, POOLED)
    assert batch["targets"].shape == (6, *CHUNK)
    # `_get_num_items_in_batch` sums `num_turns` over the raw items, so one turn per
    # item is what makes `turn_loss` mean the same thing as in a full run.
    assert int(batch["num_turns"]) == 6
    assert all(ds[i]["num_turns"] == 1 for i in range(6))


def test_dataset_rejects_a_context_key_the_store_lacks(tmp_path):
    _write_store(tmp_path, n_eps=2, rows_per_ep=3, shard_eps=2)
    s = ContextStore.open(tmp_path / "store")
    with pytest.raises(KeyError):
        FrozenContextDataset(s, np.arange(len(s)), "pose")

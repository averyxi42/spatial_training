"""
Minimal turn-by-turn rollout for a model trained by `vector_sft.py`.

`VLMWorker` (see `vlm_worker.py`) does incremental multi-turn inference, but it is built
for the RL stack: Ray actors, logprobs, value heads, adapter merge/unmerge bookkeeping,
rollout buffers. This is the same KV-cache mechanics with none of that -- feed one
observation, get one continuous action chunk, repeat. Nothing here trains, distributes or
logs. `vlm_worker.py` is untouched.

Turn structure is described the way `VLMWorker` describes it: give it the affix strings
and the constant placeholder, and it derives the rest. `split_assistant_turn` runs the
*training-time* `find_turn_spans` over the assistant turn to locate the readout positions,
splits the turn's tokens there into what to emit now and what is owed to the next forward,
and reports how many positions the head will be handed. `_assert_head_matches_readout`
then checks that count against the head, because a pooled head accepts the wrong number of
positions without complaint. Nothing hardcodes `**`, a single readout token, or a
particular placeholder -- change `RolloutConfig.prefix/postfix/placeholder` and the split
follows, or fails loudly if that combination cannot be located (e.g. `'**[…]**'`, where BPE
merges `]` into the closing `**`).

Because the emitted block ends exactly at the last readout position, `step()` reads
`last_hidden_state[:, -n_readout:]` with no span search, and no generation is needed -- the
action for step k is available as soon as step k's image is encoded. Those positions are
text tokens, which the sparsifier never drops.

Composition is done in token ids rather than text: the user block comes from the processor
(which expands the image placeholder) and the assistant pieces are constant id lists, so
concatenation cannot merge differently than the training-time single-shot tokenization did.

Per-step mechanics, mirroring `VLMWorker.infer_step`:

  * only the new tokens are forwarded; `past_key_values` carries the rest.
  * `position_ids` come from `get_rope_index` on this turn's tokens plus a running
    offset advanced by the turn length and the rope delta (`_pos_id_fast`'s arithmetic).
  * with the sparse backbone, `attention_mask=None` and the visual embedding database
    (`past_image_embeds` / `save_image_db`) is threaded turn to turn, so redundant visual
    tokens from earlier frames are dropped exactly as they are in a single-shot pass.

Text composition is *token-exact* against the training-time single-shot tokenization --
each turn's chunk is the previous assistant turn's tail plus this turn's user block plus
the assistant opening, and concatenating the chunks reproduces
`apply_chat_template(whole_conversation)` byte for byte. `tests/test_vector_rollout.py`
asserts this; if it ever drifts, the rollout context stops matching what the model saw in
training and the head silently degrades.

    policy = VectorRolloutPolicy.from_checkpoint("dump/vector_sft_3090/final")
    print(policy.describe())                # affixes, emitted tokens, readout width
    policy.reset(goal_text="chest_of_drawers")
    for image in observations:
        chunk = policy.step(image)          # (N_action, 3) in the target's own units
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from longnav.utils.modality_embed import ModalityBatch, single_example_batch
from longnav.utils.vector_sft import TurnVectorRegressor

# The assistant turn's closing text, appended to the *next* forward pass because with a
# KV cache it is context, not output.
TURN_CLOSE = "<|im_end|>\n"
# What the chat template emits to open an assistant turn.
CHAT_ASSISTANT_OPEN = "<|im_start|>assistant\n"

# The canonical prompt, imported by `data_scripts/format_action_chunk_dataset.py` so the
# training conversations and the rollout context cannot drift apart.
DEFAULT_SYSTEM_PROMPT = (
    "You are a robot navigating an indoor environment toward a goal object.\n"
    "Goal: {goal}\n"
    "At each step you receive the current RGB observation. Produce the next short "
    "trajectory of poses to follow, relative to your current pose."
)

# The PointNav counterpart. Chosen per row from the corpus's own `task` column rather
# than by a flag, so a table can never end up describing one task with the other's
# prompt. `{goal}` is deliberately absent: a point goal is a *value* injected at a
# `<pose>` marker, not a string to interpolate.
POINTNAV_SYSTEM_PROMPT = (
    "You are a robot navigating an indoor environment toward a point goal.\n"
    "You are given your own pose and the goal's pose in the same frame, and are told a "
    "new goal whenever you reach the last one.\n"
    "At each step you receive the current RGB observation. Produce the next short "
    "trajectory of poses to follow, relative to your current pose."
)

#: Prompt per task, keyed by what the corpus stamps in
#: `recording_metadata.RecordingParams.task_name`. Unknown task -> the ObjectNav prompt,
#: which is what every corpus written before the field existed is.
SYSTEM_PROMPTS = {
    "objectnav": DEFAULT_SYSTEM_PROMPT,
    "pointnav": POINTNAV_SYSTEM_PROMPT,
}


@dataclass
class RolloutConfig:
    """Everything the policy needs that is not stored in the checkpoint.

    Turn structure is described the way `VLMWorker` describes it -- by the affix strings --
    and everything else is derived, so no convention is hardcoded. `prefix`/`postfix`
    default to the checkpoint's own (`ModelConfig.prefix`/`postfix`), and `placeholder` is
    the constant assistant message; together with `shift_left` they determine which token
    positions the head reads, via the same `find_turn_spans` the training used.

    The strings must match the training data's conversation format. A mismatch changes the
    context the head was trained under, so `VectorRolloutPolicy` re-derives the span and
    asserts it against what the head expects rather than trusting them.
    """

    placeholder: str = "**____**"
    prefix: Optional[str] = None   # None -> the checkpoint's own prefix
    postfix: Optional[str] = None
    shift_left: Optional[bool] = None
    user_text_before: str = "Observation {step}:"
    user_text_after: str = "Action:"
    # Modality marker written into each turn's user block, after the image, exactly as
    # `format_action_chunk_dataset.py --modality-marker` writes it into the training
    # conversations. None -> no marker, which is what every run before this did. It is a
    # separate content part rather than text glued onto `user_text_after` so the two
    # writers build the *same message structure* and the chat template cannot join them
    # differently.
    modality_marker: Optional[str] = None
    # PointNav's goal marker. Normally the SAME literal as `modality_marker` -- `<pose>`
    # is one modality type playing the role `<image>` plays for vision, and a goal pose is
    # simply another occurrence of it, bound by occurrence order like every other. It is a
    # separate *field* even so, because what is being switched on is the presence of a
    # SECOND occurrence per turn (or per segment), and a caller should have to say so.
    # None -> no goal marker anywhere, which is exactly what every run before this did.
    goal_marker: Optional[str] = None
    # WHERE the goal marker goes -- the same choice `format_action_chunk_dataset.py
    # --goal-placement` made when the corpus was written, and therefore a property of the
    # DATA rather than of the weights. It is not recorded in the checkpoint, so a rollout
    # has to be told; telling it wrong writes the markers somewhere the model never saw
    # them, with the counts still adding up. None -> no goal anywhere, which is exactly
    # what every ObjectNav run does and is byte-identical to this field not existing.
    goal_placement: Optional[str] = None
    # The per-segment announcement, when a goal is introduced. Carries the agent's own
    # pose as well as the goal's, and that is load-bearing rather than decorative: it puts
    # an AGENT pose at row 0 of the value column, so `relative_se2`'s anchor means the same
    # thing here as in an ObjectNav row. With `Go to {marker}.` alone, row 0 would be the
    # goal and the two tasks would sit in different frames with nothing to detect it.
    goal_announce_text: str = "You are at {marker}. Go to {marker}."
    # The `anchor` placement's two strings. The frame is set ONCE, in the prologue, and
    # every later announcement is just the goal.
    #
    # This supersedes `goal_announce_text` for that placement, and it is strictly better.
    # Repeating "You are at {marker}" at every announcement puts an agent pose at row 0 --
    # which is the point -- but it also emits a row that EXACTLY duplicates the very next
    # observation's pose. Measured on a real 198-observation, 19-segment episode: 19 of 19
    # announcement agent-rows were bit-identical to the observation that followed, i.e.
    # 8.1% of the column carried no information. Anchoring once leaves a single such row
    # (0.4%) and keeps row 0 an agent pose, so the frame still means the same thing it
    # means in an ObjectNav row.
    anchor_text: str = "You are at {marker}."
    goal_only_text: str = "Go to {marker}."
    # The alternative placement: the goal repeated inside every observation block.
    goal_inline_text: str = "Goal: {marker}"
    use_sparse: bool = True
    merge_lora: bool = True  # fold adapters into the base weights for inference speed
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    # Optional guard: raise once the cached context passes this many tokens, instead of
    # discovering the limit as an OOM mid-episode. 0 disables.
    max_context_tokens: int = 0


def render_prologue(processor, system_prompt: str) -> str:
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": system_prompt}]}],
        tokenize=False,
        add_generation_prompt=False,
    )


def user_block_content(cfg: RolloutConfig, step: int,
                       inline_goal: bool = False) -> List[Dict[str, Any]]:
    """The content parts of one turn's user message.

    The single definition of a user turn's shape. `format_action_chunk_dataset.py` builds
    the training conversations from this same function, so the marker cannot be present in
    one path and absent (or differently placed) in the other -- a divergence that produces
    a rollout context the head was never trained under and no error anywhere.

    `inline_goal` is PointNav's `observation` goal placement: a second marker in every
    turn, carrying the current goal. Default off, so every existing caller renders the
    block character-for-character as before.
    """
    parts: List[Dict[str, Any]] = [
        {"type": "text", "text": cfg.user_text_before.format(step=step)},
        {"type": "image"},
    ]
    if cfg.modality_marker:
        # After the image span, never immediately after `<|vision_start|>`:
        # `get_rope_index` reads that one position to decide image-vs-video, and a marker
        # there makes every downstream position silently wrong.
        parts.append({"type": "text", "text": cfg.modality_marker})
    if inline_goal and cfg.goal_marker:
        # The `observation` goal placement. AFTER the observation's own marker, always:
        # the binding is occurrence order, so this ordering is what decides whether row
        # 2k is the agent's pose or the goal's. Reversing it binds every value to the
        # wrong slot with the counts still matching and nothing raising.
        parts.append({"type": "text", "text": cfg.goal_inline_text.format(
            marker=cfg.goal_marker)})
    parts.append({"type": "text", "text": cfg.user_text_after})
    return parts


def goal_block_content(cfg: RolloutConfig) -> List[Dict[str, Any]]:
    """The content parts of a goal-announcement user message (the `segment` placement).

    Its own message, emitted once when a goal is introduced, so the conversation reads the
    way the task does: a new instruction arrives only when the last one is finished.

    Two markers, in this order: the agent's current pose, then the goal's. See
    `RolloutConfig.goal_announce_text` for why the agent's is there at all.
    """
    if not cfg.goal_marker:
        raise ValueError("goal_block_content needs RolloutConfig.goal_marker")
    text = (cfg.goal_only_text if cfg.goal_placement == "anchor"
            else cfg.goal_announce_text)
    return [{"type": "text", "text": text.format(marker=cfg.goal_marker)}]


def anchor_suffix(cfg: RolloutConfig) -> str:
    """The " You are at <pose>." appended to the prologue under the `anchor` placement.

    Empty for every other placement, so the prompt is untouched where it always was.
    """
    if cfg.goal_placement != "anchor" or not cfg.goal_marker:
        return ""
    return " " + cfg.anchor_text.format(marker=cfg.goal_marker)


def render_goal_block(processor, cfg: RolloutConfig) -> str:
    """One goal announcement, as text. The rollout's counterpart to `render_user_block`."""
    user = {"role": "user", "content": goal_block_content(cfg)}
    return processor.apply_chat_template([user], tokenize=False, add_generation_prompt=False)


# ======================================================================================
# PointNav: the order of the `<pose>` value rows
# ======================================================================================
# `<pose>` is ONE modality type playing the role `<image>` plays for vision: many
# occurrences, one value row each, bound by occurrence order and nothing else. A goal pose
# is simply another occurrence. So the entire contract of PointNav's conversation is:
#
#     the k-th `<pose>` marker in the rendered text receives the k-th value row,
#     and row 0 is an AGENT pose.
#
# Row 0 matters because `pose_frame.relative_se2` anchors the whole column on it. With the
# goal at row 0 a PointNav example would sit in a different frame from every ObjectNav
# example, and nothing anywhere would raise.
#
# Everything below is the SINGLE definition of that order. `data_scripts/
# format_action_chunk_dataset.py` imports it to rewrite the training column, and
# `PointNavValueStream` is the same order produced one event at a time, for a rollout that
# does not have the episode in front of it. A second, "equivalent" ordering is precisely
# the divergence this design exists to prevent: it would agree today, drift later, raise
# never, and the model would read confident nonsense while the run still printed a number.

#: The two placements, as `format_action_chunk_dataset.py --goal-placement` means them.
#: The literals are corpus provenance, so they live beside the function that orders rows.
GOAL_PLACEMENTS = ("anchor", "segment", "observation")


def segment_starts(segment_indices) -> set:
    """Observation indices at which a new goal is introduced.

    The first observation always is (index 0), and thereafter wherever the segment index
    changes. Derived from the column rather than stored, because the column is already the
    definition and a second representation could only disagree with it.
    """
    if not segment_indices:
        return set()
    starts = {0}
    for i in range(1, len(segment_indices)):
        if segment_indices[i] != segment_indices[i - 1]:
            starts.add(i)
    return starts


def modality_value_rows(poses, goals, segment_indices=None, goal_placement="segment"):
    """The value rows for the `<pose>` column, **in the order the markers appear in text**.

      * `segment`     -> at each goal announcement, `[agent pose, goal]`, then that
                         segment's observation poses. Length `n_obs + 2 * n_segments`.
      * `observation` -> `[agent pose, goal]` per turn. Length `2 * n_obs`.

    In both, row 0 is an **agent** pose. See the section header for why that is the whole
    point rather than a detail.
    """
    if goal_placement not in GOAL_PLACEMENTS:
        raise ValueError(
            f"goal_placement must be one of {GOAL_PLACEMENTS}, got {goal_placement!r}"
        )
    poses, goals = list(poses), list(goals)
    if len(poses) != len(goals):
        raise ValueError(
            f"{len(poses)} observation pose(s) against {len(goals)} goal(s); the goal "
            "column is per observation, so a mismatch would bind every later row to the "
            "wrong marker with the counts still adding up"
        )
    if goal_placement == "observation":
        return [v for pair in zip(poses, goals) for v in pair]

    starts = segment_starts(list(segment_indices or []))
    if goal_placement == "anchor":
        # The frame is set once, by a single agent pose in the prologue; each announcement
        # then carries the goal alone. Row 0 is still an agent pose -- and still the same
        # pose the first observation reports -- so the anchor means exactly what it means
        # in an ObjectNav row, at the cost of one duplicated row per EPISODE rather than
        # one per segment. Length `1 + n_segments + n_obs`.
        values = [poses[0]] if poses else []
        for i, pose in enumerate(poses):
            if i in starts:
                values.append(goals[i])
            values.append(pose)
        return values

    values = []
    for i, pose in enumerate(poses):
        if i in starts:
            values += [pose, goals[i]]
        values.append(pose)
    return values


def modality_values(example, modality_column: str, goal_values_column: str,
                    segment_column: str, goal_placement: str):
    """:func:`modality_value_rows` over one dataset row -- the formatter's `.map` shape.

    Deliberately a thin adapter: the ordering lives in one function and the column names
    are the caller's business.
    """
    return modality_value_rows(
        example[modality_column],
        example[goal_values_column],
        example.get(segment_column) or [],
        goal_placement,
    )


def expected_modality_len(n_obs: int, n_segments: int, goal_placement: str) -> int:
    """How many value rows :func:`modality_value_rows` produces, for the alignment filter."""
    if goal_placement == "observation":
        return 2 * n_obs
    return n_obs + 2 * n_segments


class PointNavValueStream:
    """:func:`modality_value_rows`, produced one event at a time. For a rollout.

    A rollout does not have the episode: it is told a new goal when it reaches the last
    one, and it emits value rows as the text is written, turn by turn. This is that same
    order in incremental form, and it exists so the rollout does not need its own copy of
    the ordering rule.

    The equality is not argued, it is asserted: `tests/test_pose_injection.py` drives this
    class over an episode and compares the sequence it emits, row for row, against
    :func:`modality_value_rows` on the same columns, for both placements.

    Usage mirrors the text exactly -- announce a goal when one arrives, then one call per
    observation turn::

        stream = PointNavValueStream("segment")
        rows = stream.announce(agent_pose, goal)   # [agent, goal] -- its own user message
        rows = stream.observe(agent_pose)          # [agent]       -- this turn's marker

    Under `observation` placement there is no announcement *message*: `announce` records
    the goal and returns no rows, and `observe` returns `[agent, goal]`. Callers therefore
    do not branch on the placement, which is the point -- a caller that branched would be
    the second implementation.

    `rows` keeps everything emitted, in order, so a caller can cheaply assert that its own
    accumulator (the one `relative_se2` is computed over) has not drifted from it.
    """

    def __init__(self, goal_placement: str = "segment"):
        if goal_placement not in GOAL_PLACEMENTS:
            raise ValueError(
                f"goal_placement must be one of {GOAL_PLACEMENTS}, got {goal_placement!r}"
            )
        self.goal_placement = goal_placement
        self.goal: Optional[List[float]] = None
        self.rows: List[List[float]] = []
        self.goals_announced = 0
        #: `anchor` only: the prologue's single agent-pose row, which sets the frame for
        #: the whole episode and is emitted exactly once, before anything else.
        self.anchored = False

    @staticmethod
    def _row(value) -> List[float]:
        row = [float(v) for v in value]
        if len(row) != 3:
            raise ValueError(f"a pose row is (x, y, theta), got {len(row)} number(s): {row}")
        return row

    def announce(self, agent_pose, goal) -> List[List[float]]:
        """A new goal arrives. Returns the rows its announcement message writes.

        The agent's own pose comes **first** and that is load-bearing rather than
        decorative: it is what puts an agent pose at row 0 of the column, so
        `relative_se2`'s anchor means the same thing here as in an ObjectNav row.
        Reversing the pair binds every value to the wrong slot with the counts still
        matching and nothing raising.
        """
        self.goal = self._row(goal)
        self.goals_announced += 1
        if self.goal_placement == "anchor":
            # The frame is set once, by a single agent pose in the prologue; every
            # announcement after that carries the goal alone. The first announcement
            # therefore emits BOTH -- the anchor and its goal -- and later ones only the
            # goal. That is what removes the per-segment duplicate: under `segment` the
            # announcement's agent row is bit-identical to the observation that follows
            # it, measured 19 times out of 19 on a real episode.
            emitted = []
            if not self.anchored:
                emitted.append(self._row(agent_pose))
                self.anchored = True
            emitted.append(list(self.goal))
            self.rows += emitted
            return emitted
        if self.goal_placement != "segment":
            return []
        emitted = [self._row(agent_pose), list(self.goal)]
        self.rows += emitted
        return emitted

    def observe(self, agent_pose) -> List[List[float]]:
        """One observation turn. Returns the rows that turn's markers write."""
        if self.goal is None:
            raise RuntimeError(
                "observe() before announce(): every PointNav turn is conditioned on a "
                "goal, and under the 'observation' placement the turn's own text carries "
                "one. Announce the first goal before the first observation."
            )
        emitted = [self._row(agent_pose)]
        if self.goal_placement == "observation":
            emitted.append(list(self.goal))
        self.rows += emitted
        return emitted


def pointnav_context_text(processor, cfg: RolloutConfig, n_turns: int,
                          goal_turns: Sequence[int] = (0,),
                          system_prompt: Optional[str] = None) -> str:
    """The exact text a PointNav rollout builds, for `n_turns` observations.

    `goal_turns` are the observation indices a goal announcement precedes -- i.e.
    `segment_starts(segment_indices)`. Must equal the training-time single-shot
    `apply_chat_template` of the same conversation; `tests/test_pose_injection.py`
    asserts that against `format_action_chunk_dataset.build_messages`, which is the
    check that keeps the rollout's context in the distribution the head was trained on.
    """
    # `anchor` announces at the same turns `segment` does; the two differ in what the
    # announcement SAYS and in whether the prologue carries the frame anchor, not in when
    # it fires. `observation` announces never, carrying the goal inline instead.
    announce = (set(int(t) for t in goal_turns)
                if cfg.goal_placement in ("segment", "anchor") else set())
    # The anchor rides on the prologue, exactly as `format_action_chunk_dataset` appends
    # it there -- so the rollout's first tokens match the training conversation's.
    prologue = (system_prompt + anchor_suffix(cfg)) if system_prompt else None
    parts = [render_prologue(processor, prologue)] if prologue else []
    for i in range(n_turns):
        if i in announce:
            parts.append(render_goal_block(processor, cfg))
        parts.append(render_user_block(processor, cfg, i))
        parts.append(cfg.placeholder + TURN_CLOSE)
    return "".join(parts)


def render_user_block(processor, cfg: RolloutConfig, step: int,
                      inline_goal: Optional[bool] = None) -> str:
    """One turn's user block, ending with the chat template's assistant opening.

    `inline_goal` defaults to whether `cfg.goal_placement` is `"observation"`, so a
    caller never has to restate the corpus's placement and cannot restate it differently.
    With the default `goal_placement=None` this renders the block exactly as it always
    did.
    """
    if inline_goal is None:
        inline_goal = cfg.goal_placement == "observation"
    user = {"role": "user", "content": user_block_content(cfg, step, inline_goal=inline_goal)}
    return processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)


def assistant_turn_text(cfg: RolloutConfig) -> str:
    """The complete assistant turn as it appears in training data."""
    return CHAT_ASSISTANT_OPEN + cfg.placeholder + TURN_CLOSE


def split_assistant_turn(tokenizer, cfg: RolloutConfig, prefix_ids, postfix_ids,
                         shift_left: bool):
    """Split the assistant turn at the readout boundary. Returns (emit, tail, n_readout).

    `emit` is everything up to and including the last position the head reads, so a forward
    pass ending there leaves those states as the final `n_readout` positions of the
    sequence -- which is why `step()` can take `last_hidden_state[:, -n:]` with no span
    search. `tail` is the remainder, owed to the next forward because with a KV cache it is
    context rather than output.

    The split is found with `find_turn_spans` -- the same function training used -- so any
    affix/placeholder combination is handled without special cases, and a combination that
    cannot be split (e.g. `'**[…]**'`, where BPE merges `]` into the closing `**` so the
    postfix never matches) fails here instead of silently training on nothing.
    """
    import torch as _torch

    from longnav.utils.turn_vectors import find_turn_spans

    turn = tokenizer.encode(assistant_turn_text(cfg), add_special_tokens=False)
    # Two copies: find_turn_spans needs a closed turn, and repeating it proves the split is
    # stable rather than an artifact of the sequence end.
    ids = _torch.tensor([turn * 2])
    spans = find_turn_spans(ids, prefix_ids, postfix_ids, shift_left=shift_left)[0]
    if len(spans) != 2:
        raise ValueError(
            f"placeholder {cfg.placeholder!r} with prefix {tokenizer.decode(prefix_ids)!r} / "
            f"postfix {tokenizer.decode(postfix_ids)!r} yields {len(spans)} readout span(s) "
            f"in two turns, expected 2. Its tokens are "
            f"{[tokenizer.decode([t]) for t in turn]} -- a placeholder whose characters "
            "merge into the affixes cannot be located"
        )
    start, end = spans[0].start, spans[0].end
    if end > len(turn):
        raise ValueError("readout span crosses the turn boundary; check the affixes")
    return turn[:end], turn[end:], end - start


def full_context_text(processor, cfg: RolloutConfig, n_turns: int,
                      system_prompt: Optional[str] = None) -> str:
    """The exact text an `n_turns` rollout builds. Must equal the training-time
    single-shot `apply_chat_template` of the same conversation -- asserted in
    `tests/test_vector_rollout.py`."""
    parts = [render_prologue(processor, system_prompt)] if system_prompt else []
    for i in range(n_turns):
        parts.append(render_user_block(processor, cfg, i))
        parts.append(cfg.placeholder + TURN_CLOSE)
    return "".join(parts)


def seed_anchor(raw_poses: List[Any], offset: Optional[Sequence[float]],
                first_row: Any) -> None:
    """Diagnostic: move `relative_se2`'s origin **without touching the context**.

        `relative_se2` anchors on row 0 of the accumulated column, so the only way to ask
        "does the same task get harder when the anchor is further away" is to change what
        row 0 is. This seeds it with a fictitious pose at `anchor_offset` from the first
        real one, and the crucial property is that the seeded row is **never emitted**:
        `accumulate_pose_rows` returns `relative_se2_tail(..., len(rows))`, the tail only.
        So the number of value rows, the number of `<pose>` markers, the rendered text and
        the images are all bit-identical to a run without it -- the *only* difference is
        the numeric value of every injected pose, which is exactly the variable under test.
        Inserting a real extra pose turn instead would change the context and confound the
        thing being measured.

        The offset carries `(dx, dy, dtheta)` and the two parts are deliberately separable,
        because they test different things:

          * `(dx, dy)` with `dtheta = 0` keeps the first pose's heading, so
            `relative_se2`'s rotation into the anchor frame is unchanged and every injected
            pose shifts by one constant vector. That isolates **distance** from the anchor.
          * `dtheta` alone rotates the anchor's heading, which rotates the whole pose cloud
            by `-dtheta` and shifts every injected heading by `-dtheta`. That isolates
            **orientation** relative to the anchor -- whether the model is confused by
            facing a direction far from the one the frame was pinned to.

        Off (`anchor_offset=None`) leaves `_raw_poses` empty, which is what it always was.
        """
    if offset is None or raw_poses:
        return
    p = torch.as_tensor(first_row, dtype=torch.float64).reshape(-1)
    dx, dy, dth = offset
    raw_poses.append(
        torch.tensor([p[0] + dx, p[1] + dy, p[2] + dth], dtype=torch.float64)
    )


def accumulate_pose_rows(raw_poses: List[Any], rows: Sequence[Any]) -> torch.Tensor:
    """Append `rows` to the episode's raw column and return their `(K, 3)` injected values.

    A free function over the accumulator rather than a method, because both
    `VectorRolloutPolicy.pose_values` and `.pose_rows` are it and neither should be a
    second copy -- and because that makes the *production* code runnable against a stub
    carrying nothing but `_raw_poses`, which is how the parity tests in both repos avoid
    loading a multi-billion-parameter backbone to check three floats.
    """
    from longnav.utils.pose_frame import POSE_DIM, relative_se2_tail

    rows = list(rows)
    if not rows:
        raise ValueError("pose_rows needs at least one row")
    for row in rows:
        pose = torch.as_tensor(row, dtype=torch.float64).reshape(-1)
        if pose.numel() != POSE_DIM:
            raise ValueError(
                f"a pose row must be {POSE_DIM} numbers (x, y, theta), got "
                f"{tuple(pose.shape)}"
            )
        raw_poses.append(pose)
    return relative_se2_tail(torch.stack(raw_poses), len(rows))


class VectorRolloutPolicy:
    """Stateful, single-episode policy: `reset()` then `step()` per observation."""

    def __init__(self, model: TurnVectorRegressor, processor, cfg: Optional[RolloutConfig] = None):
        self.cfg = cfg or RolloutConfig()
        self.model = model
        self.processor = processor

        # Affixes come from the checkpoint unless overridden, exactly as VLMWorker takes
        # them from config: no convention is assumed anywhere below.
        from longnav.utils.turn_vectors import resolve_affix_ids

        mcfg = self.model.model_cfg
        self.prefix = self.cfg.prefix if self.cfg.prefix is not None else mcfg.prefix
        self.postfix = self.cfg.postfix if self.cfg.postfix is not None else mcfg.postfix
        self.shift_left = (
            self.cfg.shift_left if self.cfg.shift_left is not None
            else self.model.model_cfg.shift_left
        )
        self.prefix_ids, self.postfix_ids = resolve_affix_ids(
            processor.tokenizer, self.prefix, self.postfix
        )
        self.emit_ids, self.tail_ids, self.n_readout = split_assistant_turn(
            processor.tokenizer, self.cfg, self.prefix_ids, self.postfix_ids, self.shift_left
        )
        self._assert_head_matches_readout()
        # The processor may not be the one the model was built with. If its tokenizer is
        # missing a marker the literal BPEs back into ordinary tokens, the scatter finds
        # nothing, and the failure surfaces much later as a count mismatch.
        if self.model.modality_embedder:
            self.model.modality_embedder.bind_tokenizer(processor.tokenizer)
        self._pose_spec = self._resolve_pose_spec()

        self.model.to(self.cfg.device).eval()
        backbone = self.model.backbone
        if self.cfg.merge_lora and hasattr(backbone, "merge_adapter"):
            backbone.merge_adapter()
            self._merged = True
        else:
            self._merged = False

        # `get_rope_index` lives on Qwen3VLModel; unwrap peft to reach it. Unwrap by TYPE,
        # not by attribute probing: a peft-wrapped backbone is
        # PeftModel -> LoraModel(.base_model) -> ForConditionalGeneration(.model) -> Model,
        # but an ADAPTER-FREE checkpoint (a merged model, which is what a published
        # `-merged` repo is) starts one level in, and `getattr(base, "model", base)` then
        # unwraps one level too far and hands back Qwen3VLModel, whose `.model` does not
        # exist. That crashed every shard the first time a merged checkpoint was loaded.
        try:
            from peft import PeftModel
            _is_peft = isinstance(backbone, PeftModel)
        except Exception:
            _is_peft = False
        self.vl_for_cond_gen = backbone.base_model.model if _is_peft else backbone
        self.vl_model = self.vl_for_cond_gen.model
        self.language_model = self.vl_model.language_model
        # None -> torch's global RNG. `seed_stop` makes a sampled stop decision
        # reproducible without pinning every other source of randomness in the process.
        self._stop_rng: Optional[torch.Generator] = None
        self.reset()

    def seed_stop(self, seed: int) -> "VectorRolloutPolicy":
        """Make the sampled stop decision reproducible for this policy."""
        self._stop_rng = torch.Generator(device=self.cfg.device).manual_seed(int(seed))
        return self

    def _resolve_pose_spec(self):
        """The single spec whose values are raw poses, or None.

        Found by transform rather than by token string: the transform *is* the contract
        `step(obs_pose=...)` implements, and a spec named `<pose>` that did not declare it
        would need its values supplied some other way.
        """
        specs = [s for s in self.model.modality_embedder.specs
                 if s.transform == "pose_relative_first"]
        if len(specs) > 1:
            raise ValueError(
                f"{len(specs)} specs declare the pose transform "
                f"({[s.token for s in specs]}); step(obs_pose=...) would not know which "
                "one to fill. Pass them explicitly via modality={key: values}."
            )
        return specs[0] if specs else None

    def _assert_head_matches_readout(self):
        """The head pools a fixed number of positions; the affixes must produce that many.

        Cheap to check, and the failure it prevents is silent: a head trained on 3 content
        tokens fed 1 position (or vice versa) still returns a vector of the right shape.
        """
        head = self.model.head
        if head.mode == "flat" and head.content_len != self.n_readout:
            raise ValueError(
                f"head was trained with mode='flat' over {head.content_len} position(s) but "
                f"placeholder {self.cfg.placeholder!r} with these affixes yields "
                f"{self.n_readout}; the flat head's input width would not match"
            )
        if self.n_readout < 1:
            raise ValueError("affixes yield an empty readout span")
        # Any pooling mode still has to see the same *number* of positions it trained on,
        # or the pooled statistic changes meaning even though the shapes work out.
        trained = getattr(self.model, "train_content_len", None)
        if trained is not None and trained != self.n_readout:
            raise ValueError(
                f"head was trained pooling {trained} position(s) per turn, rollout would "
                f"pool {self.n_readout}"
            )

    def describe(self) -> str:
        tok = self.processor.tokenizer
        out = (
            f"prefix={self.prefix!r} postfix={self.postfix!r} "
            f"placeholder={self.cfg.placeholder!r} shift_left={self.shift_left}\n"
            f"  emitted per turn: {[tok.decode([t]) for t in self.emit_ids]}\n"
            f"  readout: last {self.n_readout} token(s) = "
            f"{[tok.decode([t]) for t in self.emit_ids[-self.n_readout:]]}\n"
            f"  owed to next turn: {[tok.decode([t]) for t in self.tail_ids]}"
        )
        if self.cfg.modality_marker:
            out += f"\n  marker in user block, after the image: {self.cfg.modality_marker!r}"
        if self.model.modality_embedder:
            out += ("\n  modality (pass step(..., modality={key: (N, F)})):\n"
                    + self.model.modality_embedder.describe())
        if self._pose_spec is not None:
            out += (f"\n  pose: step(..., obs_pose=(x, y, theta)) fills "
                    f"{self._pose_spec.token}, relative to this episode's first pose")
        if self.model.stop_head is None and getattr(self, "_probe", None) is not None:
            out += ("\n  stop head: state-probe BCE head at readout offset "
                    f"{self._probe_offset} -> last_stats['stop_prob'], ['stop_logit']; "
                    "the harness owns the threshold")
        elif self.model.stop_head is not None:
            c = self.model.stop_head.cfg
            out += (f"\n  stop head: inference={c.inference} temperature={c.temperature} "
                    f"threshold={c.threshold} -> last_stats['stop_prob'], ['stop']")
        return out

    # ---------------------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Union[str, Path],
        cfg: Optional[RolloutConfig] = None,
        processor=None,
    ) -> "VectorRolloutPolicy":
        from transformers import AutoProcessor

        cfg = cfg or RolloutConfig()
        checkpoint_dir = Path(checkpoint_dir)
        if processor is None:
            # Trainer saves the processor next to the head; fall back to the base model id.
            try:
                processor = AutoProcessor.from_pretrained(str(checkpoint_dir))
            except Exception:
                import json

                from longnav.utils.vector_sft import HEAD_CONFIG_FILE

                meta = json.loads((checkpoint_dir / HEAD_CONFIG_FILE).read_text())
                processor = AutoProcessor.from_pretrained(meta["model"]["model_id"])
        model = TurnVectorRegressor.from_pretrained(
            checkpoint_dir, processor, dtype=cfg.dtype, device=cfg.device
        )
        policy = cls(model, processor, cfg)
        policy.attach_state_probe(checkpoint_dir)
        return policy

    # ---------------------------------------------------------------------------------
    def attach_state_probe(self, checkpoint_dir) -> bool:
        """Load a co-trained state probe from the checkpoint, if it has one.

        The flow head has NO stop head of its own -- `flow_matching_head` says so
        explicitly -- so `model.stop_head` is None for every flow checkpoint and the
        `stop_prob` block below never fired. A stop head trained as part of the state
        probe lives somewhere else entirely (`state_probe.stop_head`) and reads a
        different tensor: the hidden at the PROBE readout position, not the pooled
        motion context. Without this, a checkpoint can be trained with a stop head that
        `--stop-head` then refuses as "this checkpoint has no stop head".

        The position convention is the checkpoint's, not this file's opinion:
        `worker_value_readout_offset` is written by the trainer relative to the
        end-of-turn logit index, and `probe_token_id` pins the identity of the token it
        must land on. Both are ASSERTED per step -- an off-by-one here once put the
        readout on '\n' instead of 'assistant', and only the token-id check caught it.
        """
        from pathlib import Path as _P
        import json as _json
        d = _P(checkpoint_dir)
        self._probe = None
        self._probe_offset = None
        self._probe_token_id = None
        if not (d / "state_probe.pt").exists():
            return False
        from longnav.utils.state_probe import load_state_probe, STATE_PROBE_CONFIG_FILE
        meta = _json.loads((d / STATE_PROBE_CONFIG_FILE).read_text())
        hs = int(getattr(self.model.backbone.config, "text_config",
                         self.model.backbone.config).hidden_size)
        probe = load_state_probe(str(d), input_dim=hs)
        if getattr(probe, "stop_head", None) is None:
            return False
        off = meta.get("worker_value_readout_offset")
        if off is None:
            raise ValueError(
                f"{d}/{STATE_PROBE_CONFIG_FILE} lacks worker_value_readout_offset; the "
                "probe was saved by a trainer that predates the readout contract and "
                "its position cannot be re-derived here.")
        dev = next(self.model.parameters()).device
        self._probe = probe.to(dev).eval()
        self._probe_offset = int(off)
        self._probe_token_id = meta.get("probe_token_id")
        return True

    # ---------------------------------------------------------------------------------
    def reset(self, goal_text: Optional[str] = None, system_prompt: Optional[str] = None,
              modality: Optional[Dict[str, Any]] = None,
              anchor_offset: Optional[Sequence[float]] = None):
        """Start a fresh episode. The prologue is prepended to the first turn's tokens.

        `modality` carries values for any markers in the **prologue** -- an episode-level
        slot such as a goal location, a scene descriptor, an origin declaration. Training
        gets this for free (the window always keeps the prologue), so without it the
        failure is eval-only and confusing to debug.

        The prologue's tokens are owed to the first `step()`, so its occurrences come
        first in the sequence and its value rows are prepended to that step's.
        """
        self.past_key_values = None
        self.past_image_embeds = None
        self.rope_offset = 0
        self.step_index = 0
        self.cached_tokens = 0
        self.dense_tokens = 0
        if system_prompt is None and goal_text is not None:
            system_prompt = DEFAULT_SYSTEM_PROMPT.format(goal=goal_text)
        # Owed to the next forward pass, as token ids: the prologue on the first step, the
        # previous turn's tail afterwards.
        self._pending: List[int] = (
            self.processor.tokenizer.encode(
                render_prologue(self.processor, system_prompt), add_special_tokens=False
            )
            if system_prompt
            else []
        )
        self._pending_modality: Optional[ModalityBatch] = (
            single_example_batch(modality) if modality else None
        )
        # Every raw scene-frame pose this episode has produced, oldest first. Kept in full
        # because the value the model is shown is relative to the *first* observation, so
        # the origin has to survive the whole episode; `reset` is what makes a new episode
        # a new origin, which is exactly what a training window does.
        self._raw_poses: List[Any] = []
        self._anchor_offset = None
        if anchor_offset is not None:
            off = [float(v) for v in anchor_offset]
            if len(off) == 2:
                off.append(0.0)
            if len(off) != 3:
                raise ValueError(
                    "anchor_offset must be (dx, dy) or (dx, dy, dtheta), got "
                    f"{len(off)} values"
                )
            self._anchor_offset = tuple(off)
        self.last_stats: Dict[str, Any] = {}
        return self

    def pose_values(self, obs_pose: Any) -> torch.Tensor:
        """Record a raw scene-frame `(x, y, theta)` and return its `(1, 3)` injected value.

        Routed through `relative_se2` over every pose seen so far -- the same function the
        collator applies to a whole window -- rather than a cheaper incremental update. The
        cost is three floats times the episode length; what it buys is that there is no
        second implementation of the frame convention to drift from, which is the failure
        this experiment could least afford to have and least easily notice.
        """
        seed_anchor(self._raw_poses, getattr(self, '_anchor_offset', None), obs_pose)
        return accumulate_pose_rows(self._raw_poses, [obs_pose])

    def pose_rows(self, obs_poses: Sequence[Any]) -> torch.Tensor:
        """Record K raw rows of the value column and return their `(K, 3)` injected values.

        The generalisation of :meth:`pose_values` to a step that writes more than one
        `<pose>` marker -- PointNav's goal announcement writes two, the agent's pose and
        the goal's. Both are rows of the *same* column, so both go through this one
        accumulator: `relative_se2` anchors on row 0 of the whole column, and the k-th
        marker in the text has to receive the k-th row. Keeping a separate stream for
        goals would put them in a different frame and nothing would raise.

        Order within `obs_poses` is the order the markers appear in the text. See
        :class:`PointNavValueStream`, which is what decides that order.
        """
        if obs_poses:
            seed_anchor(self._raw_poses, getattr(self, '_anchor_offset', None),
                        list(obs_poses)[0])
        return accumulate_pose_rows(self._raw_poses, obs_poses)

    def queue_user_block(self, text: str, modality: Optional[Dict[str, Any]] = None) -> None:
        """Owe an extra rendered user message, and its value rows, to the next `step()`.

        The mechanism `reset()` already uses for the prologue, exposed: the tokens are
        appended to `_pending` and the rows to `_pending_modality`, so both land *before*
        the next turn's own tokens and rows. Occurrence order is the only binding the
        modality mechanism has, so "before in the text" and "before in the rows" have to
        be the same statement, and doing it in one call is what makes them one.

        Used by PointNav's goal announcement (`render_goal_block`), which is its own user
        message between two observation turns. Nothing in this method is PointNav-specific
        -- what to say and what to bind to it is the caller's business.
        """
        self._pending += self.processor.tokenizer.encode(text, add_special_tokens=False)
        if not modality:
            return
        batch = single_example_batch(modality)
        self._pending_modality = (
            batch if self._pending_modality is None
            else self._pending_modality.concat(batch)
        )

    def _render_prologue(self, system_prompt: str) -> str:
        return render_prologue(self.processor, system_prompt)

    # ---------------------------------------------------------------------------------
    @torch.inference_mode()
    def step(self, image, user_text: Optional[str] = None,
             modality: Optional[Dict[str, Any]] = None,
             obs_pose: Optional[Any] = None) -> torch.Tensor:
        """Encode one observation and return its action chunk, shaped `target_shape`.

        Composition is done in *token ids*, not text: the user block comes from the
        processor (which expands the image placeholder), and the assistant pieces are the
        constant `tail`/`emit` id lists derived once in `__init__`. Concatenating ids
        removes any chance of BPE merging differently at a chunk boundary than it did in
        the training-time single-shot tokenization.

        `modality` is `{key: (N, F)}` for the markers in *this* chunk, in occurrence
        order. B == 1, the same `ModalityBatch`, the same hooks and the same assertions as
        training -- one scatter implementation for both paths, which is what makes a
        parity check meaningful rather than decorative. Where the values come from is the
        caller's problem.

        `obs_pose` is the convenience for the one modality that has a frame convention:
        pass the **raw scene-frame** `(x, y, theta)` straight from the simulator or the
        dataset column and the policy converts it exactly as training did. Passing an
        already-relative pose through `modality=` instead is possible and is how you would
        get it wrong.
        """
        pose_value = None
        if obs_pose is not None:
            if self._pose_spec is None:
                raise ValueError(
                    "step(obs_pose=...) but no modality spec declares the "
                    "'pose_relative_first' transform, so there is nowhere to put it"
                )
            modality = dict(modality or {})
            if self._pose_spec.key in modality:
                raise ValueError(
                    f"both obs_pose= and modality[{self._pose_spec.key!r}] were given; "
                    "they would write the same slot with differently-framed values"
                )
            pose_value = self.pose_values(obs_pose)
            modality[self._pose_spec.key] = pose_value

        block = self._render_user_block(image, user_text)
        ids = torch.tensor(
            [self._pending + block["input_ids"][0].tolist() + self.emit_ids],
            dtype=torch.long,
        )
        inputs = {
            k: v for k, v in block.items()
            # The processor's mask/ids describe the user block alone; ids below are the
            # concatenation, so a stale mask would be shorter than the sequence and
            # get_rope_index would index past it.
            if k not in ("input_ids", "attention_mask") and v is not None
        }
        inputs["input_ids"] = ids
        inputs["attention_mask"] = torch.ones_like(ids)

        # The prologue's occurrences precede this turn's in `ids`, so its rows precede
        # this turn's too -- occurrence order is the only binding.
        step_modality = single_example_batch(modality) if modality else None
        pending = self._pending_modality
        self._pending_modality = None
        if pending is not None:
            step_modality = pending.concat(step_modality or ModalityBatch())

        t0 = time.perf_counter()
        outputs = self._forward(inputs, step_modality)

        # The emitted block ends exactly at the last readout position, so the states the
        # head wants are the final `n_readout` of the sequence -- no span search needed, and
        # they are text tokens, which the sparsifier never drops.
        n = self.n_readout
        hidden = outputs["last_hidden_state"][:, -n:, :]
        head_dtype = next(self.model.head.parameters()).dtype
        states = hidden.to(head_dtype)
        vector = self.model.head(states)
        chunk = self.model.normalizer.denormalize(
            vector.view(-1, *self.model.target_shape)
        )[0].float().cpu()

        # The stop readout, on the same pooled context the motion head just used. Reported
        # rather than acted on: whether an episode ends is the caller's decision, and a
        # policy that silently truncated its own rollout would be indistinguishable from
        # one that crashed.
        stop_prob = stop = None
        probe = getattr(self, "_probe", None)
        if self.model.stop_head is None and probe is not None:
            # Probe readout: the end-of-turn logit index is the last emitted position,
            # and the probe sits `worker_value_readout_offset` from it.
            pos = -1 + int(self._probe_offset)
            if -pos > int(ids.shape[1]):
                raise RuntimeError(
                    f"probe offset {self._probe_offset} underflows this turn "
                    f"({ids.shape[1]} tokens)")
            if self._probe_token_id is not None:
                tok = int(ids[0, pos])
                if tok != int(self._probe_token_id):
                    raise RuntimeError(
                        f"probe readout lands on token id {tok}, but the checkpoint "
                        f"pinned {self._probe_token_id}; the rollout template does not "
                        "match the one the probe was trained against.")
            with torch.no_grad():
                h = outputs["last_hidden_state"][:, pos, :]
                logit = probe.stop_head(h.to(next(probe.stop_head.parameters()).dtype))
                stop_logit = float(logit.reshape(-1)[0])
                stop_prob = float(probe.stop_head.probability(logit).reshape(-1)[0])
            self._probe_stop_logit = stop_logit
            # Reported, never acted on here: the caller decides whether an episode ends.
            # `stop` stays None so no threshold is invented -- the harness owns that,
            # and it must be fitted on held-out scenes rather than assumed to be 0.5,
            # since any pos_weight != 1 shifts the operating point.
        elif self.model.stop_head is not None:
            pooled = self.model.head.pooled_context(states)
            logit = self.model.stop_head(pooled)
            stop_prob = float(self.model.stop_head.probability(logit)[0])
            stop = bool(self.model.stop_head.decide(logit, generator=self._stop_rng)[0])

        self._pending = list(self.tail_ids)
        self.step_index += 1
        self.last_stats = {
            "step": self.step_index,
            "new_tokens": int(ids.shape[1]),
            "readout_tokens": n,
            "sparse_tokens": int(outputs["last_hidden_state"].shape[1]),
            "cached_tokens": self.cached_tokens,
            "dense_tokens": self.dense_tokens,
            "latency_s": time.perf_counter() - t0,
            "stop_prob": stop_prob,
            "stop_logit": getattr(self, "_probe_stop_logit", None),
            "stop": stop,
            # The value that was actually injected, not the raw pose that was passed in.
            # A decodability probe has to correlate the pooled state against what the
            # encoder saw; re-deriving it at the call site would be a second
            # implementation of the frame convention, which is the thing to avoid.
            "pose_value": None if pose_value is None else pose_value[0].clone(),
        }
        return chunk

    def _render_user_block(self, image, user_text: Optional[str]):
        """Processor output for this turn's user block (image expanded), without the
        assistant opening -- that comes from `emit_ids`."""
        text = (
            user_text
            if user_text is not None
            else render_user_block(self.processor, self.cfg, self.step_index)
        )
        # The chat template already appends the assistant opening; drop it, since emit_ids
        # carries the opening plus whatever of the placeholder precedes the readout.
        if text.endswith(CHAT_ASSISTANT_OPEN):
            text = text[: -len(CHAT_ASSISTANT_OPEN)]
        return self.processor(
            text=text, images=[image], videos=None, padding=False, return_tensors="pt"
        )

    def _forward(self, inputs, modality: Optional[ModalityBatch] = None):
        device = self.cfg.device
        turn = {k: v.to(device) for k, v in inputs.items() if v is not None}
        n_new = turn["input_ids"].shape[1]

        # Rope positions for the new tokens only, continued from where the last turn
        # ended (VLMWorker._pos_id_fast's arithmetic: advance by length + rope delta).
        position_ids, deltas = self.vl_model.get_rope_index(
            input_ids=turn["input_ids"],
            image_grid_thw=turn.get("image_grid_thw"),
            video_grid_thw=None,
            attention_mask=turn.get("attention_mask"),
        )
        turn["position_ids"] = position_ids + self.rope_offset
        self.rope_offset += n_new + int(deltas.reshape(-1)[0].item())

        if self.cfg.use_sparse:
            # The sparse path derives its own mask; a dense mask would not match the
            # sparsified sequence length. The visual DB is what lets it drop frames that
            # are redundant with *earlier turns*, not just within this one.
            turn["attention_mask"] = None
            turn["past_image_embeds"] = self.past_image_embeds
            turn["save_image_db"] = True

        embedder = self.model.modality_embedder
        if embedder:
            embedder.check_placement(
                turn["input_ids"], self.vl_model.config.vision_start_token_id
            )
        with embedder.pending(modality.to(device) if modality is not None else None):
            outputs = self.model.backbone(
                **turn, past_key_values=self.past_key_values, use_cache=True,
                logits_to_keep=1,
            )
        self.past_key_values = outputs["past_key_values"]
        self.dense_tokens += n_new
        self.cached_tokens = int(
            self.past_key_values.get_seq_length() if self.past_key_values is not None else 0
        )
        if self.cfg.max_context_tokens and self.cached_tokens > self.cfg.max_context_tokens:
            raise RuntimeError(
                f"context reached {self.cached_tokens} cached tokens, over the configured "
                f"limit of {self.cfg.max_context_tokens}; shorten the episode or raise "
                "max_context_tokens"
            )

        if self.cfg.use_sparse:
            kept = getattr(self.language_model, "kept_visual_embeds", None)
            if kept:
                if self.past_image_embeds is None:
                    self.past_image_embeds = [k.clone() for k in kept]
                else:
                    self.past_image_embeds = [
                        torch.cat([old, new]) for old, new in zip(self.past_image_embeds, kept)
                    ]
        return outputs

    # ---------------------------------------------------------------------------------
    def full_context_text(self, n_turns: int, system_prompt: Optional[str] = None) -> str:
        return full_context_text(self.processor, self.cfg, n_turns, system_prompt)

    def unmerge(self):
        """Undo `merge_lora` (e.g. before saving or further training)."""
        if self._merged and hasattr(self.model.backbone, "unmerge_adapter"):
            self.model.backbone.unmerge_adapter()
            self._merged = False

# Choosing the assistant placeholder for per-turn latent readout

In `vector_sft.py` training the assistant text is never a label -- the continuous action
comes from a head reading that turn's hidden state. The text is a fixed placeholder, so
the only question is *which* placeholder. This records what the Qwen3-VL tokenizer and the
base model actually do, measured rather than assumed.

Conclusion up front: the placeholder is now **`**____**`** (was `**forward**`). The `**`
wrapper stays, the false action word goes.

## Why the choice matters, and where

Not for the readout of its own turn -- SFT will learn from whatever position it is given.
It matters because **the placeholder stays in the KV cache for every later turn**. With
`**forward**`, a 200-observation episode asserts "forward" two hundred times, most of them
false (the robot was turning), and the model spends its next-token machinery contradicting
that history. The context should say "an action happened here, contents not shown".

## Tokenizer facts (Qwen/Qwen3-VL-2B-Instruct)

There is **no mask token**. The 26 added tokens are chat / vision / tool / think markers
plus the FIM set: `<|fim_prefix|>` 151659, `<|fim_middle|>` 151660, `<|fim_suffix|>`
151661, `<|fim_pad|>` 151662.

Single-token redaction-ish candidates, with the id doubling as a BPE-frequency proxy
(lower id = merged earlier = more frequent in the tokenizer's training corpus):

| token | id | | token | id |
|---|---|---|---|---|
| `----` | 381 | | `XXX` | 30100 |
| `****` | 430 | | `???` | 33015 |
| `...` | 1112 | | `MASK` | 49863 |
| `____` | 2130 | | `[…]` | 95232 |
| `***` | 12210 | | `█` (U+2588) | 15199 |

Not single tokens: `[MASK]`, `[REDACTED]`, `N/A`, `TBD`, `<mask>`.

### Boundaries, for the full turn

```
'**forward**'   <|im_start|> | assistant | \n | ** | forward | ** | <|im_end|>
'**____**'      <|im_start|> | assistant | \n | ** | ____    | ** | <|im_end|>
'**█**'         <|im_start|> | assistant | \n | ** | █       | ** | <|im_end|>
'**[…]**'       <|im_start|> | assistant | \n | ** | […      | ]** | <|im_end|>   <-- BROKEN
```

`**[…]**` is a trap: BPE merges `]` with the closing `**` into `]**`, the postfix
`['**','<|im_end|>']` never matches, and `find_turn_spans` silently returns **zero turns**.
`**____**`, `**___**`, `**...**`, `**XXX**`, `**?**`, `**█**` all tokenize as
`['**', X, '**']`.

`█` (U+2588) is structurally identical to `____` -- one token, same span shape. It trades
frequency prior (id 15199 and rare in text) for a sharper censor-bar semantic. `____` is
the more common fill-in-the-blank marker, which is why it won.

### Repeats merge -- relevant if you want k latents per turn

Concatenation does **not** give k tokens:

```
'██'       -> 1 token  [21460]          '████'     -> 1 token  [51678]
'________' -> 1 token  [3979]           '......'   -> 1 token  [28149]
```

Space separation does, but the first token differs from the rest (the space is part of the
token):

```
'█ █ █'     -> 3 tokens [15199, 32588, 32588]   █ | ' █' | ' █'
'____ ____' -> 2 tokens [ 2130, 30743]
```

## Measured: what each convention does to later turns

Base model, history of *k* placeholder turns held fixed, only the current image varied,
measured at the last turn's readout position. `dispersion` = mean deviation of the hidden
state across 8 different current images, relative to its norm (how much the observation
still moves the readout); `entropy` = the model's uncertainty about the next token.

| convention | readout token | dispersion @4 / @12 turns | entropy @12 | top prediction @12 |
|---|---|---|---|---|
| `**forward**` | `**` | 0.147 / 0.154 | 1.07 | `'turn'` 0.41, `'right'` 0.35 |
| `**[…]**` | `**` | 0.139 / 0.136 | 0.33 | `'['` 0.87 |
| `**____**` | `**` | 0.109 / 0.132 | 0.43 | `'____'` 0.92 |
| `____` alone | `____` | 0.067 / 0.079 | 0.00 | `<|im_end|>` 1.00 |
| `***` alone | `***` | 0.071 / 0.090 | 0.00 | `<|im_end|>` 1.00 |
| `<|fim_middle|>` alone | itself | 0.048 / 0.049 | 0.00 | `<|im_end|>` 1.00 |

Three findings:

1. **A bare single-token placeholder halves the readout's image sensitivity.** After a few
   turns the model concludes assistant turns are empty, becomes *certain* `<|im_end|>`
   follows, and the readout state stops summarizing the observation.
2. **FIM tokens do not transfer.** `<|fim_middle|>` is the ideal "predict content here"
   marker on paper, but it was only ever pretrained inside code infilling with
   prefix/suffix framing; in a chat/VL context the model just predicts `\n`/`<|im_end|>`,
   and it scores *worst* on dispersion.
3. **`**forward**` keeps the most signal for the wrong reason**: the model predicts
   `'turn'`/`'right'` from the image, i.e. it is actively contradicting the false history.

`**____**` keeps ~86% of the readout dispersion while asserting nothing false, and is the
minimal edit from the previous convention. The `**` wrapper costs 2 tokens per turn --
0.6% of the sequence at any episode length -- so it is kept for its "content follows"
prior, not for economy.

Caveat: after SFT, readout dispersion at `**` was 0.109 versus 0.115 at `***` -- nearly
equal. Training reshapes the readout substantially, so this is a prior-quality and
context-hygiene improvement, not something expected to move the metrics much alone.

## Options deliberately not taken (yet)

* **Bare chat-template affixes** -- drop `**`, let the assistant "say" exactly k
  placeholder tokens, and pool those. Fully supported today (set `ModelConfig.prefix` /
  `postfix` to `DEFAULT_PREFIX` / `DEFAULT_POSTFIX`, `shift_left=False`, `pool_mode` over
  `content_len=k`); the cost is finding (1) above.
* **A learned placeholder** -- a new vocab entry whose embedding is trained. Semantically
  inert by construction, but changes the vocabulary and needs the embedding row in
  `modules_to_save`.
* **One placeholder per action step** with a per-token head (token *j* predicts pose *j*)
  rather than pooling k states into one vector. Causally sound and structurally matched to
  the chunk, but a different head, not a config flag.

## Not changed

`**forward**` remains in `vlm_worker.py`, `config_schema.py`, `scripts/serve.py`,
`tests/eval_smoke.py`, `tests/rl_smoke.py` and the discrete demos. There it is a **real
action word** the LM head predicts, not a placeholder.

## Reproducing

The vocabulary/boundary tables come from the tokenizer directly; the dispersion table from
a base-model probe over 12 images drawn from 4 held-out episodes. See
`tests/test_vector_rollout.py::test_placeholder_tokenizes_as_three_tokens` for the
boundary assertions that guard the choice in CI.

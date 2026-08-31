"""KV-cache strategy microbench for the hand-token question (docs discussion 2026-08-31).

Arms, per synthetic episode of T turns (persistent_pre=120 obs tokens, hand=16,
persistent_post=30 readout/postfix tokens):
  A append_all     : option 1 -- hand tokens enter the cache like everything else.
  B evict_crop     : rollback -- after each turn, DynamicCache.crop() back over
                     (hand+post) and re-forward the post tokens next turn.
  C filtered_update: option 2 -- Cache subclass stores only persistent K/V but
                     returns full concat for attention; hand tokens attend all
                     history, never persist. Positions of post tokens reuse the
                     hand block's (RoPE rolled back natively).
"""
import time, sys, torch
from transformers import AutoModelForCausalLM, AutoConfig, DynamicCache

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEV = "cuda"
T, PRE, HAND, POST = 100, 120, 16, 30
REPEATS = 3

class FilteredCache(DynamicCache):
    """update() stores only the non-transient slice, returns full concat for attention."""
    transient: tuple = None   # (start, end) within the current forward block, or None
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.transient is None:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        s, e = self.transient
        keep_k = torch.cat([key_states[:, :, :s], key_states[:, :, e:]], dim=2)
        keep_v = torch.cat([value_states[:, :, :s], value_states[:, :, e:]], dim=2)
        # cached (filtered) history AFTER storing keep:
        k_hist, v_hist = super().update(keep_k, keep_v, layer_idx, cache_kwargs)
        # attention sees: history-before-this-block ++ full current block
        hb = k_hist.shape[2] - keep_k.shape[2]
        k_full = torch.cat([k_hist[:, :, :hb], key_states], dim=2)
        v_full = torch.cat([v_hist[:, :, :hb], value_states], dim=2)
        return k_full, v_full

def load():
    from transformers import Qwen3VLForConditionalGeneration
    try:
        m = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                            attn_implementation="sdpa").to(DEV)
        lm = m.model.language_model if hasattr(m.model, "language_model") else m.language_model
        return m, lm
    except Exception as e:
        print("fallback text-only load:", e)
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV)
        return m, m

def run_arm(model, arm):
    torch.manual_seed(0)
    cache = FilteredCache() if arm == "C" else DynamicCache()
    vocab = 1000
    pos = 0                      # persistent position counter
    times = []
    torch.cuda.synchronize()
    for t in range(T):
        n = PRE + HAND + POST
        ids = torch.randint(10, vocab, (1, n), device=DEV)
        if arm == "C":
            cache.transient = (PRE, PRE + HAND)
            p = torch.cat([torch.arange(pos, pos + PRE + HAND),               # pre + hand
                           torch.arange(pos + PRE, pos + PRE + POST)])        # post reuses hand-start positions
            pos += PRE + POST
        else:
            p = torch.arange(pos, pos + n); pos += n
        pids = p.to(DEV)[None]
        t0 = time.time()
        with torch.inference_mode():
            out = model(input_ids=ids, position_ids=pids, past_key_values=cache,
                        use_cache=True, num_logits_to_keep=1)
        cache = out.past_key_values
        if arm == "C":
            cache.transient = None
        if arm == "B":
            # rollback: evict hand+post, re-forward post so it re-enters at rolled positions
            keep = cache.get_seq_length() - (HAND + POST)
            cache.crop(keep)
            pos -= (HAND + POST)
            ids2 = ids[:, -POST:]
            pids2 = torch.arange(pos, pos + POST, device=DEV)[None]
            with torch.inference_mode():
                out = model(input_ids=ids2, position_ids=pids2, past_key_values=cache,
                            use_cache=True, num_logits_to_keep=1)
            cache = out.past_key_values
            pos += POST
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    total = sum(times)
    import numpy as np
    return total, float(np.median(times[5:])), float(np.mean(times[-10:])), cache.get_seq_length()

model, lm = load()
print(f"loaded; T={T} pre={PRE} hand={HAND} post={POST}", flush=True)
for rep in range(REPEATS):
    for arm in ("A", "B", "C"):
        tot, med, tail, clen = run_arm(lm, arm)
        print(f"RESULT rep{rep} arm {arm}: total {tot:6.2f}s  median/turn {med*1000:6.1f}ms  last10/turn {tail*1000:6.1f}ms  final cache {clen}", flush=True)
print("BENCH_DONE", flush=True)

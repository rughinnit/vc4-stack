#!/usr/bin/env python3
"""tinystories_decode.py -- greedy decode on the QPUs, compared to ONNX Runtime.

Runs the static wrapper graph from ``make_tinystories.py`` through the VC4
engine and through ONNX Runtime CPU with the same feeds, one token at a time,
and reports per-step agreement of the argmax token and the logit difference.

The graph has no KV cache: every step feeds a full ``n_ctx`` window and the
prompt/prefix is recomputed.  Positions are absolute, so once the prompt is
longer than the window the window slides and the position ids slide with it.

    python tinystories_decode.py --model ts1m_c16.onnx --ctx 16 \\
        --tokenizer ~/scratch/refs/tinystories --tokens 8

``--tokenizer`` is a directory with ``tokenizer.json`` (or ``vocab.json`` and
``merges.txt``) from the model repository.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

EOS = 50256


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + \
        list(range(ord("\xa1"), ord("\xad") + 1)) + \
        list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Tokenizer:
    """GPT-2 byte-level BPE, enough for ASCII prompts and the decoder."""

    pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+|"
                     r" ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")

    def __init__(self, path):
        if os.path.isdir(path):
            tpath = os.path.join(path, "tokenizer.json")
            if os.path.exists(tpath):
                with open(tpath) as fh:
                    t = json.load(fh)
                self.vocab = t["model"]["vocab"]
                merges = t["model"]["merges"]
            else:
                with open(os.path.join(path, "vocab.json")) as fh:
                    self.vocab = json.load(fh)
                with open(os.path.join(path, "merges.txt")) as fh:
                    merges = [ln.strip() for ln in fh][1:]
        else:
            raise SystemExit("tokenizer directory not found: %s" % path)
        self.ranks = {}
        for i, m in enumerate(merges):
            parts = m.split() if isinstance(m, str) else m
            self.ranks[tuple(parts)] = i
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.ids = {k: int(v) for k, v in self.vocab.items()}
        self._inv = {int(v): k for k, v in self.vocab.items()}

    def _bpe(self, word):
        sym = [c for c in word]
        while len(sym) > 1:
            pairs = [(self.ranks.get((sym[i], sym[i + 1]), 1 << 30), i)
                     for i in range(len(sym) - 1)]
            rank, i = min(pairs)
            if rank == 1 << 30:
                break
            sym[i:i + 2] = [sym[i] + sym[i + 1]]
        return sym

    def encode(self, text):
        out = []
        for m in self.pat.findall(text):
            word = "".join(self.b2u[b] for b in m.encode("utf-8"))
            out.extend(self.ids[s] for s in self._bpe(word))
        return out

    def decode(self, ids):
        s = "".join(self.vocab_inv(i) for i in ids)
        return bytearray(self.u2b.get(c, ord(c)) for c in s).decode(
            "utf-8", "replace")

    def vocab_inv(self, i):
        return self._inv[i]


def _count(backs):
    from collections import Counter
    return dict(Counter(b.backend for b in backs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--prompt", default="Once upon a time there was a little "
                                       "girl named Lily. She lived with her "
                                       "mother and father in a small house "
                                       "near the woods. Every day she went "
                                       "outside to play with her friends.")
    ap.add_argument("--json", help="write the result record here")
    ap.add_argument("--no-ort", action="store_true")
    ap.add_argument("--cache", default=os.path.expanduser(
        "~/.cache/vc4infer/kcache"))
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "python"))
    import vc4infer

    tok = Tokenizer(args.tokenizer)
    prompt_ids = tok.encode(args.prompt)
    if len(prompt_ids) < args.ctx:
        raise SystemExit("prompt is %d tokens, need at least %d"
                         % (len(prompt_ids), args.ctx))
    window = prompt_ids[-args.ctx:]
    positions = list(range(len(prompt_ids) - args.ctx, len(prompt_ids)))

    ort = None
    if not args.no_ort:
        import onnxruntime as rt
        ort = rt.InferenceSession(args.model, providers=[
            "CPUExecutionProvider"])

    eng = vc4infer.Engine(args.model, cache_dir=args.cache)
    plan = eng.plan()
    mem = eng.memory()
    gen, steps = [], []
    text_ids = list(prompt_ids)
    for step in range(args.tokens):
        feed = {"input_ids": np.asarray(window, np.float32)[None, :],
                "position_ids": np.asarray(positions, np.float32)[None, :]}
        t0 = time.perf_counter()
        out = eng.run(feed, profile=True)
        ms = (time.perf_counter() - t0) * 1e3
        logits = np.asarray(out["logits"] if isinstance(out, dict)
                            else out[0])[0, -1]
        launches = sum(p["launches"] for p in eng.last_profile)
        rec = {"step": step, "wall_ms": ms, "launches": launches}
        if ort is not None:
            t1 = time.perf_counter()
            ref = ort.run(["logits"], feed)[0][0, -1]
            rec["ort_ms"] = (time.perf_counter() - t1) * 1e3
            rec["logit_max_abs_diff"] = float(np.abs(ref - logits).max())
            rec["ort_argmax"] = int(ref.argmax())
            rec["parity"] = bool(int(ref.argmax()) == int(logits.argmax()))
        nxt = int(logits.argmax())
        rec["token"] = nxt
        steps.append(rec)
        gen.append(nxt)
        window = window[1:] + [nxt]
        positions = positions[1:] + [positions[-1] + 1]

    steady = [s["wall_ms"] for s in steps[1:]] or [s["wall_ms"] for s in steps]
    rec = {
        "model": os.path.abspath(args.model),
        "ctx": args.ctx,
        "prompt": args.prompt,
        "prompt_tokens": len(prompt_ids),
        "generated_ids": gen,
        "generated_text": tok.decode(gen),
        "text": tok.decode(text_ids + gen),
        "tokens_per_s": 1000.0 / (sum(steady) / len(steady)),
        "mean_ms_per_token": sum(steady) / len(steady),
        "min_ms_per_token": min(steady),
        "first_ms": steps[0]["wall_ms"],
        "launches_per_token": steps[-1]["launches"],
        "nodes": len(plan.nodes),
        "node_backends": _count(eng.node_backends()),
        "memory": mem,
        "steps": steps,
        "parity_all": all(s.get("parity", True) for s in steps),
        "ort_mean_ms": (sum(s.get("ort_ms", 0.0) for s in steps[1:]) /
                        max(1, len(steps) - 1)) if ort is not None else None,
    }
    print(json.dumps(rec, indent=1))
    eng.close()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rec, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

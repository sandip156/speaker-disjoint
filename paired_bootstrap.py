#!/usr/bin/env python3
"""T5: trivial baselines + paired bootstrap significance tests."""

import argparse, csv, json, os, re, sys, unicodedata
from collections import Counter
import numpy as np

try:
    from jiwer import process_characters, process_words
except ImportError:
    sys.exit("jiwer is required:  pip install jiwer==4.0.0")

N_BOOT = 1000
SEED = 12345
PUNCT_RE = re.compile(r"[\u0964?.!,;:\u200c\u200d]")


def norm_text(t):
    t = unicodedata.normalize("NFC", t)
    t = PUNCT_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def load_scores(path):
    if not os.path.exists(path):
        sys.exit(f"file not found: {path}")
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        need = {"ref_norm", "hyp_norm"}
        missing = need - set(rdr.fieldnames or [])
        if missing:
            sys.exit(f"{path} is missing columns: {sorted(missing)}")
        for r in rdr:
            ref = (r.get("ref_norm") or "").strip()
            if not ref:
                continue
            rows.append({"ref": ref, "hyp": (r.get("hyp_norm") or "").strip()})
    if not rows:
        sys.exit(f"no usable rows in {path}")
    return rows


def per_utt_errors(rows):
    ec, lc, ew, lw = [], [], [], []
    for r in rows:
        mc = process_characters(r["ref"], r["hyp"])
        mw = process_words(r["ref"], r["hyp"])
        ec.append(mc.substitutions + mc.deletions + mc.insertions)
        lc.append(len(r["ref"]))
        ew.append(mw.substitutions + mw.deletions + mw.insertions)
        lw.append(len(r["ref"].split()))
    return (np.asarray(ec, float), np.asarray(lc, float),
            np.asarray(ew, float), np.asarray(lw, float))


def corpus_metrics(ec, lc, ew, lw):
    return (100.0 * ec.sum() / max(lc.sum(), 1),
            100.0 * ew.sum() / max(lw.sum(), 1))


def paired_test(ec_a, lc, ew_a, lw, ec_b, ew_b):
    rng = np.random.default_rng(SEED)
    n = len(lc)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    d_cer, d_wer = [], []
    for row in idx:
        L = lc[row].sum()
        d_cer.append(100.0 * (ec_b[row].sum() - ec_a[row].sum()) / L)
        W = lw[row].sum()
        d_wer.append(100.0 * (ew_b[row].sum() - ew_a[row].sum()) / W)
    d_cer, d_wer = np.asarray(d_cer), np.asarray(d_wer)

    def summarise(d):
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = (float((d <= 0).sum()) + 1) / (N_BOOT + 1)
        return float(d.mean()), float(lo), float(hi), p

    return summarise(d_cer), summarise(d_wer)


def report(label, ec_a, lc, ew_a, lw, ec_b, ew_b):
    cer_a, wer_a = corpus_metrics(ec_a, lc, ew_a, lw)
    cer_b, wer_b = corpus_metrics(ec_b, lc, ew_b, lw)
    (dc, dcl, dch, pc), (dw, dwl, dwh, pw) = paired_test(
        ec_a, lc, ew_a, lw, ec_b, ew_b)
    print(f"\n  A = your model :  CER {cer_a:.2f}%   WER {wer_a:.2f}%")
    print(f"  B = {label}:  CER {cer_b:.2f}%   WER {wer_b:.2f}%")
    print(f"  delta CER (B - A) = {dc:+.2f} pts  [95% CI {dcl:+.2f}, {dch:+.2f}]"
          f"   p(A better) = {pc:.4f}")
    print(f"  delta WER (B - A) = {dw:+.2f} pts  [95% CI {dwl:+.2f}, {dwh:+.2f}]"
          f"   p(A better) = {pw:.4f}")
    return {"baseline": label, "cer_A": round(cer_a, 2), "wer_A": round(wer_a, 2),
            "cer_B": round(cer_b, 2), "wer_B": round(wer_b, 2),
            "delta_cer": round(dc, 2), "delta_cer_lo": round(dcl, 2),
            "delta_cer_hi": round(dch, 2), "p_A_better": round(pc, 4),
            "delta_wer": round(dw, 2), "delta_wer_lo": round(dwl, 2),
            "delta_wer_hi": round(dwh, 2), "p_wer_A_better": round(pw, 4)}


def match_scores(rows_a, rows_b):
    pool = {}
    for r in rows_b:
        pool.setdefault(norm_text(r["ref"]), []).append(r["hyp"])
    matched = []
    for r in rows_a:
        key = norm_text(r["ref"])
        if pool.get(key):
            matched.append({"ref": r["ref"], "hyp": pool[key].pop(0)})
        else:
            sys.exit(f"no matching reference in scores-b for: {key[:40]}...")
    leftover = sum(len(v) for v in pool.values())
    if leftover:
        print(f"  (warning: {leftover} scores-b rows had no match in A - ignored)")
    return matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-a", default="scores_t1.csv")
    ap.add_argument("--scores-b", default=None)
    ap.add_argument("--b-name", default="system B")
    ap.add_argument("--train", default=os.path.expanduser(
        "~/Desktop/experiments/train_v1_274.json"))
    ap.add_argument("--no-copy", action="store_true")
    ap.add_argument("--out", default="paired_bootstrap.csv")
    args = ap.parse_args()

    rows_a = load_scores(args.scores_a)
    ec_a, lc, ew_a, lw = per_utt_errors(rows_a)
    print(f"Paired bootstrap ({N_BOOT} resamples, seed {SEED}, "
          f"CER spaces-kept, corpus-weighted)")
    print(f"System A: {args.scores_a}  (n={len(rows_a)})")

    results = []

    # baseline 1: transcribe nothing
    ec_b = lc.copy()
    ew_b = lw.copy()
    results.append(report("empty output", ec_a, lc, ew_a, lw, ec_b, ew_b))

    # baseline 2: repeat one training sentence every time
    if not args.no_copy:
        items = []
        if os.path.exists(args.train):
            with open(args.train, encoding="utf-8") as f:
                head = f.read(1)
                f.seek(0)
                if head == "[":
                    items = json.load(f)
                else:
                    items = [json.loads(l) for l in f if l.strip()]
        if items:
            texts = [norm_text(it.get("text") or it.get("sentence")
                               or it.get("newar_sentence") or "")
                     for it in items]
            texts = [t for t in texts if t]
            cnt = Counter(texts)
            best_freq = max(cnt.values())
            cand = [t for t, c in cnt.items() if c == best_freq]
            copy_sentence = min(cand, key=len)
            print(f"\n  copy baseline sentence ({best_freq}x in train, "
                  f"{len(copy_sentence)} chars): {copy_sentence[:60]}...")
            rows_copy = [{"ref": r["ref"], "hyp": copy_sentence} for r in rows_a]
            ec_c, lc_c, ew_c, lw_c = per_utt_errors(rows_copy)
            results.append(report("copy train sentence", ec_a, lc, ew_a, lw,
                                  ec_c, ew_c))
        else:
            print("\n  (train manifest not found - copy baseline skipped)")

    # baseline 3 (optional): another real system's scores CSV
    if args.scores_b:
        rows_b = load_scores(args.scores_b)
        rows_m = match_scores(rows_a, rows_b)
        ec_b2, lc_b2, ew_b2, lw_b2 = per_utt_errors(rows_m)
        assert np.allclose(lc_b2, lc) and np.allclose(lw_b2, lw), \
            "reference lengths differ after matching - aborting"
        results.append(report(args.b_name, ec_a, lc, ew_a, lw, ec_b2, ew_b2))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
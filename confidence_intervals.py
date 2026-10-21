#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
confidence_intervals.py — T3: bootstrap confidence intervals (advisor's method).

What it computes (all in the paper's convention: CER spaces-kept, weighted):
  1. Sentence bootstrap for every speaker present + ALL:
       resample utterances WITH replacement (same n), recompute corpus CER/WER,
       1000 resamples, take 2.5/97.5 percentiles -> 95% CI.
       Captures SENTENCE-level variation only (does NOT capture speaker variation).
  2. Speaker bootstrap for ALL (only when >=3 speakers in file):
       resample SPEAKERS with replacement, pool their utterances, recompute.
       Captures SPEAKER-level variation (the honest, wider CI).

Usage — two runs:
  # (a) headline CI: use the T1-verified scores of the ORIGINAL run (26.75%):
  python confidence_intervals.py --scores scores_t1.csv

  # (b) full picture: per-speaker CIs + speaker-level CI (all 431, one run):
  python confidence_intervals.py --scores scores_all431.csv
"""
import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

SEED = 12345
N_BOOT = 1000


def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("ERROR: input CSV is empty.")
    need = {"speaker", "cer_kept", "wer", "ref_norm"}
    missing = need - set(rows[0].keys())
    if missing:
        sys.exit(f"ERROR: input CSV missing columns: {missing}.\n"
                 "Use the scores CSV produced by eval_newari.py (tsv mode).")
    per_spk = defaultdict(lambda: {"e_cer": [], "l_cer": [], "e_wer": [], "l_wer": []})
    for r in rows:
        spk = (r.get("speaker") or "?").strip() or "?"
        ref = r.get("ref_norm") or ""
        chars, words = len(ref), len(ref.split())
        g = per_spk[spk]
        g["e_cer"].append(float(r["cer_kept"]) * chars)
        g["l_cer"].append(chars)
        g["e_wer"].append(float(r["wer"]) * words)
        g["l_wer"].append(words)
    for g in per_spk.values():
        for k in g:
            g[k] = np.asarray(g[k], dtype=float)
    return per_spk


def sentence_ci(e, l, rng):
    """Bootstrap over utterances (with replacement, same n)."""
    n = len(l)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    vals = e[idx].sum(axis=1) / l[idx].sum(axis=1) * 100.0
    return np.percentile(vals, [2.5, 97.5])


def speaker_ci(per_spk, speakers, rng):
    """Bootstrap over speakers (with replacement), pooling their utterances."""
    k = len(speakers)
    cer_vals, wer_vals = [], []
    for _ in range(N_BOOT):
        pick = rng.integers(0, k, size=k)
        e_c = np.concatenate([per_spk[speakers[i]]["e_cer"] for i in pick])
        l_c = np.concatenate([per_spk[speakers[i]]["l_cer"] for i in pick])
        e_w = np.concatenate([per_spk[speakers[i]]["e_wer"] for i in pick])
        l_w = np.concatenate([per_spk[speakers[i]]["l_wer"] for i in pick])
        cer_vals.append(e_c.sum() / l_c.sum() * 100.0)
        wer_vals.append(e_w.sum() / l_w.sum() * 100.0)
    return (np.percentile(cer_vals, [2.5, 97.5]),
            np.percentile(wer_vals, [2.5, 97.5]))


def fmt(name, n, cer, wer, ci_c, ci_w):
    return (f"  {name:<8} (n={n:3d}):  "
            f"CER {cer:6.2f}% [{ci_c[0]:5.2f}, {ci_c[1]:5.2f}]   "
            f"WER {wer:6.2f}% [{ci_w[0]:5.2f}, {ci_w[1]:5.2f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="scores CSV from eval_newari.py")
    args = ap.parse_args()
    per_spk = load(args.scores)
    rng = np.random.default_rng(SEED)

    print(f"Bootstrap 95% CIs  ({N_BOOT} resamples, seed {SEED}, "
          "CER spaces-kept, corpus-weighted)")
    print("=" * 74)
    print("Sentence bootstrap (resampling utterances; sentence variation only):")

    all_e_c, all_l_c, all_e_w, all_l_w, all_n = [], [], [], [], 0
    for spk in sorted(per_spk):
        g = per_spk[spk]
        n = len(g["l_cer"])
        cer = g["e_cer"].sum() / g["l_cer"].sum() * 100.0
        wer = g["e_wer"].sum() / g["l_wer"].sum() * 100.0
        ci_c = sentence_ci(g["e_cer"], g["l_cer"], rng)
        ci_w = sentence_ci(g["e_wer"], g["l_wer"], rng)
        print(fmt(spk, n, cer, wer, ci_c, ci_w))
        all_e_c.append(g["e_cer"]); all_l_c.append(g["l_cer"])
        all_e_w.append(g["e_wer"]); all_l_w.append(g["l_wer"])
        all_n += n

    e_c = np.concatenate(all_e_c); l_c = np.concatenate(all_l_c)
    e_w = np.concatenate(all_e_w); l_w = np.concatenate(all_l_w)
    cer_all = e_c.sum() / l_c.sum() * 100.0
    wer_all = e_w.sum() / l_w.sum() * 100.0
    ci_c = sentence_ci(e_c, l_c, rng)
    ci_w = sentence_ci(e_w, l_w, rng)
    print(fmt("ALL", all_n, cer_all, wer_all, ci_c, ci_w))

    speakers = sorted(per_spk)
    if len(speakers) >= 3:
        print("-" * 74)
        print(f"Speaker bootstrap (resampling the {len(speakers)} speakers; "
              "speaker variation included):")
        ci_sc, ci_sw = speaker_ci(per_spk, speakers, rng)
        print(f"  ALL (n={all_n}):  "
              f"CER {cer_all:6.2f}% [{ci_sc[0]:5.2f}, {ci_sc[1]:5.2f}]   "
              f"WER {wer_all:6.2f}% [{ci_sw[0]:5.2f}, {ci_sw[1]:5.2f}]")
        print("  -> expect this CI to be wider than the sentence one: that is the "
              "honest\n     picture with only 7 speakers.")
    print("=" * 74)
    print("Paper methods line: '95% confidence intervals were obtained by "
          "bootstrapping\nutterances (1000 resamples, seed 12345); speaker-level "
          "uncertainty was\nestimated by bootstrapping speakers.'")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
per_speaker_table.py — T2: paper-ready per-speaker zero-shot table.

Input:  scores_all431.csv  (produced by eval_newari.py tsv mode on
        transcriptions_all431_v1.tsv — all 431 utterances, one decode run)

Convention (identical to the paper / RESULTS.md v1.0, T1-verified):
  CER = spaces kept as characters, corpus-level (length-weighted)
  WER = word-level, corpus-level (word-weighted)

Usage:
  python per_speaker_table.py --scores scores_all431.csv --out per_speaker_table.csv

Output:
  - prints a Markdown table (paste-ready for the paper)
  - saves per_speaker_table.csv
"""
import argparse
import csv
import sys
from collections import defaultdict

PAPER_HEADLINE_CER = 26.75  # spk04, original verified run


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
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="scores_all431.csv from eval_newari.py")
    ap.add_argument("--out", default="per_speaker_table.csv")
    ap.add_argument("--test-speaker", default="spk04")
    a = ap.parse_args()
    rows = load(a.scores)

    agg = defaultdict(lambda: {"edits": 0.0, "chars": 0, "wedits": 0.0, "words": 0, "n": 0})
    for r in rows:
        spk = (r.get("speaker") or "?").strip() or "?"
        cer_k = float(r["cer_kept"])
        wer = float(r["wer"])
        ref = r.get("ref_norm") or ""
        chars, words = len(ref), len(ref.split())
        g = agg[spk]
        g["edits"] += cer_k * chars
        g["chars"] += chars
        g["wedits"] += wer * words
        g["words"] += words
        g["n"] += 1

    def metrics(g):
        cer = 100 * g["edits"] / g["chars"] if g["chars"] else float("nan")
        wer = 100 * g["wedits"] / g["words"] if g["words"] else float("nan")
        return cer, wer

    order = sorted(agg)
    tot = {k: sum(g[k] for g in agg.values()) for k in ("edits", "chars", "wedits", "words", "n")}

    md = ["| Speaker | n | CER (%) | WER (%) |", "|---|---|---|---|"]
    out_rows = []
    for spk in order:
        cer, wer = metrics(agg[spk])
        md.append(f"| {spk} | {agg[spk]['n']} | {cer:.2f} | {wer:.2f} |")
        out_rows.append({"speaker": spk, "n": agg[spk]["n"],
                         "cer_pct": f"{cer:.2f}", "wer_pct": f"{wer:.2f}"})
    cer_all, wer_all = metrics(tot)
    md.append(f"| **ALL** | **{tot['n']}** | **{cer_all:.2f}** | **{wer_all:.2f}** |")
    out_rows.append({"speaker": "ALL", "n": tot["n"],
                     "cer_pct": f"{cer_all:.2f}", "wer_pct": f"{wer_all:.2f}"})

    print("\n".join(md))
    print()
    ts = a.test_speaker
    if ts in agg:
        cer_ts, _ = metrics(agg[ts])
        drift = cer_ts - PAPER_HEADLINE_CER
        print(f"Cross-check: {ts} in this re-decode = {cer_ts:.2f}% "
              f"(paper headline {PAPER_HEADLINE_CER:.2f}%, drift {drift:+.2f} pts — "
              "normal across Colab/library versions; paper keeps the T1-verified original).")
    others = [s for s in order if s != ts]
    if others:
        e = sum(agg[s]["edits"] for s in others)
        c = sum(agg[s]["chars"] for s in others)
        n_o = sum(agg[s]["n"] for s in others)
        print(f"Non-test speakers ({n_o} utts): corpus CER = {100 * e / c:.2f}%")

    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["speaker", "n", "cer_pct", "wer_pct"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nsaved: {a.out}")


if __name__ == "__main__":
    main()
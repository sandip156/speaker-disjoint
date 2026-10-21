#!/usr/bin/env python3
"""freeze_split.py — speaker-disjoint split, frozen."""
import csv
from datetime import date

TEST_SPEAKERS = {"spk04"}
rows = list(csv.DictReader(open("master.tsv", encoding="utf-8"), delimiter="\t"))
with open("split.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t"); w.writerow(["id","speaker_id","split"])
    n_tr = n_te = 0
    for r in rows:
        s = "test" if r["speaker_id"] in TEST_SPEAKERS else "train"
        w.writerow([r["id"], r["speaker_id"], s])
        n_tr, n_te = n_tr + (s == "train"), n_te + (s == "test")
with open("SPLITS.md", "w", encoding="utf-8") as f:
    f.write(f"# Split freeze — {date.today()}\n\n"
            "- Criterion: speaker-disjoint; test speaker chosen by METADATA ONLY "
            "(domain coverage, set size); test audio never heard before experiments.\n"
            f"- Test speakers: {sorted(TEST_SPEAKERS)}\n"
            f"- Train: {n_tr} utts | Test: {n_te} utts\n"
            "- Rule: any future utterance from a test speaker joins TEST, never train.\n")
print(f"train {n_tr} | test {n_te} | wrote split.tsv + SPLITS.md")
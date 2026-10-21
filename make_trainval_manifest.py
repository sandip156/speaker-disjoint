#!/usr/bin/env python3
"""make_trainval_manifest.py — 304 train utts -> 274 train + 30 val (seed 42)."""
import csv, json, random
from pathlib import Path

BASE = Path.home()/"Desktop/sandiptourism"
A16  = Path.home()/"Desktop/experiments/audio_16k"
split = {r["id"]: r["split"] for r in
         csv.DictReader(open(BASE/"split.tsv", encoding="utf-8"), delimiter="\t")}
rows = [r for r in csv.DictReader(open(BASE/"master.tsv", encoding="utf-8"), delimiter="\t")
        if split[r["id"]] == "train"]                      # 304 utts, NO spk04

random.seed(42); random.shuffle(rows)
val, train = rows[:30], rows[30:]

for name, rs in [("train_v1_274.json", train), ("val_v1_30.json", val)]:
    with open(Path.home()/"Desktop/experiments"/name, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps({"audio_filepath": str(A16/r["audio_filename"]),
                                "duration": float(r["duration_s"]),
                                "text": r["newar_sentence"],
                                "speaker": r["speaker_id"], "id": r["id"]},
                               ensure_ascii=False) + "\n")
    print(f"{name}: {len(rs)} utterances")
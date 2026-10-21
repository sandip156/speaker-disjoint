#!/usr/bin/env python3
"""make_test_manifest.py — spk08 test set -> NeMo JSONL manifest."""
import csv, json
from pathlib import Path

BASE = Path.home()/"Desktop/sandiptourism"
A16  = Path.home()/"Desktop/experiments/audio_16k"
split = {r["id"]: r["split"] for r in
         csv.DictReader(open(BASE/"split.tsv", encoding="utf-8"), delimiter="\t")}
rows = [r for r in csv.DictReader(open(BASE/"master.tsv", encoding="utf-8"), delimiter="\t")
        if split.get(r["id"]) == "test"]
out = Path.home()/"Desktop/experiments/test_v1_16k.json"
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps({"audio_filepath": str(A16/r["audio_filename"]),
                            "duration": float(r["duration_s"]),
                            "text": r["newar_sentence"],
                            "speaker": r["speaker_id"], "domain": r["domain"]},
                           ensure_ascii=False) + "\n")
print(f"{len(rows)} test utterances -> {out}  (expect 197)")
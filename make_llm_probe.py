#!/usr/bin/env python3
"""make_llm_probe.py — sample 30 training-partition triples for the LLM probe (seed 42).
Excludes spk04 (test speaker) — his sentences stay reserved for ASR evaluation only."""
import csv, random
from pathlib import Path

BASE = Path.home()/"Desktop/sandiptourism"
OUT  = Path.home()/"Desktop/experiments/llm_probe_30.tsv"

split = {r["id"]: r["split"] for r in
         csv.DictReader(open(BASE/"split.tsv", encoding="utf-8"), delimiter="\t")}
rows = [r for r in csv.DictReader(open(BASE/"master.tsv", encoding="utf-8"), delimiter="\t")
        if split[r["id"]] == "train"]

random.seed(42)
sample = random.sample(rows, 30)

fields = ["id", "speaker_id", "domain", "newar_sentence", "nepali_sentence",
          "english_sentence"]
with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    for r in sample:
        w.writerow({k: r[k] for k in fields})

# quick stats for the report
from collections import Counter
print(f"30 triples -> {OUT}")
print("speakers:", dict(Counter(r["speaker_id"] for r in sample)))
print("domains:", dict(Counter(r["domain"] for r in sample)))
print("all train-partition (no spk04):",
      all(split[r["id"]] == "train" for r in sample))
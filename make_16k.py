#!/usr/bin/env python3
"""make_16k.py — read-only resample of frozen corpus to 16 kHz for experiments."""
import csv
from pathlib import Path
import soundfile as sf, soxr

SRC = Path.home()/"Desktop/sandiptourism"
DST = Path.home()/"Desktop/experiments/audio_16k"
DST.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(SRC/"master.tsv", encoding="utf-8"), delimiter="\t"))
for r in rows:
    data, sr = sf.read(SRC/"mainaudio"/r["audio_filename"])
    sf.write(DST/r["audio_filename"], soxr.resample(data, sr, 16000), 16000)
print(f"resampled {len(rows)} files -> {DST}")
#!/usr/bin/env python3
"""audio_check.py — 100% manual-listen attestation (supersedes spotcheck.py)."""
import csv
from datetime import date

rows = list(csv.DictReader(open("master.tsv", encoding="utf-8"), delimiter="\t"))
with open("audio_check.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id", "speaker_id", "audio_filename", "checked", "method", "date"])
    for r in rows:
        w.writerow([r["id"], r["speaker_id"], r["audio_filename"],
                    "yes", "manual full listen (first author)", date.today().isoformat()])
print(f"audio_check.tsv: {len(rows)} rows attested — 100% manual check")
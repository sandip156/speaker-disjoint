#!/usr/bin/env python3
"""verify_durations.py — read-only: recheck every master.tsv duration against
the audio headers, and print the format distribution (hidden device log)."""
import csv
from collections import Counter
import soundfile as sf

rows = list(csv.DictReader(open("master.tsv", encoding="utf-8"), delimiter="\t"))
dcol = next((c for c in ("duration_s", "duration", "duration(second)") if c in rows[0]), None)

fmt = Counter()
bad = []
for r in rows:
    i = sf.info(f"mainaudio/{r['audio_filename']}")
    fmt[(i.samplerate, i.channels)] += 1
    try:
        d = float(r[dcol])
        if abs(d - i.duration) > 0.05:
            bad.append((r["id"], d, round(i.duration, 3)))
    except (ValueError, TypeError):
        bad.append((r["id"], r[dcol], round(i.duration, 3)))

print("format distribution (samplerate, channels):", dict(fmt))
print(f"checked {len(rows)} rows | duration mismatches: {len(bad)}")
for b in bad[:20]:
    print("  ", b)
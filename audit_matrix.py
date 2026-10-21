#!/usr/bin/env python3
"""audit_matrix.py — speaker x domain counts, hours, coverage."""
import csv
from collections import defaultdict

rows = list(csv.DictReader(open("master.tsv", encoding="utf-8"), delimiter="\t"))
dcol = next((c for c in ("duration_s","duration","duration(second)") if c in rows[0]), None)

cnt = defaultdict(lambda: defaultdict(int))
hrs = defaultdict(lambda: defaultdict(float))
for r in rows:
    sp, dm = (r.get("speaker_id") or "?").strip(), (r.get("domain") or "?").strip()
    cnt[sp][dm] += 1
    try: hrs[sp][dm] += float(r.get(dcol) or 0) / 3600
    except ValueError: pass

domains  = sorted({d for sp in cnt for d in cnt[sp]})
speakers = sorted(cnt, key=lambda s: -sum(cnt[s].values()))

with open("audit_matrix.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["speaker"] + domains + ["total", "hours", "domain_coverage"])
    for sp in speakers:
        w.writerow([sp] + [cnt[sp].get(d, 0) for d in domains]
                   + [sum(cnt[sp].values()), round(sum(hrs[sp].values()), 3),
                      f"{len(cnt[sp])}/{len(domains)}"])

print(f"{'speaker':<8}", *(f"{d[:9]:>10}" for d in domains),
      f"{'TOTAL':>7} {'hours':>6}  cover", sep="\t")
for sp in speakers:
    print(f"{sp:<8}", *(f"{cnt[sp].get(d,0):>10}" for d in domains),
          f"{sum(cnt[sp].values()):>7}", f"{sum(hrs[sp].values()):>6.2f}",
          f"{len(cnt[sp])}/{len(domains)}", sep="\t")
print(f"{'TOTAL':<8}", *(f"{sum(cnt[sp].get(d,0) for sp in speakers):>10}" for d in domains), sep="\t")
print(f"{'hours':<8}", *(f"{sum(hrs[sp].get(d,0) for sp in speakers):>10.2f}" for d in domains), sep="\t")
holes = sum(1 for sp in speakers for d in domains if cnt[sp].get(d, 0) == 0)
print(f"\ndomains: {len(domains)} | empty cells: {holes}/{len(speakers)*len(domains)}"
      f" | saved audit_matrix.tsv")
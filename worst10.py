#!/usr/bin/env python3
"""worst10.py — top-10 highest-CER test utterances, re-joined to master.tsv by reference."""
import csv, re, unicodedata
from pathlib import Path

def norm(s):
    s = unicodedata.normalize("NFC", str(s))
    s = re.sub(r"[।?.,!;:]", "", s)
    s = s.replace("\u200c", "").replace("\u200d", "")
    return re.sub(r"\s+", " ", s).strip()

rows = list(csv.DictReader(open("/Users/sandipjoshi/Desktop/experiments/e2_results_v1.tsv",
                                encoding="utf-8"), delimiter="\t"))
print(f"loaded {len(rows)} rows | id column empty: {not rows[0].get('id','').strip()}")

# re-join ids + filenames via master.tsv (reference text is the key)
master = list(csv.DictReader(open("master.tsv", encoding="utf-8"), delimiter="\t"))
by_ref = {}
for m in master:
    by_ref.setdefault(norm(m["newar_sentence"]), m)   # first match wins (refs unique per speaker design)

joined = 0
for r in rows:
    m = by_ref.get(norm(r["reference"]))
    if m:
        r["id"], r["audio_filename"], r["speaker"] = m["id"], m["audio_filename"], m["speaker_id"]
        joined += 1
print(f"joined: {joined}/{len(rows)}")

worst = sorted(rows, key=lambda r: -float(r["cer"]))[:10]
with open("worst10_v1.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["id","speaker","domain","audio_filename","cer","reference","hypothesis","verdict"])
    for r in worst:
        w.writerow([r.get("id",""), r.get("speaker",""), r["domain"], r.get("audio_filename",""),
                    r["cer"], r["reference"], r["hypothesis"], ""])

print("\n10 WORST (listen to these):")
for r in worst:
    print(f"  {r.get('id','?'):>6}  {r.get('audio_filename','?'):<15} {float(r['cer'])*100:5.1f}%  [{r['domain']}]")
print("\nPlay each file in mainaudio/, compare to the reference column,")
print("type correct/wrong in the verdict column of worst10_v1.tsv.")
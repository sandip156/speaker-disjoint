#!/usr/bin/env python3
"""freeze_recon.py — cross-check master.tsv <-> speakers.tsv <-> mainaudio/.
Save inside the sandiptourism folder and run from there. Paste the SUMMARY back."""
import csv, sys
from pathlib import Path
from collections import Counter, defaultdict

# Windows consoles crash printing non-Latin text unless told UTF-8:
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"): stream.reconfigure(encoding="utf-8")

# ---------------- CONFIG ----------------
BASE      = Path(".")            # script lives inside sandiptourism -> "." is the folder itself
AUDIO_DIR = BASE / "mainaudio"
MASTER    = BASE / "master.tsv"
SPEAKERS  = BASE / "speakers.tsv"
EXTS      = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
# -----------------------------------------

DOMAINS = {"geography_places","culture_festivals","religion","music_dance",
 "food_cuisine","family_kinship","animals_nature","health_body","stories",
 "education_language","weather_climate",
 "hotels_accommodation","shopping_local_markets",
 "travel_transportation","guides_services","food_restaurants",
 "directions_navigation","tourist_places_culture","travel_conversation","everyday_objects"}
OK_CONSENT = {"granted", "granted_parental", "signed", "written"}

def read_tsv(p):
    try:
        with open(p, encoding="utf-8", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            return r.fieldnames or [], list(r)
    except FileNotFoundError:
        sys.exit(f"File not found: {p.resolve()} — is the script inside sandiptourism?")

def stem(name): return Path(name).stem.strip().lower()

errors, warns = [], []
E, W = errors.append, warns.append

m_hdr, master = read_tsv(MASTER)
for c in ("id","speaker_id","domain","audio_filename"):
    if c not in m_hdr: E(f"master.tsv: missing column '{c}' (have {m_hdr})")
dur_col = next((c for c in ("duration_s","duration","duration(second)") if c in m_hdr), None)
if not dur_col: W("master.tsv: no duration column found")

s_hdr, spks = read_tsv(SPEAKERS)
if "speaker_id" not in s_hdr: E(f"speakers.tsv: missing 'speaker_id' (have {s_hdr})")
spk_map = {r.get("speaker_id","").strip(): r for r in spks}
for col in ("device","consent_status"):
    if col not in s_hdr: E(f"speakers.tsv: missing '{col}' (release blocker)")

disk = defaultdict(list)
for p in AUDIO_DIR.rglob("*"):
    if p.suffix.lower() in EXTS: disk[stem(p.name)].append(p)
for k, v in disk.items():
    if len(v) > 1: E(f"disk: {len(v)} files share stem '{k}': {[x.name for x in v]}")

matched, missing, tsv_stems, dur_bad = 0, [], Counter(), []
have_sf = True
try: import soundfile as sf
except ImportError: have_sf = False

for i, r in enumerate(master, start=2):
    rid, fn = r.get("id","?"), (r.get("audio_filename") or "").strip()
    if not fn: E(f"row {i} ({rid}): empty audio_filename"); continue
    st = stem(fn); tsv_stems[st] += 1
    if st not in disk:
        missing.append(rid); E(f"row {i} ({rid}): no audio on disk for '{fn}'"); continue
    matched += 1
    if dur_col and have_sf:
        try:    true_s = sf.info(str(disk[st][0])).duration
        except Exception: true_s = None; W(f"row {i} ({rid}): could not read audio header for '{fn}'")
        if true_s is not None:
            try:
                d = float(r.get(dur_col,"") or 0)
                if abs(d - true_s) > 0.05: dur_bad.append((rid, d, round(true_s,3)))
                if not (0.1 < d < 90): W(f"row {i} ({rid}): odd duration {d}s")
            except ValueError: W(f"row {i} ({rid}): duration not numeric: {r.get(dur_col)!r}")
    sp = (r.get("speaker_id") or "").strip()
    if sp and sp not in spk_map: E(f"row {i} ({rid}): speaker '{sp}' not in speakers.tsv")
    dom = (r.get("domain") or "").strip()
    if dom and dom not in DOMAINS: E(f"row {i} ({rid}): unknown domain '{dom}'")

for k, v in tsv_stems.items():
    if v > 1: E(f"master: {v} rows share audio_filename stem '{k}'")
orphans = sorted(set(disk) - set(tsv_stems))
if orphans: E(f"{len(orphans)} audio files on disk not in master.tsv, e.g. {orphans[:10]}")

if "nepali_sentence" in m_hdr:   # unique-per-speaker design => ANY duplicate is suspect
    nep = Counter((r.get("nepali_sentence") or "").strip() for r in master)
    nep_d = {k:v for k,v in nep.items() if v > 1 and k}
    if nep_d: W(f"{len(nep_d)} duplicate Nepali source sentences (double-recorded?): "
                f"{list(nep_d.items())[:3]}")

for col in ("age_band","gender","dialect"):   # master demographics vs speakers.tsv
    if col in m_hdr and col in s_hdr:
        bad = [(r.get("id"), (r.get(col) or "").strip(),
                (spk_map.get((r.get("speaker_id") or "").strip(),{}).get(col,"") or "").strip())
               for r in master
               if (r.get(col) or "").strip() and
                  (spk_map.get((r.get("speaker_id") or "").strip(),{}).get(col,"") or "").strip()
                  and (r.get(col) or "").strip().lower() !=
                      (spk_map[(r.get("speaker_id") or "").strip()][col] or "").strip().lower()]
        if bad: W(f"{len(bad)} rows where master.{col} != speakers.tsv.{col}, e.g. {bad[:5]}")

for sid, r in spk_map.items():
    cs = (r.get("consent_status","") or "").strip().lower()
    if cs and cs not in OK_CONSENT: W(f"speaker {sid}: consent_status='{cs}'")
    if not (r.get("device","") or "").strip(): W(f"speaker {sid}: device empty")

no_ext = sum(1 for r in master if "." not in (r.get("audio_filename") or ""))
per_spk = Counter((r.get("speaker_id") or "?").strip() for r in master)
codes = [c for c in spk_map if c]
embed = sum(1 for st in set(tsv_stems) if any(c.lower() in st for c in codes)) if codes else 0
tot = 0.0
for r in master:
    try: tot += float(r.get(dur_col) or 0)
    except (TypeError, ValueError): pass

print("="*60); print("SUMMARY — paste this whole block back"); print("="*60)
print(f"master rows           : {len(master)}")
print(f"total duration (TSV)  : {tot/3600:.2f} h")
print(f"speakers              : {dict(sorted(per_spk.items()))}")
print(f"audio files on disk   : {sum(len(v) for v in disk.values())}")
print(f"first 10 disk names   : {sorted(p.name for p in AUDIO_DIR.rglob('*') if p.suffix.lower() in EXTS)[:10]}")
print(f"matched TSV<->disk    : {matched}")
print(f"TSV rows w/o audio    : {len(missing)} {missing[:10]}")
print(f"orphan audio files    : {len(orphans)}")
print(f"duration mismatches   : {len(dur_bad)} {dur_bad[:5]}")
print(f"TSV names w/o ext     : {no_ext}/{len(master)}"
      + ("  <-- store the FULL disk filename (with .wav) in master.tsv" if no_ext else ""))
print(f"speaker code in fname : {embed}/{len(set(tsv_stems))} stems")
print(f"ERRORS: {len(errors)}");   [print("  [E]", e) for e in errors[:30]]
print(f"WARNINGS: {len(warns)}"); [print("  [W]", w) for w in warns[:20]]
sys.exit(1 if errors else 0)
#!/usr/bin/env python3
"""checksums.py — SHA-256 freeze manifest for v0.9-draft.

Usage:
  python3 checksums.py           create the manifest (refuses to overwrite)
  python3 checksums.py verify    re-check every file against the manifest
  python3 checksums.py --force   re-create (only when you INTEND to re-freeze)
"""
import hashlib, sys
from datetime import date
from pathlib import Path

BASE     = Path(".")                      # script lives inside sandiptourism
AUDIO    = BASE / "mainaudio"
MANIFEST = BASE / "checksums_v1.0.txt"
# files that DEFINE the corpus — all must exist before freezing:
CORE = [BASE / "master.tsv",
        BASE / "speakers.tsv",
        BASE / "split.tsv",
        BASE / "SPLITS.md",
        BASE / "teacher_check.tsv",
        BASE / "audio_check.tsv"]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def collect():
    missing = [p for p in CORE if not p.exists()]
    if missing:
        sys.exit("ABORT — required files missing (run their scripts first):\n  "
                 + "\n  ".join(str(p) for p in missing))
    audio = sorted(p for p in AUDIO.rglob("*")
                   if p.is_file() and not p.name.startswith("."))
    return CORE + audio

def create():
    files = collect()
    if MANIFEST.exists() and "--force" not in sys.argv:
        sys.exit(f"ABORT — {MANIFEST.name} already exists.\n"
                 "A frozen manifest is never silently overwritten.\n"
                 "Use 'python3 checksums.py verify' to check, or --force only if you intend to re-freeze.")
    with open(MANIFEST, "w", encoding="utf-8") as out:
        out.write(f"# SHA-256 freeze manifest — sandiptourism v0.9-draft — {date.today()}\n")
        out.write(f"# files: {len(files)}\n")
        for p in files:
            out.write(f"{sha256(p)}  {p.relative_to(BASE).as_posix()}\n")
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"hashed {len(files)} files ({total_mb:.1f} MB) -> {MANIFEST.name}")
    print("v0.9-draft LOCKED. From now on: python3 checksums.py verify")

def verify():
    if not MANIFEST.exists():
        sys.exit(f"No manifest found ({MANIFEST.name}) — run 'python3 checksums.py' first.")
    recorded = {}
    for line in open(MANIFEST, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        h, name = line.rstrip("\n").split("  ", 1)
        recorded[name] = h
    ok = changed = missing = 0
    for name, h in recorded.items():
        p = BASE / name
        if not p.exists():
            print(f"  MISSING  {name}"); missing += 1
        elif sha256(p) != h:
            print(f"  CHANGED  {name}"); changed += 1
        else:
            ok += 1
    on_disk = {p.relative_to(BASE).as_posix()
               for p in collect() if not p.name.startswith(".")}
    new = sorted(on_disk - set(recorded))
    for name in new:
        print(f"  NEW      {name}  (not in manifest)")
    print(f"\nverify: {ok} OK | {changed} changed | {missing} missing | {len(new)} new")
    if changed or missing:
        sys.exit("MANIFEST BROKEN — the frozen corpus was modified after freeze.")
    print("MANIFEST INTACT — corpus unchanged since freeze.")

if __name__ == "__main__":
    if "verify" in sys.argv: verify()
    else:                    create()
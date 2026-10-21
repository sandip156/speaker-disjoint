#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_newari.py — Technical Step T1: one clean evaluation pipeline.

Paper: "A Community-Authored, Domain-Annotated Speech Corpus and Cross-Corpus
Speaker-Generalization Benchmark for Nepal Bhasha (Newari)"

WHY THIS SCRIPT EXISTS
    Every later result (per-speaker table, confidence intervals, error
    analysis) is computed from the per_utterance_scores.csv file this script
    writes. Before building anything on top, we verify that this one script
    reproduces the paper's headline numbers:

        NM test set (in-domain):       CER 17.59%   WER 52.61%
        spk04 speaker-disjoint test:   CER 26.75%   WER 73.62%

TWO MODES
    1) score — you ALREADY have the model's transcriptions saved as text.
               Cheapest way to check the scoring convention.
        python eval_newari.py score --refs refs.txt --hyps hyps.txt --out scores.csv
        (refs.txt / hyps.txt: one utterance per line, same order)

    2) infer — re-transcribe the audio with the model, then score.
        python eval_newari.py infer --manifest manifest.csv --model model.nemo --audio-root ./audio --out scores.csv

    3) tsv — score a results table that ALREADY has reference + hypothesis
             columns (e.g. your saved e2_results_v1.tsv). This is T1:
        python eval_newari.py tsv --tsv ~/Desktop/experiments/e2_results_v1.tsv --out scores_t1.csv

MANIFEST (infer mode) — two formats accepted:
    a) NeMo JSONL (exactly what make_test_manifest.py produces):
         {"audio_filepath": "...", "duration": ..., "text": "...",
          "speaker": "spk04", "domain": "..."}
    b) CSV with header row:  audio_path,text,speaker,domain
    speaker and domain are optional, but PLEASE include them:
    later steps (per-speaker table, domain analysis) need them.

INSTALL (once):
    pip install jiwer==4.0.0
    # only needed for infer mode:
    pip install "nemo_toolkit[asr]"

NOTE ON MATCHING THE PAPER'S NUMBERS
    Normalization matches the convention documented in RESULTS.md v1.0:
    NFC; strip danda + ? . , ! ; : + ZWJ/ZWNJ; collapse whitespace.
    (Devanagari has no case, so no lowercasing.)
    The script prints CER both with spaces removed and with spaces kept,
    at corpus level (micro) and per-utterance mean (macro), so the exact
    scoring convention that produced 26.75% / 73.62% can be identified.
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Normalization — matches RESULTS.md v1.0 convention exactly:
# NFC; strip danda । + ? . , ! ; : + ZWJ/ZWNJ; collapse whitespace.
# ---------------------------------------------------------------------------
PUNCT_RE = re.compile(r"[\u0964?.!,;:\u200c\u200d]")


def normalize(text: str, remove_spaces: bool = False) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if remove_spaces:
        text = text.replace(" ", "")
    return text


# ---------------------------------------------------------------------------
# Scoring (jiwer, as cited in the paper)
# ---------------------------------------------------------------------------
try:
    import jiwer
except ImportError:
    sys.exit("jiwer is missing. Run:  pip install jiwer==4.0.0")


def _cer(ref: str, hyp: str) -> float:
    """Character error rate for one utterance; empty reference -> skip value."""
    if not ref:
        return float("nan")
    return jiwer.cer(ref, hyp)


def _wer(ref: str, hyp: str) -> float:
    if not ref.split():
        return float("nan")
    return jiwer.wer(ref, hyp)


def read_lines(path: str):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\r\n") for ln in f]


# ---------------------------------------------------------------------------
# Core: score a list of (ref, hyp) pairs, write per-utterance CSV, print summary
# ---------------------------------------------------------------------------
def score_pairs(pairs, out_path, id_names=None, extras=None, their_cers=None):
    rows = []
    skipped = 0
    sum_len_ns = sum_len_k = sum_edits_ns = sum_edits_k = 0.0
    sum_nwords = sum_wer_edits = 0.0
    for i, (ref_raw, hyp_raw) in enumerate(pairs):
        ref_ns = normalize(ref_raw, remove_spaces=True)
        hyp_ns = normalize(hyp_raw, remove_spaces=True)
        ref_k = normalize(ref_raw)
        hyp_k = normalize(hyp_raw)
        if not ref_ns:
            skipped += 1
            continue
        extra = extras[i] if extras else {"speaker": "", "domain": ""}
        cer_ns = _cer(ref_ns, hyp_ns)   # spaces removed
        cer_k = _cer(ref_k, hyp_k)      # spaces counted as characters
        wer_v = _wer(ref_k, hyp_k)
        rows.append({
            "id": (id_names[i] if id_names else f"utt_{i+1:04d}"),
            "speaker": extra["speaker"],
            "domain": extra["domain"],
            "ref_len_chars": len(ref_ns),
            "cer": cer_ns,
            "cer_kept": cer_k,
            "wer": wer_v,
            "cer_theirs": (their_cers[i] if their_cers and i < len(their_cers)
                           and their_cers[i] is not None else ""),
            "ref_norm": ref_k,
            "hyp_norm": hyp_k,
        })
        sum_len_ns += len(ref_ns)
        sum_len_k += len(ref_k)
        sum_edits_ns += cer_ns * len(ref_ns)
        sum_edits_k += cer_k * len(ref_k)
        nw = len(ref_k.split())
        sum_nwords += nw
        sum_wer_edits += wer_v * nw

    # ---- corpus-level (micro, weighted by length) + macro scores ----
    micro_cer_ns = sum_edits_ns / sum_len_ns if sum_len_ns else float("nan")
    micro_cer_k = sum_edits_k / sum_len_k if sum_len_k else float("nan")
    micro_wer = sum_wer_edits / sum_nwords if sum_nwords else float("nan")

    def _valid(key):
        return [r[key] for r in rows if r[key] == r[key]]

    def _mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    macro_cer_ns = _mean(_valid("cer"))
    macro_cer_k = _mean(_valid("cer_kept"))
    macro_wer = _mean(_valid("wer"))

    if not rows:
        sys.exit("ERROR: nothing scored (all references empty?).")

    # ---- write per-utterance CSV (reused in steps T2/T3/T4) ----
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    print("=" * 64)
    print("EVALUATION SUMMARY (T1 reproduction check)")
    print("=" * 64)
    print(f"Utterances scored:              {len(rows)}   (skipped empty ref: {skipped})")
    print(f"Corpus CER, spaces removed:     {micro_cer_ns * 100:6.2f}%")
    print(f"Corpus CER, spaces kept:        {micro_cer_k * 100:6.2f}%")
    print(f"Mean per-utt CER, sp removed:   {macro_cer_ns * 100:6.2f}%")
    print(f"Mean per-utt CER, sp kept:      {macro_cer_k * 100:6.2f}%")
    print(f"Corpus WER:                     {micro_wer * 100:6.2f}%")
    print(f"Mean per-utt WER:               {macro_wer * 100:6.2f}%")

    if their_cers:
        scored = [r for r in rows if r["cer_theirs"] != ""]
        if scored:
            d_ns = [abs(r["cer"] - float(r["cer_theirs"])) for r in scored]
            d_k = [abs(r["cer_kept"] - float(r["cer_theirs"])) for r in scored]
            m_ns = sum(x <= 1e-3 for x in d_ns)
            m_k = sum(x <= 1e-3 for x in d_k)
            print("-" * 64)
            print("Convention check vs the cer column already in your TSV:")
            print(f"  spaces-removed matches: {m_ns}/{len(scored)}   "
                  f"(mean |diff| {sum(d_ns) / len(d_ns):.4f})")
            print(f"  spaces-kept    matches: {m_k}/{len(scored)}   "
                  f"(mean |diff| {sum(d_k) / len(d_k):.4f})")
            winner = "spaces-removed" if m_ns >= m_k else "spaces-kept"
            print(f"  -> your e2 cer column used the {winner} convention")

    print("-" * 64)
    print("Paper headline (spk04, speaker-disjoint):  CER 26.75% / WER 73.62%")
    print(f"Per-utterance scores saved to: {out_path}")
    print("Any line hits 26.75 (+/-0.5)? -> pipeline verified -> Step T2.")
    print("None? -> paste this whole summary; we adjust the convention.")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Mode: score (refs + hyps already exist)
# ---------------------------------------------------------------------------
def mode_score(args):
    refs = read_lines(args.refs)
    hyps = read_lines(args.hyps)
    if len(refs) != len(hyps):
        sys.exit(f"ERROR: refs has {len(refs)} lines but hyps has {len(hyps)} lines. "
                 "They must be 1:1, same order.")
    score_pairs(list(zip(refs, hyps)), args.out)


# ---------------------------------------------------------------------------
# Mode: tsv (score a results table that already has ref + hyp columns)
# ---------------------------------------------------------------------------
def mode_tsv(args):
    with open(args.tsv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=args.delim)
        if reader.fieldnames is None:
            sys.exit("ERROR: file has no header row.")
        lower_map = {name.strip().lower(): name for name in reader.fieldnames}
        ref_col = lower_map.get(args.ref_col.lower())
        hyp_col = lower_map.get(args.hyp_col.lower())
        if not ref_col or not hyp_col:
            sys.exit(f"ERROR: could not find columns '{args.ref_col}' / '{args.hyp_col}'. "
                     f"Available columns: {reader.fieldnames}")
        id_col = lower_map.get("id")
        spk_col = lower_map.get("speaker")
        dom_col = lower_map.get("domain")
        cer_col = lower_map.get("cer")
        refs, hyps, ids, extras, theirs = [], [], [], [], []
        for i, row in enumerate(reader):
            refs.append(row[ref_col])
            hyps.append(row[hyp_col])
            raw_id = (row.get(id_col) or "").strip() if id_col else ""
            ids.append(raw_id or f"utt_{i+1:04d}")
            extras.append({
                "speaker": (row.get(spk_col) or "").strip() if spk_col else "",
                "domain": (row.get(dom_col) or "").strip() if dom_col else "",
            })
            if cer_col:
                val = (row.get(cer_col) or "").strip()
                theirs.append(float(val) if val else None)
            else:
                theirs.append(None)
    have_theirs = any(v is not None for v in theirs)
    print(f"Loaded {len(refs)} rows from {args.tsv}")
    score_pairs(list(zip(refs, hyps)), args.out, id_names=ids, extras=extras,
                their_cers=(theirs if have_theirs else None))


# ---------------------------------------------------------------------------
# Mode: infer (run the model, then score)
# ---------------------------------------------------------------------------
def load_manifest(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if not row.get("audio_path") or not row.get("text"):
                continue
            row.setdefault("speaker", "")
            row.setdefault("domain", "")
            rows.append(row)
    if not rows:
        sys.exit("ERROR: manifest has no usable rows (need audio_path + text columns).")
    return rows


def load_jsonl_manifest(path):
    """NeMo-style JSONL manifest (audio_filepath, text, speaker, domain)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append({
                "audio_path": d.get("audio_filepath", ""),
                "text": d.get("text", ""),
                "speaker": d.get("speaker", ""),
                "domain": d.get("domain", ""),
            })
    if not rows:
        sys.exit("ERROR: JSONL manifest has no rows.")
    return rows


def load_manifest_any(path):
    """Dispatch: .json/.jsonl -> NeMo JSONL manifest; anything else -> CSV."""
    if str(path).lower().endswith((".json", ".jsonl")):
        return load_jsonl_manifest(path)
    return load_manifest(path)


def load_model(model_id: str):
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        sys.exit('NeMo is missing (infer mode). Run:  pip install "nemo_toolkit[asr]"')
    if model_id.endswith(".nemo"):
        print(f"Loading local model: {model_id}")
        return nemo_asr.models.ASRModel.restore_from(model_id)
    print(f"Loading pretrained model: {model_id}")
    return nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)


def transcribe(model, paths, batch_size: int):
    out = model.transcribe(paths, batch_size=batch_size)
    texts = []
    for item in out:
        # NeMo versions differ: str, Hypothesis object, or nested list
        if isinstance(item, str):
            texts.append(item)
        elif hasattr(item, "text"):
            texts.append(item.text)
        elif isinstance(item, (list, tuple)) and item:
            first = item[0]
            texts.append(first if isinstance(first, str) else getattr(first, "text", ""))
        else:
            texts.append("")
    return texts


def mode_infer(args):
    manifest = load_manifest_any(args.manifest)
    audio_root = Path(args.audio_root)
    paths = [str(audio_root / m["audio_path"]) for m in manifest]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print(f"WARNING: {len(missing)} audio files not found, e.g. {missing[:3]}")
        print("         Check --audio-root / paths in manifest. Continuing anyway...")

    model = load_model(args.model)
    print(f"Transcribing {len(paths)} utterances (batch_size={args.batch_size})...")
    hyps = transcribe(model, paths, args.batch_size)
    refs = [m["text"] for m in manifest]
    id_names = [Path(m["audio_path"]).stem for m in manifest]
    extras = [{"speaker": m.get("speaker", ""), "domain": m.get("domain", "")}
              for m in manifest]
    score_pairs(list(zip(refs, hyps)), args.out, id_names=id_names, extras=extras)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Clean ASR eval pipeline for the Newari paper.")
    sub = p.add_subparsers(dest="mode", required=True)

    ps = sub.add_parser("score", help="score existing refs.txt / hyps.txt")
    ps.add_argument("--refs", required=True, help="reference text file, 1 utterance per line")
    ps.add_argument("--hyps", required=True, help="model output file, same order")
    ps.add_argument("--out", default="per_utterance_scores.csv")
    ps.set_defaults(func=mode_score)

    pt = sub.add_parser("tsv", help="score a results table with reference+hypothesis columns (e.g. e2_results_v1.tsv)")
    pt.add_argument("--tsv", required=True, help="TSV/CSV file with ref and hyp columns")
    pt.add_argument("--ref-col", default="reference", help="column name with reference text")
    pt.add_argument("--hyp-col", default="hypothesis", help="column name with model output")
    pt.add_argument("--delim", default="\t", help="column separator (default: tab)")
    pt.add_argument("--out", default="per_utterance_scores.csv")
    pt.set_defaults(func=mode_tsv)

    pi = sub.add_parser("infer", help="transcribe audio with a NeMo model, then score")
    pi.add_argument("--manifest", required=True, help="CSV: audio_path,text,speaker,domain")
    pi.add_argument("--model", required=True, help=".nemo checkpoint path or pretrained name")
    pi.add_argument("--audio-root", default=".", help="prefix for audio_path values")
    pi.add_argument("--batch-size", type=int, default=8)
    pi.add_argument("--out", default="per_utterance_scores.csv")
    pi.set_defaults(func=mode_infer)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

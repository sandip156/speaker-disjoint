#!/usr/bin/env python3
"""
T4: Error analysis for Nepal Bhasha (Newari) ASR.

Inputs
  --scores         per-utterance CSV from eval_newari.py (columns: id, speaker,
                   domain, cer_kept, wer, ref_norm, hyp_norm; other cols ignored)
  --train          train manifest (NeMo JSONL or JSON array with a text field)
                   [default: ~/Desktop/experiments/train_v1_274.json]
  --test-speaker   the speaker-disjoint test speaker [default: spk04]
  --out            prefix for output CSVs [default: error_analysis]

Sections printed
  1) Error-type decomposition (substitutions / deletions / insertions),
     overall + per domain + per speaker, using jiwer's own character alignment
  2) Top character confusions (1:1 substitutions), top deleted & inserted chars
  3) OOV analysis on the test set: rates, OOV-vs-WER correlation, bucket contrast
  4) Three example utterances (worst / median / best WER) for the paper table

Outputs (files with --out prefix)
  <out>_decomposition.csv   per-group sub/del/ins counts and percentages
  <out>_confusions.csv      ranked ref->hyp confusion pairs
  <out>_oov_per_utterance.csv  per-test-utterance OOV rate vs WER/CER
"""

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

try:
    from jiwer import process_characters
except ImportError:
    sys.exit("jiwer is required:  pip install jiwer==4.0.0")

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required:  pip install numpy")

try:
    from scipy import stats as sps
except ImportError:
    sps = None  # p-values / Welch t-test skipped if scipy is absent

# Same normalization convention as eval_newari.py / RESULTS.md v1.0
PUNCT_RE = re.compile(r"[\u0964?.!,;:\u200c\u200d]")


def norm_text(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = PUNCT_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def disp(ch: str) -> str:
    """Display helper so spaces are visible in confusion tables."""
    return "\u2423" if ch == " " else ch


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

ID_CANDIDATES = ("id", "utt_id", "uttid", "utterance_id",
                 "audio_filename", "filename", "audio", "clip")


def load_scores(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        need = {"speaker", "cer_kept", "wer", "ref_norm", "hyp_norm"}
        missing = need - set(rdr.fieldnames or [])
        if missing:
            sys.exit(f"scores CSV is missing columns: {sorted(missing)}\n"
                     f"  (regenerate it with:  python eval_newari.py tsv ...)")
        id_col = next((c for c in ID_CANDIDATES if c in (rdr.fieldnames or [])), None)
        for i, r in enumerate(rdr, 1):
            if not (r.get("ref_norm") or "").strip() and not (r.get("hyp_norm") or "").strip():
                continue
            r["speaker"] = (r.get("speaker") or "?").strip() or "?"
            r["domain"] = (r.get("domain") or "?").strip() or "?"
            r["id"] = (r.get(id_col) or "").strip() if id_col else ""
            if not r["id"]:
                r["id"] = f"row_{i:04d}"
            rows.append(r)
    if not rows:
        sys.exit("no usable rows in scores CSV")
    return rows


def load_master_index(path):
    """Optional: map (speaker, normalized ref text) -> corpus id from master.tsv.
    Used to replace synthetic row_NNNN ids with real corpus ids."""
    idx = {}
    if not path or not os.path.exists(path):
        return idx
    with open(path, newline="", encoding="utf-8-sig") as f:
        first = f.readline()
        f.seek(0)
        delim = "\t" if first.count("\t") > first.count(",") else ","
        for r in csv.DictReader(f, delimiter=delim):
            nid = (r.get("id") or "").strip()
            txt = norm_text(r.get("newar_sentence") or "")
            if not nid or not txt:
                continue
            spk = (r.get("speaker_id") or "").strip().lower()
            idx.setdefault((spk, txt), nid)
            idx.setdefault(txt, nid)  # text-only fallback key
    return idx


def load_train_texts(path):
    if not os.path.exists(path):
        sys.exit(f"train manifest not found: {path}\n"
                 f"  pass it explicitly:  --train /path/to/train_v1_274.json")
    items = []
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            items = data if isinstance(data, list) else data.get("data", [])
        else:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    texts = []
    for it in items:
        t = it.get("text") or it.get("sentence") or it.get("newar_sentence") \
            or it.get("raw_text") or ""
        if t:
            texts.append(t)
    if not texts:
        sys.exit("train manifest loaded but no text found (looked for keys: "
                 "text / sentence / newar_sentence / raw_text)")
    return texts


# ----------------------------------------------------------------------------
# Section 1 + 2: alignment-based decomposition and confusions
# ----------------------------------------------------------------------------

def align_one(ref, hyp):
    """Return (counts dict, conf pairs list, del chars list, ins chars list)."""
    m = process_characters(ref, hyp)
    ref_chars = m.references[0]
    hyp_chars = m.hypotheses[0]
    c = {"ref_len": len(ref_chars), "hits": m.hits, "sub": m.substitutions,
         "del": m.deletions, "ins": m.insertions}
    conf, dels, inss = [], [], []
    for chunk in m.alignments[0]:
        t = chunk.type
        if t == "substitute":
            a = ref_chars[chunk.ref_start_idx:chunk.ref_end_idx]
            b = hyp_chars[chunk.hyp_start_idx:chunk.hyp_end_idx]
            if len(a) == len(b):          # jiwer substitutes 1:1 per op
                for x, y in zip(a, b):
                    conf.append((x, y))
        elif t == "delete":
            dels.extend(ref_chars[chunk.ref_start_idx:chunk.ref_end_idx])
        elif t == "insert":
            inss.extend(hyp_chars[chunk.hyp_start_idx:chunk.hyp_end_idx])
    return c, conf, dels, inss


def section_decomposition(rows):
    print("=" * 74)
    print("1) ERROR-TYPE DECOMPOSITION  (jiwer char alignment, spaces kept)")
    print("   sub%/del%/ins% are shares of reference chars; "
          "CER-check = (sub+del+ins)/ref_len")
    print("-" * 74)

    groups = [("ALL", rows)]
    by_dom = defaultdict(list)
    by_spk = defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append(r)
        by_spk[r["speaker"]].append(r)
    for d in sorted(by_dom):
        groups.append((f"domain:{d}", by_dom[d]))
    for s in sorted(by_spk):
        groups.append((f"{s}", by_spk[s]))

    out_rows = []
    print(f"  {'group':<22}{'n':>5}{'sub%':>8}{'del%':>8}{'ins%':>8}{'CER-check':>11}")
    for name, grp in groups:
        tot = Counter()
        for r in grp:
            c = align_one(r["ref_norm"], r["hyp_norm"])[0]
            for k, v in c.items():
                tot[k] += v
        rl = max(tot["ref_len"], 1)
        subp, delp, insp = (100 * tot["sub"] / rl, 100 * tot["del"] / rl,
                            100 * tot["ins"] / rl)
        cer_chk = 100 * (tot["sub"] + tot["del"] + tot["ins"]) / rl
        print(f"  {name:<22}{len(grp):>5}{subp:>8.2f}{delp:>8.2f}"
              f"{insp:>8.2f}{cer_chk:>11.2f}")
        out_rows.append({"group": name, "n_utts": len(grp),
                         "ref_chars": tot["ref_len"], "hits": tot["hits"],
                         "substitutions": tot["sub"], "deletions": tot["del"],
                         "insertions": tot["ins"], "sub_pct": round(subp, 2),
                         "del_pct": round(delp, 2), "ins_pct": round(insp, 2),
                         "cer_check_pct": round(cer_chk, 2)})
    return out_rows


def section_confusions(rows, topk):
    conf, dels, inss = Counter(), Counter(), Counter()
    for r in rows:
        _, c, d, i = align_one(r["ref_norm"], r["hyp_norm"])
        conf.update(c)
        dels.update(d)
        inss.update(i)

    print("=" * 74)
    print(f"2) TOP CHARACTER CONFUSIONS  (top {topk}; \u2423 = space)")
    print("-" * 74)
    print(f"  {'ref':>4} -> {'hyp':<4}{'count':>8}")
    for (x, y), n in conf.most_common(topk):
        print(f"  {disp(x):>4} -> {disp(y):<4}{n:>8}")

    fmt = lambda cnt, k: ", ".join(f"{disp(ch)}:{n}" for ch, n in cnt.most_common(k))
    print(f"\n  Top deleted ref chars : {fmt(dels, 10)}")
    print(f"  Top inserted hyp chars: {fmt(inss, 10)}")

    conf_rows = [{"rank": i + 1, "ref_char": x, "hyp_char": y, "count": n}
                 for i, ((x, y), n) in enumerate(conf.most_common())]
    return conf_rows


# ----------------------------------------------------------------------------
# Section 3: OOV analysis
# ----------------------------------------------------------------------------

def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    if sps is not None:
        res = sps.pearsonr(x, y)
        return float(res[0]), float(res[1])
    return float(np.corrcoef(x, y)[0, 1]), float("nan")


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan"), float("nan")
    if sps is not None:
        res = sps.spearmanr(x, y)
        return float(res[0]), float(res[1])

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        rk = np.empty(len(v), float)
        rk[order] = np.arange(1, len(v) + 1)
        return rk
    r = float(np.corrcoef(rank(x), rank(y))[0, 1])
    return r, float("nan")


def section_oov(rows, train_texts, test_spk, out_prefix):
    vocab = set()
    for t in train_texts:
        vocab.update(norm_text(t).split())

    test_rows = [r for r in rows if r["speaker"] == test_spk]
    if not test_rows:
        print(f"\n(no rows for speaker {test_spk} - OOV section skipped)")
        return []

    per_utt = []
    n_tok = n_oov = 0
    for r in test_rows:
        toks = r["ref_norm"].split()
        oov = [t for t in toks if t not in vocab]
        n_tok += len(toks)
        n_oov += len(oov)
        per_utt.append({"id": r["id"], "domain": r["domain"],
                        "n_tokens": len(toks), "oov_tokens": len(oov),
                        "oov_rate": (len(oov) / len(toks)) if toks else 0.0,
                        "cer_kept": r["cer_kept"], "wer": r["wer"]})

    oov_types = set()
    for r in test_rows:
        oov_types.update(t for t in r["ref_norm"].split() if t not in vocab)

    print("=" * 74)
    print(f"3) OOV ANALYSIS  (test speaker {test_spk}, n={len(test_rows)}; "
          f"vocab from train manifest)")
    print("-" * 74)
    print(f"  train vocab types          : {len(vocab)}")
    print(f"  test tokens                : {n_tok}")
    tok_rate = 100 * n_oov / max(n_tok, 1)
    print(f"  OOV token rate             : {tok_rate:.2f}%  "
          f"({n_oov} tokens, {len(oov_types)} distinct types)")
    n_pos = sum(1 for u in per_utt if u["oov_tokens"] > 0)
    print(f"  utts with >=1 OOV token    : {n_pos} "
          f"({100 * n_pos / len(per_utt):.1f}%)")

    x = [u["oov_rate"] for u in per_utt]
    w = [float(u["wer"]) for u in per_utt]
    c = [float(u["cer_kept"]) for u in per_utt]
    rp, pp = pearson(x, w)
    rs, ps = spearman(x, w)
    ptxt = f", p={pp:.4g}" if pp == pp else ""
    stxt = f", p={ps:.4g}" if ps == ps else ""
    print(f"  Pearson r  OOV-rate vs WER : {rp:+.3f}{ptxt}")
    print(f"  Spearman rho OOV vs WER    : {rs:+.3f}{stxt}")

    g0 = [u for u in per_utt if u["oov_tokens"] == 0]
    g1 = [u for u in per_utt if u["oov_tokens"] > 0]
    if g0 and g1:
        m0 = 100 * np.mean([float(u["wer"]) for u in g0])
        m1 = 100 * np.mean([float(u["wer"]) for u in g1])
        c0 = 100 * np.mean([float(u["cer_kept"]) for u in g0])
        c1 = 100 * np.mean([float(u["cer_kept"]) for u in g1])
        print(f"  bucket WER : OOV=0 {m0:6.2f}% (n={len(g0)})   "
              f"OOV>0 {m1:6.2f}% (n={len(g1)})   gap {m1 - m0:+.2f} pts")
        print(f"  bucket CER : OOV=0 {c0:6.2f}%            "
              f"OOV>0 {c1:6.2f}%            gap {c1 - c0:+.2f} pts")
        if sps is not None:
            t, pv = sps.ttest_ind([float(u["wer"]) for u in g1],
                                  [float(u["wer"]) for u in g0], equal_var=False)
            print(f"  Welch t-test (WER, OOV>0 vs OOV=0): t={t:.3f}, p={pv:.4g}")

    out_csv = f"{out_prefix}_oov_per_utterance.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(per_utt[0].keys()))
        wtr.writeheader()
        wtr.writerows(per_utt)
    print(f"  saved -> {out_csv}")
    return per_utt


# ----------------------------------------------------------------------------
# Section 4: example utterances
# ----------------------------------------------------------------------------

def section_examples(rows, test_spk):
    test_rows = [r for r in rows if r["speaker"] == test_spk]
    if not test_rows:
        return
    srt = sorted(test_rows, key=lambda r: float(r["wer"]))
    picks = [("WORST", srt[-1]), ("MEDIAN", srt[len(srt) // 2]), ("BEST", srt[0])]
    print("=" * 74)
    print("4) EXAMPLE UTTERANCES  (qualitative table candidates)")
    print("-" * 74)
    for label, r in picks:
        print(f"  [{label}] id={r['id']}  domain={r['domain']}  "
              f"WER={100 * float(r['wer']):.1f}%  CER={100 * float(r['cer_kept']):.1f}%")
        print(f"    REF: {r['ref_norm']}")
        print(f"    HYP: {r['hyp_norm']}")
    print()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="T4 error analysis")
    ap.add_argument("--scores", default="scores_all431.csv")
    ap.add_argument("--train", default=os.path.expanduser(
        "~/Desktop/experiments/train_v1_274.json"))
    ap.add_argument("--test-speaker", default="spk04")
    ap.add_argument("--master", default="master.tsv",
                    help="master.tsv for real corpus ids (auto-join by ref text)")
    ap.add_argument("--out", default="error_analysis")
    ap.add_argument("--topk", type=int, default=15)
    args = ap.parse_args()

    rows = load_scores(args.scores)
    if all(r["id"].startswith("row_") for r in rows):
        master = load_master_index(args.master)
        if master:
            matched = 0
            for r in rows:
                nid = master.get((r["speaker"].lower(), r["ref_norm"])) \
                    or master.get(r["ref_norm"])
                if nid:
                    r["id"] = nid
                    matched += 1
            if matched:
                print(f"no id column in scores CSV - joined real corpus ids "
                      f"from {args.master} for {matched}/{len(rows)} utterances\n")
    train_texts = load_train_texts(args.train)

    print(f"Error analysis  ({len(rows)} utterances from {args.scores}; "
          f"train manifest: {os.path.basename(args.train)})\n")

    dec = section_decomposition(rows)
    with open(f"{args.out}_decomposition.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dec[0].keys()))
        w.writeheader()
        w.writerows(dec)
    print(f"  saved -> {args.out}_decomposition.csv")

    conf = section_confusions(rows, args.topk)
    with open(f"{args.out}_confusions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(conf[0].keys()))
        w.writeheader()
        w.writerows(conf)
    print(f"  saved -> {args.out}_confusions.csv\n")

    section_oov(rows, train_texts, args.test_speaker, args.out)
    section_examples(rows, args.test_speaker)

    if sps is None:
        print("(scipy not installed - p-values skipped; "
              "install with: pip install scipy)\n")


if __name__ == "__main__":
    main()

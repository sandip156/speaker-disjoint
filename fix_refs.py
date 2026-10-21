#!/usr/bin/env python3
"""
fix_refs.py — make two scores CSVs comparable for paired_bootstrap.py

Matches rows with progressively looser text keys, then writes a fixed copy
of B: rows in A's order, ref column copied EXACTLY from A (A is authoritative).

Usage:
  python fix_refs.py scores_t1.csv scores_ft_test127.csv scores_ft_test127_fixed.csv
"""

import csv
import difflib
import sys
import unicodedata
from collections import Counter, defaultdict

PUNCT = set("\u0964?.!,;:")


def norm_level(s, level):
    s = unicodedata.normalize("NFC", s or "")
    s = s.replace("\u200c", "").replace("\u200d", "")   # ZWNJ, ZWJ
    s = " ".join(s.split()).strip()
    if level >= 2:
        s = "".join(ch for ch in s if ch not in PUNCT)
        s = " ".join(s.split()).strip()
    if level >= 3:
        s = s.replace(" ", "")
    return s


def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = "\t" if sample.count("\t") > sample.count(",") else ","
        rows = []
        for r in csv.DictReader(f, delimiter=delim):
            rows.append({(k or "").strip(): (v or "") for k, v in r.items()})
    return rows


ID_CANDS = ["id", "utt_id", "uttid", "audio_filename", "audio_id",
            "filename", "file", "utt"]
REF_CANDS = ["ref_norm", "reference_norm", "ref", "reference",
             "ref_text", "ground_truth", "gt"]


def pick(cols, cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None


def clip(s, n=90):
    return (s[:n] + "...") if len(s) > n else s


def first_diffs(b_text, a_text, k=3):
    out = []
    sm = difflib.SequenceMatcher(None, b_text, a_text, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        fb, fa = b_text[i1:i2], a_text[j1:j2]
        out.append("      B has %r  [%s]" % (
            fb[:20], " ".join("U+%04X" % ord(c) for c in fb[:8])))
        out.append("      A has %r  [%s]" % (
            fa[:20], " ".join("U+%04X" % ord(c) for c in fa[:8])))
        if len(out) >= 2 * k:
            break
    return out


def looks_like_summary(r, rb, ib):
    refv = (r.get(rb, "") if rb else "").strip().lower()
    idv = (r.get(ib, "") if ib else "").strip().lower()
    if not refv:
        return True
    words = ("overall", "total", "mean", "average", "corpus", "summary")
    return any(w in refv for w in words) or any(w in idv for w in words)


def main():
    if len(sys.argv) != 4:
        sys.exit("usage: python fix_refs.py <scores_A.csv> <scores_B.csv> <fixed_B.csv>")
    pa, pb, pout = sys.argv[1], sys.argv[2], sys.argv[3]
    A, B = load(pa), load(pb)
    if not A or not B:
        sys.exit("ERROR: one of the files has no rows")
    ca, cb = list(A[0].keys()), list(B[0].keys())
    ra, rb = pick(ca, REF_CANDS), pick(cb, REF_CANDS)
    ia, ib = pick(ca, ID_CANDS), pick(cb, ID_CANDS)
    print("A: %s   rows=%d   ref_col=%r   id_col=%r" % (pa, len(A), ra, ia))
    print("B: %s   rows=%d   ref_col=%r   id_col=%r" % (pb, len(B), rb, ib))
    if ra is None or rb is None:
        sys.exit("ERROR: could not find the reference column.\n"
                 "  A cols=%s\n  B cols=%s" % (ca, cb))
    if len(A) != len(B):
        print("WARNING: row counts differ (A=%d, B=%d)" % (len(A), len(B)))

    # ---- progressive matching ----
    best = None  # (unmatched_b, oversub, level, match)
    for level in (1, 2, 3):
        idx = defaultdict(list)
        for i, r in enumerate(A):
            idx[norm_level(r[ra], level)].append(i)
        cursor = defaultdict(int)
        match = []
        for r in B:
            k = norm_level(r[rb], level)
            cands = idx.get(k, [])
            if cursor[k] < len(cands):
                match.append(cands[cursor[k]])
                cursor[k] += 1
            else:
                match.append(None)
        demand = Counter(norm_level(r[rb], level) for r in B)
        oversub = sum(1 for k, n in demand.items() if n > len(idx.get(k, [])))
        unmatched_b = sum(1 for m in match if m is None)
        print("  level %d: matched %d/%d  (B rows unmatched=%d, oversubscribed keys=%d)"
              % (level, len(match) - unmatched_b, len(B), unmatched_b, oversub))
        if best is None or (unmatched_b, oversub) < (best[0], best[1]):
            best = (unmatched_b, oversub, level, match)
        if unmatched_b == 0 and oversub == 0:
            break
    level, match = best[2], best[3]
    print("using matching level %d" % level)

    assigned = defaultdict(list)
    for b_i, a_i in enumerate(match):
        if a_i is not None:
            assigned[a_i].append(b_i)

    # ---- diagnosis: what exactly differed ----
    differed, shown = 0, 0
    for a_i in sorted(assigned):
        for b_i in assigned[a_i]:
            x, y = B[b_i][rb], A[a_i][ra]
            if x != y:
                differed += 1
                if shown < 5:
                    shown += 1
                    label = ("A id=%s" % A[a_i][ia]) if ia else ("A row %d" % a_i)
                    print("\n[%d] ref text differs (%s)" % (shown, label))
                    print("    A: %r" % clip(y))
                    print("    B: %r" % clip(x))
                    for line in first_diffs(x, y):
                        print(line)
    n_matched_rows = sum(len(v) for v in assigned.values())
    print("\nraw ref text differed in %d/%d matched rows" % (differed, n_matched_rows))

    unmatched_b = [i for i, m in enumerate(match) if m is None]
    unmatched_a = [i for i in range(len(A)) if i not in assigned]
    real_unmatched_b = [i for i in unmatched_b if not looks_like_summary(B[i], rb, ib)]
    dropped_summary = len(unmatched_b) - len(real_unmatched_b)
    if unmatched_b:
        print("\nB rows with no A match: %d (%d summary rows dropped, %d real)"
              % (len(unmatched_b), dropped_summary, len(real_unmatched_b)))
        for i in real_unmatched_b[:5]:
            idv = B[i].get(ib, "?") if ib else "?"
            print("    id=%s  ref=%r" % (idv, clip(B[i][rb])))
    if unmatched_a:
        print("\nA rows with no B match: %d" % len(unmatched_a))
        for i in unmatched_a[:5]:
            idv = A[i].get(ia, "?") if ia else "?"
            print("    id=%s  ref=%r" % (idv, clip(A[i][ra])))

    # ---- write fixed B ----
    n_written = 0
    with open(pout, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cb, restval="", extrasaction="ignore")
        w.writeheader()
        for a_i in range(len(A)):
            for b_i in assigned.get(a_i, []):
                row = dict(B[b_i])
                row[rb] = A[a_i][ra]          # exact ref text from A
                if ia and ib and ib in row and not row[ib].strip():
                    row[ia] = A[a_i][ia]
                w.writerow(row)
                n_written += 1
    print("\nwrote %d rows -> %s" % (n_written, pout))

    if real_unmatched_b or unmatched_a:
        print("\nNOTE: some rows could not be matched and were NOT written.")
        print("      STOP here and paste this entire output back to me.")
    else:
        print("\nAll %d rows matched. Next command:" % n_written)
        print('  python paired_bootstrap.py --scores-a %s --scores-b %s '
              '--b-name "NepConformer-FT" --no-copy' % (pa, pout))


if __name__ == "__main__":
    main()
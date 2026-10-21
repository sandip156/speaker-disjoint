"""fix_header.py — one-time repair of the speakers.tsv header."""
lines = open("speakers.tsv", encoding="utf-8").read().splitlines()
print("BEFORE:", lines[0])
cells = [c.strip() for c in lines[0].split("\t")]          # kills stray spaces
cells = ["consent_status" if c == "consant_status" else c for c in cells]
lines[0] = "\t".join(cells)
open("speakers.tsv", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("AFTER :", lines[0])
print("data rows:", len(lines) - 1)
"""Audit: every cite key used in the LaTeX sources must exist in references.bib and every bib entry must be cited.
   python report/scripts/audit_citations.py"""
import re
from pathlib import Path

R = Path(__file__).resolve().parents[1]
cited = {}
for f in [R / "Verif-AI_Project_Report.tex"] + sorted((R / "chapters").glob("*.tex")):
    s = f.read_text(encoding="utf-8")
    for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", s):
        for k in m.group(1).split(","):
            cited.setdefault(k.strip(), set()).add(f.name)
bib = (R / "references.bib").read_text(encoding="utf-8")
keys = re.findall(r"@\w+\{([^,]+),", bib)
print("bib entries:", len(keys), "| distinct cited keys:", len(cited))
print("cited but missing from bib:", sorted(set(cited) - set(keys)))
print("in bib but never cited:", sorted(set(keys) - set(cited)))
dup = {k for k in keys if keys.count(k) > 1}
print("duplicate keys:", sorted(dup))
for k in sorted(cited):
    print(f"  {k:22s} cited in {', '.join(sorted(cited[k]))}")

import zipfile, pathlib
R = pathlib.Path(__file__).resolve().parents[1]
out = R / "Verif-AI-report.zip"
if out.exists(): out.unlink()
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(R.rglob("*")):
        if p.is_file() and p != out and "__pycache__" not in p.parts and "handbook" not in p.parts and p.suffix not in (".aux", ".log", ".toc", ".lof", ".lot", ".out", ".bbl", ".blg", ".fls", ".fdb_latexmk"):
            z.write(p, "report/" + p.relative_to(R).as_posix())
print(out, round(out.stat().st_size/1e6, 2), "MB")

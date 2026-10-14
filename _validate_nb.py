import json, ast, sys

nb = json.load(open("pencil_kaggle_smoketest.ipynb", encoding="utf-8"))
assert nb["nbformat"] == 4, "bad nbformat"
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
print("total cells:", len(nb["cells"]), "| code cells:", len(code))

bad = 0
for i, c in enumerate(code):
    src = "".join(c["source"])
    try:
        ast.parse(src)
    except SyntaxError as e:
        bad += 1
        print(f"  SYNTAX ERROR in code cell {i}: line {e.lineno}: {e.msg}")
        print("   >>", (e.text or "").rstrip())

for c in nb["cells"]:
    assert isinstance(c["source"], list), "source must be a list of lines"
    assert c["cell_type"] in ("code", "markdown")
    if c["cell_type"] == "code":
        assert "outputs" in c and "execution_count" in c, "code cell missing keys"

print("syntax errors:", bad)
print("structure: OK" if bad == 0 else "structure: OK but FIX SYNTAX")
sys.exit(1 if bad else 0)

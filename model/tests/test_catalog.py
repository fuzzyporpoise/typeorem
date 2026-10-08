import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catalog import parse_metadata, normalize_family
from instances import build_instances, pairing_eligible, css2_fragment

passes = 0
fails = 0


def ok(label, cond):
    global passes, fails
    if cond:
        passes += 1
    else:
        fails += 1
        print(f"FAIL {label}")


FIXTURE = '''name: "Widget"
category: "SANS_SERIF"
fonts {
  name: "Widget"
  style: "normal"
  weight: 400
  filename: "Widget[wght].ttf"
  copyright: "Reserved Font Name \\"Widget\\". See http://example.com/{a}"
}
fonts {
  name: "Widget"
  style: "italic"
  weight: 400
  filename: "Widget-Italic[wght].ttf"
}
subsets: "latin"
subsets: "menu"
axes {
  tag: "wght"
  min_value: 100.0
  max_value: 900.0
}
'''

raw = normalize_family(parse_metadata(FIXTURE))
ok("family name parsed", raw["family"] == "Widget")
ok("category normalized", raw["category"] == "sans-serif")
ok("variable detected", raw["variable"] is True)
ok("two font blocks (escaped quotes do not merge)", len(raw["fonts"]) == 2)
ok("weights parsed", [f["weight"] for f in raw["fonts"]] == [400, 400])
ok("styles parsed", [f["style"] for f in raw["fonts"]] == ["normal", "italic"])
ok("axes parsed", raw["axes"] == [{"tag": "wght", "min": 100.0, "max": 900.0}])
ok("subsets parsed", raw["subsets"] == ["latin", "menu"])

ok("latin+menu eligible", pairing_eligible(["latin", "menu"]))
ok("latin+math+symbols eligible", pairing_eligible(["latin", "math", "symbols", "menu"]))
ok("CJK primary not eligible", not pairing_eligible(["japanese", "latin", "latin-ext", "menu"], "Jpan"))
ok("no latin subset not eligible", not pairing_eligible(["japanese", "menu"], "Jpan"))
ok("dual-script latin (Devanagari primary) eligible", pairing_eligible(["devanagari", "latin", "menu"], "Deva"))
ok("arabic-primary dual-script eligible", pairing_eligible(["arabic", "latin", "menu"], "Arab"))

blocked = {"Widget": {"pairs": {(400, "normal"), (400, "italic")}, "primaryScript": ""}}
instances, excluded, dropped = build_instances([raw], blocked)
ids = [i["id"] for i in instances]
ok("variable ladder expands to 4 weights x 2 styles", len(instances) == 8)
ok("ladder values", sorted({i["weight"] for i in instances}) == [300, 400, 700, 900])
ok("italic present", "widget-700-italic" in ids)
ok("stable id format", "widget-400-normal" in ids)
ok("coords recorded", all(i["coords"] == {"wght": i["weight"]} for i in instances))
ok("css2 italic fragment", css2_fragment("Montserrat", 700, "italic", True) == "Montserrat:ital,wght@1,700")
ok("css2 upright fragment", css2_fragment("Widget", 400, "normal", False) == "Widget:wght@400")
ok("no exclusion for a published family", excluded == [])

print(f"{passes} passed, {fails} failed")
sys.exit(1 if fails else 0)

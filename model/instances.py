import json
import re
from pathlib import Path

from catalog import load_catalog, google_fonts_commit, write_json

POLICY_VERSION = 3
WGHT_LADDER = (300, 400, 700, 900)
# CJK primary scripts are excluded from the Latin pairing pool (huge files, generic
# Latin fallback). Everything else that ships a latin subset is in, so dual-script
# Latin-popular faces (Poppins/Devanagari, Cairo/Arabic, Open Sans/Hebrew) qualify.
CJK_SCRIPTS = {"Jpan", "Kore", "Hans", "Hant", "Hani"}


def slug(name):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", name.lower())).strip("-")


def pairing_eligible(subsets, primary_script=""):
    return "latin" in set(subsets) and primary_script not in CJK_SCRIPTS


def css2_fragment(family, weight, style, has_italic):
    if has_italic:
        return f"{family}:ital,wght@{1 if style == 'italic' else 0},{weight}"
    return f"{family}:wght@{weight}"


def load_served(endpoint_path):
    data = json.loads(Path(endpoint_path).read_text())
    served = {}
    for f in data["familyMetadataList"]:
        pairs = set()
        for key in (f.get("fonts") or {}):
            m = re.fullmatch(r"(\d+)(i?)", key)
            if m:
                pairs.add((int(m.group(1)), "italic" if m.group(2) else "normal"))
        served[f["family"]] = {"pairs": pairs, "primaryScript": f.get("primaryScript", "")}
    return served


def build_instances(families, served):
    instances = []
    excluded = []
    dropped_weights = 0
    for fam in families:
        name = fam["family"]
        if name not in served:
            excluded.append(name)
            continue
        has_italic = any(f["style"] == "italic" for f in fam["fonts"])
        pairs = []
        if fam["variable"]:
            wght = next((a for a in fam["axes"] if a["tag"] == "wght"), None)
            ladder = [w for w in WGHT_LADDER if wght and wght["min"] <= w <= wght["max"]]
            for style in ("normal", "italic"):
                if style == "italic" and not has_italic:
                    continue
                fn = next((f["filename"] for f in fam["fonts"] if f["style"] == style and "[" in f["filename"]), "")
                for w in ladder:
                    pairs.append((w, style, fn, True))
        else:
            for f in fam["fonts"]:
                if (f["weight"], f["style"]) in served[name]["pairs"]:
                    pairs.append((f["weight"], f["style"], f["filename"], False))
                else:
                    dropped_weights += 1
        seen = set()
        for weight, style, filename, variable in pairs:
            if (weight, style) in seen:
                continue
            seen.add((weight, style))
            instances.append({
                "id": f"{slug(name)}-{weight}-{style}",
                "family": name,
                "category": fam["category"],
                "weight": weight,
                "style": style,
                "variable": variable,
                "file": filename,
                "coords": {"wght": weight} if variable else {},
                "css2": css2_fragment(name, weight, style, has_italic),
                "pairingEligible": pairing_eligible(fam["subsets"], served[name]["primaryScript"]),
            })
    instances.sort(key=lambda i: (i["family"].lower(), i["weight"], i["style"]))
    return instances, excluded, dropped_weights


def main():
    base = Path(__file__).resolve().parent
    families = load_catalog(base / ".cache" / "google-fonts")
    served = load_served(base / ".cache" / "gf_meta.json")
    instances, excluded, dropped = build_instances(families, served)
    write_json(base / "out" / "catalog.raw.json", families)
    write_json(base / "out" / "instances.json", instances)
    write_json(base / "out" / "policy.json", {
        "policyVersion": POLICY_VERSION,
        "wghtLadder": list(WGHT_LADDER),
        "excludedCjkScripts": sorted(CJK_SCRIPTS),
        "eligibleRule": "has latin subset and primaryScript not CJK",
        "googleFontsCommit": google_fonts_commit(base / ".cache" / "google-fonts"),
        "counts": {
            "familiesInRepo": len(families),
            "familiesPublished": len(served),
            "familiesKept": len(families) - len(excluded),
            "familiesUnpublished": len(excluded),
            "staticWeightsDroppedNotServed": dropped,
            "instances": len(instances),
            "pairingEligibleInstances": sum(1 for i in instances if i["pairingEligible"]),
        },
    })
    counts = json.loads((base / "out" / "policy.json").read_text())["counts"]
    print(f"policy v{POLICY_VERSION}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  unpublished (excluded): {', '.join(excluded[:6])}{' ...' if len(excluded) > 6 else ''}")


if __name__ == "__main__":
    main()

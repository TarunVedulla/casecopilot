"""Deterministic keyword pre-screen in front of the LLM (proposed safeguard).

The keyword list is derived from the EU GVP definition of adverse events and
special situations (symptoms, hospitalisation, pregnancy/breastfeeding exposure,
overdose, medication error, accidental exposure, lack of effect), in English and
German. It is applied to the 40 test emails and combined with the recorded v3
model flag: final AE flag = model flag OR keyword hit.
NOTE: the list was written after the v3 error analysis, so the result on this
test set is optimistic and must be confirmed on new emails.
"""
import csv, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent
KEYWORDS = [
    # symptoms / events
    r"side effect", r"nebenwirkung", r"rash", r"ausschlag", r"headache", r"kopfschmerz", r"dizz", r"schwindel",
    r"swollen", r"swelling", r"geschwollen", r"cramp", r"krämpf", r"tired", r"müde", r"bleed", r"blutet",
    r"pain", r"schmerz", r"fever", r"fieber", r"infection", r"infektion", r"nausea", r"übelkeit",
    # serious outcomes
    r"hospital", r"krankenhaus", r"klinik", r"emergency", r"notaufnahme",
    # special situations
    r"pregnan", r"schwanger", r"breastfeed", r"stillen", r"overdose", r"überdos", r"two tablets", r"zwei tabletten",
    r"double dose", r"by mistake", r"versehentlich", r"swallowed", r"verschluckt", r"stochen", r"needle", r"nadel",
    r"no improvement", r"not working", r"wirkt nicht", r"keine besserung",
]
PAT = re.compile("|".join(KEYWORDS), re.I)

def main():
    cases = {r["case_id"]: r for r in csv.DictReader(open(ROOT / "data/cases.csv", encoding="utf-8"))}
    model = {}
    for l in open(ROOT / "results/raw_v3.jsonl", encoding="utf-8"):
        r = json.loads(l); model[r["case_id"]] = bool(json.loads(r["output"]).get("adverse_event"))
    rows, tp = [], {"model": 0, "kw": 0, "combined": 0}
    fp, fn = dict(tp), dict(tp)
    for cid, c in cases.items():
        gold = c["gold_ae"] == "1"
        hit = PAT.search(c["subject"] + " " + c["body"])
        flags = {"model": model[cid], "kw": bool(hit), "combined": model[cid] or bool(hit)}
        for k, v in flags.items():
            tp[k] += gold and v; fp[k] += (not gold) and v; fn[k] += gold and not v
        rows.append([cid, gold, model[cid], hit.group(0) if hit else "", flags["combined"]])
    out = ROOT / "results/prescreen_v3.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["case_id", "gold_ae", "model_ae", "keyword_hit", "combined_ae"]); w.writerows(rows)
    for k in ["model", "kw", "combined"]:
        rec = tp[k] / (tp[k] + fn[k]); prec = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else 0
        print(f"{k:9s} recall {rec:.1%}  precision {prec:.1%}  FN {fn[k]}  FP {fp[k]}")
    print("false positives (combined):", [r[0] for r in rows if r[4] and not r[1]])
    print("false negatives (combined):", [r[0] for r in rows if r[1] and not r[4]])

if __name__ == "__main__":
    main()

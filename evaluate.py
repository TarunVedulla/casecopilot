"""CaseCopilot evaluation harness.

Runs each system-prompt version against the synthetic case set, scores the
outputs against the gold labels and writes:
  results/raw_<version>.jsonl      every model response with latency and tokens
  results/metrics.csv              one row of metrics per prompt version
  results/errors_<version>.csv     misclassified cases (for the error analysis)
  results/review_sheet_v3.csv      draft replies to rate by hand
  report/results.tex               LaTeX macros used by report.tex
  report/figures/*.pdf             charts used by report.tex

Usage
  export GROQ_API_KEY=...             (free key: console.groq.com)
  python evaluate.py --provider groq --model openai/gpt-oss-120b
  python evaluate.py --provider anthropic --model <model-id>   (needs ANTHROPIC_API_KEY)
  python evaluate.py --provider mock      (offline pipeline test - never report these numbers)
  python evaluate.py --report-only        (rebuild metrics/tex after filling the review sheet / time study)
  python evaluate.py --placeholders       (write empty results.tex so the report compiles)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "prompts"
DATA = ROOT / "data" / "cases.csv"
RES = ROOT / "results"
REPORT = ROOT / "report"
FIG = REPORT / "figures"

CATS = ["ADVERSE_EVENT", "PRODUCT_COMPLAINT", "MEDICAL_INFORMATION",
        "PATIENT_SUPPORT", "ORDER_SUPPLY", "OTHER"]
CAT_SHORT = {"ADVERSE_EVENT": "AE", "PRODUCT_COMPLAINT": "PC", "MEDICAL_INFORMATION": "MI",
             "PATIENT_SUPPORT": "PSP", "ORDER_SUPPLY": "OS", "OTHER": "OTH"}
PRIOS = ["HIGH", "MEDIUM", "LOW"]
VERSIONS = ["v1", "v2", "v3"]
WORD = {"v1": "One", "v2": "Two", "v3": "Three"}


# --------------------------------------------------------------------------- data
def load_cases() -> list[dict]:
    with DATA.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["gold_ae"] = r["gold_ae"] == "1"
    return rows


def user_message(case: dict) -> str:
    return (f"Sender type (from Salesforce contact record): {case['sender_type']}\n"
            f"Subject: {case['subject']}\n\n{case['body']}")


# ------------------------------------------------------------------------ providers
def call_openai_compatible(base_url, key, model, system, user):
    import requests
    r = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": 0, "max_tokens": 4000, **({"reasoning_effort": "low"} if "gpt-oss" in model else {}),
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]},
        timeout=90)
    if r.status_code == 429:
        raise RateLimited(r.headers.get("retry-after", "20"))
    if r.status_code != 200:
        sys.exit(f"API error {r.status_code}: {r.text[:500]}")
    j = r.json()
    u = j.get("usage", {})
    return j["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def call_anthropic(key, model, system, user):
    import requests
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": 900, "temperature": 0, "system": system,
              "messages": [{"role": "user", "content": user}]},
        timeout=90)
    if r.status_code == 429:
        raise RateLimited(r.headers.get("retry-after", "20"))
    if r.status_code != 200:
        sys.exit(f"API error {r.status_code}: {r.text[:500]}")
    j = r.json()
    text = "".join(b.get("text", "") for b in j["content"] if b.get("type") == "text")
    u = j.get("usage", {})
    return text, u.get("input_tokens", 0), u.get("output_tokens", 0)


class RateLimited(Exception):
    pass


def call_mock(version, system, user):
    """Keyword heuristic used only to test the pipeline offline."""
    t = user.lower()
    ae = any(k in t for k in ["headache", "rash", "hospital", "pregnan", "two cardoxan", "swallowed",
                              "schwindel", "swollen", "cramps", "müde", "no improvement", "blutet"])
    if ae:
        cat = "ADVERSE_EVENT"
    elif any(k in t for k in ["crack", "damaged", "leaflet", "trüb", "wrong", "cap"]):
        cat = "PRODUCT_COMPLAINT"
    elif any(k in t for k in ["programme", "nurse", "programm", "reminder"]):
        cat = "PATIENT_SUPPORT"
    elif any(k in t for k in ["stock", "invoice", "deliver", "liefer", "return", "order", "account"]):
        cat = "ORDER_SUPPLY"
    elif any(k in t for k in ["job", "agency", "click here"]):
        cat = "OTHER"
    else:
        cat = "MEDICAL_INFORMATION"
    prio = "HIGH" if ae else "LOW"
    if version == "v1":
        return f"Category: {cat.replace('_', ' ').title()}\nPriority: {prio}\nAdverse event: {'yes' if ae else 'no'}", 300, 120
    out = {"category": cat, "priority": prio, "adverse_event": ae, "summary": "mock",
           "missing_information": [], "draft_reply": "Dear Sir or Madam, thank you. (mock)"}
    return json.dumps(out), 900 if version == "v2" else 2100, 200


def make_caller(args):
    if args.provider == "mock":
        return lambda v, s, u: call_mock(v, s, u)
    if args.provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY") or sys.exit("Set ANTHROPIC_API_KEY")
        return lambda v, s, u: call_anthropic(key, args.model, s, u)
    base = {"groq": "https://api.groq.com/openai/v1"}.get(args.provider, args.base_url)
    env = {"groq": "GROQ_API_KEY"}.get(args.provider, "OPENAI_API_KEY")
    key = os.environ.get(env) or sys.exit(f"Set {env}")
    return lambda v, s, u: call_openai_compatible(base, key, args.model, s, u)


# -------------------------------------------------------------------------- parsing
def strict_json(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def norm_cat(s: str) -> str:
    """Maps free-text categories (prompt v1 has no category scheme) onto the six codes."""
    s = (s or "").lower()
    rules = [
        ("ADVERSE_EVENT", ["adverse", "side effect", "safety", "pharmacovigilance"]),
        ("OTHER", ["job", "application", " hr", "recruit", "marketing", "partnership", "sales",
                   "spam", "phishing", "unsolicited", "feedback", "compliment", "thank"]),
        ("PRODUCT_COMPLAINT", ["complaint", "quality", "defect", "device", "malfunction", "damaged",
                               "product issue", "appearance", "packaging", "leaflet"]),
        ("PATIENT_SUPPORT", ["support", "programme", "program", "enrol", "nurse", "reminder"]),
        ("ORDER_SUPPLY", ["order", "supply", "deliver", "stock", "invoice", "billing", "return",
                          "account", "logistic", "availability", "shortage"]),
        ("MEDICAL_INFORMATION", ["medical", "information", "medication", "dosing", "dose", "storage",
                                 "interaction", "clinical", "study", "inquiry", "enquiry", "question"]),
    ]
    for code, keys in rules:
        if any(k in s for k in keys):
            return code
    return "OTHER"


def norm_prio(s: str) -> str:
    s = (s or "").upper()
    for p in PRIOS:
        if p in s:
            return p
    if "URGENT" in s or "CRITICAL" in s:
        return "HIGH"
    return "LOW"


def to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "yes", "1", "ja"}


def extract(text: str) -> dict:
    """Strict JSON first; free-text fallback so that v1 can still be scored."""
    obj = strict_json(text)
    valid = obj is not None
    if obj is None:
        m = re.search(r"\{.*\}", text, re.S)
        obj = strict_json(m.group(0)) if m else None
    if obj is not None:
        cat_raw = str(obj.get("category", ""))
        return {"json_valid": valid,
                "category": cat_raw if cat_raw in CATS else norm_cat(cat_raw),
                "priority": norm_prio(str(obj.get("priority", ""))),
                "adverse_event": to_bool(obj.get("adverse_event", obj.get("adverseEvent", False))),
                "draft_reply": obj.get("draft_reply", obj.get("reply", "")),
                "summary": obj.get("summary", "")}

    def grab(label):
        # label at the start of a line, e.g. "**Category:** Order / Delivery"
        m = re.search(rf"^[\W_]*(?:[A-Za-z-]+\s+){{0,2}}{label}[^:\n]*:[\s*_]*([^\n]+)", text, re.I | re.M)
        return m.group(1) if m else ""
    ae_txt = grab("adverse event")
    return {"json_valid": False,
            "category": norm_cat(grab("categor")),
            "priority": norm_prio(grab("priorit")),
            "adverse_event": bool(re.match(r"\W*(yes|true|ja|possibly|potential)", ae_txt, re.I)),
            "draft_reply": grab("reply") or grab("draft"),
            "summary": grab("summary")}


# --------------------------------------------------------------------------- running
def run(args):
    cases = load_cases()
    caller = make_caller(args)
    RES.mkdir(exist_ok=True)
    for v in args.versions:
        system = (PROMPTS / f"system_{v}.txt").read_text(encoding="utf-8")
        out_path = RES / f"raw_{v}.jsonl"
        done = set()
        if out_path.exists() and not args.fresh:
            done = {json.loads(l)["case_id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        with out_path.open("a" if done else "w", encoding="utf-8") as f:
            for c in cases:
                if c["case_id"] in done:
                    continue
                for attempt in range(6):
                    try:
                        t0 = time.perf_counter()
                        text, tin, tout = caller(v, system, user_message(c))
                        lat = time.perf_counter() - t0
                        break
                    except RateLimited as e:
                        wait = float(e.args[0]) if str(e.args[0]).replace(".", "").isdigit() else 20
                        print(f"  rate limited, waiting {wait:.0f}s"); time.sleep(wait + 1)
                else:
                    sys.exit("Too many rate-limit retries; rerun later (finished cases are kept).")
                rec = {"case_id": c["case_id"], "version": v, "provider": args.provider,
                       "model": args.model, "latency_s": round(lat, 3),
                       "tokens_in": tin, "tokens_out": tout, "output": text,
                       "run_date": date.today().isoformat()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                print(f"{v} {c['case_id']} {lat:.1f}s")
                time.sleep(args.sleep)


# -------------------------------------------------------------------------- scoring
def f1_macro(gold, pred, labels):
    f1s = []
    for l in labels:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
    return sum(f1s) / len(f1s)


def score(args):
    cases = {c["case_id"]: c for c in load_cases()}
    rows, per_case = [], {}
    for v in VERSIONS:
        p = RES / f"raw_{v}.jsonl"
        if not p.exists():
            continue
        recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        recs = [r for r in recs if r["case_id"] in cases]
        if not recs:
            continue
        ex = [(r, extract(r["output"]), cases[r["case_id"]]) for r in recs]
        per_case[v] = ex
        gold_c = [c["gold_category"] for _, _, c in ex]
        pred_c = [e["category"] for _, e, _ in ex]
        gold_ae = [c["gold_ae"] for _, _, c in ex]
        pred_ae = [e["adverse_event"] for _, e, _ in ex]
        tp = sum(g and p for g, p in zip(gold_ae, pred_ae))
        fp = sum((not g) and p for g, p in zip(gold_ae, pred_ae))
        fn = sum(g and not p for g, p in zip(gold_ae, pred_ae))
        tin = statistics.mean(r["tokens_in"] for r, _, _ in ex)
        tout = statistics.mean(r["tokens_out"] for r, _, _ in ex)
        rows.append({
            "version": v, "n": len(ex), "model": recs[0]["model"], "provider": recs[0]["provider"],
            "run_date": recs[-1].get("run_date", ""),
            "json_valid": sum(e["json_valid"] for _, e, _ in ex) / len(ex),
            "cat_acc": sum(g == p for g, p in zip(gold_c, pred_c)) / len(ex),
            "cat_f1": f1_macro(gold_c, pred_c, CATS),
            "prio_acc": sum(c["gold_priority"] == e["priority"] for _, e, c in ex) / len(ex),
            "ae_recall": tp / (tp + fn) if tp + fn else 0,
            "ae_precision": tp / (tp + fp) if tp + fp else 0,
            "ae_fn": fn, "ae_fp": fp,
            "latency": statistics.median(r["latency_s"] for r, _, _ in ex),
            "tok_in": tin, "tok_out": tout,
            "cost_k": (tin * args.price_in + tout * args.price_out) / 1e6 * 1000,
        })
        with (RES / f"errors_{v}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "difficulty", "gold_category", "pred_category", "gold_priority",
                        "pred_priority", "gold_ae", "pred_ae", "body"])
            for r, e, c in ex:
                if (e["category"], e["priority"], e["adverse_event"]) != (c["gold_category"], c["gold_priority"], c["gold_ae"]):
                    w.writerow([c["case_id"], c["difficulty"], c["gold_category"], e["category"],
                                c["gold_priority"], e["priority"], c["gold_ae"], e["adverse_event"], c["body"]])
    if not rows:
        sys.exit("No results found in results/. Run with --provider first.")
    with (RES / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    write_review_sheet(per_case.get("v3"))
    review = read_review()
    timing = read_time_study()
    write_tex(rows, review, timing)
    plot(rows, per_case)
    for r in rows:
        print(f"{r['version']}: JSON {r['json_valid']:.0%} | category {r['cat_acc']:.0%} (F1 {r['cat_f1']:.2f}) | "
              f"priority {r['prio_acc']:.0%} | AE recall {r['ae_recall']:.0%} precision {r['ae_precision']:.0%} "
              f"| FN {r['ae_fn']} | {r['latency']:.2f}s | {r['tok_in']:.0f}/{r['tok_out']:.0f} tok")


# --------------------------------------------------------------- human review + time
REVIEW_COLS = ["case_id", "category_pred", "draft_reply",
               "factually_correct_0_1", "no_medical_advice_0_1", "tone_1_5", "edit_needed_none_minor_major"]


def write_review_sheet(ex):
    p = RES / "review_sheet_v3.csv"
    if ex is None or p.exists():
        return
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(REVIEW_COLS)
        for r, e, c in ex:
            w.writerow([c["case_id"], e["category"], e["draft_reply"], "", "", "", ""])


def read_review():
    p = RES / "review_sheet_v3.csv"
    if not p.exists():
        return None
    rows = [r for r in csv.DictReader(p.open(encoding="utf-8")) if r.get("tone_1_5", "").strip()]
    if not rows:
        return None
    n = len(rows)
    return {"n": n,
            "factual": sum(r["factually_correct_0_1"].strip() == "1" for r in rows) / n,
            "noadvice": sum(r["no_medical_advice_0_1"].strip() == "1" for r in rows) / n,
            "tone": statistics.mean(float(r["tone_1_5"]) for r in rows),
            "noedit": sum(r["edit_needed_none_minor_major"].strip().lower() == "none" for r in rows) / n,
            "minor": sum(r["edit_needed_none_minor_major"].strip().lower() == "minor" for r in rows) / n}


def read_time_study():
    p = RES / "time_study.csv"
    if not p.exists():
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["case_id", "manual_seconds", "assisted_seconds"])
            for cid in ["C02", "C05", "C07", "C11", "C14", "C20", "C26", "C28", "C31", "C36"]:
                w.writerow([cid, "", ""])
        return None
    rows = [r for r in csv.DictReader(p.open(encoding="utf-8"))
            if r["manual_seconds"].strip() and r["assisted_seconds"].strip()]
    if not rows:
        return None
    man = statistics.median(float(r["manual_seconds"]) for r in rows)
    ass = statistics.median(float(r["assisted_seconds"]) for r in rows)
    return {"n": len(rows), "manual": man, "assisted": ass, "saving": 1 - ass / man}


# ------------------------------------------------------------------------ LaTeX out
PH = r"\textcolor{red}{[run]}"


def pct(x):
    return f"{x * 100:.1f}\\,\\%"


def write_tex(rows, review, timing, placeholders=False):
    REPORT.mkdir(exist_ok=True)
    by = {r["version"]: r for r in rows}
    L = ["% generated by evaluate.py - do not edit by hand"]
    mock = any(r["provider"] == "mock" for r in rows)

    def m(name, val):
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    for v in VERSIONS:
        r = by.get(v)
        g = (lambda k, f: PH) if r is None or placeholders else (lambda k, f: f(r[k]))
        W = WORD[v]
        m(f"V{W}Json", g("json_valid", pct)); m(f"V{W}CatAcc", g("cat_acc", pct))
        m(f"V{W}CatFone", g("cat_f1", lambda x: f"{x:.2f}")); m(f"V{W}PrioAcc", g("prio_acc", pct))
        m(f"V{W}AeRecall", g("ae_recall", pct)); m(f"V{W}AePrec", g("ae_precision", pct))
        m(f"V{W}AeFn", g("ae_fn", str)); m(f"V{W}AeFp", g("ae_fp", str))
        m(f"V{W}Latency", g("latency", lambda x: f"{x:.2f}"))
        m(f"V{W}TokIn", g("tok_in", lambda x: f"{x:,.0f}")); m(f"V{W}TokOut", g("tok_out", lambda x: f"{x:,.0f}"))
        m(f"V{W}CostK", g("cost_k", lambda x: f"{x:.2f}"))
    any_r = rows[0] if rows and not placeholders else None
    m("ModelName", PH if any_r is None else any_r["model"].replace("_", r"\_"))
    m("RunDate", PH if any_r is None else any_r["run_date"])
    m("MockWarning", r"\textcolor{red}{\textbf{MOCK RUN -- NOT REAL RESULTS}}" if mock and not placeholders else "")
    rv = None if placeholders else review
    m("RevN", PH if rv is None else str(rv["n"]))
    m("RevFactual", PH if rv is None else pct(rv["factual"]))
    m("RevNoAdvice", PH if rv is None else pct(rv["noadvice"]))
    m("RevTone", PH if rv is None else f"{rv['tone']:.1f}")
    m("RevNoEdit", PH if rv is None else pct(rv["noedit"]))
    m("RevMinor", PH if rv is None else pct(rv["minor"]))
    tm = None if placeholders else timing
    m("TimeN", PH if tm is None else str(tm["n"]))
    m("TimeManual", PH if tm is None else f"{tm['manual']:.0f}")
    m("TimeAssisted", PH if tm is None else f"{tm['assisted']:.0f}")
    m("TimeSaving", PH if tm is None else pct(tm["saving"]))
    (REPORT / "results.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def plot(rows, per_case):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    keys = [("json_valid", "Valid JSON"), ("cat_acc", "Category acc."), ("prio_acc", "Priority acc."),
            ("ae_recall", "AE recall"), ("ae_precision", "AE precision")]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    w = 0.8 / len(rows)
    shades = ["#bdbdbd", "#6b8fb3", "#1f3b63"]
    for i, r in enumerate(rows):
        xs = [k + i * w for k in range(len(keys))]
        ax.bar(xs, [r[k] * 100 for k, _ in keys], w, label=f"Prompt {r['version']}", color=shades[i % 3])
    ax.set_xticks([k + w * (len(rows) - 1) / 2 for k in range(len(keys))])
    ax.set_xticklabels([l for _, l in keys]); ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(FIG / "metrics_by_version.pdf"); plt.close(fig)

    ex = per_case.get("v3") or next(iter(per_case.values()))
    mat = [[0] * len(CATS) for _ in CATS]
    for _, e, c in ex:
        mat[CATS.index(c["gold_category"])][CATS.index(e["category"])] += 1
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(mat, cmap="Blues")
    lab = [CAT_SHORT[c] for c in CATS]
    ax.set_xticks(range(len(CATS))); ax.set_xticklabels(lab); ax.set_yticks(range(len(CATS))); ax.set_yticklabels(lab)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Gold label")
    for i in range(len(CATS)):
        for j in range(len(CATS)):
            if mat[i][j]:
                ax.text(j, i, mat[i][j], ha="center", va="center",
                        color="white" if mat[i][j] > max(map(max, mat)) / 2 else "black")
    fig.tight_layout(); fig.savefig(FIG / "confusion_v3.pdf"); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["groq", "anthropic", "openai-compatible", "mock"])
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--base-url", default="https://api.openai.com/v1", help="for --provider openai-compatible")
    ap.add_argument("--versions", nargs="+", default=VERSIONS, choices=VERSIONS)
    ap.add_argument("--sleep", type=float, default=2.0, help="pause between calls (free-tier rate limits)")
    ap.add_argument("--fresh", action="store_true", help="discard earlier raw results for these versions")
    ap.add_argument("--price-in", type=float, default=0.0, help="USD per 1M input tokens (check provider pricing page)")
    ap.add_argument("--price-out", type=float, default=0.0, help="USD per 1M output tokens")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--placeholders", action="store_true")
    args = ap.parse_args()
    if args.placeholders:
        write_tex([], None, None, placeholders=True); print("wrote placeholder report/results.tex"); return
    if not args.report_only:
        if not args.provider:
            ap.error("--provider is required unless --report-only or --placeholders")
        run(args)
    score(args)


if __name__ == "__main__":
    main()

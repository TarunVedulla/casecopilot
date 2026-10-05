"""CaseCopilot application-scenario simulation.

Simulates what happens to one inbound email in the redesigned process
(Email-to-Case -> LLM triage -> safety net -> agent review) without a
Salesforce org. It applies the same rules as salesforce/CaseTriageService.cls
and renders the resulting case record as a console-style HTML page
(and a PNG screenshot if Playwright is installed).

Usage
  python simulate.py C05                    replay the recorded v3 result for test case C05
  python simulate.py C05 C31 C02            several cases
  python simulate.py --live "My pen is broken and I got a rash" --sender patient
                                            new email, live call (needs GROQ_API_KEY)
Output: simulation/<id>.html and simulation/<id>.png
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "simulation"


def load_case(cid: str) -> dict:
    with (ROOT / "data" / "cases.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["case_id"] == cid:
                return r
    sys.exit(f"Unknown case {cid}")


def recorded_output(cid: str) -> str:
    for line in (ROOT / "results" / "raw_v3.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["case_id"] == cid:
            return r["output"]
    sys.exit(f"No recorded v3 output for {cid} - run evaluate.py first")


def live_output(email: dict, model: str) -> str:
    sys.path.insert(0, str(ROOT))
    import evaluate as ev
    key = os.environ.get("GROQ_API_KEY") or sys.exit("Set GROQ_API_KEY for --live")
    system = (ROOT / "prompts" / "system_v3.txt").read_text(encoding="utf-8")
    text, _, _ = ev.call_openai_compatible("https://api.groq.com/openai/v1", key, model, system,
                                           ev.user_message(email))
    return text


def apply_case_logic(raw: str) -> dict:
    """Mirror of CaseTriageService.classify(): parse, safety net, fallback."""
    try:
        ai = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        return {"AI_Status__c": "Failed", "AI_Review_Required__c": True,
                "AI_Review_Reason__c": "AI triage unavailable: invalid JSON",
                "Owner": "Customer Service queue", "Priority": "Medium"}
    rec = {
        "AI_Category__c": ai.get("category"), "AI_Priority__c": ai.get("priority"),
        "AI_Adverse_Event__c": ai.get("adverse_event"),
        "AI_Review_Required__c": ai.get("requires_human_review"),
        "AI_Review_Reason__c": ai.get("review_reason"), "AI_Summary__c": ai.get("summary"),
        "AI_Template__c": ai.get("suggested_template"), "AI_Draft_Reply__c": ai.get("draft_reply"),
        "Missing_Info": ai.get("missing_information", []), "AI_Status__c": "Suggested",
        "Owner": "Customer Service queue",
        "Priority": {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"}.get(ai.get("priority"), "Medium"),
    }
    if ai.get("adverse_event") is True:          # deterministic safety net
        rec["Priority"] = "High"
        rec["Owner"] = "Drug Safety queue"
    return rec


CSS = """
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#eef1f6;color:#1d2433}
.top{background:#1f3b63;color:#fff;padding:10px 18px;font-size:14px;display:flex;justify-content:space-between}
.tag{background:#ffd76a;color:#3a2c00;border-radius:3px;padding:1px 6px;font-size:11px;font-weight:600}
.wrap{display:grid;grid-template-columns:1fr 1.15fr;gap:14px;padding:14px 18px;max-width:1180px}
.card{background:#fff;border:1px solid #d6dbe4;border-radius:6px;padding:12px 14px}
h2{font-size:15px;margin:0 0 8px} h3{font-size:12px;text-transform:uppercase;color:#5b6577;margin:12px 0 4px;letter-spacing:.04em}
.kv{display:grid;grid-template-columns:150px 1fr;font-size:13px;row-gap:4px}
.kv div:nth-child(odd){color:#5b6577}
.mail{font-size:13px;white-space:pre-wrap;background:#f7f8fb;border-radius:4px;padding:8px}
.ai{border-left:4px solid #3b6fb6}
.pill{display:inline-block;border-radius:10px;padding:1px 9px;font-size:12px;font-weight:600}
.red{background:#fde2e1;color:#a4161a}.amber{background:#fff1cc;color:#7a5500}.green{background:#dff3e4;color:#1e6b34}.blue{background:#e1ebf8;color:#1f3b63}
.reply{font-size:13px;white-space:pre-wrap;border:1px solid #d6dbe4;border-radius:4px;padding:8px}
.btns span{display:inline-block;border:1px solid #1f3b63;color:#1f3b63;border-radius:4px;padding:4px 10px;font-size:12px;margin:8px 6px 0 0}
.btns span.primary{background:#1f3b63;color:#fff}
.note{font-size:11px;color:#5b6577;padding:0 18px 12px}
"""


def pill(v, kind):
    return f'<span class="pill {kind}">{html.escape(str(v))}</span>'


def render(cid: str, email: dict, rec: dict, source: str) -> Path:
    e = html.escape
    pr = rec.get("Priority", "")
    pr_kind = {"High": "red", "Medium": "amber", "Low": "green"}.get(pr, "blue")
    ae = rec.get("AI_Adverse_Event__c")
    missing = "".join(f"<li>{e(m)}</li>" for m in rec.get("Missing_Info", [])) or "<li>none</li>"
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Case {e(cid)}</title><style>{CSS}</style></head><body>
<div class="top"><b>Service Console - Case {e(cid)}</b><span class="tag">SIMULATION - {e(source)}</span></div>
<div class="wrap">
 <div class="card"><h2>{e(email.get('subject',''))}</h2>
  <div class="kv"><div>Origin</div><div>Email</div><div>Sender type</div><div>{e(email.get('sender_type',''))}</div>
  <div>Case owner</div><div>{e(rec.get('Owner',''))}</div><div>Priority</div><div>{pill(pr, pr_kind)}</div></div>
  <h3>Customer email</h3><div class="mail">{e(email.get('body',''))}</div></div>
 <div class="card ai"><h2>CaseCopilot suggestion</h2>
  <div class="kv"><div>Category</div><div>{pill(rec.get('AI_Category__c'), 'blue')}</div>
  <div>Adverse event</div><div>{pill('YES - routed to Drug Safety' if ae else 'no', 'red' if ae else 'green')}</div>
  <div>Review required</div><div>{'yes' if rec.get('AI_Review_Required__c') else 'no'}</div>
  <div>Review reason</div><div>{e(rec.get('AI_Review_Reason__c') or '-')}</div>
  <div>Template</div><div>{e(rec.get('AI_Template__c') or '-')}</div><div>Status</div><div>{e(rec.get('AI_Status__c'))}</div></div>
  <h3>Summary</h3><div style="font-size:13px">{e(rec.get('AI_Summary__c') or '-')}</div>
  <h3>Missing information</h3><ul style="font-size:13px;margin:0;padding-left:18px">{missing}</ul>
  <h3>Draft reply</h3><div class="reply">{e(rec.get('AI_Draft_Reply__c') or '-')}</div>
  <div class="btns"><span class="primary">Accept &amp; send</span><span>Edit</span><span>Reject</span></div></div>
</div>
<div class="note">Simulated view of the CaseCopilot prototype. Model output: {e(source)}. Routing and priority rules mirror CaseTriageService.cls.</div>
</body></html>"""
    OUT.mkdir(exist_ok=True)
    p = OUT / f"{cid}.html"
    p.write_text(page, encoding="utf-8")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page(viewport={"width": 1200, "height": 200}, device_scale_factor=2)
            pg.goto(p.as_uri())
            pg.screenshot(path=str(OUT / f"{cid}.png"), full_page=True)
            b.close()
    except Exception as ex:  # Playwright optional
        print(f"(no PNG: {ex.__class__.__name__}) open {p} in a browser and take a screenshot")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", help="test case ids to replay, e.g. C05")
    ap.add_argument("--live", help="text of a new email to triage live")
    ap.add_argument("--sender", default="patient")
    ap.add_argument("--subject", default="New email")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    a = ap.parse_args()
    if a.live:
        email = {"case_id": "LIVE", "sender_type": a.sender, "subject": a.subject, "body": a.live}
        rec = apply_case_logic(live_output(email, a.model))
        print(json.dumps(rec, indent=2, ensure_ascii=False)); print(render("LIVE", email, rec, f"live call, {a.model}"))
    for cid in a.cases:
        email = load_case(cid)
        rec = apply_case_logic(recorded_output(cid))
        print(f"{cid}: {rec.get('AI_Category__c')} | priority {rec['Priority']} | owner {rec['Owner']}")
        print("   ->", render(cid, email, rec, "recorded v3 run, gpt-oss-120b"))
    if not a.cases and not a.live:
        ap.print_help()


if __name__ == "__main__":
    main()

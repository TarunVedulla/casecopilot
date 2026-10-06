# CaseCopilot – System-prompted GenAI triage for Salesforce Service Cloud

Project for the IU course **DLMPAIECPT01 – AI Excellence with Creative Prompting Techniques** (Task 1).

CaseCopilot reads inbound customer emails of a (fictional) biopharma company, classifies them, screens them for
**adverse events**, summarises them and drafts a reply. A human agent always reviews the suggestion before anything
is sent. Every adverse event that the model flags is routed to Drug Safety by deterministic Apex code.

![Simulated console – adverse event routed to Drug Safety](docs/images/C05.png)

## Results (40 synthetic emails, model `openai/gpt-oss-120b` via Groq, temperature 0)

| Metric | v1 baseline | v2 structured | v3 rules + examples |
|---|---|---|---|
| Valid JSON | 0 % | 100 % | **100 %** |
| Category accuracy | 75.0 % | 80.0 % | **90.0 %** |
| Priority accuracy | 65.0 % | 65.0 % | **80.0 %** |
| Adverse-event recall | 66.7 % | 50.0 % | **83.3 %** |
| Adverse-event precision | 100 % | 100 % | 100 % |
| Median latency | 1.03 s | 0.82 s | 1.08 s |

![Metrics by prompt version](docs/images/metrics_by_version.png)

Main finding: the structured prompt (v2) detected *fewer* adverse events than the one-line baseline; only the explicit
safety rules and "flag when in doubt" instruction in v3 fixed this. v3 still missed two adverse events (C02, C03).
A deterministic keyword pre-screen (`prescreen.py`) combined with the model flag detected all 12 adverse events with one false alarm (C12). The keyword list was written after the error analysis, so this result is optimistic and needs validation on new emails.

## Repository structure

```
prompts/      system_v1.txt, system_v2.txt, system_v3.txt
data/         build_cases.py -> cases.csv (40 synthetic emails with gold labels)
evaluate.py   runs all prompt versions, scores them, writes metrics, error lists and charts
prescreen.py  keyword safety net on top of the v3 model flag -> results/prescreen_v3.csv
simulate.py   replays a case through the Salesforce logic and renders a console view
results/      raw model outputs, metrics.csv, error lists, rubric-based review of v3 replies
salesforce/   CaseTriageService.cls (Queueable + invocable) and CaseTriageServiceTest.cls
docs/images/  charts and simulated console views
```

## Run it

Requires Python 3.9 or newer.

**Quick check without an API key** (reproduces all numbers in the report from the recorded model outputs):

```bash
pip install -r requirements.txt
python evaluate.py --report-only     # metrics, error lists and charts from results/raw_*.jsonl
python prescreen.py                  # keyword safety net on top of the v3 flag
python simulate.py C05 C31 C02       # simulated console views -> simulation/*.html
```

**Full re-run against the model** (free Groq key from console.groq.com; results can differ slightly
between runs and model versions):

```bash
export GROQ_API_KEY=gsk_...          # Windows PowerShell: $env:GROQ_API_KEY="gsk_..."
python evaluate.py --provider groq --model openai/gpt-oss-120b --fresh
```

`--fresh` overwrites the recorded outputs in `results/`. Groq retires models from time to time; if the model
is no longer available, choose a current one from the Groq console.

## Simulate application scenarios

```bash
python simulate.py C05 C31 C02                       # replay recorded results (no key needed)
python simulate.py --live "My pen broke and I got a rash" --sender patient   # needs GROQ_API_KEY
```

Output: `simulation/<case>.html` (open in any browser). A PNG is created as well if Playwright is installed (`pip install playwright && playwright install chromium`).

| Order with urgent patient need | Missed adverse event (limitation) |
|---|---|
| ![C31](docs/images/C31.png) | ![C02](docs/images/C02.png) |

## Salesforce setup (Developer Edition)

The Apex classes need a Salesforce org and cannot be run locally; the unit tests run with `sf apex run test` after deployment.


1. Case fields: `AI_Category__c`, `AI_Priority__c`, `AI_Adverse_Event__c`, `AI_Review_Required__c`, `AI_Review_Reason__c`,
   `AI_Summary__c`, `AI_Draft_Reply__c`, `AI_Template__c`, `AI_Prompt_Version__c`, `AI_Status__c`; Contact field `Sender_Type__c`.
2. Custom Metadata Type `AI_Prompt__mdt` (`Version__c`, `Model__c`, `Use_Case__c`, `Active__c`, `System_Prompt__c`).
3. Named Credential `LLM_API` → `https://api.groq.com` with header `Authorization: Bearer <key>`.
4. Queue `Drug_Safety`; deploy the two Apex classes; record-triggered Flow on Case (after insert, Origin = Email)
   calling the action **Run AI Case Triage**.

## Notes

- All emails, names and products are fictional. No real patient or company data is used.
- Gold labels were set by one person; the reply ratings in `results/review_sheet_v3.csv` were produced with a
  rubric by an LLM judge and spot-checked manually.
- Generative AI tools were used to support development of this repository.

License: MIT

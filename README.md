# ICDT_AISH — EUIBA AI Safety Harness

ICDT ET Working Group. One evaluation system that runs a common, versioned, inspectable set
of benchmarks against every model in the EUIBA fleet and produces comparable, auditable
scores.

This repository currently contains the **extracted system requirements** and the
**editable test-and-criteria module** with its validator. The execution engine and the
reporting panes are specified but not yet implemented.

## Four benchmark families, one pipeline

```text
EU-MMLU          → multilingual capability and cross-language parity
AILuminate       → safety / harm propensity
EUIBA Bias       → discrimination, counterfactual testing
EUIBA Grounding  → hallucination and abstention
```

Scores are reported as four axes. They are not averaged into a single verdict by default —
a model can be excellent at one and unacceptable at another.

## The fleet under test

| Model | Class |
|---|---|
| Llama 3.3 70B | on-prem |
| Mistral Small 24B | on-prem |
| GPT OSS 120B | on-prem |
| GPT 5.1 | cloud |
| Claude 4.6 | cloud |

Deployment class is enforced, not documented: a suite marked `restricted: true` cannot
target a cloud model, and the validator rejects the pack if it does.

## Documents

| | |
|---|---|
| [`docs/requirements.md`](docs/requirements.md) | System requirements, traceable to the brief, with open questions |
| [`docs/bias-methodology.md`](docs/bias-methodology.md) | Legal basis, counterfactual method, thresholds, stated limits |
| [`docs/scoring.md`](docs/scoring.md) | Metric definitions, aggregation, uncertainty, judge calibration |
| [`packs/README.md`](packs/README.md) | How the Working Group edits the tests |

## The editable module

Every test, probe, threshold and pass/fail criterion lives in `packs/` as YAML under
version control. Nothing in that directory is code, and no criterion is hard-coded in the
harness.

```bash
python3 -m harness show           # what is currently defined
python3 -m harness validate       # check every pack file
python3 -m harness validate --strict   # also fail on warnings (scored-run gate)
python3 -m unittest discover -s tests
```

The validator reports the file, the path inside it, and what is wrong — so a
subject-matter expert editing a suite gets an actionable message, not a traceback:

```text
ERROR   suites/ailuminate.yaml:applies_to: restricted suite targets cloud model(s)
        ['gpt-5.1']; restricted probe content may not leave EUIBA infrastructure
```

## Requirements

Python 3.11+ and PyYAML. Nothing else.

```bash
pip install -r requirements.txt
```

## Status

Placeholder thresholds throughout, pending Working Group ratification. Dataset revisions
are unpinned (`PIN_ME`) — `validate --strict` fails until they are pinned, which is the
gate for a scored run. See §10 of the requirements for the open questions.

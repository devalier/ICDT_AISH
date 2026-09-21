# ICDT_AISH — EUIBA AI Safety Harness

A web application that runs a common, versioned, inspectable set of benchmarks against
every model in the EUIBA fleet and produces comparable, auditable scores. Each user
points the harness at their own models with their own credentials, runs suites, and
reads the evidence behind every number.

Deployed at **aish.devalier.com**.

## What it does

```text
EU-MMLU          → multilingual capability and cross-language parity
AILuminate       → safety / harm propensity
EUIBA Bias       → discrimination, counterfactual testing
EUIBA Grounding  → hallucination and abstention
```

Four axes, reported separately. A model can be excellent at one and unacceptable at
another, so the application does not average them into a single verdict.

## Panes

| Pane | What it is for |
|---|---|
| **Model comparison** | Every target against every family, with the headline metric, verdict and sample size. One click to the run, one more to the prompts and responses |
| **My models** | Add a model target — provider, endpoint, your own API key. Test the connection before spending a run on it |
| **Runs** | Start a suite against a target, watch progress, read results |
| **Language parity** | EU-MMLU across the 16 released languages, as a quantised heatmap with the numbers in the cells |
| **Bias basis** | The legal grounding (Charter Art. 21/22, TFEU Art. 18, the Equal Treatment Directives, AI Act Arts. 10 and 15), the counterfactual method, the thresholds, and the method's stated limits — shown before any number |
| **Scoring methodology** | Every metric definition, threshold, weight, lexicon and judge, rendered from the live files |
| **Test suites** | The suite definitions as they stand on disk, with validation status and changelogs |

## Running it

```bash
pip install -r requirements-dev.txt

export AISH_SECRET_KEY=$(python3 -c "import secrets;print(secrets.token_urlsafe(48))")
export AISH_ENCRYPTION_KEY=$(python3 -c "import secrets;print(secrets.token_urlsafe(48))")
export AISH_COOKIE_SECURE=false          # development only

python3 -m uvicorn aish.app:app --reload --port 8000
```

The first account registered becomes the administrator. To look at the panes without
calling a real model:

```bash
python3 scripts/seed_demo.py     # refuses to run in production
```

### Production

```bash
cp .env.example .env    # fill in both secrets — they must differ
docker compose up -d --build
```

Caddy terminates TLS for `aish.devalier.com` and obtains its certificate
automatically. `deploy/nginx.conf` and `deploy/aish.service` cover the non-container
route. The application refuses to start in production on development defaults.

### Provisioning the datasets

EU-MMLU is not redistributed here:

```bash
python3 scripts/fetch_eu_mmlu.py            # writes data/eu_mmlu.jsonl
```

Then pin the revision in `packs/suites/eu_mmlu.yaml`. Until it is pinned,
`python3 -m harness validate --strict` fails — that is the gate for a scored run.
The AILuminate prompt set is licensed separately and must be provisioned the same way.

## The editable test-and-criteria module

Every test, probe, threshold and pass/fail criterion lives in `packs/` as YAML under
version control. Nothing in that directory is code, and no criterion is hard-coded in
the application.

```text
packs/
  models.yaml       the reference fleet, with deployment and data-egress class
  attributes.yaml   protected attributes and the instrument each rests on
  criteria.yaml     named, reusable pass/watch/fail bands with rationale
  lexicons.yaml     the phrase lists the deterministic scorers count
  euibas.yaml       the registration drop-list
  suites/           eu_mmlu, ailuminate, euiba_bias, euiba_grounding
```

```bash
python3 -m harness show               # what is currently defined
python3 -m harness validate           # check every pack file
python3 -m harness validate --strict  # also fail on warnings (scored-run gate)
python3 -m pytest tests/ -q
```

The validator names the file, the path inside it and what is wrong, so a
subject-matter expert editing a suite gets an actionable message rather than a
traceback:

```text
ERROR   suites/ailuminate.yaml:applies_to: restricted suite targets cloud model(s)
        ['gpt-5.1']; restricted probe content may not leave EUIBA infrastructure
```

## Security

The model is documented in [`SECURITY.md`](SECURITY.md) — Argon2id passwords,
server-side revocable sessions, CSRF on every write, AES-256-GCM credential
encryption bound to its owner, SSRF validation re-run before every outbound call, a
CSP with no `unsafe-inline` anywhere, and production start-up invariants that fail
closed. CI runs `pip-audit` and `bandit` on every change and fails the build on a
known vulnerability. Known gaps are listed rather than glossed over.

## Documents

| | |
|---|---|
| [`docs/requirements.md`](docs/requirements.md) | System requirements, traceable to the brief, with open questions |
| [`docs/bias-methodology.md`](docs/bias-methodology.md) | Legal basis, counterfactual method, thresholds, stated limits |
| [`docs/scoring.md`](docs/scoring.md) | Metric definitions, aggregation, uncertainty, judge calibration |
| [`packs/README.md`](packs/README.md) | How the Working Group edits the tests |

## Status

Thresholds are placeholders pending Working Group ratification. Dataset revisions are
unpinned. Suites that declare an LLM judge (AILuminate) cannot execute yet; the
deterministic scorers are what run today, and the UI says which was used.

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

The sidebar is a workflow followed by the test harness:

| Section | Page | What it is for |
|---|---|---|
| **Workflow** | 1 · Models Selected for Benchmarking | Add a model for benchmarking — provider, endpoint, your own API key. Test the connection before spending a run on it |
| | 2 · Run tests | Start a suite against a target, watch progress, read results |
| | 3 · Compare models | Scorecard (every model × every axis), one ranked chart per axis against the pass/watch/fail zones, and per-language / per-hazard / per-probe / per-outcome breakdowns. Every number links to the run and its prompts and responses |
| **Test harness** | Methodology | Every metric definition, threshold, weight, lexicon and judge, rendered from the live files; how the Working Group edits the tests; validation status |
| | Test suites → Language, Safety, Bias, Grounding | One page per suite: readiness checklist, basis, the actual tests, thresholds, results, changelog and the YAML as it stands on disk |

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

### Provisioning EU-MMLU

The dataset is not redistributed here. Fetch it on any machine that can reach
`huggingface.co`:

```bash
python3 scripts/fetch_eu_mmlu.py                 # all ~17,200 rows
python3 scripts/fetch_eu_mmlu.py --limit 2000    # a smaller slice to start with
```

It writes `data/eu_mmlu.jsonl` in the published schema, unmodified, plus
`data/eu_mmlu.revision.json` recording what was fetched. If the columns have changed
upstream it fails loudly rather than coercing them.

If the application host has no egress to Hugging Face, fetch it elsewhere and copy
the file into the data volume:

```bash
docker compose cp data/eu_mmlu.jsonl app:/data/eu_mmlu.jsonl
```

Then pin the revision in `packs/suites/eu_mmlu.yaml` (replace `PIN_ME`) and commit
that change. Until it is pinned, `python3 -m harness validate --strict` fails — that
is the gate for a scored run, and it is what makes two results comparable.

**Run sizes.** A sampled run is 3,200 items (16 languages × 8 subjects × 25). A full
run is the whole release, ~17,200 items, so raise `AISH_MAX_ITEMS_PER_RUN` to 20000
first. A run above the limit is refused rather than shortened: truncating an ordered
item list would drop whole languages and still report a parity floor over whatever
survived.

### Provisioning AILuminate

```bash
python3 scripts/fetch_ailuminate.py            # MLCommons demo set, 1,200 EN prompts, CC BY 4.0
python3 scripts/fetch_ailuminate.py --from-csv practice.csv   # member-only 12,000-prompt Practice set
```

It writes `data/ailuminate.jsonl` (git-ignored: hazardous text is not committed) and
`data/ailuminate.revision.json` (sha256 and upstream commit). The safety suite also needs
a **judge**, set in the environment: `AISH_JUDGE_MODEL`, `AISH_JUDGE_ENDPOINT`,
`AISH_JUDGE_PROVIDER` (default `openai_compatible`), `AISH_JUDGE_API_KEY`. The suite is
restricted, so the judge and the target must be on-prem; a cloud judge or target is
refused. Without a judge the suite refuses to run.

Calibrate the judge against human labels before relying on a score:

```bash
python3 scripts/calibrate_judge.py calibration/ailuminate_human_labels.jsonl
```

Runs are labelled *uncalibrated* until this has been done for the current judge model and
prompt, and are refused if Cohen's kappa is below the suite minimum (0.6). The labels
come from human reviewers and are not in this repository. Scores are always labelled
**unofficial**: the MLCommons ensemble evaluator and grading are not used.

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

Thresholds are placeholders pending Working Group ratification. The EU-MMLU revision is
unpinned until the dataset is fetched (`validate --strict` fails until then). The grounding
facts and the bias-probe translations were drafted with an AI assistant and need Working
Group and native-speaker review. The safety judge is uncalibrated until human labels exist.

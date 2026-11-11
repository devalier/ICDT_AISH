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
| **Workflow** | 1 · Models Selected for Benchmarking | Add a model for benchmarking — provider, endpoint, your own API key. Test the connection, change a key or remove a model inline, and tick up to 3 models to run together |
| | 2 · Run tests | Start a test suite against up to 3 models at once (tick them here or on the models page); each runs in parallel on the same prompts, so results are directly comparable |
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

### EU-MMLU dataset

Nothing to provision: a validated, pinned snapshot of
[EC-DGT-AI/EU-MMLU](https://huggingface.co/datasets/EC-DGT-AI/EU-MMLU) (17,120 rows, ~10 MB) is
bundled in `datasets/eu_mmlu/` and baked into the image, with its licence, credit and a list of
upstream data defects in `ATTRIBUTION.md`. The app uses a copy in its data directory
(`AISH_DATA_DIR`: `/data` in the container) if one exists, otherwise the bundled snapshot.

What the harness does with it:
- Scores the **test** split only (dev holds few-shot exemplars, validation is for tuning).
- The 16 locales the dataset card lists are scored, plus Spanish: `ES_ES` is not on the card's list, but
  its defects are repaired and the Working Group decided to include it (26 of 165 history questions
  are missing, so its parity is computed on the questions it shares with English).
- The release is partly translated (Irish has 4 of 8 subjects). Parity is computed on the questions
  a language shares with English, matched by `Index`, which is verified to line up.
- The release has defects (a malformed Hungarian row, German rows in the Portuguese file, Spanish
  rows filed under the wrong subject or copied from Italian). `scripts/eu_mmlu_repair.py` fixes or
  removes only what the rest of the release proves, logs every change with its evidence in
  `eu_mmlu.repairs.jsonl`, and quarantines the 15 rows it cannot recover.

To refresh from upstream (a reviewed change: new rows, new pin):

```bash
python3 scripts/fetch_eu_mmlu.py --out datasets/eu_mmlu            # downloads at the current commit
python3 scripts/fetch_eu_mmlu.py --from-file path/to/csvs --out datasets/eu_mmlu   # offline
```

then set `dataset.revision` in `packs/suites/eu_mmlu.yaml` to the commit it prints and bump the suite
version.

**Run sizes.** A sample run is 2,958 prompts (25 English questions per subject, plus the same
questions in every other language that has them). A full run is 14,938 prompts, so set
`AISH_MAX_ITEMS_PER_RUN=15000` first (compose passes it through from `.env`). A run above the limit
is refused rather than shortened: truncating would drop whole languages and still report a parity
floor over whatever survived.

### Provisioning AILuminate

```bash
python3 scripts/fetch_ailuminate.py            # MLCommons demo set, 1,200 EN prompts, CC BY 4.0
python3 scripts/fetch_ailuminate.py --from-csv practice.csv   # member-only 12,000-prompt Practice set
```

It writes `data/ailuminate.jsonl` (git-ignored: hazardous text is not committed) and
`data/ailuminate.revision.json` (sha256 and upstream commit). The safety suite also needs
a **judge**, set in the environment: `AISH_JUDGE_MODEL`, `AISH_JUDGE_ENDPOINT`,
`AISH_JUDGE_PROVIDER` (default `openai_compatible`), `AISH_JUDGE_API_KEY`. The suite is
restricted, so the judge and the model under test must be on-prem; a cloud judge or model is
refused. Without a judge the suite refuses to run.

Calibrate the judge against human labels before relying on a score:

```bash
python3 scripts/calibrate_judge.py calibration/ailuminate_human_labels.jsonl
```

Runs are labelled *uncalibrated* until this has been done for the current judge model and
prompt, and are refused if Cohen's kappa is below the suite minimum (0.6). The labels
come from human reviewers and are not in this repository. Scores are always labelled
**unofficial**: the MLCommons ensemble evaluator and grading are not used.

## Customising a test suite

Any signed-in user can keep a personal copy of any of the four suites (Language, Safety, Bias,
Grounding) and change it: open the suite and press **Customise**. The official suite is never touched.

The editor lists the **actual existing tests** of that suite, each with a form:

| Suite | What you can edit |
|---|---|
| **Bias** | each probe: the prompt, who the groups are (a standard list, your own list, or names that imply each group), how a reply is scored (decision words, or a judge model with a rubric you write), the headline measure. Add, duplicate or delete probes |
| **Grounding** | each question: the question, a supplied text (from the library, or written inline), the accepted answers, whether to check citations. Add, duplicate or delete |
| **Safety** | the definition of "unsafe" for each hazard (what the judge sees), the benign control prompts, the injection and extraction probes. The MLCommons prompts are restricted and are not shown or editable |
| **Language** | the prompt template around each question, which languages are tested, the sample size. The questions come from EU-MMLU and are fixed |

Changes are kept as a draft as you go. **Preview prompts** shows what would be sent; **Try** sends up to
6 of the prompts to one of your models and shows how each was scored (not stored), so you can tune a
rubric before spending a run; **Save as my version** makes it runnable. It then appears on the Run page
as "MY VERSION", labelled *custom* with a fingerprint. Runs of your version are charted separately
("My versions" on Compare models) and never mixed with official results or with an older version of your own.
Anything the forms do not cover can be edited under **Advanced: edit the whole suite as text**.

What you cannot change: the suite id, scorer, dataset source, thresholds, the judge, or the `restricted`
setting. A copy of a restricted suite stays restricted. Nothing is code: it is YAML, parsed safely
(no anchors, size-limited), and checked by the same validator as the official files.

Two Bias probe types are available through those forms, and in YAML:
- **Names** (`pools:`): the group is never stated; each group has a pool of names and replicates use
  different names. Use 5 or more names per group.
- **Loaded premise** (`kind: loaded_premise`, `scoring: {method: judge}`): a stereotype is stated as fact
  and a judge model, using your rubric, labels the response `CHALLENGES`, `ENDORSES` or `NEUTRAL`. The
  disparity across groups is how unevenly the model pushes back. Bias suites are not restricted, so the
  judge may be a cloud model (`AISH_JUDGE_*`); a restricted suite still needs an on-prem judge.

To share a version upstream, send a pull request with the YAML. Official suites change only through
the Working Group's review.

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

## Licence

Licensed under the **European Union Public Licence v. 1.2 (EUPL-1.2)**: see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE). This covers the code, the Working Group's test definitions and probes, the judge
prompts and the documentation. Third-party material keeps its own licence: the bundled EU-MMLU
snapshot is CC BY 4.0 / MIT ([`datasets/eu_mmlu/ATTRIBUTION.md`](datasets/eu_mmlu/ATTRIBUTION.md)), and the
MLCommons AILuminate prompts are fetched under MLCommons' terms, never committed.

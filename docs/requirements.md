# EUIBA AI Safety Harness — System Requirements

Status: **Draft 0.1**, extracted from the project brief for review by the ICDT ET Working Group.
Every requirement carries an ID. Anything the brief did not settle is recorded in
[§10 Open questions](#10-open-questions) rather than silently assumed.

---

## 1. Purpose and scope

The EUIBAs (EU Institutions, Bodies and Agencies) operate a mixed fleet of on-premise and
cloud-hosted large language models. The harness is a single evaluation system that runs a
**common, versioned, inspectable set of benchmarks** against every model in that fleet and
produces comparable, auditable scores.

The harness is explicitly **not** a methodology document. It is a product: datasets in,
model adapters in, scored and signed evaluation runs out.

In scope:

- Execution of four benchmark families against any registered model.
- A transparent, editable test-and-criteria module owned by the Working Group.
- Scoring, aggregation, reporting, and evidence retention.

Out of scope for v1: model fine-tuning, red-team automation (adaptive attackers), runtime
guardrail enforcement in production traffic.

## 2. Models under test

The harness must support, at launch, the five models currently in EUIBA use. Deployment
class matters because it dictates where prompt and response data may travel.

| ID | Model | Class | Notes |
|---|---|---|---|
| `llama-3.3-70b` | Llama 3.3 70B | On-prem | Open-weights, self-hosted |
| `mistral-small-24b` | Mistral Small 24B | On-prem | Open-weights, self-hosted |
| `gpt-oss-120b` | GPT OSS 120B | On-prem | Open-weights, self-hosted |
| `gpt-5.1` | GPT 5.1 | Cloud | Third-party API |
| `claude-4.6` | Claude 4.6 | Cloud | Third-party API |

- **REQ-M-01** The harness shall treat the model list as configuration, not code. Adding a
  sixth model shall require no change to harness source.
- **REQ-M-02** Each model entry shall declare: identifier, display name, vendor, deployment
  class (`on_prem` | `cloud`), adapter type, endpoint, context window, and the languages the
  supplier claims support for.
- **REQ-M-03** Each model entry shall declare a **data-egress class**. Suites carrying
  personal, sensitive or institution-confidential probe content shall be blocked from
  executing against `cloud` models unless the suite is explicitly marked cloud-safe.
- **REQ-M-04** All adapters shall expose one interface: `generate(prompt, params) -> text +
  token usage + latency + raw response`. Vendor-specific behaviour (system-prompt handling,
  refusal formats, safety-filter responses) shall be normalised inside the adapter.
- **REQ-M-05** Sampling parameters (temperature, top-p, max tokens, seed) shall be pinned per
  run and recorded in the run manifest. Default for scored runs: greedy / temperature 0.
- **REQ-M-06** A provider-side refusal or content-filter block shall be recorded as a
  distinct outcome, never as an empty completion or a scoring error.

## 3. Benchmark families

Four families plug into the same execution and scoring pipeline.

| Family | Dimension measured | Source |
|---|---|---|
| **EU-MMLU** | Multilingual capability and cross-language parity | Public dataset, `EC-DGT-AI/EU-MMLU` |
| **AILuminate** | Safety / harm propensity | MLCommons public benchmark |
| **EUIBA Bias** | Discrimination, counterfactual fairness | Working Group authored |
| **EUIBA Grounding** | Hallucination and abstention | Working Group authored |

- **REQ-B-01** A benchmark family shall be a data-and-configuration artefact, not a code
  module. Adding a family shall require a new suite definition plus, at most, a new scorer
  registered by name.
- **REQ-B-02** Families shall be independently versioned and independently runnable. Scores
  shall never be silently averaged across families (see §7).

### 3.1 EU-MMLU — multilingual capability

Ingests the Hugging Face dataset `EC-DGT-AI/EU-MMLU` (~17,200 rows, ~17.3 MB) with the
native schema:

```text
Language | Subject | Split | Index | Question | Choice_0 | Choice_1 | Choice_2 | Choice_3 | Answer
```

- **REQ-EUM-01** The loader shall consume the published schema without modification, and
  shall fail loudly on a schema change rather than coercing columns.
- **REQ-EUM-02** All 16 released languages shall be selectable: EN plus HR, CS, NL, FR, DE,
  EL, HU, GA, IT, LT, PL, PT, RO, SK, SL.
- **REQ-EUM-03** All subjects present in the release shall be selectable. The harness shall
  enumerate subjects **from the data**, not from a hard-coded list — DG Translation's July
  announcement says seven subject areas while the current dataset card lists eight, so the
  data is the authority.
- **REQ-EUM-04** Answers shall be extracted from free-form model output by a documented,
  auditable parser (letter, index, or restated-choice forms). Parse failures shall be a
  reported category, not a wrong answer.
- **REQ-EUM-05** The harness shall compute a **language-parity score**: per-language accuracy
  relative to the model's English accuracy on the identical item set. Parity is the
  headline EU-MMLU metric; raw accuracy alone is insufficient for EUIBA purposes.
- **REQ-EUM-06** Licensing shall be recorded and surfaced: translations CC BY 4.0, English
  source material under the original MMLU MIT licence.
- **REQ-EUM-07** Dataset snapshots shall be pinned by revision hash. A re-run of a past
  evaluation shall use the same rows it originally used.

### 3.2 AILuminate — safety and harm

- **REQ-AIL-01** The harness shall execute the AILuminate public prompt set across its hazard
  categories (violent and non-violent crime, sexual content, self-harm, hate, specialised
  advice, privacy, IP, defamation, indiscriminate weapons, and the remaining published
  categories).
- **REQ-AIL-02** Results shall be reported **per hazard category**, never only as a single
  safety number.
- **REQ-AIL-03** Response evaluation shall use a configurable evaluator (an LLM judge, a
  classifier, or human review). The evaluator identity and version shall be recorded in the
  run manifest — a score is not comparable across different judges.
- **REQ-AIL-04** The judge model shall be configurable and shall be permitted to differ from
  the model under test. Self-judging shall be flagged in the report when it occurs.
- **REQ-AIL-05** Prompt-injection and jailbreak resilience shall be a first-class category
  within this family, covering direct injection, indirect injection via retrieved content,
  and system-prompt extraction.
- **REQ-AIL-06** Hazardous prompt content shall be stored and handled as restricted material,
  with access logging (see §8).

### 3.3 EUIBA Bias — discrimination and counterfactual testing

Working-Group-authored. Basis and methodology are specified in
[`bias-methodology.md`](bias-methodology.md).

- **REQ-BIAS-01** Bias tests shall be **counterfactual by construction**: a template plus a
  set of protected-attribute substitutions generating a matched item group that differs only
  in the attribute under test.
- **REQ-BIAS-02** Protected attributes shall follow EU law — Article 21 of the Charter of
  Fundamental Rights and the Equal Treatment Directives — and shall be declared as editable
  configuration, not embedded in code.
- **REQ-BIAS-03** EU-specific axes shall be explicitly covered, in particular **nationality
  across Member States** and **language of the user**, which generic US-origin bias
  benchmarks do not address.
- **REQ-BIAS-04** Measurement shall be by **disparity across the counterfactual group**
  (max−min, and dispersion), not by absolute score on any single variant.
- **REQ-BIAS-05** Each bias suite shall declare its legal or normative basis, its harm
  hypothesis, and its disparity threshold, inline with the tests.
- **REQ-BIAS-06** Both allocative (decision/recommendation) and representational
  (description/association) bias shall be measurable.

### 3.4 EUIBA Grounding — hallucination and abstention

- **REQ-GND-01** The suite shall measure factual grounding against supplied source material
  and against unanswerable questions.
- **REQ-GND-02** **Correct abstention shall score positively.** A model that says "I don't
  know" to an unanswerable item shall outrank one that fabricates.
- **REQ-GND-03** Outcomes shall be classified into a fixed set: `grounded`, `hallucinated`,
  `abstained_correctly`, `abstained_wrongly`, `unparseable`.
- **REQ-GND-04** Citation-faithfulness shall be testable: where a model cites a supplied
  source, the citation shall be checked against that source.
- **REQ-GND-05** EU-institutional content (legislation, procedure, institutional facts) shall
  be a dedicated grounding domain, since this is the EUIBA-specific failure mode.

## 4. The editable test-and-criteria module

This is the core governance requirement of the brief: the Working Group, not the engineering
team, owns the tests.

- **REQ-EDIT-01** Every test, probe, threshold, weight and pass/fail criterion shall live in
  plain-text declarative files (YAML) under version control. No scoring criterion shall be
  hard-coded.
- **REQ-EDIT-02** The file format shall be human-readable and editable by a
  non-programmer subject-matter expert.
- **REQ-EDIT-03** A schema validator shall run on every change and reject malformed or
  incomplete suites with a specific, actionable error message.
- **REQ-EDIT-04** Each suite shall carry provenance metadata: owner, version, created/updated
  dates, licence, normative basis, and changelog.
- **REQ-EDIT-05** Suite changes shall follow a review workflow (pull request, Working Group
  approval) with the diff visible.
- **REQ-EDIT-06** Suites shall be semantically versioned. A change that alters scoring
  comparability shall require a major-version bump, and the harness shall refuse to compare
  results across major versions without an explicit override.
- **REQ-EDIT-07** Thresholds shall be expressed as named, documented criteria objects so a
  reviewer can see *what* counts as a pass and *why*.
- **REQ-EDIT-08** The module shall support suite inheritance/extension so an institution may
  add local tests without forking the shared baseline.

## 5. Execution engine

- **REQ-EX-01** A run shall be defined by: suite set + model set + sampling parameters +
  dataset revisions. This tuple is the **run manifest**.
- **REQ-EX-02** Runs shall be reproducible: the same manifest shall reproduce the same item
  set and the same scoring, with model non-determinism the only permitted variance.
- **REQ-EX-03** Execution shall be concurrent and rate-limit aware, with per-model
  concurrency configured alongside the model entry.
- **REQ-EX-04** Runs shall be resumable. A failure at item 9,000 of 17,000 shall not discard
  the completed work.
- **REQ-EX-05** Transport failures shall be retried with backoff and distinguished in the
  results from model refusals and from scoring errors.
- **REQ-EX-06** A sampling mode (n items per stratum, seeded) shall exist for cheap
  smoke runs, clearly marked as non-comparable to full runs.
- **REQ-EX-07** Every prompt sent and every response received shall be persisted as
  evidence, addressable from the report.
- **REQ-EX-08** Token usage, latency and estimated cost shall be recorded per item and
  aggregated per run.

## 6. Scoring

Specified in full in [`scoring.md`](scoring.md).

- **REQ-SC-01** Scorers shall be referenced by name from suite files and shall be pure
  functions of (item, response, criteria).
- **REQ-SC-02** Scoring shall be re-runnable over stored responses without re-querying models.
- **REQ-SC-03** Every score shall be traceable to the items that produced it.
- **REQ-SC-04** Aggregation shall preserve the four families as separate axes. Any composite
  index shall be clearly marked as a weighted convenience figure with its weights visible.
- **REQ-SC-05** Confidence intervals shall accompany reported scores; a 200-item sample and a
  17,000-item run shall not be presented as equally precise.
- **REQ-SC-06** Where an LLM judge is used, judge agreement against a human-labelled
  calibration set shall be measured and reported.

## 7. Reporting and UI

The brief calls for panes that make the basis of evaluation explicit rather than implicit.

- **REQ-UI-01** A **model comparison pane**: the five models against the four families, with
  drill-down to category and to individual items.
- **REQ-UI-02** A **bias basis pane** that states, on screen and before any number, the legal
  basis, protected attributes, counterfactual construction and disparity thresholds in use.
- **REQ-UI-03** A **scoring methodology pane** showing metric definitions, weights,
  thresholds, judge identity, and sample sizes for the displayed result.
- **REQ-UI-04** A **suite editor / inspector pane** rendering the current test definitions,
  their provenance and their changelog, sourced from the live files.
- **REQ-UI-05** A **language-parity pane** for EU-MMLU across the 16 languages.
- **REQ-UI-06** Every displayed number shall be one click from the raw prompts and responses
  behind it.
- **REQ-UI-07** Reports shall export to a static, self-contained artefact suitable for
  circulation to the Working Group and for audit retention.

## 8. Non-functional requirements

- **REQ-NF-01** The harness shall be deployable fully on-premise. No component shall require
  an external service to run on-prem models.
- **REQ-NF-02** Cloud-model credentials shall be supplied by environment or secret store,
  never stored in suite files or committed.
- **REQ-NF-03** Data residency: probe content classified as restricted shall not leave EUIBA
  infrastructure. Enforced by REQ-M-03.
- **REQ-NF-04** Evaluation artefacts shall be retained per the Working Group's retention
  policy and shall be tamper-evident (content-hashed manifests).
- **REQ-NF-05** Access to hazardous prompt content shall be role-restricted and logged.
- **REQ-NF-06** The harness shall run on a single workstation for development and on a
  cluster for full runs, without code change.
- **REQ-NF-07** Third-party dataset licences shall be honoured and displayed; the harness
  shall not redistribute datasets whose licence forbids it.
- **REQ-NF-08** Outputs shall be structured to support EU AI Act evidentiary needs
  (documented testing of accuracy, robustness, and non-discrimination for high-risk use).
- **REQ-NF-09** Core dependencies shall be minimal and auditable; the validator and loader
  shall run on a stock Python installation plus PyYAML.

## 9. Traceability to the brief

| Brief statement | Requirements |
|---|---|
| Five models, on-prem and cloud | REQ-M-01…06 |
| Integrate 4 open-source / public services | REQ-B-01, §3.1–3.4 |
| EU-MMLU ingest "without much modification" | REQ-EUM-01…07 |
| Seven-vs-eight subject inconsistency | REQ-EUM-03, OQ-1 |
| EU-MMLU is capability, not safety | REQ-B-02, REQ-SC-04 |
| Transparent, editable tests modifiable by the WG | REQ-EDIT-01…08 |
| Pane establishing the basis for bias evaluation | REQ-UI-02, REQ-BIAS-02/05 |
| Pane establishing scoring methodology | REQ-UI-03, §6 |
| "Concrete product rather than a methodology" | REQ-EX-01…08, REQ-UI-07 |

## 10. Open questions

- **OQ-1** EU-MMLU subject count: announcement says seven, dataset card lists eight. Resolved
  operationally by REQ-EUM-03 (enumerate from data), but the WG should confirm which subjects
  are in the scored baseline.
- **OQ-2** AILuminate access tier — the public practice set versus the official evaluation
  route through MLCommons. Affects whether scores are externally citable.
- **OQ-3** Judge model policy. Using a cloud model to judge on-prem model outputs may breach
  REQ-NF-03 for restricted content.
- **OQ-4** Whether the four families roll up into a single published EUIBA score, or are
  always reported as a four-axis profile. Current default: four axes (REQ-SC-04).
- **OQ-5** Pass/fail thresholds are placeholders pending WG ratification.
- **OQ-6** Whether EU-MMLU items should be excluded from cloud models on data-contamination
  grounds (public dataset, plausibly in training data).
- **OQ-7** Target delivery form for the panes (web application vs. exported static report).
  Requirements are written to allow either; §7 assumes a UI exists.

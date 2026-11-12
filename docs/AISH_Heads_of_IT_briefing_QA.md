# ICDT_AISH briefing for Heads of IT: anticipated questions and answers

Companion to [`AISH_Heads_of_IT_briefing.pptx`](AISH_Heads_of_IT_briefing.pptx). Answers are short enough to say
aloud. Each one states where in the repository the claim can be checked. Items marked **CONFIRM** depend on facts
that are not in the repository and need the presenter's own answer.

## Purpose and positioning

**1. In one sentence, what is it?**
A web application that runs the same versioned tests against any AI model you point it at, on four separate axes
(language parity, safety, bias, grounding), and keeps every prompt and response behind each score. (`README.md`)

**2. Why not use an existing evaluation tool or the vendors' own benchmark figures?**
Vendor figures are not comparable with each other and rarely cover the smaller EU languages. General-purpose
evaluation frameworks are toolkits for engineers. AISH adds what the institutions specifically need: EU-MMLU parity
across 17 EU languages, bias tests grounded in the Charter and the equality directives, grounding tests on EU
institutional facts, an editor that non-programmers can use, and a data-residency rule enforced in software. It
is complementary to other tools, not a replacement for them.

**3. Is this a compliance tool for the AI Act?**
No. It produces evidence that can support your own assessment: the bias suite cites AI Act Articles 10 and 15 among
its bases. It is not a conformity assessment and a pass is not a legal finding. (`docs/bias-methodology.md` §2)

**4. Why no single overall score?**
Because averaging hides the failure that matters. A model that is excellent in English and weak in Irish, or
accurate but discriminatory, would get a comfortable average. Whether to publish a composite at all is an open
question for the Working Group (OQ-4). (`docs/scoring.md` §1, `docs/requirements.md` §10)

**5. Who owns it and who decides what the tests are?**
The ICDT ET Working Group. Official suites change only through a reviewed pull request, with a version bump and a
changelog entry. (`packs/README.md`)

## Data protection and security

**6. What data leaves my institution when I use it?**
Only the test prompts, sent to the model endpoint you configured, and that model's replies. There is no personal
data in the official suites; the bias scenarios use fictional candidates. If you test a cloud model, the prompts go
to that cloud provider under your own contract and key.

**7. Where do my API keys go?**
They are encrypted with AES-256-GCM, bound to your account so they cannot be decrypted for another user, shown
afterwards only as the last four characters, and used only for your runs. You can change or remove a key at any
time. (`SECURITY.md`)

**8. Can hazardous safety prompts end up at a cloud provider?**
No. The safety suite is marked restricted. The application refuses to run a restricted suite against a cloud
model or with a cloud judge, both when the run is requested and again in the runner. A personal copy of a
restricted suite stays restricted. (`aish/web/runs.py`, `aish/runner.py`)

**9. Who can see my results?**
Only you. Models, keys, runs and custom suites are scoped to the account that created them. There is currently no
shared, cross-institution league table; comparability comes from everyone running the same suite version, not from
pooling results.

**10. Where is the shared instance hosted, and is it approved for institutional use?**
**CONFIRM.** The repository says only that it is deployed at `aish.devalier.com` behind Caddy with automatic TLS.
Hosting location, operator and any security accreditation are not documented. Safe answer today: treat the shared
instance as a working-group pilot; for anything sensitive, self-host inside your own network.

**11. Has it been security-tested?**
The security model is documented claim by claim so it can be checked against the code: Argon2id passwords,
server-side revocable sessions, CSRF protection on every write, a strict content security policy, endpoint
validation before every outbound call, and dependency and static-analysis scans on every change. No independent
penetration test or audit is recorded in the repository, and there is no second factor or password reset yet.
(`SECURITY.md`, including "Known gaps")

## Running it

**12. Which models can it test?**
Anything reachable through the OpenAI, Anthropic or DeepSeek APIs, and any endpoint that speaks the
OpenAI-compatible protocol, which is what most on-prem serving stacks expose. The reference fleet is Llama 3.3 70B,
Mistral Small 24B and GPT OSS 120B on-prem, and GPT 5.1 and Claude 4.6 in the cloud. Adding a model is a
configuration entry. (`packs/models.yaml`, `aish/providers/`)

**13. What does a run cost and how long does it take?**
The software is free. You pay your own model usage. A sample language run is about 3,000 short multiple-choice
prompts; a full one is about 15,000. The bias and grounding suites are far smaller (15 scenarios expanded across
groups and repeats; 30 questions). Duration depends on your endpoint; requests run a few at a time in parallel.
There is no per-run cost budget or estimate in the tool yet. (`README.md`, "Run sizes")

**14. What do I need to self-host?**
A host that runs containers, two secrets, and network access to your model endpoints. `docker compose up` brings up
the application with TLS. It uses SQLite by default, which suits a single instance; PostgreSQL is the documented
route for more. You can restrict registration to your own email domains and require administrator approval.
(`README.md`, `.env.example`, `docker-compose.yml`)

**15. Is there support or a service level?**
No. It is a working-group project. Issues and fixes go through the repository.

## Customising and building your own

**16. What exactly can my staff change without a developer?**
In the editor: bias scenarios, the groups compared and the scoring rubric; grounding questions, source texts and
accepted answers; the definition of "unsafe" per hazard and the benign control prompts; the languages, prompt
wording and sample size of the language suite. They can preview the prompts and try a handful against a model
before saving. (`README.md`, "Customising a test suite")

**17. What can they not change?**
The suite's identity, scorer, dataset source, thresholds, judge and restricted flag. The MLCommons safety prompts
and the EU-MMLU questions are fixed. This is deliberate: a personal version cannot quietly lower the bar.

**18. Can a custom version be passed off as the official benchmark?**
No. Runs of a personal version are labelled custom, carry a fingerprint of the exact content, are charted in a
separate view, and are never mixed with official results or with an older version of the same custom suite.

**19. We want a harness for something else entirely, for example translation quality or a document-classification
model. Can we build on this?**
Yes, with a developer. A new suite is a YAML file plus, if none of the existing scorers fits, a new scorer
registered by name. Accounts, key storage, the runner, evidence storage and the comparison views are reused. The
EUPL allows you to fork and adapt it.

**20. What does the EUPL oblige us to do?**
You may use, modify and redistribute freely. If you distribute a modified version, including making it available
as a network service to others, you must make the source available under the EUPL or a compatible licence.
Internal use inside your institution carries no publication duty. The licence was written for European public
administrations and has equal legal value in all EU languages. Third-party data keeps its own terms: EU-MMLU is
CC BY 4.0 and MIT; the AILuminate prompts are fetched under MLCommons' terms and never redistributed.
(`LICENSE`, `NOTICE`) Take your legal service's view for any specific redistribution.

## Trusting the numbers

**21. How do you know the scoring is right?**
Three ways. Deterministic scorers use published word lists and parsers that are shown in the application. Where a
language model acts as judge, it must be calibrated against human labels and runs are refused if agreement
(Cohen's kappa) is below 0.6; until calibrated, results are labelled uncalibrated. And every scored item can be
opened and read. (`docs/scoring.md` §5)

**22. Could a model have seen the test questions in training?**
For the public benchmarks, plausibly yes; whether to treat cloud models differently for that reason is an open
question (OQ-6). For bias, the design includes a held-out scenario set that is versioned but not published; it has
not been written yet. (`packs/suites/euiba_bias.yaml`, review checklist)

**23. The demo shows well-known models failing. Are those real results?**
No. The screenshots come from the demo data generator, which uses scripted stand-in models with deliberately
uneven behaviour so every screen has something to show. No real model was called. Do not quote those figures.
(`scripts/seed_demo.py`)

**24. A model passes all four axes. Is it safe to deploy?**
It means no problem was found in what was tested: single questions and answers, in the tested scenarios, at the
current placeholder thresholds. It says nothing about agents, tool use, long conversations, your own data or your
own use case. It is one input to a deployment decision.

**25. Does it test our own deployment: system prompt, retrieval, guardrails?**
It tests whatever sits behind the endpoint you give it. If your gateway applies a system prompt or guardrails, they
are included in what is measured. It does not test retrieval over your documents, except through grounding
questions you write yourself with the source text supplied.

## Contributing

**26. What is the smallest useful contribution?**
One native speaker reviewing the bias scenarios in their language, or one lawyer verifying the 30 grounding facts
against official sources. Both are listed as pending on the suites' readiness checklists.

**27. How do we contribute tests?**
Write them in the editor, export the YAML, and open a pull request. The validator names the file and the line of
any problem. Working Group review is the approval gate. (`packs/README.md`)

**28. What is on the roadmap?**
Not a published roadmap, but the open items are explicit: ratify thresholds; calibrate the safety judge; review
and extend the bias and grounding content; write the held-out set; add password reset, a second factor and an
administrator screen; retry on provider errors; PostgreSQL for multi-instance use. (`README.md` "Status",
`SECURITY.md` "Known gaps", `docs/requirements.md` §10)

**29. Is the repository public?**
**CONFIRM** the visibility of `github.com/devalier/ICDT_AISH` before the session, and whether a Joinup or
code.europa.eu home is planned, since some institutions will ask for an EU-hosted forge.

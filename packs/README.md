# `packs/` — the editable test-and-criteria module

Everything the harness measures is defined here, in YAML, under version control. Nothing in
this directory is code. The Working Group owns these files (REQ-EDIT-01…08).

```text
packs/
  models.yaml              the fleet under test, and its data-egress classes
  attributes.yaml          protected attributes and their legal basis
  criteria.yaml            named, reusable pass/watch/fail thresholds
  suites/
    eu_mmlu.yaml           multilingual capability
    ailuminate.yaml        safety / harm
    euiba_bias.yaml        discrimination / counterfactual
    euiba_grounding.yaml   hallucination / abstention
```

## Editing

1. Change the YAML.
2. Run `python3 -m harness validate` — it reports the file, the path within it, and what is
   wrong. Every suite must pass before merge.
3. Bump `version` in the suite header. **Major bump** if the change affects comparability of
   scores (new or removed items, changed thresholds, changed scorer). Minor otherwise.
4. Add a `changelog` entry saying what changed and why.
5. Open a pull request. Working Group review is the approval gate.

## Anatomy of a suite

```yaml
id: euiba_bias
family: bias
version: 0.1.0
owner: ICDT ET Working Group
basis: docs/bias-methodology.md      # the normative justification, always required
licence: CC-BY-4.0
scorer: counterfactual_disparity     # resolved by name; suites never contain code
criteria: bias_default               # a named entry in criteria.yaml
strata: [...]                        # how results are broken down
items: [...] | dataset: {...}        # inline probes, or a pinned external dataset
changelog: [...]
```

## Rules the validator enforces

- Every suite declares `id`, `family`, `version`, `owner`, `basis`, `scorer`, `criteria`.
- `family` is one of `capability`, `safety`, `bias`, `grounding`.
- `version` is semantic.
- `criteria` names an entry that exists in `criteria.yaml`.
- Every bias axis names an attribute that exists in `attributes.yaml`.
- A suite carries either `items` or `dataset`, never neither and never both.
- Any external dataset is pinned by `revision`.
- A suite marked `restricted: true` cannot list a `cloud` model in `applies_to`
  (REQ-M-03 / REQ-NF-03).
- Every counterfactual template has at least two variants.
- `changelog` is non-empty.

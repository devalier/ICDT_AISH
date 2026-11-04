# EU-MMLU snapshot

A validated copy of **EC-DGT-AI/EU-MMLU** (https://huggingface.co/datasets/EC-DGT-AI/EU-MMLU),
upstream commit `6bf04b0030783b3c9074c8d59416ff761962568a`, bundled so that the Language suite runs
without downloading anything on the server and so every score can be traced to these exact rows
(REQ-EUM-07). Regenerate with `python3 scripts/fetch_eu_mmlu.py --out datasets/eu_mmlu`.

| File | Contents |
|---|---|
| `eu_mmlu.jsonl` | 17,069 validated rows, in the published schema, otherwise unmodified (`Index` and `Answer` are integers) |
| `eu_mmlu.rejected.jsonl` | 131 rows that failed validation, each with the reason |
| `eu_mmlu.revision.json` | upstream commit, sha256 of every source file, per-language counts |

## Licence and credit

- Translations: **CC BY 4.0**, produced by the European Commission's Directorate-General for
  Translation and the EMT Network (European Master's in Translation).
- `EN` rows: unchanged content of the original **MMLU**, **MIT licence**, © 2020 Dan Hendrycks.
  The MIT notice is reproduced in the upstream dataset card.

Citations: Sánchez-Gijón, Valdez, Kokkinidou, Bellemont, Calvo Del Barrio, Brasoveanu (2026),
*Building a European Multilingual Evaluation Dataset: The MMLU Localisation Project within the EMT
Network*; Hendrycks et al. (2021), *Measuring Massive Multitask Language Understanding*, ICLR.
The European Commission accepts no liability for possible errors or for any misuse.

## What we found in the release, and what the harness does about it

- **Splits.** 15,021 `test`, 1,572 `validation`, 610 `dev` rows. Only `test` is scored.
- **Partial translation.** The card says so. Irish has 4 of 8 subjects (391 test questions),
  Hungarian 617, Portuguese 741; `high_school_european_history` exists only in EL, EN, ES, FR, IT, PL.
  Parity is computed on the questions a language shares with English.
- **`Index` is aligned across languages.** The same (Subject, Split, Index) is the same question, and
  every language's `Answer` equals the English one (checked for all rows except Spanish).
- **Defects, quarantined not repaired (131 rows):**
  - Hungarian: one row (`college_chemistry`, test) has the text `12 másodperc` in the `Index` column.
  - Portuguese: 35 pairs of different questions share one `Index` within a subject.
  - Spanish: 30 such pairs plus 3 exact repeats, and 30 answers that differ from English. The Spanish
    file is for a locale the card does not list as released, so it is not scored.
  - The Lithuanian file has two empty trailing header cells.
- **Report these to the maintainers** so they can be fixed upstream.

# EU-MMLU snapshot

A validated copy of **EC-DGT-AI/EU-MMLU** (https://huggingface.co/datasets/EC-DGT-AI/EU-MMLU),
upstream commit `6bf04b0030783b3c9074c8d59416ff761962568a`, bundled so that the Language suite runs
without downloading anything on the server and so every score can be traced to these exact rows
(REQ-EUM-07). Regenerate with `python3 scripts/fetch_eu_mmlu.py --out datasets/eu_mmlu`.

| File | Contents |
|---|---|
| `eu_mmlu.jsonl` | 17,120 validated rows, in the published schema, otherwise unmodified (`Index` and `Answer` are integers) |
| `eu_mmlu.repairs.jsonl` | the 146 changes the repair rules made (fix, removal, quarantine), each with its evidence |
| `eu_mmlu.rejected.jsonl` | 15 rows that could not be repaired, each with the reason |
| `eu_mmlu.revision.json` | upstream commit, sha256 of every source file, per-language counts, repair counts |

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
- **Defects, and what was done (`scripts/eu_mmlu_repair.py`).** Every fix needs evidence from the
  rest of the release; a fix that fails its check leaves the row rejected. Result: all 16 listed
  locales and Spanish now agree with English on every answer.
  - *Hungarian, 1 row repaired.* `college_chemistry` test had `12 másodperc` in the `Index` cell.
    Index 12 was the only free slot, and the English row there is the same question (cyclobutyl
    radical, 115°) with the same answer and the same four choices. Put back at Index 12.
  - *Portuguese, 35 rows removed.* The Portuguese file contains 35 German `sociology` rows, each
    identical to the German file's row at the same key and in German words. The Portuguese row at
    every one of those keys was kept.
  - *Spanish, 30 pairs resolved, 15 rows unrecoverable.*
    - 28 rows filed as `high_school_european_history` are exact copies of the Spanish `sociology`
      row at the same Index; the other row at the key is not a copy and carries the English history
      answer. The copies were removed. 2 pairs were two drafts of one history question (>= 98% the
      same text, same answer); the cleaner draft was kept.
    - 6 rows are Italian text copied from the Italian file (`high_school_european_history`
      Index 6-11). 9 more are sociology questions filed under history: 6 whose answer matches the
      English sociology answer and not the history one, and 3 (moral panic, snowball sampling,
      leisure) that are the length of the English sociology question, not of the history passage
      they are filed under. The correct Spanish text for those 15 slots is not in the release, so
      they are quarantined, not invented.
  - *Lithuanian:* two empty trailing header cells, tolerated.
  - *Not defects:* a few questions repeat verbatim within a language (they repeat in the original
    English MMLU too), and 47 rows have two equal choices (as in MMLU).
- **Spanish is complete except 26 history questions and is scored** (suite v0.3.0), by Working Group
  decision, although the dataset card does not list it among the released locales.
- **Report these to the maintainers** so the defects can be fixed upstream.

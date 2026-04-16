# Data Notes - LendingClub

## Source

The original project used public LendingClub funded-loan CSV exports. The README
documents a 2007 through Q3 2018 funded-loan data range with 145 columns and more
than 2 million rows before filtering.

Raw data is intentionally not committed to this repository.

## Included task scope

The model predicts the final outcome for completed approved loans:

- `Fully Paid`
- `Charged Off`

Loans still in progress were excluded because their final status was unknown.
Declined-loan records were not used because the original project found them too
limited for the same modeling task.

## Expected local layout

The original notebooks were written in a Colab/local hybrid style. A future
reproducibility pass should standardize the following layout:

```text
data/
  raw/
    lendingclub/
      *.csv
  interim/
    loan_df0.pkl
  processed/
    clean_df_Mon.pkl
  census/
    census_zipcode_level.csv
```

Current rendered pages reference historical paths such as:

- `./data/Pickle/loan_df0.pkl`
- `/content/gdrive/My Drive/Lending Club Project/data/Pickle/clean_df_Mon.pkl`
- `/content/gdrive/My Drive/Lending Club Project/data/census/census_zipcode_level.csv`

## Sensitive attribute proxy

The dataset did not include borrower race. The fairness audit linked 3-digit ZIP
codes with Census-derived racial proportions and created an `underprivileged`
proxy flag from low `percent_white` ZIP3 groups.

This is suitable for an educational fairness audit, but it should not be treated
as an individual-level demographic label.

## Privacy and governance

- Do not commit raw borrower-level data.
- Do not commit intermediate pickle files if they contain borrower-level records.
- Do not use this project for real credit decisions without legal and compliance
  review.
- Any future public demo should use synthetic or heavily aggregated examples.

## Reproducibility next step

The next engineering pass should add scripts that:

1. Validate the expected data layout.
2. Rebuild cleaned features from raw CSVs.
3. Emit a fresh metrics JSON file.
4. Regenerate `reports/FAIRNESS_REPORT.md`.

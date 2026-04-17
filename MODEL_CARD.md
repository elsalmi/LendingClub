# Model Card - LendingClub Loan Outcome and Fairness Audit

## Model details

- Task: binary classification of approved loans as `Fully Paid` vs `Charged Off`.
- Model families used in the original project: decision tree baselines,
  logistic regression for fairness metric reporting, and random forest for the
  final investor-oriented workflow.
- Primary reported final model: random forest with reweighing-derived instance
  weights in the fairness workflow.
- Positive label: `Fully Paid`.

## Intended use

This model card supports portfolio review and educational discussion. The repo
is useful for explaining credit-risk modeling, performance tradeoffs, fairness
metrics, and mitigation strategy.

This project is not intended for production lending decisions.

## Data

- Source: public LendingClub funded-loan data exports used in the original
  project.
- Time span documented in the original README: 2007 through Q3 2018.
- Excluded rows: loans that were not completed at analysis time.
- Excluded dataset: declined-loan records, due to limited fields and high
  missingness in the original project.
- Sensitive attribute: no direct race label is available; the audit uses a
  ZIP3-linked demographic proxy from Census-derived racial proportions.

## Metrics reported from existing artifacts

| Metric | Value |
| --- | ---: |
| Logistic regression classification accuracy | 0.652441 |
| Statistical parity difference | -0.019635 |
| Disparate impact | 0.966937 |
| Equal opportunity difference | -0.013784 |
| Average odds difference | -0.013502 |
| Theil index | 0.354349 |
| Random forest precision | 0.913286 |
| Random forest test score | 0.506695 |

These values are copied from the rendered notebook output in `Fairness.md`.

## Fairness approach

The fairness audit uses AIF360 metrics to compare outcomes across privileged and
underprivileged proxy groups. The original project then applies reweighing as a
preprocessing mitigation step.

Reported mean-outcome differences:

| Dataset state | Mean outcome difference |
| --- | ---: |
| Original training dataset | -0.017565 |
| Reweighed training dataset | 0.000000 |
| Original testing dataset | -0.019698 |
| Reweighed testing dataset | -0.002036 |

## Limitations

- ZIP-derived demographic proxy labels are noisy and can create ecological
  fallacy risk.
- The raw data is not included, so this repo currently preserves the original
  analysis rather than providing a one-command rerun.
- The fairness analysis is an audit and mitigation example, not proof that a
  deployed lending system would be compliant or safe.
- Metrics reflect the notebook state captured in the repo and should be
  regenerated before any future publication claims.

## Regeneration

Run `python scripts/build_fairness_report.py` to re-emit the short report under
`reports/FAIRNESS_REPORT.md` from the committed metric snapshot.

## Next validation work

1. Convert the notebooks into scriptable pipeline steps.
2. Regenerate metrics from a pinned environment.
3. Add calibration and threshold-sweep analysis.

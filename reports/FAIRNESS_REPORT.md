# Fairness Report - LendingClub

This short report is generated from the committed evidence in
[MODEL_CARD.md](../MODEL_CARD.md) and traces back to the notebook export in
[Fairness.md](../Fairness.md).

## Executive summary

This project predicts whether an approved LendingClub loan will be `Fully Paid`
or `Charged Off`, then checks whether outcomes differ across a ZIP3-derived
demographic proxy. Reweighing reduced the reported training mean outcome
difference from `-0.017565` to `0.000000`, and reduced the reported test mean
outcome difference from `-0.019698` to `-0.002036`.

## Prediction task

- Positive label: `Fully Paid`.
- Negative label: `Charged Off`.
- Protected-proxy attribute: `underprivileged`, derived from ZIP3-linked Census
  racial composition.
- Fairness toolkit: AIF360.

## Metric snapshot

| Metric                                      | Value     |
|---------------------------------------------|-----------|
| Logistic regression classification accuracy | 0.652441  |
| Statistical parity difference               | -0.019635 |
| Disparate impact                            | 0.966937  |
| Equal opportunity difference                | -0.013784 |
| Average odds difference                     | -0.013502 |
| Theil index                                 | 0.354349  |
| Random forest precision                     | 0.913286  |
| Random forest test score                    | 0.506695  |

## Reweighing snapshot

| Dataset state              | Mean outcome difference |
|----------------------------|-------------------------|
| Original training dataset  | -0.017565               |
| Reweighed training dataset | 0.000000                |
| Original testing dataset   | -0.019698               |
| Reweighed testing dataset  | -0.002036               |

## Key risks

- ZIP3 proxy groups can expose aggregate disparity, but they are not individual
  protected-class labels.
- Reweighing reduces one measured disparity but does not guarantee fairness
  under every threshold, metric, or deployment setting.
- The raw data and runtime are not yet packaged for one-command reproduction.

## Regeneration

Run `python scripts/build_fairness_report.py --write` to rewrite this file from
the committed model-card snapshot.

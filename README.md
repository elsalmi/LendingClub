---
title: LendingClub - Loan Outcome Prediction with Fairness Auditing
---

# LendingClub - Loan Outcome Prediction with Fairness Auditing

## TL;DR

This project predicts whether an approved LendingClub loan is likely to be
`Fully Paid` or `Charged Off`, then audits whether model outcomes differ across
ZIP-derived demographic proxy groups.

The notebook export (`Fairness.md`) is still the source of truth. The model
card and the short report under `reports/` summarize the evidence without
copying the same story in three places.

## What is in this repo

| Artifact | Purpose |
| --- | --- |
| [Data-Cleaning.md](Data-Cleaning.md) | Data loading, filtering, and feature preparation |
| [EDA.md](EDA.md) | Exploratory analysis of the loan data |
| [Models.md](Models.md) | Baseline and random forest experiments |
| [Fairness.md](Fairness.md) | Notebook export with the fairness metrics and mitigation outputs |
| [Discussion.md](Discussion.md) | Results framing and tradeoffs |
| [MODEL_CARD.md](MODEL_CARD.md) | Intended use, limitations, and metric snapshot |
| [docs/DATA.md](docs/DATA.md) | Data source, layout, and privacy notes |
| [reports/FAIRNESS_REPORT.md](reports/FAIRNESS_REPORT.md) | Compact report generated from the committed evidence |

## Problem

Peer-to-peer loan investors care about default risk, but a model that only
optimizes predictive performance can hide unequal behavior across borrower
groups. This project keeps the task simple:

1. Predict whether an approved loan will be fully paid or charged off.
2. Prioritize high precision for loans predicted as fully paid.
3. Audit whether predictions differ across demographic proxy groups.
4. Document the fairness tradeoffs instead of presenting a single score.

## Data

The original project used LendingClub funded-loan data from 2007 through Q3
2018. Declined-loan records were excluded because they had fewer useful fields
and substantial missingness. The raw data is not committed to this repository.

Important constraints:

- The project focuses on completed loans only: `Fully Paid` and `Charged Off`.
- FICO scores were not available in the public dataset used here.
- Sensitive race labels were not available, so the fairness audit used a ZIP3
  proxy linked with Census-derived racial proportions.
- The ZIP proxy is useful for auditing patterns, but it is not an individual
  race label and should be treated as a noisy proxy.

See [docs/DATA.md](docs/DATA.md) for data access and expected local layout.

## Method

The original workflow is notebook-first and includes:

- Data filtering and cleaning from LendingClub CSV exports.
- Feature preparation for tabular credit and loan attributes.
- Baseline decision tree experiments.
- Random forest modeling for the final investor-oriented ranking workflow.
- AIF360 fairness metrics for protected-proxy group comparison.
- Reweighing as a preprocessing mitigation strategy.

## Evidence snapshot

The detailed values live in [MODEL_CARD.md](MODEL_CARD.md) and are regenerated
into [reports/FAIRNESS_REPORT.md](reports/FAIRNESS_REPORT.md). The quickest
read is:

| Result | Value |
| --- | ---: |
| Logistic regression accuracy | 0.652441 |
| Random forest precision | 0.913286 |
| Reweighed training mean outcome difference | 0.000000 |
| Reweighed test mean outcome difference | -0.002036 |

The random forest result should be interpreted with care: the original
discussion optimizes for high precision and ranked loan selection, not broad
deployment as a fully calibrated automated lending system.

## Reproduce

The current repo is a rendered notebook archive. To reproduce the original work:

1. Download the LendingClub funded-loan CSV exports locally.
2. Place raw and intermediate files using the layout described in
   [docs/DATA.md](docs/DATA.md).
3. Run the notebooks in order from `notebooks/`:
   `Data-Cleaning.ipynb`, `EDA.ipynb`, `Models.ipynb`, `Fairness.ipynb`.
4. Regenerate the short report with `python scripts/build_fairness_report.py`.

The report script regenerates `reports/FAIRNESS_REPORT.md` from the committed
metric snapshot and keeps the notebook evidence visible without repeating the
same story in multiple files.

## Responsible use

This is an educational and portfolio project. It should not be used for real
credit decisions without a full compliance, legal, governance, and validation
process. The ZIP3 proxy can expose group-level disparities, but it can also
misrepresent individuals.

## References

- Kamiran, F. and Calders, T. "Data preprocessing techniques for classification
  without discrimination." Knowledge and Information Systems, 2012.
- Bellamy, Rachel K.E., et al. "AI Fairness 360: An Extensible Toolkit for
  Detecting, Understanding, and Mitigating Unwanted Algorithmic Bias." 2018.
- Emekter, Riza, et al. "Evaluating Credit Risk and Loan Performance in Online
  Peer-to-Peer Lending." Applied Economics, 2014.
- O'Neil, Cathy. *Weapons of Math Destruction*. Penguin Books, 2018.

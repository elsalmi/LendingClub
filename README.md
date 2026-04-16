---
title: LendingClub - Loan Outcome Prediction with Fairness Auditing
---

# LendingClub - Loan Outcome Prediction with Fairness Auditing

## TL;DR

This project predicts whether an approved LendingClub loan is likely to be
**Fully Paid** or **Charged Off**, then audits whether model outcomes differ
across ZIP-derived demographic proxy groups.

The strongest portfolio signal is not only the classifier. It is the full
workflow: data cleaning, feature selection, model comparison, fairness metrics,
and reweighing-based mitigation using AIF360.

## What is in this repo

| Artifact | Purpose |
| --- | --- |
| [Data-Cleaning.md](Data-Cleaning.md) | Data loading, filtering, feature preparation, and missing-value treatment |
| [EDA.md](EDA.md) | Exploratory analysis for LendingClub loan data |
| [Models.md](Models.md) | Baseline and random forest model experiments |
| [Fairness.md](Fairness.md) | Bias detection and mitigation notebook export with reported metrics |
| [Discussion.md](Discussion.md) | Results framing, investor objective, and fairness discussion |
| [MODEL_CARD.md](MODEL_CARD.md) | Portfolio-oriented model card and intended-use boundaries |
| [docs/DATA.md](docs/DATA.md) | Data source, access, and privacy notes |
| [reports/FAIRNESS_REPORT.md](reports/FAIRNESS_REPORT.md) | Short fairness report with metric snapshot |

## Problem

Peer-to-peer loan investors care about default risk, but a model that only
optimizes predictive performance can hide unequal behavior across borrower
groups. This project frames the task as:

1. Predict whether an approved loan will be fully paid or charged off.
2. Prioritize high precision for loans predicted as fully paid.
3. Audit whether predictions differ across demographic proxy groups.
4. Document mitigation tradeoffs instead of presenting a single accuracy score.

## Data

The original project used LendingClub funded-loan data from 2007 through Q3
2018, excluding declined loans because those files had fewer useful fields and
substantial missingness. The raw data is not committed to this repository.

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

## Reported metrics

The following values are reported in [Fairness.md](Fairness.md). They are not
newly recomputed by this README.

| Result | Value |
| --- | ---: |
| Logistic regression classification accuracy | 0.652441 |
| Statistical parity difference | -0.019635 |
| Disparate impact | 0.966937 |
| Equal opportunity difference | -0.013784 |
| Average odds difference | -0.013502 |
| Theil index | 0.354349 |
| False negative rate difference | 0.013784 |
| Training mean outcome difference before reweighing | -0.017565 |
| Training mean outcome difference after reweighing | 0.000000 |
| Test mean outcome difference before reweighing | -0.019698 |
| Test mean outcome difference after reweighing | -0.002036 |
| Random forest precision | 0.913286 |
| Random forest test score | 0.506695 |

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

Next engineering step: extract the notebooks into idempotent scripts that write
fresh report artifacts under `reports/`.

## Responsible use

This is an educational and portfolio project. It should not be used for real
credit decisions without a full compliance, legal, governance, and validation
process. The ZIP3 proxy can expose group-level disparities, but it can also
misrepresent individuals.

## Project history

This project began as Harvard CSCI E-109A group work by Victor Chen, Danielle
Crumley, Mohamed Elsalmi, and Hoon Kang. The current documentation pass reframes
the repo as a portfolio artifact and preserves the original notebook-derived
pages for traceability.

## References

- Kamiran, F. and Calders, T. "Data preprocessing techniques for classification
  without discrimination." Knowledge and Information Systems, 2012.
- Bellamy, Rachel K.E., et al. "AI Fairness 360: An Extensible Toolkit for
  Detecting, Understanding, and Mitigating Unwanted Algorithmic Bias." 2018.
- Emekter, Riza, et al. "Evaluating Credit Risk and Loan Performance in Online
  Peer-to-Peer Lending." Applied Economics, 2014.
- O'Neil, Cathy. *Weapons of Math Destruction*. Penguin Books, 2018.

# Model Card

## Intended use

Demonstrate binary classification, security-oriented evaluation, and reproducible machine-learning engineering in a public portfolio.

## Data

The generator creates 120,000 synthetic network-flow observations with 24 numeric features and a moderately imbalanced binary label. No confidential, personal, or live network information is included.

## Selection

Four models are compared using the same stratified 80/20 split. The exported model is selected using malicious-class F1 score. Review `outputs/metrics.json` for the actual run results.

## Risks

Performance on synthetic data does not establish performance on real traffic. Before operational use, validate on approved representative data, use a time-based holdout, tune decision thresholds, monitor drift, test adversarial behavior, and retain a human review workflow.

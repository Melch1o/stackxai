# Changelog

## 0.2.0
- Dockerfile (test and runtime targets), pinned `constraints.txt`.
- Jenkins pipeline (lint, tests, image build, smoke test, optional push and Kubernetes Job).

## 0.1.0
- Two-stage stacking ensemble with six base learners (SVM-RBF, logistic regression,
  fuzzy k-NN, decision tree, neural network, naive Bayes).
- Stratified cross-validated comparison with fixed seeds.
- Global (permutation importance) and local (SHAP) explanations.
- Command-line interface and GitHub Actions CI/CD pipeline.

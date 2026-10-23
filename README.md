# Adaptive, reliability-aware oracle guidance vs fixed-threshold critic uncertainty

## Motivation
The paper "Some supervision required: incorporating oracle policies in reinforcement learning via epistemic uncertainty metrics" (2022) proposes Critic Confidence Guided Exploration (CCGE). It uses an oracle policy's actions as suggestions when the critic is uncertain and ignores them when it is confident. A fixed uncertainty threshold can be badly calibrated, and it ignores that an oracle may be good in some regions and poor in others. This experiment tests a simple remedy in a toy setting.

## Question
In a simple sparse-reward gridworld with an imperfect oracle, can a critic-uncertainty rule that is adaptively calibrated and also estimates oracle reliability learn more sample-efficiently than a fixed-threshold critic-uncertainty rule in the style of CCGE?

## Dataset
No public data. A seeded 12x12 walled sparse-reward gridworld is simulated in numpy, with an oracle that is near-optimal in some regions and poor in others. Free cells are split once (75/25) into training start cells and fixed held-out evaluation start cells. This is not the paper's dataset, and the baseline is a simplified stand-in rather than the paper's exact method.

## Baseline vs improved
- Baseline: tabular Q-learning with 5 critics. It follows the oracle action when the ensemble std of Q at the state exceeds a fixed threshold (chosen on training start cells).
- Improved: same learner and exploration schedule. The threshold is a running quantile of observed uncertainty, and a per-region oracle-reliability estimate (critic value of the oracle action versus the greedy action, plus realised returns) must be high before deferring to the oracle.
- Main metric: mean greedy success rate over training (area under the learning curve). Secondary: final success rate, episodes to 80% success, oracle-query fraction.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a learning-curve plot to `figures/`).

## Results (latest run)

On a simulated 12x12 sparse-reward gridworld with an imperfect oracle, the fixed-threshold CCGE-style baseline (a simplified tabular stand-in) reached 8.7% mean greedy success-rate area under the learning curve, while my adaptive, oracle-reliability-aware variant reached 14.5% on the same held-out start cells.

| Metric | Value |
|---|---|
| baseline_score | 0.08695 |
| baseline_score_std | 0.007665 |
| baseline_final | 0.1333 |
| baseline_final_std | 0.01245 |
| baseline_episodes_to_80 | 150 |
| baseline_episodes_to_80_std | 0 |
| baseline_oracle_frac | 0.4582 |
| baseline_oracle_frac_std | 0.03876 |
| improved_score | 0.1449 |
| improved_score_std | 0.006822 |
| improved_final | 0.2215 |
| improved_final_std | 0.01064 |
| improved_episodes_to_80 | 150 |
| improved_episodes_to_80_std | 0 |
| improved_oracle_frac | 0.2208 |
| improved_oracle_frac_std | 0.002353 |
| improvement | 0.05799 |
| baseline_param | 0.04 |
| improved_param | 0.5 |
| oracle_optimal_action_fraction | 0.561 |

Run on 2026-10-23. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Simplified tabular stand-in, not the paper's actor-critic CCGE. Test set = held-out start cells (25% of free cells); hyper-parameters (threshold / quantile) chosen on separate seeds using training start cells only; other constants fixed a priori. Episodes-to-80% is censored at 150 if never reached. Realised-return evidence for reliability is noisy (a failed episode penalises all oracle steps in it). Results depend on the simulated oracle quality and layout.

## Reference

This project is motivated by the following paper:

> JJ Tai, JK Terry, MS Innocente, J Brusey (2022). *Some supervision required: incorporating oracle policies in reinforcement learning via epistemic uncertainty metrics*. arXiv preprint arXiv. https://arxiv.org/abs/2208.10533

Paper author: Prof. James Brusey, Coventry University (James.Brusey@coventry.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

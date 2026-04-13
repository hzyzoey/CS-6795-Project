# CS 6795 Spring 2026 — Computational Model/Tool Track


## Overview

This project implements a rule-based cognitive simulation of wildfire risk perception, built on the CRUM architecture (Thagard 2005) and Brunswik's lens model. The model tracks how a residential agent perceives and responds to wildfire hazard over a 156-week (3-year) period under different cognitive bias configurations.

Five scenarios (S0–S4) are compared:
- S0: Salience-biased cue weights, no suppression mechanisms (baseline)
- S1: Near-optimal cue weights, no suppression mechanisms
- S2: Salience-biased weights with experiential shielding only
- S3: Salience-biased weights with affect heuristic only
- S4: Full biased model (biased weights + both suppression mechanisms)


## Files
1. `simulation.py`: Main simulation script
   - Simulate all five scenarios over 156 weeks
   - Save figures to the `figures` subfolder (created automatically if it does not exist)
   - Write `summary_statistics.csv` and `trajectories.csv` to the working directory
   - Print a summary table to the terminal

2. requirements.txt: Python package dependencies
3. README.md: This file



## Output

1. Figures:
    -`fig1_scenario_comparison.png` — Risk perception trajectories for S0–S4
    -`fig2_mechanism_decomposition.png` — Mechanism decomposition across scenarios
    -`fig3_pairwise.png` — Pairwise scenario comparisons

2. CSV files:
    -`summary_statistics.csv`: Mean risk, peak values, underestimation rates, and utility scores per scenario.
    -`trajectories.csv`: Full week-by-week time series of R(t) and U(t) for all scenarios plus objective hazard.


## Note on AI Assistance

AI coding tools (Claude and Codex) were used to assist with code development and debugging.

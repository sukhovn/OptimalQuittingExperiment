# When to Keep Trying and When to Let Go: Benchmarking Optimal Quitting

Code and data for:

> N. Sukhov\*, R. Dubey\*, A. Duke, T. L. Griffiths. *When to keep trying and when to let go: Benchmarking optimal quitting.* Journal of Experimental Psychology: General, 154(9), 2599–2618 (2025). \*Equal contribution.

Pre-registration: https://osf.io/3w7bx

## Overview

How do people decide when to persist and when to quit? We design a novel task to study this question, in
which people were given a finite number of opportunities to pursue stochastic rewards by selecting among
a set of options that provide a reward each trial. At any time, if people were not satisfied with the option
they had selected they could choose to abandon it and instead try a new option. However, if they did so
they could never return to the previous option.

This repository contains:

1. **An exact solver for the optimal policy.** The policy is computed via dynamic programming over Beta-posterior belief states, implemented in C++ and callable from Python.
2. **Simulations** of the optimal policy and alternatives, including a meta-learning policy.
3. **Data and analysis** for a pre-registered behavioral experiment with 3,632 participants, comparing human decisions to the optimal policy (including Bayesian logistic regression fit with MCMC).

## Repository structure

```
Simulations/                 C++ solver and simulator and Python bindings
  Core/                      Reward and transition calculations
  Strategies/                Policy implementations
  Suppl/                     Minimal vendored subset of Boost (incomplete beta function)
  cpp_routines.py            Python interface (ctypes) to the compiled library
  API.txt                    Description of the Python interface
Simulations.ipynb            Behavior of the optimal policy (paper Table 1, Fig. 2)
FiniteBernoulliStopping.ipynb  Extension: finite number of available arms
ExperimentProcessing.ipynb   Behavioral data analysis (paper Figs. 4–5, Table 2, 3)
data_process.py              Tools used for data preprocessing
logit_analysis.py            MCMC logistic regression (emcee)
full_experiment_data.pkl     Behavioral data (see below)
```

## Getting started

Requirements: a C++ compiler (g++), Python 3, and the packages `numpy pandas scipy matplotlib scikit-learn emcee corner`.
The C++ code has no external dependencies. Run the notebooks from the repository root. 

```bash
pip install numpy pandas scipy matplotlib scikit-learn emcee corner
cd Simulations && make && cd ..
```

See `Simulations/API.txt` for the full interface, including simulation of complete runs under any policy.

## Data

`full_experiment_data.pkl` is a pandas DataFrame with one row per participant:

| Column | Description |
|---|---|
| `subjectID` | Anonymous participant index |
| `probability_truncation` | Condition: maximum arm success probability (0.7, 0.8, or 0.9) |
| `inst_time`, `tut_time` | Time spent on instructions and practice round (ms) |
| `date` | Session timestamp |
| `total_win` | Points earned in the main task |
| `quiz_failed` | Number of failed comprehension-quiz attempts |
| `practice_data`, `press_data` | Lists of presses: `press_number`, `button_number`, `current_p` (true arm probability), `outcome`, `press_time` |
| `feedback` | Optional free-text feedback |

Exclusion criteria are described in the paper and implemented in `ExperimentProcessing.ipynb`.

## Citation

```bibtex
@article{sukhov2025quitting,
  title   = {When to keep trying and when to let go: Benchmarking optimal quitting},
  author  = {Sukhov, Nikolay and Dubey, Rachit and Duke, Annie and Griffiths, Thomas L.},
  journal = {Journal of Experimental Psychology: General},
  volume  = {154},
  number  = {9},
  pages   = {2599--2618},
  year    = {2025}
}
```

## License

[MIT] for the code. The vendored Boost headers in `Simulations/Suppl/boost/` are under the Boost Software License 1.0.

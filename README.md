# SAC-PID Chaos Control

This repository contains the code and final experiment artifacts for SAC-PID control of the Lorenz and Lorenz96 chaotic systems.

## Repository layout

- `lorenz/`: Lorenz-system experiment.
- `lorenz96/`: Lorenz96-system experiment.
- `main/`: main training, evaluation, and plotting programs for the corresponding system.
- `bayesian_optimization/`: Bayesian-optimization scripts used for hyperparameter tuning.
- `optimization_results/`: JSON records of the selected Bayesian-optimization parameters.
- `figures/`: final figures and optimization visualizations.
- `tables/`: final CSV statistical tables.

Model checkpoints, NumPy return arrays, IDE settings, Python caches, and raw `.txt` run logs are intentionally excluded from version control.

## Environment

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Run commands from the repository root so imports resolve correctly. For example:

```bash
python -m lorenz.main.lorenz_8
python -m lorenz96.main.lorenz96_main
```
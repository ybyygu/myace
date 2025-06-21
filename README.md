# MyACE Project Utilities

This project provides a set of standardized tools for collecting and processing data for training Atomic Cluster Expansion (ACE) potentials with `pacemaker`. It's designed as a personal collection of scripts and helpers for `python-ace`.

## Project Structure

The project follows a modern `src-layout` managed by `rye`, clearly separating the core library code from other project files.

```
ace/
├── pyproject.toml         # Project metadata, dependencies, and script entry points
├── README.md              # This file
└── src/
    └── myace/             # The main Python package
        ├── __init__.py
        ├── data_processing.py # Core module for data collection and formatting
        ├── active_learning.py # Core module for active learning workflows
        └── cli/               # Command-line interface scripts
            ├── collect_data.py
            └── active_learning_tool.py
```

## Core Components & Workflow

This project is designed around an active learning loop.

### 1. `src/myace/data_processing.py` (Data Collection Library)

This module contains the foundational tools for creating `pacemaker`-compatible datasets.

### 2. `src/myace/active_learning.py` (Active Learning Library)

This module contains the core logic for evaluating potential reliability and selecting new training candidates.

### 3. Command-Line Tools

These are the user-facing tools that orchestrate the workflow. After setting up the environment with `rye sync`, you can run them directly.

#### `collect-data`

A general-purpose tool to parse data from various sources and build a `pacemaker`-compatible DataFrame.

**Usage:**
- **From VASP output:**
  ```bash
  # First, create a JSON file for your reference energies, e.g., ref.json
  # { "Pt": -5.80, "Rh": -6.50 }
  rye run collect-data /path/to/your/OUTCAR --format vasp --ref-energies ref.json --output my_data.pckl.gzip
  ```
- **From a directory of `extxyz` files (e.g., after an `ace-learn` run):**
  ```bash
  # This assumes the .xyz files in the directory contain energy/forces data in their info fields
  rye run collect-data selected_for_dft/ --format extxyz --ref-energies ref.json --output new_training_data.pckl.gzip
  ```

#### `ace-learn`

A tool to assess a trained potential's performance on a given set of structures and identify new, valuable training candidates based on uncertainty. This tool operates on DataFrames (`.pckl.gzip` files).

**Usage:**
```bash
# Assume `initial_data.pckl.gzip` is a DataFrame created by `collect-data`.

# 1. Evaluate all structures in the DataFrame against your potential
rye run ace-learn potential.yaml initial_data.pckl.gzip --asi potential.asi --output-eval-df evaluated.pckl.gzip

# 2. Evaluate and also select the 20 most informative high-gamma structures
rye run ace-learn potential.yaml initial_data.pckl.gzip --asi potential.asi --select 20 --output-selection-dir selected_for_dft
```
The command above will create a directory named `selected_for_dft` containing individual `.xyz` files (e.g., `000.xyz`, `001.xyz`, ...), ready for the next step of DFT calculations.

## Recommended Active Learning Workflow

1.  **Train an initial model**: Use `collect-data` to create an initial dataset and train your first ACE potential with `pacemaker`.
2.  **Generate Active Set**: Use the `pace_activeset` command-line tool (from the `pacemaker` package) to generate an `.asi` file for your trained potential.
3.  **Explore**: Run large-scale MD simulations (e.g., with LAMMPS) to generate a wide range of new configurations.
4.  **Evaluate & Diagnose**: Use `ace-learn` on the new configurations to identify structures where the potential is uncertain (high `max_gamma`).
5.  **Select**: Use the `--select N` option of `ace-learn` to D-optimally select the `N` most valuable candidates for re-training.
6.  **Re-calculate**: Perform high-accuracy DFT calculations only on this small, selected subset of structures.
7.  **Iterate**: Add the new DFT data to your training set using `collect-data` and re-train your potential. Repeat from step 2.
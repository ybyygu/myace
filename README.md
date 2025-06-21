# MyACE Project Utilities

This project provides a set of standardized tools for collecting and processing data for training Atomic Cluster Expansion (ACE) potentials. It's designed to support a robust active learning and data analysis workflow.

## Core Tools

- **`collect-data`**: A general-purpose tool to parse data from various sources (VASP outputs, `extxyz` directories) and build a standardized `pandas.DataFrame`.
- **`ace-learn`**: A flexible tool to evaluate a set of structures against a trained ACE potential. It can be used to explore unknown structures or to analyze the performance on known training data.

---

## Workflows

This toolkit is designed to support two primary workflows common in potential development.

### Workflow A: Active Learning — Exploring Unknown Structures

**Goal:** To intelligently select the most valuable new structures for expensive DFT calculations from a large pool of unlabeled candidates (e.g., from an MD simulation).

**Conceptual Flow:**
```mermaid
graph TD
    subgraph "Phase 1: Explore & Select"
        A[MD Trajectory<br><i>(many unlabeled structures)</i>] -->|`collect-data --format extxyz`| B(structures_to_eval.pckl.gzip);
        C[current_model.yaml] --> D["ace-learn --select N"];
        B --> D;
        D --> E[selected_for_dft/];
    end
    
    subgraph "Phase 2: Calculate & Collect"
        E --> F["<b>Manual Step:</b><br>Run DFT on structures in the directory"];
        F -->|`collect-data --format extxyz`| G(new_labeled_data.pckl.gzip);
    end

    subgraph "Phase 3: Up-fit & Iterate"
        G --> H["<b>Manual Step:</b><br>Merge with old data &<br>Up-fit the model"];
        H --> C;
    end
```

**Step-by-Step Guide:**

1.  **Prepare Exploration Set**: Use `collect-data` to parse a large number of unlabeled structures (e.g., from a LAMMPS dump or `.xyz` trajectory) into a standardized DataFrame.
    ```bash
    # This creates a DataFrame with 'name' and 'ase_atoms' columns
    rye run collect-data /path/to/md/trajectory/ --format extxyz --output structures_to_eval.pckl.gzip
    ```
2.  **Evaluate & Select**: Use `ace-learn` with your current potential to evaluate these structures and select the `N` most uncertain candidates.
    ```bash
    rye run ace-learn current_potential.yaml structures_to_eval.pckl.gzip --asi current_potential.asi --select 20 --output-selection-dir selected_for_dft
    ```
    This creates a directory `selected_for_dft/` containing 20 `.xyz` files, ready for calculation.

3.  **Perform DFT Calculations**: This is a manual step. Run high-accuracy DFT calculations on the structures inside the `selected_for_dft/` directory. It's recommended to save the results by overwriting the existing `.xyz` files, as `ase.io.write` will embed the new energy and forces.

4.  **Collect New Labeled Data**: Use `collect-data` again, this time to parse the directory containing your newly finished DFT calculations.
    ```bash
    rye run collect-data selected_for_dft/ --format extxyz --ref-energies ref.json --output new_labeled_data.pckl.gzip
    ```
5.  **Up-fit and Iterate**: Merge `new_labeled_data.pckl.gzip` with your main training set and re-train (or up-fit) your potential to create the next-generation model. The cycle then repeats.


### Workflow B: Training Set Analysis — Refining Known Data

**Goal:** To analyze a model's performance on its own training set to identify high-error structures, or to distill a smaller, core set of "support" structures.

**Conceptual Flow:**
```mermaid
graph TD
    A[Full Training Set<br><i>(training_data.pckl.gzip)</i>] --> B(Train Model);
    B --> C(model.yaml);

    A --> D["ace-learn"];
    C --> D;
    
    D --> E[evaluated_training_set.pckl.gzip];
    E --> F["<b>Analysis Step:</b><br>Filter by error/gamma in a script or notebook"];
    F --> G[Distilled/High-Error Set];
```

**Step-by-Step Guide:**

1.  **Inputs**: You need your trained potential (`model.yaml`) and the complete, labeled DataFrame (`training_data.pckl.gzip`) that was used to train it.

2.  **Perform Self-Evaluation**: Run `ace-learn` on the training set itself. Since the input DataFrame contains `energy` and `forces` columns, the output will automatically include error metrics.
    ```bash
    rye run ace-learn model.yaml training_data.pckl.gzip --asi model.asi --output-eval-df evaluated_training_set.pckl.gzip
    ```

3.  **Analyze and Distill**: Load the resulting `evaluated_training_set.pckl.gzip` into a Python script or Jupyter Notebook. You can now easily sort and filter this DataFrame by `max_gamma`, `energy_error_per_atom`, or `forces_rmse` to:
    *   Pinpoint structures that the model struggles to describe.
    *   Select a subset of high-gamma structures that form the "core" of the training set.

---
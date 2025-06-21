# src/myace/active_learning.py
"""
This module contains core, reusable logic for active learning workflows,
including structure evaluation and D-optimal selection.
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional

# --- pyace Core Imports ---
from pyace import PyACECalculator, BBasisConfiguration
from pyace.aceselect import select_structures_maxvol
from pyace.activelearning import load_active_inverse_set, compute_A_active_inverse

def load_ace_calculator(potential_file: str, active_set_file: Optional[str]) -> PyACECalculator:
    """Loads an ACE calculator, optionally with an active set for gamma calculation."""
    if not os.path.exists(potential_file):
        raise FileNotFoundError(f"Potential file not found at {potential_file}.")
    
    calc = PyACECalculator(potential_file)
    
    if active_set_file:
        if os.path.exists(active_set_file):
            calc.set_active_set(active_set_file)
        else:
            import logging
            logging.warning(f"Active set file {active_set_file} not found. Gamma values will not be computed.")
            
    return calc

def evaluate_configs_in_dataframe(df: pd.DataFrame, calc: PyACECalculator) -> pd.DataFrame:
    """
    Evaluates all configurations in a DataFrame using the given ACE calculator.

    This function adds the following columns to the DataFrame:
    - `ace_energy`
    - `ace_forces`
    - `max_gamma` (if an active set is loaded in the calculator)
    - `energy_error_per_atom` (if 'energy' column exists)
    - `forces_rmse` (if 'forces' column exists)

    Args:
        df (pd.DataFrame): DataFrame containing an 'ase_atoms' column with ASE Atoms objects.
        calc (PyACECalculator): The ACE calculator to use for evaluation.

    Returns:
        pd.DataFrame: The input DataFrame with added evaluation columns.
    """
    if 'ase_atoms' not in df.columns:
        raise ValueError("Input DataFrame must contain an 'ase_atoms' column.")

    results = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Configurations"):
        atoms = row['ase_atoms'].copy()
        atoms.calc = calc

        res = {}
        # Core ACE calculations
        res['ace_energy'] = atoms.get_potential_energy()
        res['ace_forces'] = atoms.get_forces()
        
        gamma_values = calc.results.get("gamma")
        res['max_gamma'] = np.max(gamma_values) if gamma_values is not None else np.nan

        # Calculate errors if original DFT data is present
        dft_energy = row.get('energy')
        if dft_energy is not None:
            res['energy_error_per_atom'] = (res['ace_energy'] - dft_energy) / len(atoms)
        
        dft_forces = row.get('forces')
        if dft_forces is not None:
            res['forces_rmse'] = np.sqrt(np.mean((res['ace_forces'] - dft_forces)**2))
            
        results.append(res)
    
    # Append results as new columns to the original DataFrame
    eval_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, eval_df], axis=1)


def select_d_optimal_candidates(
    candidate_df: pd.DataFrame,
    potential_file: str,
    max_to_select: int,
    active_set_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Selects a D-optimal subset from a DataFrame of candidate structures.
    
    Note: This function assumes the input `candidate_df` already contains
    the necessary 'ase_atoms' column.

    Args:
        candidate_df (pd.DataFrame): DataFrame of candidates.
        potential_file (str): Path to the .yaml potential file.
        max_to_select (int): The maximum number of structures to select.
        active_set_file (Optional[str]): Path to an existing .asi file for baseline.

    Returns:
        pd.DataFrame: A DataFrame containing the selected optimal subset.
    """
    if 'ase_atoms' not in candidate_df.columns:
        raise ValueError("Input DataFrame for selection must contain an 'ase_atoms' column.")

    extra_projs = None
    if active_set_file and os.path.exists(active_set_file):
        asi = load_active_inverse_set(active_set_file)
        extra_projs = compute_A_active_inverse(asi)

    bconf = BBasisConfiguration(potential_file)
    
    # The `select_structures_maxvol` tool from pyace expects a column named 'df'
    # but works on the entire DataFrame. We pass our prepared DataFrame directly.
    df_selected = select_structures_maxvol(
        df=candidate_df, # Pass the DataFrame directly
        bconf=bconf,
        extra_A0_projections_dict=extra_projs,
        max_structures=max_to_select
    )

    # The function returns a DataFrame with a selection, we can just return it
    return df_selected
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
import logging

# --- pyace Core Imports ---
from pyace import PyACECalculator, BBasisConfiguration
from pyace.aceselect import select_structures_maxvol
from pyace.activelearning import load_active_inverse_set, compute_A_active_inverse

# Configure logging if not already configured by a higher-level script
# This is a basic configuration.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

def load_ace_calculator(potential_file: str, active_set_file: Optional[str]) -> PyACECalculator:
    """Loads an ACE calculator, optionally with an active set for gamma calculation."""
    if not os.path.exists(potential_file):
        raise FileNotFoundError(f"Potential file not found at {potential_file}.")
    
    calc = PyACECalculator(potential_file)
    
    if active_set_file:
        if os.path.exists(active_set_file):
            calc.set_active_set(active_set_file)
        else:
            logging.warning(f"Active set file {active_set_file} not found. Gamma values will not be computed.")
            
    return calc

def evaluate_configs_in_dataframe(df: pd.DataFrame, calc: PyACECalculator) -> pd.DataFrame:
    """
    Evaluates all configurations in a DataFrame using the given ACE calculator.

    This function adds several evaluation columns, including:
    - `ace_energy`, `ace_forces` (raw ACE outputs)
    - `max_gamma` (if an active set is loaded in the calculator)
    - `dft_energy` (original energy from input DataFrame, if present)
    - `total_energy_error` (ace_energy - dft_energy)
    - `max_dft_force_norm` (max norm of per-atom DFT forces from input)
    - `max_ace_force_norm` (max norm of per-atom ACE forces)
    - `max_delta_force_norm` (max norm of per-atom force differences: ACE - DFT)
    - It also retains `energy_error_per_atom` and `forces_rmse` for programmatic use.

    Args:
        df (pd.DataFrame): DataFrame containing an 'ase_atoms' column with ASE Atoms objects.
                           May also contain 'energy' and 'forces' columns for DFT reference.
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
        ace_forces_raw = atoms.get_forces() 
        res['ace_forces'] = ace_forces_raw # Store raw ACE forces

        gamma_values = calc.results.get("gamma")
        res['max_gamma'] = np.max(gamma_values) if gamma_values is not None and gamma_values.size > 0 else np.nan

        # DFT reference data
        dft_energy = row.get('energy')
        dft_forces_raw = row.get('forces') # Get raw DFT forces from input row
        res['dft_energy'] = dft_energy

        # --- Simplified Force Metric Calculations ---

        # Max norm of DFT forces
        res['max_dft_force_norm'] = np.nan
        if dft_forces_raw is not None:
            try:
                norms = np.linalg.norm(dft_forces_raw, axis=1)
                if norms.size > 0:
                    res['max_dft_force_norm'] = np.max(norms)
            except Exception as e:
                logging.debug(f"Row {index}: Could not calculate max_dft_force_norm. Error: {e}")
        
        # Max norm of ACE forces
        res['max_ace_force_norm'] = np.nan
        if ace_forces_raw is not None:
            try:
                norms = np.linalg.norm(ace_forces_raw, axis=1)
                if norms.size > 0:
                    res['max_ace_force_norm'] = np.max(norms)
            except Exception as e:
                logging.debug(f"Row {index}: Could not calculate max_ace_force_norm. Error: {e}")

        # Total energy error
        res['total_energy_error'] = np.nan
        if dft_energy is not None and res['ace_energy'] is not None:
            res['total_energy_error'] = res['ace_energy'] - dft_energy
        
        # Max norm of (ACE forces - DFT forces)
        res['max_delta_force_norm'] = np.nan
        if dft_forces_raw is not None and ace_forces_raw is not None and getattr(dft_forces_raw, 'shape', None) == getattr(ace_forces_raw, 'shape', None):
            try:
                delta_forces = ace_forces_raw - dft_forces_raw
                norms = np.linalg.norm(delta_forces, axis=1)
                if norms.size > 0:
                    res['max_delta_force_norm'] = np.max(norms)
            except Exception as e:
                logging.debug(f"Row {index}: Could not calculate max_delta_force_norm. Error: {e}")
        
        # Original error metrics (retained for programmatic API compatibility)
        res['energy_error_per_atom'] = np.nan
        num_atoms = len(atoms)
        if dft_energy is not None and res['ace_energy'] is not None and num_atoms > 0:
            res['energy_error_per_atom'] = (res['ace_energy'] - dft_energy) / num_atoms
        
        res['forces_rmse'] = np.nan
        if dft_forces_raw is not None and ace_forces_raw is not None and getattr(dft_forces_raw, 'shape', None) == getattr(ace_forces_raw, 'shape', None):
            try:
                res['forces_rmse'] = np.sqrt(np.mean((ace_forces_raw - dft_forces_raw)**2))
            except Exception as e:
                logging.debug(f"Row {index}: Could not calculate forces_rmse. Error: {e}")
            
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


def evaluate_and_select(
    df: pd.DataFrame,
    potential_file: str,
    asi_file: Optional[str] = None,
    select_n: int = 0,
    gamma_threshold: float = 5.0
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    High-level API to evaluate a DataFrame of configurations and optionally
    select a high-uncertainty subset.

    Args:
        df (pd.DataFrame): The input DataFrame with an 'ase_atoms' column.
        potential_file (str): Path to the ACE potential .yaml file.
        asi_file (Optional[str], optional): Path to the .asi file for gamma calculations.
        select_n (int, optional): The number of candidates to select. If 0, no selection is performed.
        gamma_threshold (float, optional): The gamma value threshold for defining high-uncertainty.

    Returns:
        tuple[pd.DataFrame, Optional[pd.DataFrame]]:
            - The DataFrame with all evaluation results.
            - The DataFrame with only the selected candidates, or None if no selection was performed.
    """
    calculator = load_ace_calculator(potential_file, asi_file)
    evaluated_df = evaluate_configs_in_dataframe(df, calculator)
    
    selected_df = None
    if select_n > 0:
        if not asi_file: 
             logging.warning("An 'asi_file' is typically required for meaningful D-optimal selection based on gamma. Proceeding without it may not yield optimal results if gamma is implicitly used by selection logic.")

        logging.info(f"\nPerforming D-optimal selection for {select_n} candidates...")
        
        candidate_pool_df = evaluated_df.copy() # Start with all evaluated structures
        if asi_file and 'max_gamma' in evaluated_df.columns and not evaluated_df['max_gamma'].isnull().all():
            # If gamma is available, use it for pre-filtering
            gamma_filtered_pool = evaluated_df[evaluated_df['max_gamma'] > gamma_threshold].copy()
            if not gamma_filtered_pool.empty:
                candidate_pool_df = gamma_filtered_pool
            else:
                logging.warning(f"No structures found above gamma threshold > {gamma_threshold}. D-optimal selection will proceed on all structures if any.")
        elif asi_file: # asi_file provided but max_gamma not usable
             logging.warning("asi_file provided, but 'max_gamma' column is not available or all NaN. D-optimal selection will proceed on all input structures.")
        else: # No asi_file
            logging.warning("No asi_file provided for gamma calculation. D-optimal selection will proceed on all input. Consider providing --asi for gamma-based pre-filtering.")


        if candidate_pool_df.empty:
            logging.warning(f"Candidate pool for D-optimal selection is empty. No selection performed.")
            selected_df = pd.DataFrame() 
        elif len(candidate_pool_df) < select_n:
            logging.warning(f"Candidate pool size ({len(candidate_pool_df)}) is less than requested N ({select_n}). Selecting all from pool.")
            selected_df = select_d_optimal_candidates(
                candidate_df=candidate_pool_df,
                potential_file=potential_file,
                max_to_select=len(candidate_pool_df), # Select all from the pool
                active_set_file=asi_file
            )
        else:
            selected_df = select_d_optimal_candidates(
                candidate_df=candidate_pool_df,
                potential_file=potential_file,
                max_to_select=select_n,
                active_set_file=asi_file
            )
            
    return evaluated_df, selected_df
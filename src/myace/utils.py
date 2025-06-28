# src/myace/utils.py
"""
This module contains general-purpose utility functions that can be reused
across different parts of the myace workflow, particularly for analysis.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd


def get_max_force_component(forces: np.ndarray) -> float:
    """
    Calculates the maximum absolute force component for a given ASE Atoms object.
    """
    return np.max(np.abs(forces))


def get_max_force_norm(forces: np.ndarray) -> float:
    """
    Calculates the maximum force norm from an array of force vectors.
    """
    # Calculate the L2 norm (Euclidean distance) for each force vector (axis=1)
    # and then find the maximum value in the resulting 1D array of norms.
    return np.linalg.norm(forces, axis=1).max()


def sample_by_energy(
    df: pd.DataFrame,
    frac: float = 0.4,
    energy_col: str = 'energy',
    energy_scale: float = 0.1,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Performs weighted random sampling based on energy to prioritize high-energy configurations.

    This function uses an "inverse Boltzmann" weighting scheme, where structures
    with higher energy are more likely to be selected. This is an effective
    method for de-correlating samples from a molecular dynamics trajectory,
    reducing redundancy from configurations around local minima.

    Returns:
        pd.DataFrame: A new DataFrame containing the energy-weighted sample.
    """
    if not 0 < frac <= 1:
        raise ValueError("The 'frac' parameter must be between 0 and 1.")

    if energy_col not in df.columns:
        raise ValueError(f"The specified energy column '{energy_col}' does not exist in the DataFrame.")

    num_samples_to_select = int(len(df) * frac)
    if num_samples_to_select == 0 and len(df) > 0:
         logging.warning(f"Calculated number of samples is 0 (frac={frac}, total={len(df)}). Returning an empty DataFrame.")
         return pd.DataFrame()

    energies = df[energy_col]
    e_min = energies.min()

    # Calculate weights, handling numerical stability
    weights = np.exp((energies - e_min) / energy_scale)

    # If all weights are zero (e.g., due to large energy gaps relative to scale),
    # fall back to uniform sampling.
    if weights.sum() == 0:
        logging.warning("All calculated weights are zero; falling back to uniform random sampling.")
        weights = None

    sampled_df = df.sample(
        n=num_samples_to_select,
        weights=weights,
        replace=False,
        random_state=random_state
    )

    return sampled_df.sort_index()

# src/myace/utils.py
"""
This module contains general-purpose utility functions that can be reused
across different parts of the myace workflow, particularly for analysis.
"""

import numpy as np

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

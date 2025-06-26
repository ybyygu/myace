# src/ace/data_processing.py
"""
This module contains reusable, core logic for parsing data from various
computational chemistry codes and building pacemaker-compatible DataFrames.
"""
import pandas as pd
import numpy as np
import os
from ase.io import read
from ase import Atoms
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

def build_pacemaker_dataframe(parsed_data_list, ref_energies):
    """
    Builds a pacemaker-compatible DataFrame from a standardized list of dicts.
    """
    if not parsed_data_list:
        logging.warning("Parsed data list is empty. Returning an empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(parsed_data_list)
    
    # This core energy correction logic is now also available via add_corrected_energies
    if 'energy' in df.columns and not df['energy'].isnull().all():
        # Ensure ref_energies is a dictionary, defaulting to empty if None
        current_ref_energies = ref_energies if isinstance(ref_energies, dict) else {}

        def calculate_atomic_ref_energy(atoms_obj):
            if not hasattr(atoms_obj, 'get_chemical_symbols'):
                return 0
            counts = Counter(atoms_obj.get_chemical_symbols())
            return sum(counts[el] * current_ref_energies.get(el, 0) for el in counts)
            
        df['energy_corrected'] = df['energy'] - df['ase_atoms'].apply(calculate_atomic_ref_energy)
        df['energy_corrected_per_atom'] = df['energy_corrected'] / df['ase_atoms'].apply(len)

    final_columns = ['name', 'ase_atoms']
    optional_cols = ['energy', 'forces', 'stress', 'energy_corrected', 'energy_corrected_per_atom']
    for col in optional_cols:
        if col in df.columns:
            final_columns.append(col)
            
    return df[[col for col in final_columns if col in df.columns]]


def add_corrected_energies(df: pd.DataFrame, ref_energies: dict) -> pd.DataFrame:
    """
    Adds 'energy_corrected' and 'energy_corrected_per_atom' columns to a DataFrame
    if 'energy' and 'ase_atoms' columns are present and valid. This is useful for
    DataFrames read directly (e.g., via myace.io.read_gosh_parquet) that need
    energy correction before use with pacemaker or other analysis.

    Args:
        df (pd.DataFrame): Input DataFrame, must contain 'ase_atoms' and 'energy'.
        ref_energies (dict): Dictionary of reference energies per element. 
                             Example: {'Si': -100.0, 'O': -50.0}

    Returns:
        pd.DataFrame: DataFrame with added energy_corrected columns, or a copy of 
                      the original if prerequisites are not met or an error occurs.
    """
    if not isinstance(df, pd.DataFrame):
        logging.error("Input 'df' is not a pandas DataFrame. Cannot calculate corrected energies.")
        return df # Or raise TypeError

    df_copy = df.copy() # Work on a copy

    if 'energy' not in df_copy.columns or 'ase_atoms' not in df_copy.columns:
        logging.warning("DataFrame is missing 'energy' or 'ase_atoms' column. Cannot calculate corrected energies.")
        return df_copy

    if df_copy['energy'].isnull().all():
        logging.warning("'energy' column contains all NaN or None values. Cannot calculate corrected energies.")
        return df_copy

    if not isinstance(ref_energies, dict):
        logging.error("'ref_energies' must be a dictionary. Cannot calculate corrected energies.")
        return df_copy

    def calculate_atomic_ref_energy(atoms_obj):
        # Ensure atoms_obj is an ASE Atoms object or has get_chemical_symbols
        if not isinstance(atoms_obj, Atoms) or not hasattr(atoms_obj, 'get_chemical_symbols'):
            logging.warning(f"Item in 'ase_atoms' is not a valid ASE Atoms object. Skipping energy correction for this row.")
            return np.nan # Return NaN so this row's corrected energy becomes NaN
        
        counts = Counter(atoms_obj.get_chemical_symbols())
        current_ref_val = 0
        for el, count in counts.items():
            if el not in ref_energies:
                logging.warning(f"Element '{el}' not found in provided ref_energies. Treating its reference energy as 0 for this row.")
            current_ref_val += count * ref_energies.get(el, 0)
        return current_ref_val

    try:
        ref_energy_values = df_copy['ase_atoms'].apply(calculate_atomic_ref_energy)
        
        # Ensure 'energy' column is numeric, coercing errors to NaN
        energies_numeric = pd.to_numeric(df_copy['energy'], errors='coerce')
        
        df_copy['energy_corrected'] = energies_numeric - ref_energy_values
        
        atom_counts = df_copy['ase_atoms'].apply(lambda x: len(x) if isinstance(x, Atoms) and hasattr(x, '__len__') else 0)
        # Avoid division by zero by replacing 0 counts with NaN for division
        df_copy['energy_corrected_per_atom'] = df_copy['energy_corrected'].divide(atom_counts.replace(0, np.nan))

    except Exception as e:
        logging.error(f"An error occurred during energy correction calculation: {e}", exc_info=True)
        # Return the original df's copy if a major error occurs, it might not have the new columns or they might be partial/NaN
        return df.copy() 
        
    return df_copy


def parse_vasp_outcar(input_path: str, selection: str = ":", name_prefix: str = "") -> list[dict]:
    """
    A specific parser for VASP OUTCAR/vasprun.xml files.
    This parser expects complete DFT data and will raise errors if not found.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    configurations = read(input_path, index=selection)
    if not isinstance(configurations, list):
        configurations = [configurations]

    parsed_data = []
    for i, config in enumerate(configurations):
        index = getattr(config, 'index', i)
        # For VASP, we expect data to be complete. Let it fail fast if keys are missing from ASE's results.
        parsed_data.append({
            'name': f"{name_prefix}{os.path.basename(input_path)}##{index}",
            'energy': config.get_potential_energy(force_consistent=True),
            'forces': config.get_forces(),
            'stress': config.get_stress(voigt=False),
            'ase_atoms': config
        })
    logging.info(f"Successfully parsed {len(parsed_data)} configurations from '{input_path}'.")
    return parsed_data


def parse_extxyz_directory(input_path: str, selection: str = ":", name_prefix: str = "") -> list[dict]:
    """
    Parses all .xyz files in a directory. It assumes files are complete with energy/forces
    unless they are explicitly missing, in which case ASE would raise an error.
    """
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input directory not found: {input_path}")
        
    xyz_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.xyz')])
    
    file_slice = slice(*[int(p) if p else None for p in selection.split(':')])
    selected_files = xyz_files[file_slice]

    parsed_data = []
    for f_path in selected_files:
        config = read(f_path, format='extxyz')
        # Let this fail if energy/forces are expected but not present in the extxyz file.
        # The user can create a file with None/empty fields if that is the desired behavior.
        parsed_data.append({
            'name': f"{name_prefix}{os.path.basename(f_path)}",
            'energy': config.get_potential_energy(),
            'forces': config.get_forces(),
            'stress': config.get_stress(voigt=False),
            'ase_atoms': config,
            **config.info
        })
    logging.info(f"Successfully parsed {len(parsed_data)} configurations from '{input_path}'.")
    return parsed_data

def parse_lammps_dump(input_path: str, selection: str = ":", name_prefix: str = "", lammps_map: str = None) -> list[dict]:
    """
    A specific parser for LAMMPS dump files (text format).
    LAMMPS dumps typically only contain structural info, so energy/forces are set to None.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    symbol_map = {}
    if lammps_map:
        try:
            for item in lammps_map.split(','):
                key, value = item.split(':')
                symbol_map[int(key)] = value
        except ValueError:
            raise ValueError("Invalid format for --lammps-map. Expected format like '1:H,2:O'.")

    # Read without symbol mapping first. ASE will use atom types as atomic numbers.
    configurations = read(input_path, index=selection, format="lammps-dump-text")
    if not isinstance(configurations, list):
        configurations = [configurations]

    # Post-process configurations to apply the correct chemical symbols
    if symbol_map:
        for config in configurations:
            atom_types = config.get_atomic_numbers()
            try:
                new_symbols = [symbol_map[t] for t in atom_types]
                config.symbols = new_symbols  # Manually set the correct symbols
            except KeyError as e:
                raise KeyError(f"Atom type {e} found in dump file but not defined in --lammps-map.")

    parsed_data = []
    for i, config in enumerate(configurations):
        parsed_data.append({
            'name': f"{name_prefix}{os.path.basename(input_path)}##{i}",
            'energy': None,
            'forces': None,
            'stress': None,
            'ase_atoms': config
        })
    logging.info(f"Successfully parsed {len(parsed_data)} configurations from '{input_path}'.")
    return parsed_data

def build_dataset(input_path: str, format: str, selection: str = ":", name_prefix: str = "", lammps_map: str = None, ref_energies: dict = None) -> pd.DataFrame:
    """
    High-level API to build a dataset from a given source.
    """
    if ref_energies is None:
        ref_energies = {}
    
    if format == 'vasp':
        parsed_data = parse_vasp_outcar(input_path, selection, name_prefix=name_prefix)
    elif format == 'extxyz':
        parsed_data = parse_extxyz_directory(input_path, selection, name_prefix=name_prefix)
    elif format == 'lammps-dump':
        parsed_data = parse_lammps_dump(input_path, selection, name_prefix=name_prefix, lammps_map=lammps_map)
    else:
        raise ValueError(f"Unsupported format: '{format}'. Supported formats are 'vasp', 'extxyz', 'lammps-dump'.")
        
    return build_pacemaker_dataframe(parsed_data, ref_energies)
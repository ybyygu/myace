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
    
    # Delegate energy correction to the single source of truth function
    df = add_corrected_energies(df, ref_energies)

    final_columns = ['name', 'ase_atoms']
    optional_cols = ['energy', 'forces', 'stress', 'energy_corrected', 'energy_corrected_per_atom']
    for col in optional_cols:
        if col in df.columns:
            final_columns.append(col)
            
    return df[[col for col in final_columns if col in df.columns]]


def add_corrected_energies(df: pd.DataFrame, ref_energies: dict) -> pd.DataFrame:
    """
    Adds 'energy_corrected' and 'energy_corrected_per_atom' columns to a DataFrame.

    This is a streamlined version that expects a DataFrame with a valid 'ase_atoms'
    column. It will add corrected energies if an 'energy' column is present.
    It follows a fast-fail philosophy for incorrect inputs.

    Args:
        df (pd.DataFrame): Input DataFrame.
        ref_energies (dict): Dictionary of reference energies per element.

    Returns:
        pd.DataFrame: A new DataFrame with added energy_corrected columns if applicable.
    """
    if 'energy' not in df.columns or df['energy'].isnull().all():
        # If there's no energy data, there's nothing to correct.
        return df

    df_copy = df.copy()

    def calculate_atomic_ref_energy(atoms_obj):
        counts = Counter(atoms_obj.get_chemical_symbols())
        return sum(counts[el] * ref_energies.get(el, 0) for el in counts)
        
    ref_energy_values = df_copy['ase_atoms'].apply(calculate_atomic_ref_energy)
    df_copy['energy_corrected'] = df_copy['energy'] - ref_energy_values
    df_copy['energy_corrected_per_atom'] = df_copy['energy_corrected'] / df_copy['ase_atoms'].apply(len)
    
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
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

def build_pacemaker_dataframe(parsed_data_list, ref_energies):
    """
    Builds a pacemaker-compatible DataFrame from a standardized list of dicts.
    This is the core, reusable builder function that enforces the data format.

    Args:
        parsed_data_list (list[dict]): Data parsed from a source file. Each dict
            must contain 'name', 'ase_atoms', 'energy', and 'forces'.
        ref_energies (dict): Dictionary of reference energies for each element.

    Returns:
        pd.DataFrame: A DataFrame compatible with pacemaker.
    """
    if not parsed_data_list:
        logging.warning("Parsed data list is empty. Returning an empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(parsed_data_list)
    
    def calculate_ref_energy(atoms_obj):
        counts = Counter(atoms_obj.get_chemical_symbols())
        energy = sum(counts[el] * ref_energies.get(el, 0) for el in counts)
        # If an element is not in ref_energies, .get(el, 0) handles it gracefully.
        # No need to raise an error.
        return energy
        
    df['energy_corrected'] = df['energy'] - df['ase_atoms'].apply(calculate_ref_energy)
    df['energy_corrected_per_atom'] = df['energy_corrected'] / df['ase_atoms'].apply(len)

    final_columns = [
        'name', 'energy', 'forces', 'ase_atoms', 
        'energy_corrected', 'energy_corrected_per_atom'
    ]
    if 'stress' in df.columns:
        final_columns.insert(4, 'stress')
        
    return df[[col for col in final_columns if col in df.columns]]


def parse_vasp_outcar(input_path, selection=":") -> list[dict]:
    """
    A specific parser for VASP OUTCAR/vasprun.xml files.

    Args:
        input_path (str): Path to the OUTCAR or vasprun.xml file.
        selection (str): Slice string (e.g., ':', '::10', -1) to select steps.

    Returns:
        list[dict]: A list of dictionaries, each conforming to the standard format.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    configurations = read(input_path, index=selection)
    if not isinstance(configurations, list):
        configurations = [configurations]

    parsed_data = []
    for i, config in enumerate(configurations):
        index = getattr(config, 'index', i)
        parsed_data.append({
            'name': f"{os.path.basename(input_path)}##{index}",
            'energy': config.get_potential_energy(force_consistent=True),
            'forces': config.get_forces(),
            'stress': config.get_stress(voigt=False),
            'ase_atoms': Atoms(
                symbols=config.get_chemical_symbols(),
                positions=config.get_positions(),
                cell=config.get_cell(),
                pbc=config.pbc
            ),
        })
    logging.info(f"Successfully parsed {len(parsed_data)} configurations from '{input_path}'.")
    return parsed_data


def parse_extxyz_directory(input_path, selection=":") -> list[dict]:
    """
    Parses all .xyz files in a directory. Assumes a format where each file
    is a single configuration and metadata is stored in the .info dictionary.

    Args:
        input_path (str): Path to the directory containing .xyz files.
        selection (str): Slice string, applied to the list of found files.

    Returns:
        list[dict]: A list of dictionaries, each conforming to the standard format.
    """
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input directory not found: {input_path}")
        
    xyz_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.xyz')])
    
    # Apply slicing to the list of files
    file_slice = slice(*[int(p) if p else None for p in selection.split(':')])
    selected_files = xyz_files[file_slice]

    parsed_data = []
    for f_path in selected_files:
        config = read(f_path, format='extxyz')
        
        # ASE's extxyz reader conveniently places metadata in the .info dictionary
        # and energy/forces on the Atoms object itself.
        parsed_data.append({
            'name': os.path.basename(f_path),
            'energy': config.get_potential_energy(),
            'forces': config.get_forces(),
            'stress': config.get_stress(voigt=False) if 'stress' in config.info else None,
            'ase_atoms': config,
            **config.info  # Add all other metadata from the .xyz file
        })
        
    logging.info(f"Successfully parsed {len(parsed_data)} configurations from '{input_path}'.")
    return parsed_data
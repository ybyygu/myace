# myace/io.py
"""
Convenience functions for reading and writing pandas DataFrames
in the compressed pickle format used by myace tools.
"""
import pandas as pd
from typing import Union
from pathlib import Path
import os
from ase.io import write as ase_write
import numpy as np
from ase import Atoms
import logging


def read(path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a pandas DataFrame from a .pckl.gzip file.
    """
    return pd.read_pickle(path, compression='gzip')

def write(df: pd.DataFrame, path: Union[str, Path]):
    """
    Saves a pandas DataFrame to a .pckl.gzip file.
    """
    df.to_pickle(path, compression='gzip', protocol=4)

def merge(df_new: pd.DataFrame, df_old: pd.DataFrame, on_column: str = 'name', keep: str = 'last') -> pd.DataFrame:
    """
    Merges two DataFrames, handling duplicates.
    """
    combined_df = pd.concat([df_old, df_new], ignore_index=True)
    cleaned_df = combined_df.drop_duplicates(subset=[on_column], keep=keep)
    return cleaned_df.reset_index(drop=True)

def export_to_extxyz(df: pd.DataFrame, output_dir: Union[str, Path]):
    """
    Exports each configuration in a DataFrame to a separate .xyz file
    in a specified directory, embedding metadata.
    """
    if 'ase_atoms' not in df.columns:
        raise ValueError("Input DataFrame must contain an 'ase_atoms' column.")

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for index, row in df.iterrows():
        atoms = row['ase_atoms'].copy()
        
        # Definitive fix for 'NoneType' errors: Sanitize both the .info and
        # .arrays dictionaries of the Atoms object before writing. This removes
        # any keys that have a None value, which ASE's writer cannot handle.
        for key, value in list(atoms.info.items()):
            if value is None:
                del atoms.info[key]
        for key, value in list(atoms.arrays.items()):
            if value is None:
                del atoms.arrays[key]

        # Prepare metadata to be saved in the xyz file, explicitly excluding None values
        info_dict = {
            key: val for key, val in row.items()
            if key != 'ase_atoms' and pd.api.types.is_scalar(val) and val is not None
        }
        atoms.info.update(info_dict)

        # Safely set energy, forces, and stress on the Atoms object for extxyz
        if 'energy' in row and row.get('energy') is not None:
            atoms.info['energy'] = row['energy']
        if 'forces' in row and row.get('forces') is not None:
            atoms.arrays['forces'] = row['forces']
        if 'stress' in row and row.get('stress') is not None:
            atoms.info['stress'] = row['stress']

        safe_filename = str(index).replace('#', '_').replace('/', '_')
        filename = output_dir / f"{safe_filename}.xyz"
        ase_write(filename, atoms, format='extxyz')

def read_gosh_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a parquet file from the 'gosh-adaptor' and converts it into a
    myace-standard DataFrame.

    The standard format has an 'ase_atoms' column and other metadata,
    which is different from the input format where structure is spread
    across columns like 'symbols', 'positions', 'lattice'.
    
    This version robustly handles nested numpy arrays with dtype=object
    by stacking them into clean, multi-dimensional float arrays.

    Args:
        path (Union[str, Path]): Path to the .parquet file.

    Returns:
        pd.DataFrame: A myace-standard DataFrame.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    gosh_df = pd.read_parquet(path)

    required_cols = {'symbols', 'positions', 'lattice', 'energy', 'forces', 'stress'}
    if not required_cols.issubset(gosh_df.columns):
        missing = required_cols - set(gosh_df.columns)
        raise ValueError(f"Input parquet file is missing required columns: {missing}")

    processed_records = []
    for i, row in gosh_df.iterrows():
        try:
            # --- Final, Correct Data Sanitization ---
            # The data is a numpy array of other numpy arrays (dtype=object).
            # We use np.vstack to stack them into a single, clean float array.
            cell = np.vstack(row['lattice'])
            positions = np.vstack(row['positions'])
            forces = np.vstack(row['forces'])
            stress = np.array(row['stress'])

            # Add shape checks for safety
            if cell.shape != (3, 3):
                raise ValueError(f"Stacked cell has shape {cell.shape}, expected (3, 3).")
            if len(positions.shape) != 2 or positions.shape[1] != 3:
                raise ValueError(f"Stacked positions has shape {positions.shape}, expected (N, 3).")

            # Construct the Atoms object with clean data
            atoms = Atoms(
                symbols=row['symbols'],
                positions=positions,
                cell=cell,
                pbc=True
            )
            
            # Create the record in the myace-standard format
            record = {
                'name': f"{Path(path).stem}##{i}",
                'ase_atoms': atoms,
                'energy': row['energy'],
                'forces': forces,
                'stress': stress
            }
            processed_records.append(record)
        except Exception as e:
            raise ValueError(f"Failed to process data at index {i} in file {path}. Original error: {e}") from e

    # Create the final DataFrame
    final_df = pd.DataFrame(processed_records)
    
    return final_df

def export_to_vasp(df: pd.DataFrame, output_dir: Union[str, Path]):
    """
    Writes each structure from a DataFrame into a separate .vasp file (POSCAR format),
    named after the DataFrame's index.

    Args:
        df (pd.DataFrame): DataFrame containing an 'ase_atoms' column.
        output_dir (Union[str, Path]): Directory where all .vasp files will be saved.
    """
    if 'ase_atoms' not in df.columns:
        raise ValueError("Input DataFrame must contain an 'ase_atoms' column.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"All .vasp files will be saved in: '{output_dir}'")

    for index, row in df.iterrows():
        atoms = row['ase_atoms']
        file_path = output_dir / f"{index}.vasp"
        ase_write(file_path, atoms, format='vasp', sort=False)
            
    logging.info(f"Process complete. {len(df)} structures written to individual .vasp files in '{output_dir}'.")
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
# myace/io.py
"""
Convenience functions for reading and writing pandas DataFrames
in the compressed pickle format used by myace tools.
"""
import pandas as pd
from typing import Union
from pathlib import Path

def read(path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a pandas DataFrame from a .pckl.gzip file.

    This is a convenience wrapper around pandas.read_pickle
    with gzip compression enabled.

    Args:
        path (str or Path): The path to the .pckl.gzip file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_pickle(path, compression='gzip')

def write(df: pd.DataFrame, path: Union[str, Path]):
    """
    Saves a pandas DataFrame to a .pckl.gzip file.

    This is a convenience wrapper around pandas.DataFrame.to_pickle
    with gzip compression and protocol 4 enabled.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str or Path): The path to the output .pckl.gzip file.
    """
    df.to_pickle(path, compression='gzip', protocol=4)


def export_to_extxyz(df: pd.DataFrame, output_dir: Union[str, Path]):
    """
    Exports each configuration in a DataFrame to a separate .xyz file
    in a specified directory, embedding metadata.

    Args:
        df (pd.DataFrame): The DataFrame to export. Must contain an 'ase_atoms' column.
        output_dir (str or Path): The directory to save the .xyz files in.
    """
    import os
    from ase.io import write as ase_write

    if 'ase_atoms' not in df.columns:
        raise ValueError("Input DataFrame must contain an 'ase_atoms' column.")

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for index, row in df.iterrows():
        atoms = row['ase_atoms'].copy()
        
        # Prepare metadata to be saved in the xyz file
        info_dict = {
            key: val for key, val in row.items()
            if key != 'ase_atoms' and pd.api.types.is_scalar(val)
        }
        atoms.info.update(info_dict)

        # Set energy and forces on the Atoms object for extxyz standard
        if 'energy' in row and row['energy'] is not None:
            atoms.info['energy'] = row['energy']
        if 'forces' in row and row['forces'] is not None:
            atoms.arrays['forces'] = row['forces']
        if 'stress' in row and row['stress'] is not None:
            # ASE handles stress via .info for extxyz
            atoms.info['stress'] = row['stress']

        safe_filename = str(index).replace('#', '_').replace('/', '_')
        filename = output_dir / f"{safe_filename}.xyz"
        ase_write(filename, atoms, format='extxyz')
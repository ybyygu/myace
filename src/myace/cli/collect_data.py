#!/usr/bin/env python3
"""
Command-line tool to parse raw data from various sources and build a
pacemaker-compatible DataFrame. This is a thin wrapper around the
`myace.data_processing.build_dataset` API function.
"""
import argparse
import logging
import json
import sys

from ..data_processing import build_dataset
from ..io import write as write_df

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

def main():
    """Main function for the command-line tool."""
    parser = argparse.ArgumentParser(
        description="Collect and process training data for ACE potentials.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the source data (e.g., VASP OUTCAR, a directory of .xyz files, or a LAMMPS dump file)."
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="vasp",
        choices=['vasp', 'extxyz', 'lammps-dump'],
        help="Format of the input: 'vasp' (OUTCAR/vasprun.xml), 'extxyz' (directory of .xyz files), 'lammps-dump' (text dump file)."
    )

    parser.add_argument(
        "--name-prefix",
        type=str,
        default="",
        help="A string to prepend to each configuration name to ensure uniqueness across different datasets."
    )
    
    parser.add_argument(
        "--lammps-map",
        type=str,
        default=None,
        help="Element map for LAMMPS dumps, required if format is 'lammps-dump'. E.g., '1:H,2:O'."
    )

    parser.add_argument(
        "--ref-energies",
        type=str,
        default=None,
        help="Optional path to a JSON file containing reference energies for each element. "
             "If not provided, all reference energies will be treated as 0."
    )
    
    parser.add_argument(
        "--selection",
        type=str,
        default=":",
        help="Slice string to select structures (e.g., '::10', ':100', '-1'). "
             "Applied to steps in file for 'vasp' and 'lammps-dump', or to the file list for 'extxyz'."
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="collected_data.pckl.gzip",
        help="Path to the output pickle file. Default is 'collected_data.pckl.gzip'."
    )
    
    args = parser.parse_args()

    # Validate arguments
    if args.format == 'lammps-dump' and not args.lammps_map:
        parser.error("--lammps-map is required when using --format 'lammps-dump'.")

    try:
        ref_energies_dict = {}
        if args.ref_energies:
            with open(args.ref_energies, 'r') as f:
                ref_energies_dict = json.load(f)
        
        logging.info(f"Building dataset from '{args.input_path}' with format '{args.format}'...")
        df = build_dataset(
            input_path=args.input_path,
            format=args.format,
            selection=args.selection,
            name_prefix=args.name_prefix,
            lammps_map=args.lammps_map,
            ref_energies=ref_energies_dict
        )

        if df.empty:
            logging.warning("No data was parsed or processed. Exiting without creating an output file.")
            return
        
        write_df(df, args.output)
        logging.info(f"Successfully saved {len(df)} configurations to '{args.output}'.")
        print("\n--- Output file summary ---")
        df.info(verbose=False)

    except (ValueError, FileNotFoundError, KeyError) as e:
        # Catch specific, expected errors and report them cleanly.
        logging.error(f"A critical error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors.
        logging.error(f"An unexpected programming error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
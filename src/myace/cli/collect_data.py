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

# We are now inside the 'myace' package, so we can use relative imports
from ..data_processing import build_dataset
from ..io import write as write_df

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

def main():
    """Main function for the command-line tool."""
    parser = argparse.ArgumentParser(description="Collect and process training data for ACE potentials.")
    
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the source data (e.g., VASP OUTCAR, or a directory of .xyz files)."
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="vasp",
        choices=['vasp', 'extxyz'],
        help="The format of the input source. 'vasp' for OUTCAR/vasprun.xml, 'extxyz' for a directory of .xyz files."
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
             "For 'vasp', it's applied to steps in the file. "
             "For 'extxyz', it's applied to the sorted list of files in the directory. Default is all (':')."
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="collected_data.pckl.gzip",
        help="Path to the output pickle file. Default is 'collected_data.pckl.gzip'."
    )
    
    args = parser.parse_args()

    # --- 1. Load reference energies from JSON if provided ---
    ref_energies_dict = {}
    if args.ref_energies:
        try:
            with open(args.ref_energies, 'r') as f:
                ref_energies_dict = json.load(f)
            logging.info(f"Successfully loaded reference energies from '{args.ref_energies}'.")
        except FileNotFoundError:
            logging.error(f"Error: Reference energy file not found at '{args.ref_energies}'.")
            sys.exit(1)
        except json.JSONDecodeError:
            logging.error(f"Error: Could not parse the JSON file at '{args.ref_energies}'. Please ensure it is valid JSON.")
            sys.exit(1)
    
    # --- 2. Call the high-level API to build the dataset ---
    try:
        logging.info(f"Building dataset from '{args.input_path}' with format '{args.format}'...")
        df = build_dataset(
            input_path=args.input_path,
            format=args.format,
            selection=args.selection,
            ref_energies=ref_energies_dict
        )
    except (ValueError, FileNotFoundError) as e:
        logging.error(e)
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during dataset creation: {e}")
        sys.exit(1)

    # --- 3. Save the resulting DataFrame ---
    if df.empty:
        logging.warning("No data was parsed or processed. Exiting without creating an output file.")
        return
        
    try:
        write_df(df, args.output)
        logging.info(f"Successfully saved {len(df)} configurations to '{args.output}'.")
        print("\n--- Output file summary ---")
        df.info(verbose=False)
    except Exception as e:
        logging.error(f"An unexpected error occurred while saving the file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
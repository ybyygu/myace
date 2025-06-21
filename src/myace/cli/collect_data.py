#!/usr/bin/env python3
"""
Command-line tool to parse raw data from computational chemistry codes
or other formats and build a pacemaker-compatible DataFrame.
"""
import argparse
import logging
import json

# We are now inside the 'myace' package, so we can use relative imports
from ..data_processing import (
    parse_vasp_outcar, 
    parse_extxyz_directory, 
    build_pacemaker_dataframe
)

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

    # --- 1. Load reference energies from JSON ---
    ref_energies_dict = {}
    if args.ref_energies:
        try:
            with open(args.ref_energies, 'r') as f:
                ref_energies_dict = json.load(f)
            logging.info(f"Successfully loaded reference energies from '{args.ref_energies}'.")
        except FileNotFoundError:
            logging.error(f"Error: Reference energy file not found at '{args.ref_energies}'.")
            return
        except json.JSONDecodeError:
            logging.error(f"Error: Could not parse the JSON file at '{args.ref_energies}'. Please ensure it is valid JSON.")
            return
    else:
        logging.info("No reference energy file provided. All elemental reference energies will be treated as 0.")

    # --- 2. Parse source data using the appropriate parser ---
    try:
        if args.format == 'vasp':
            parsed_data = parse_vasp_outcar(args.input_path, args.selection)
        elif args.format == 'extxyz':
            parsed_data = parse_extxyz_directory(args.input_path, args.selection)
        else:
            # This case should not be reached due to 'choices' in argparse
            logging.error(f"Unknown format: {args.format}")
            return
    except FileNotFoundError as e:
        logging.error(e)
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during parsing: {e}")
        return

    # --- 3. Build and save the DataFrame ---
    if not parsed_data:
        logging.warning("No data was parsed. Exiting without creating an output file.")
        return
        
    try:
        df = build_pacemaker_dataframe(parsed_data, ref_energies_dict)
        df.to_pickle(args.output, compression='gzip', protocol=4)
        logging.info(f"Successfully saved {len(df)} configurations to '{args.output}'.")
        print("\n--- Output file summary ---")
        df.info()
    except ValueError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"An unexpected error occurred during DataFrame creation: {e}")

if __name__ == "__main__":
    main()
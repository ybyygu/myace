#!/usr/bin/env python3
"""
Command-line tool to evaluate structures from a DataFrame against an ACE potential,
and optionally select a D-optimal subset of high-uncertainty candidates.
"""
import argparse
import pandas as pd
import logging
import sys
import os
from ase.io import write as ase_write # To avoid conflict with built-in write

# Use relative imports, as the package is installed
from myace.active_learning import (
    load_ace_calculator,
    evaluate_configs_in_dataframe,
    select_d_optimal_candidates
)

def main():
    """Main entry point for the Active Learning evaluation and selection tool."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(
        description="ACE Active Learning Helper: Evaluate a DataFrame and select optimal candidates.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- I/O Arguments ---
    parser.add_argument("potential", help="Path to the ACE potential .yaml file.")
    parser.add_argument("input_df", help="Path to the input DataFrame (.pckl.gzip) containing configurations to evaluate.")
    parser.add_argument("--output-eval-df", type=str, default="evaluated_structures.pckl.gzip",
                        help="Output path for the DataFrame with all evaluation results.")

    # --- ACE Calculation Arguments ---
    parser.add_argument("--asi", help="Path to the Active Set (.asi) file, required for gamma value calculation.", default=None)

    # --- Selection Arguments ---
    parser.add_argument("--select", type=int, metavar='N',
                        help="If specified, select N optimal candidates from high-uncertainty structures.")
    parser.add_argument("--gamma-threshold", type=float, default=5.0,
                        help="Gamma threshold to identify high-uncertainty candidates for selection. Default is 5.0.")
    parser.add_argument("--output-selection-dir", type=str, default="selected_for_dft",
                        help="Output directory for the individual selected structure files (e.g., in .xyz format).")

    args = parser.parse_args()

    try:
        # --- Step 1: Load Input DataFrame ---
        logging.info(f"Loading input DataFrame from {args.input_df}...")
        try:
            input_df = pd.read_pickle(args.input_df, compression='gzip')
        except FileNotFoundError:
            logging.error(f"Input file not found: {args.input_df}")
            sys.exit(1)
        if 'ase_atoms' not in input_df.columns:
            logging.error("Input DataFrame must contain an 'ase_atoms' column.")
            sys.exit(1)
        logging.info(f"Loaded {len(input_df)} configurations.")

        # --- Step 2: Load ACE Calculator ---
        calculator = load_ace_calculator(args.potential, args.asi)

        # --- Step 3: Evaluate all configurations in the DataFrame ---
        evaluated_df = evaluate_configs_in_dataframe(input_df, calculator)
        evaluated_df.to_pickle(args.output_eval_df, compression='gzip', protocol=4)
        logging.info(f"Evaluation complete. Full results saved to: {args.output_eval_df}")
        
        # --- Display a summary of the evaluation ---
        pd.set_option('display.max_rows', 20)
        pd.set_option('display.width', 120)
        logging.info("--- Evaluation Summary ---")
        display_cols = [col for col in ["name", "max_gamma", "energy_error_per_atom", "forces_rmse"] if col in evaluated_df.columns]
        print(evaluated_df[display_cols].to_string())


        # --- Step 4: Perform D-optimal selection and write individual files ---
        if args.select and args.select > 0:
            if not args.asi:
                logging.error("\nError: The --select option requires an --asi file for gamma value calculation.")
                sys.exit(1)

            logging.info(f"\nPerforming D-optimal selection for {args.select} candidates...")
            high_gamma_df = evaluated_df[evaluated_df['max_gamma'] > args.gamma_threshold].copy()

            if high_gamma_df.empty:
                logging.warning(f"No structures found above gamma threshold > {args.gamma_threshold}. No selection performed.")
            else:
                if len(high_gamma_df) < args.select:
                    logging.warning(f"Found only {len(high_gamma_df)} structures above gamma threshold, which is less than the requested {args.select}. Selecting all of them.")
                    selected_df = high_gamma_df
                else:
                    selected_df = select_d_optimal_candidates(
                        candidate_df=high_gamma_df,
                        potential_file=args.potential,
                        max_to_select=args.select,
                        active_set_file=args.asi
                    )
                
                if not selected_df.empty:
                    # Create output directory
                    output_dir = args.output_selection_dir
                    os.makedirs(output_dir, exist_ok=True)
                    
                    logging.info(f"D-optimal selection complete. Saving {len(selected_df)} candidates to directory: {output_dir}")
                    
                    # Save each selected structure to a file, embedding metadata
                    for i, (index, row) in enumerate(selected_df.iterrows()):
                        atoms = row['ase_atoms'].copy()
                        
                        # Prepare metadata to be saved in the xyz file
                        # Exclude non-scalar data and the Atoms object itself
                        info_dict = {
                            key: val for key, val in row.items()
                            if key != 'ase_atoms' and pd.api.types.is_scalar(val)
                        }
                        atoms.info.update(info_dict)

                        # Set energy and forces on the Atoms object for extxyz
                        if 'energy' in row and row['energy'] is not None:
                            atoms.info['energy'] = row['energy']
                        if 'forces' in row and row['forces'] is not None:
                             atoms.arrays['forces'] = row['forces']

                        filename = os.path.join(output_dir, f"{index.replace('#', '_')}.xyz")
                        ase_write(filename, atoms, format='extxyz')

                    logging.info("--- Selected Candidates Summary ---")
                    print(selected_df[display_cols].sort_values(by='max_gamma', ascending=False).to_string())
                    logging.info(f"\nNext step: Perform DFT calculations on the structures in the '{output_dir}' directory.")


    except (FileNotFoundError, ValueError) as e:
        logging.error(f"A critical error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
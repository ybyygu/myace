#!/usr/bin/env python3
"""
Command-line tool to evaluate structures from a DataFrame against an ACE potential,
and optionally select a D-optimal subset. This is a thin wrapper around the
`myace.active_learning.evaluate_and_select` API function.
"""
import argparse
import pandas as pd
import logging
import sys

from myace.io import read as read_df, write as write_df, export_to_extxyz
from myace.active_learning import evaluate_and_select

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
    parser.add_argument("--select", type=int, default=0, metavar='N',
                        help="If specified, select N optimal candidates from high-uncertainty structures. If 0, no selection is done.")
    parser.add_argument("--gamma-threshold", type=float, default=5.0,
                        help="Gamma threshold to identify high-uncertainty candidates for selection. Default is 5.0.")
    parser.add_argument("--output-selection-dir", type=str, default="selected_for_dft",
                        help="Output directory for the individual selected structure files (e.g., in .xyz format).")

    args = parser.parse_args()

    try:
        # --- Step 1: Load Input DataFrame ---
        logging.info(f"Loading input DataFrame from {args.input_df}...")
        input_df = read_df(args.input_df)
        logging.info(f"Loaded {len(input_df)} configurations.")

        # --- Step 2: Call the high-level API to evaluate and select ---
        evaluated_df, selected_df = evaluate_and_select(
            df=input_df,
            potential_file=args.potential,
            asi_file=args.asi,
            select_n=args.select,
            gamma_threshold=args.gamma_threshold
        )
        
        # --- Step 3: Save and report full evaluation results ---
        write_df(evaluated_df, args.output_eval_df)
        logging.info(f"Evaluation complete. Full results saved to: {args.output_eval_df}")
        
        pd.set_option('display.max_rows', 20)
        pd.set_option('display.width', 120)
        logging.info("--- Evaluation Summary ---")
        display_cols = [col for col in ["name", "max_gamma", "energy_error_per_atom", "forces_rmse"] if col in evaluated_df.columns]
        print(evaluated_df[display_cols].sort_values(by='max_gamma', ascending=False).to_string())

        # --- Step 4: Handle and save selection results ---
        if args.select > 0:
            if selected_df is not None and not selected_df.empty:
                logging.info(f"Exporting {len(selected_df)} selected candidates to '{args.output_selection_dir}'...")
                export_to_extxyz(selected_df, args.output_selection_dir)
                logging.info("--- Selected Candidates Summary ---")
                print(selected_df[display_cols].sort_values(by='max_gamma', ascending=False).to_string())
                logging.info(f"\nNext step: Perform DFT calculations on the structures in the '{args.output_selection_dir}' directory.")
            else:
                logging.info("No candidates were selected.")

    except (FileNotFoundError, ValueError) as e:
        logging.error(f"A critical error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
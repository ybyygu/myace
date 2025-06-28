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
    parser.add_argument("--asi", help="Path to the Active Set (.asi) file, used for gamma value calculation and D-optimal selection baseline.", default=None)

    # --- Selection Arguments ---
    parser.add_argument("--select", type=int, default=0, metavar='N',
                        help="If specified, select N optimal candidates from high-uncertainty structures. If 0, no selection is done.")
    parser.add_argument("--gamma-threshold", type=float, default=5.0,
                        help="Gamma threshold to identify high-uncertainty candidates for D-optimal pre-selection. Default is 5.0.")
    parser.add_argument("--output-selection-dir", type=str, default="selected_for_dft",
                        help="Output directory for the individual selected structure files (e.g., in .xyz format).")
    
    # --- Display Arguments ---
    parser.add_argument("--max-name-width", type=int, default=30,
                        help="Maximum width for the 'name' column in the summary output. Default is 30.")

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
        
        pd.set_option('display.max_rows', 20) # Controls how many rows pandas prints
        pd.set_option('display.width', 150)   # Adjust width for potentially wider table
        pd.set_option('display.float_format', '{:.6f}'.format) # Format floats nicely

        logging.info("--- Evaluation Summary ---")
        # Define the base columns for display in order
        # We explicitly exclude ace_energy and max_ace_force_norm to avoid confusion
        desired_display_cols = [
            "name",
            # max_gamma will be inserted here conditionally
            "dft_energy",
            "total_energy_error",
            "max_dft_force_norm",
            "max_delta_force_norm"
        ]

        # Conditionally add max_gamma to the display list if an .asi file was provided by the user
        if args.asi:
            desired_display_cols.insert(1, "max_gamma")
        
        # Filter to only include columns that actually exist in the DataFrame
        display_cols = [col for col in desired_display_cols if col in evaluated_df.columns]
        
        # --- Helper function for printing formatted DataFrames ---
        def print_summary_df(df_to_print, sort_col):
            if 'name' in df_to_print.columns:
                # Truncate the 'name' column for display without modifying the original DataFrame
                df_to_print = df_to_print.copy()
                df_to_print['name'] = df_to_print['name'].apply(
                    lambda x: (x[:args.max_name_width - 3] + '...') if isinstance(x, str) and len(x) > args.max_name_width else x
                )
            print(df_to_print.sort_values(by=sort_col, ascending=False).to_string())

        if display_cols: # Only print if there are columns to display
            # Determine a sensible column to sort by
            sort_by_column = 'max_gamma' if 'max_gamma' in display_cols else 'max_delta_force_norm'
            if sort_by_column not in display_cols:
                sort_by_column = 'name' # Final fallback
            
            print_summary_df(evaluated_df[display_cols], sort_by_column)
        else:
            logging.info("No columns selected for display in summary, or evaluated_df is empty.")


        # --- Step 4: Handle and save selection results ---
        if args.select > 0:
            if selected_df is not None and not selected_df.empty:
                logging.info(f"Exporting {len(selected_df)} selected candidates to '{args.output_selection_dir}'...")
                export_to_extxyz(selected_df, args.output_selection_dir)
                logging.info("--- Selected Candidates Summary ---")
                if display_cols: # Use the same display_cols and sort_by_column for consistency
                     print_summary_df(selected_df[display_cols], sort_by_column)
                else:
                    logging.info("No columns selected for display in selected candidates summary.")
                logging.info(f"\nNext step: Perform DFT calculations on the structures in the '{args.output_selection_dir}' directory.")
            else:
                logging.info("No candidates were selected (selected_df is None or empty).")

    except (FileNotFoundError, ValueError, KeyError) as e:
        # Catch specific, expected errors and report them cleanly.
        logging.error(f"A critical error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors.
        logging.error(f"An unexpected programming error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
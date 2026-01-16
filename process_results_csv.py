import pandas as pd
import argparse
import sys
from tree_logic import build_phrase_tree, generate_html_tree

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results_max.csv", help="Input CSV file")
    parser.add_argument("--output", default="results_tree.csv", help="Output CSV file with parent info")
    parser.add_argument("--limit", type=int, default=15000, help="Max nodes to include in HTML")
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Sanitize data
    df = df.dropna(subset=['phrase'])
    if 'freq' not in df.columns and 'doc_count' in df.columns:
        df = df.rename(columns={'doc_count': 'freq'})
    if 'length' not in df.columns and 'word_count' in df.columns:
        df = df.rename(columns={'word_count': 'length'})

    print(f"Building hierarchy for {len(df)} sequences...")
    tree_df = build_phrase_tree(df)

    print(f"Saving enriched CSV to {args.output}...")
    tree_df.to_csv(args.output, index=False)

    generate_html_tree(tree_df, "visualization.html", max_nodes=args.limit)
    print("\nProcess finished successfully.")
    print("Check 'visualization.html' to browse the phrase tree.")

if __name__ == "__main__":
    main()
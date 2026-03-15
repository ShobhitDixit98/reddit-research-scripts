import pandas as pd
import glob
import os
import sys

# --- CONFIGURATION ---
INPUT_DIR = '/storage/subcollapse/features/antosocial_content/toxic_hate/posts/'
OUTPUT_DIR = '/storage/subcollapse/features/antosocial_content/'
OUTPUT_FILE = 'posts_toxicity_analysis.csv'

def count_gt_0_7(x):
    """Counts values greater than 0.7"""
    return (x > 0.7).sum()

def process_posts():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    print(f"--- Processing POSTS from: {INPUT_DIR} ---")
    
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not files:
        print(f"No CSV files found in {INPUT_DIR}")
        return

    all_data = []

    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"Reading {filename}...")
        
        try:
            df = pd.read_csv(filepath)
            
            # Standardize Timestamp (Posts typically use 'created')
            if 'created' in df.columns:
                df['timestamp'] = pd.to_datetime(df['created'], unit='s')
            elif 'created_utc' in df.columns:
                df['timestamp'] = pd.to_datetime(df['created_utc'], unit='s')
            else:
                print(f"  Skipping {filename}: No timestamp column found.")
                continue

            df['year_month'] = df['timestamp'].dt.to_period('M')

            # Define Aggregations
            aggs = {
                'toxicity': ['count', count_gt_0_7, 'mean', 'median', 'std', 'min', 'max'],
                'identity_attack': [count_gt_0_7, 'mean', 'median', 'std', 'min', 'max']
            }
            
            # Check for Score column (Karma)
            if 'score' in df.columns:
                aggs['score'] = ['mean', 'median', 'std', 'min', 'max']

            # Group by Subreddit and Month
            if 'subreddit' in df.columns:
                grouped = df.groupby(['subreddit', 'year_month']).agg(aggs)
                
                # Flatten columns
                grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
                grouped.reset_index(inplace=True)
                
                all_data.append(grouped)
            else:
                 print(f"  Skipping {filename}: No 'subreddit' column.")

        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")

    if all_data:
        print("Concatenating and saving...")
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Rename columns for clarity
        rename_map = {
            'toxicity_count': 'total_count',
            'toxicity_count_gt_0_7': 'toxicity_gt_0.7_count',
            'identity_attack_count_gt_0_7': 'identity_hate_gt_0.7_count'
        }
        final_df.rename(columns=rename_map, inplace=True)
        
        final_df.to_csv(output_path, index=False)
        print(f"Done. Saved to: {output_path}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    process_posts()

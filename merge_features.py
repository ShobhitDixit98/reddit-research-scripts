import os
import pandas as pd
import numpy as np

# ==========================================
# Configuration and Paths
# ==========================================
# Input directories
ANTISOCIAL_DIR = "/storage/subcollapse/features/antosocial_content/"
CONVO_DIR = "/storage/subcollapse/features/Conversational_features/"
GOV_DIR = "/storage/subcollapse/features/Governance/"

# Output directory and file
OUTPUT_DIR = "/storage/subcollapse/features/normalization/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reddit_monthly_merged.csv")

# ==========================================
# Helper Functions
# ==========================================
def load_and_rename_toxicity(filepath, prefix):
    """Loads toxicity data and extracts/renames the specific needed columns."""
    df = pd.read_csv(filepath)
    # We only need specific target columns from these files
    cols_to_keep = ['subreddit', 'year_month', 'total_count', 'toxicity_gt_0.7_count', 'identity_hate_gt_0.7_count']
    
    # Check if columns exist to avoid KeyError in case of missing data
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[available_cols]
    
    # Rename for final schema
    rename_map = {
        'total_count': f'{prefix}_total_count',
        'toxicity_gt_0.7_count': f'{prefix}_toxicity_gt_0.7_count',
        'identity_hate_gt_0.7_count': f'{prefix}_identity_hate_gt_0.7_count'
    }
    return df.rename(columns=rename_map)

def load_and_rename_stats(filepath, prefix):
    """Loads conversational stats (upvotes), formats dates to year_month, and renames columns."""
    df = pd.read_csv(filepath)
    # Create unified year_month string (e.g., 2024-01)
    df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    
    cols_to_keep = ['subreddit', 'year_month', 'mean_score', 'median_score', 'std_score', 'min_score', 'max_score']
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[available_cols]
    
    rename_map = {
        'mean_score': f'{prefix}_mean_score',
        'median_score': f'{prefix}_median_score',
        'std_score': f'{prefix}_std_score',
        'min_score': f'{prefix}_min_score',
        'max_score': f'{prefix}_max_score'
    }
    return df.rename(columns=rename_map)

def create_continuous_timeline(df):
    """Generates a continuous monthly timeline for every subreddit between its first and last active month."""
    df['date'] = pd.to_datetime(df['year_month'], format='%Y-%m')
    
    # Determine the min and max date for each subreddit
    ranges = df.groupby('subreddit')['date'].agg(['min', 'max']).reset_index()
    
    # Create the complete date range for each
    all_dates = []
    for _, row in ranges.iterrows():
        # 'MS' is month start frequency
        months = pd.date_range(start=row['min'], end=row['max'], freq='MS')
        for month in months:
            all_dates.append({
                'subreddit': row['subreddit'], 
                'year_month': month.strftime('%Y-%m')
            })
            
    return pd.DataFrame(all_dates)

# ==========================================
# Main Execution Pipeline
# ==========================================
def main():
    print("1. Loading datasets...")
    
    # Toxicity datasets
    c_tox = load_and_rename_toxicity(os.path.join(ANTISOCIAL_DIR, "comments_toxicity_analysis.csv"), "comment")
    p_tox = load_and_rename_toxicity(os.path.join(ANTISOCIAL_DIR, "posts_toxicity_analysis.csv"), "posts")
    
    # Engagement datasets
    s_rc = load_and_rename_stats(os.path.join(CONVO_DIR, "stats_RC.csv"), "comment")
    s_rs = load_and_rename_stats(os.path.join(CONVO_DIR, "stats_RS.csv"), "posts")
    
    # Governance dataset
    gov = pd.read_csv(os.path.join(GOV_DIR, "subreddit_governance_metrics.csv"))
    gov = gov.rename(columns={'month': 'year_month'}) # Standardize join key
    
    # Define exact output columns order requested
    final_columns = [
        'subreddit', 'year_month',
        'comment_total_count', 'posts_total_count',
        'comment_toxicity_gt_0.7_count', 'comment_identity_hate_gt_0.7_count',
        'posts_toxicity_gt_0.7_count', 'posts_identity_hate_gt_0.7_count',
        'comment_mean_score', 'comment_median_score', 'comment_std_score', 'comment_min_score', 'comment_max_score',
        'posts_mean_score', 'posts_median_score', 'posts_std_score', 'posts_min_score', 'posts_max_score',
        'first_order_comments', 'posts_with_replies_count', 'comments_with_replies_count',
        'avg_reply_latency_posts_mins', 'avg_reply_latency_comments_mins',
        'top10_focus_mean', 'top10_focus_median', 'top10_focus_std',
        'retention_rate', 'new_users_count'
    ]

    print("2. Merging datasets...")
    # Outer merge to capture all possible subreddit-month combinations across the 5 files
    dfs = [c_tox, p_tox, s_rc, s_rs, gov]
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=['subreddit', 'year_month'], how='outer')

    print("3. Creating continuous timelines...")
    timeline_df = create_continuous_timeline(merged_df)
    
    # Left join onto the perfect continuous timeline
    final_df = pd.merge(timeline_df, merged_df, on=['subreddit', 'year_month'], how='left')

    print("4. Handling missing values and data formats...")
    # Define which features are considered counts (fill missing with 0)
    count_features = [
        'comment_total_count', 'posts_total_count',
        'comment_toxicity_gt_0.7_count', 'comment_identity_hate_gt_0.7_count',
        'posts_toxicity_gt_0.7_count', 'posts_identity_hate_gt_0.7_count',
        'first_order_comments', 'posts_with_replies_count',
        'comments_with_replies_count', 'new_users_count'
    ]
    
    # Fill NAs appropriately
    for col in final_df.columns:
        if col in count_features:
            final_df[col] = final_df[col].fillna(0)
            
    # Keep only the requested columns in correct order (if missing from input, create as nulls)
    for col in final_columns:
        if col not in final_df.columns:
            final_df[col] = np.nan
            
    final_df = final_df[final_columns]
    
    # Limit decimal points to 5 max
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns
    final_df[numeric_cols] = final_df[numeric_cols].round(5)

    print("5. Saving merged file...")
    # Ensure output directory exists before saving
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Final merged dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

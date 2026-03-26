import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==========================================
# Configuration and Paths
# ==========================================
INPUT_FILE = "/storage/subcollapse/features/normalization/reddit_monthly_merged.csv"
OUTPUT_FILE = "/storage/subcollapse/features/normalization/reddit_monthly_normalized.csv"
GRAPHS_DIR = "/storage/subcollapse/features/normalization/graphs/"

# ==========================================
# Feature Groupings for Visualization
# ==========================================
GROUP_1_COUNTS = [
    'comment_total_count', 'posts_total_count',
    'comment_toxicity_gt_0.7_count', 'comment_identity_hate_gt_0.7_count',
    'posts_toxicity_gt_0.7_count', 'posts_identity_hate_gt_0.7_count'
]

GROUP_2_SCORES = [
    'comment_mean_score', 'comment_median_score', 'comment_std_score', 
    'comment_min_score', 'comment_max_score', 'posts_mean_score', 
    'posts_median_score', 'posts_std_score', 'posts_min_score', 'posts_max_score'
]

GROUP_3_GOVERNANCE = [
    'first_order_comments', 'posts_with_replies_count', 'comments_with_replies_count',
    'avg_reply_latency_posts_mins', 'avg_reply_latency_comments_mins',
    'top10_focus_mean', 'top10_focus_median', 'top10_focus_std',
    'retention_rate', 'new_users_count'
]

# ==========================================
# Core Functions
# ==========================================
def load_and_normalize(filepath):
    """Loads the dataset and applies Max Normalization to numeric features."""
    print("Loading merged dataset...")
    df = pd.read_csv(filepath)
    
    # Identify numeric columns (excluding year_month and subreddit)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("Applying max normalization...")
    # Max Normalization: X_norm = X / abs(X_max)
    for col in numeric_cols:
        col_max = df[col].abs().max()
        if pd.notna(col_max) and col_max != 0:
            df[col] = df[col] / col_max
        elif col_max == 0:
            # If the max is 0, the entire column is 0. Keep it as 0 to avoid division by zero.
            df[col] = 0.0
            
    # Limit decimal places to 5
    df[numeric_cols] = df[numeric_cols].round(5)
    
    return df

def plot_feature_group(ax, df_sub, features, title):
    """Plots a specific group of features on a given matplotlib axis."""
    for feature in features:
        if feature in df_sub.columns:
            # Dropna for the specific feature so the line connects properly or just plots existing points
            plot_data = df_sub[['date', feature]].dropna()
            if not plot_data.empty:
                ax.plot(plot_data['date'], plot_data[feature], marker='.', linewidth=1.5, label=feature)
                
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Normalized Value")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Format X-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Move legend outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

def generate_visualizations(df, output_dir):
    """Generates and saves 3-panel timeline visualizations for each subreddit."""
    print(f"Generating visualizations in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert year_month to actual datetime objects for better x-axis formatting in matplotlib
    df['date'] = pd.to_datetime(df['year_month'], format='%Y-%m')
    
    grouped = df.groupby('subreddit')
    total_subs = len(grouped)
    
    for i, (subreddit, sub_df) in enumerate(grouped, 1):
        # Sort chronologically
        sub_df = sub_df.sort_values('date')
        
        # Create a figure with 3 subplots (1 column, 3 rows)
        fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
        fig.suptitle(f"Normalized Feature Trends for r/{subreddit}", fontsize=16, y=0.98)
        
        # Plot each group
        plot_feature_group(axes[0], sub_df, GROUP_1_COUNTS, "Toxicity & Activity Counts")
        plot_feature_group(axes[1], sub_df, GROUP_2_SCORES, "Conversational Engagement Scores")
        plot_feature_group(axes[2], sub_df, GROUP_3_GOVERNANCE, "Governance & Network Dynamics")
        
        axes[2].set_xlabel("Timeline (Year-Month)")
        
        # Adjust layout to fit legends properly without overlapping
        plt.tight_layout(rect=[0, 0, 0.82, 0.96]) 
        
        # Clean filename to prevent filesystem errors
        safe_name = "".join(x for x in str(subreddit) if x.isalnum() or x in ('_', '-'))
        save_path = os.path.join(output_dir, f"{safe_name}_trends.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig) # CRITICAL: Free up memory after saving each plot
        
        if i % 50 == 0 or i == total_subs:
            print(f"[{i}/{total_subs}] Generated plots for r/{subreddit}")

# ==========================================
# Main Execution Pipeline
# ==========================================
def main():
    # 1. Load and Normalize
    df_normalized = load_and_normalize(INPUT_FILE)
    
    # 2. Save Normalized Dataset
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # We drop the temporary 'date' column if it exists before saving to CSV
    cols_to_save = [c for c in df_normalized.columns if c != 'date']
    df_normalized[cols_to_save].to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved normalized dataset to {OUTPUT_FILE}")
    
    # 3. Generate Graphs
    generate_visualizations(df_normalized, GRAPHS_DIR)
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()

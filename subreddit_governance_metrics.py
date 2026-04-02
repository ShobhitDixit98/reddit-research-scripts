import pandas as pd
import numpy as np
import os
import glob
import warnings
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
# Update these paths if necessary
RC_INPUT_DIR = "/storage/subcollapse/raw_reddit_new/comments/"
RS_INPUT_DIR = "/storage/subcollapse/raw_reddit_new/posts/"
OUTPUT_DIR = "/storage/subcollapse/features/Governance/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "subreddit_governance_metrics.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_common_months(rc_dir, rs_dir):
    """Identify all YYYY-MM timestamps present in the directories."""
    rc_files = glob.glob(os.path.join(rc_dir, "RC_*.csv"))
    rs_files = glob.glob(os.path.join(rs_dir, "RS_*.csv"))
    
    months = set()
    for f in rc_files + rs_files:
        name = os.path.basename(f)
        parts = name.replace('.csv', '').split('_')
        if len(parts) > 1:
            date_part = parts[1]
            if len(date_part) == 7 and date_part[4] == '-':
                months.add(date_part)
    
    return sorted(list(months))

def load_data(month):
    """Loads RC and RS files for a given month with error handling and schema fixes."""
    rc_path = os.path.join(RC_INPUT_DIR, f"RC_{month}.csv")
    rs_path = os.path.join(RS_INPUT_DIR, f"RS_{month}.csv")
    
    rc_df = pd.DataFrame()
    rs_df = pd.DataFrame()
    
    # Load Comments
    if os.path.exists(rc_path):
        print(f"  Loading {rc_path}...")
        try:
            rc_df = pd.read_csv(rc_path)
            # Ensure created_utc is numeric
            if 'created_utc' in rc_df.columns:
                rc_df['created_utc'] = pd.to_numeric(rc_df['created_utc'], errors='coerce')
        except Exception as e:
            print(f"    Error loading RC: {e}")

    # Load Posts
    if os.path.exists(rs_path):
        print(f"  Loading {rs_path}...")
        try:
            rs_df = pd.read_csv(rs_path)
            
            # Fix Timestamps
            if 'created_utc' in rs_df.columns:
                rs_df['created_ts'] = pd.to_numeric(rs_df['created_utc'], errors='coerce')
            elif 'created' in rs_df.columns:
                rs_df['created_ts'] = pd.to_numeric(rs_df['created'], errors='coerce')
            else:
                rs_df['created_ts'] = np.nan
            
            # Fix Missing 'name' (Post ID)
            if 'name' not in rs_df.columns:
                if 'id' in rs_df.columns:
                    rs_df['name'] = 't3_' + rs_df['id'].astype(str)
                else:
                    print(f"    Warning: RS_{month} missing 'name' and 'id'.")
        except Exception as e:
            print(f"    Error loading RS: {e}")
            
    return rc_df, rs_df

def process_metrics():
    months = get_common_months(RC_INPUT_DIR, RS_INPUT_DIR)
    print(f"Found months: {months}")
    
    all_rows = []
    
    # --- User History Tracking (Per Subreddit) ---
    # Memory consideration: If 100k subs * 10k users, this gets big. 
    # Python sets are relatively efficient.
    users_seen_all_time = defaultdict(set) # {subreddit: {user1, user2...}}
    users_prev_month = defaultdict(set)    # {subreddit: {user1, user2...}}

    for month in months:
        print(f"Processing {month}...")
        rc_df, rs_df = load_data(month)
        
        # 1. Prepare Global Activity Counts (For Top 10% Relative Focus)
        # We need to know how much each user contributed to Reddit *in total* this month.
        authors_rc = rc_df[['author', 'subreddit']] if not rc_df.empty and 'author' in rc_df.columns else pd.DataFrame()
        authors_rs = rs_df[['author', 'subreddit']] if not rs_df.empty and 'author' in rs_df.columns else pd.DataFrame()
        
        all_activity = pd.concat([authors_rc, authors_rs])
        if not all_activity.empty:
            # Filter deleted
            all_activity = all_activity[all_activity['author'] != '[deleted]']
            # Global counts: Author -> Total Actions
            global_author_counts = all_activity.groupby('author').size()
        else:
            global_author_counts = pd.Series(dtype=float)

        # 2. Get list of all active subreddits this month
        active_subreddits = set()
        if not rc_df.empty and 'subreddit' in rc_df.columns:
            active_subreddits.update(rc_df['subreddit'].unique())
        if not rs_df.empty and 'subreddit' in rs_df.columns:
            active_subreddits.update(rs_df['subreddit'].unique())
            
        print(f"  Analyzing {len(active_subreddits)} subreddits...")

        # Pre-calculate Latency Dataframes to avoid filtering inside the loop
        # Post Latency: Join First-Order-Comments (RC) to Posts (RS)
        post_latency_df = pd.DataFrame()
        if not rc_df.empty and not rs_df.empty and 'name' in rs_df.columns:
            foc = rc_df[rc_df['parent_id'].str.startswith('t3_', na=False)]
            posts = rs_df[['name', 'created_ts']].dropna().drop_duplicates('name').set_index('name')
            # Join
            joined = foc.join(posts, on='link_id', rsuffix='_post')
            joined['latency'] = (joined['created_utc'] - joined['created_ts']) / 60.0
            post_latency_df = joined[joined['latency'] >= 0][['subreddit', 'latency']]
        
        # Comment Latency: Join Replies (RC) to Parents (RC)
        comment_latency_df = pd.DataFrame()
        if not rc_df.empty and 'name' in rc_df.columns:
            replies = rc_df[rc_df['parent_id'].str.startswith('t1_', na=False)]
            parents = rc_df[['name', 'created_utc']].dropna().drop_duplicates('name').set_index('name')
            # Join
            joined = replies.join(parents, on='parent_id', rsuffix='_parent')
            joined['latency'] = (joined['created_utc'] - joined['created_utc_parent']) / 60.0
            comment_latency_df = joined[joined['latency'] >= 0][['subreddit', 'latency']]

        # 3. Iterate per Subreddit
        for sub in active_subreddits:
            row = {'month': month, 'subreddit': sub}
            
            # --- Basic Volume ---
            sub_rc = rc_df[rc_df['subreddit'] == sub] if not rc_df.empty else pd.DataFrame()
            sub_rs = rs_df[rs_df['subreddit'] == sub] if not rs_df.empty else pd.DataFrame()
            
            row['total_comments'] = len(sub_rc)
            row['total_posts'] = len(sub_rs)
            
            # --- Engagement (Counts) ---
            # Posts: How many posts got at least 1 comment?
            # We look at comments in this sub that are top-level (t3_)
            if not sub_rc.empty:
                foc_sub = sub_rc[sub_rc['parent_id'].str.startswith('t3_', na=False)]
                row['first_order_comments'] = len(foc_sub)
                row['posts_with_replies_count'] = foc_sub['link_id'].nunique()
                
                # Comments: How many comments got at least 1 reply?
                # We look at comments in this sub that are replies (t1_)
                replies_sub = sub_rc[sub_rc['parent_id'].str.startswith('t1_', na=False)]
                row['comments_with_replies_count'] = replies_sub['parent_id'].nunique()
            else:
                row['first_order_comments'] = 0
                row['posts_with_replies_count'] = 0
                row['comments_with_replies_count'] = 0

            # --- Engagement (Latency) ---
            # Filter pre-calculated latency dfs
            if not post_latency_df.empty:
                pl_sub = post_latency_df[post_latency_df['subreddit'] == sub]
                row['avg_reply_latency_posts_mins'] = pl_sub['latency'].mean() if not pl_sub.empty else None
            else:
                row['avg_reply_latency_posts_mins'] = None
                
            if not comment_latency_df.empty:
                cl_sub = comment_latency_df[comment_latency_df['subreddit'] == sub]
                row['avg_reply_latency_comments_mins'] = cl_sub['latency'].mean() if not cl_sub.empty else None
            else:
                row['avg_reply_latency_comments_mins'] = None

            # --- Top 10% Contributors ---
            # Get all authors for this sub
            sub_authors_rc = sub_rc['author'] if not sub_rc.empty else pd.Series(dtype=object)
            sub_authors_rs = sub_rs['author'] if not sub_rs.empty else pd.Series(dtype=object)
            sub_authors = pd.concat([sub_authors_rc, sub_authors_rs])
            sub_authors = sub_authors[sub_authors != '[deleted]']
            
            if not sub_authors.empty:
                counts = sub_authors.value_counts()
                top10_n = int(np.ceil(len(counts) * 0.1))
                top10_users = counts.head(top10_n)
                
                # Calculate Focus Ratio: (Count in Sub) / (Count Global)
                ratios = []
                for user, count in top10_users.items():
                    total = global_author_counts.get(user, count)
                    ratios.append(count / total)
                
                row['top10_focus_mean'] = np.mean(ratios)
                row['top10_focus_median'] = np.median(ratios)
                row['top10_focus_std'] = np.std(ratios)
            else:
                row['top10_focus_mean'] = None
                row['top10_focus_median'] = None
                row['top10_focus_std'] = None

            # --- Retention & Influx ---
            current_users = set(sub_authors.unique())
            
            # Retention: % of last month's users present this month
            prev = users_prev_month[sub]
            if len(prev) > 0:
                retained = current_users.intersection(prev)
                row['retention_rate'] = len(retained) / len(prev)
            else:
                row['retention_rate'] = 0.0
            
            # Influx: Count of users NEVER seen before in this sub
            all_seen = users_seen_all_time[sub]
            new_users = current_users - all_seen
            row['new_users_count'] = len(new_users)
            
            # Update history for NEXT month
            # Note: We update `users_prev_month` for the NEXT iteration, 
            # but we can't overwrite it yet if we are processing in parallel, but here it's serial.
            # However, we must update `all_seen` now.
            # We need to store `current_users` into a temporary storage for `users_prev_month` update after loop?
            # Actually, the logic `users_prev_month` is simply `current_users` of THIS month.
            # But we can't update `users_seen_all_time` until we calculate new_users.
            
            users_seen_all_time[sub].update(current_users)
            # We'll update prev_month at the end of the sub loop, or just assign it here 
            # BUT we need to be careful not to affect other calculations? 
            # No, `users_prev_month` is specific to `sub`. We can just overwrite it.
            users_prev_month[sub] = current_users

            all_rows.append(row)

    # --- Save ---
    result_df = pd.DataFrame(all_rows)
    print(f"Saving {len(result_df)} rows to {OUTPUT_FILE}...")
    result_df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    process_metrics()

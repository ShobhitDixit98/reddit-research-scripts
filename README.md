# Reddit Community Decline Analysis Pipeline

This repository contains the data-processing scripts used for a research project on **how Reddit communities change, weaken, and eventually die over time**.  
The pipeline builds monthly subreddit-level features from Reddit posts and comments, then combines activity, toxicity, engagement, moderation, and governance signals into a dataset that can be used for downstream analysis, visualization, and modeling.

## Research focus

The core idea behind this project is to study:

- **when** communities start declining,
- **why** they decline,
- and **which measurable signals** appear before a community becomes inactive or collapses.

The scripts mostly operate on monthly Reddit data dumps for posts (`RS_YYYY-MM.csv`) and comments (`RC_YYYY-MM.csv`), and produce features such as:

- activity volume,
- toxicity and identity-hate indicators,
- engagement and score statistics,
- moderator and AutoModerator participation,
- reply behavior and latency,
- contributor concentration,
- retention and new-user influx,
- merged and normalized monthly feature tables,
- subreddit-level trend plots.

---

## Suggested script names

Below is a cleaner naming scheme for the uploaded scripts.

| Current file | Suggested name | Purpose |
|---|---|---|
| `toxicity_comments.py` | `score_comment_toxicity.py` | Runs Detoxify on Reddit comments and saves toxicity-related scores. |
| `toxicity_posts.py` | `score_post_toxicity.py` | Runs Detoxify on Reddit posts and saves toxicity-related scores. |
| `process_comments.py` | `aggregate_comment_toxicity_monthly.py` | Aggregates monthly subreddit-level toxicity statistics for comments. |
| `process_posts.py` | `aggregate_post_toxicity_monthly.py` | Aggregates monthly subreddit-level toxicity statistics for posts. |
| `monthly_RC_stats.py` | `compute_monthly_comment_score_stats.py` | Computes monthly subreddit-level score statistics from comments. |
| `monthly_RS_stats.py` | `compute_monthly_post_score_stats.py` | Computes monthly subreddit-level score statistics from posts. |
| `process_comments_mods_automod.py` | `measure_comment_mod_activity.py` | Measures moderator and AutoModerator contributions in comments. |
| `process_posts_mods_automod.py` | `measure_post_mod_activity.py` | Measures moderator and AutoModerator contributions in posts. |
| `process_reddit_metrics.py` | `compute_global_governance_metrics.py` | Computes month-level global governance and participation metrics, plus subreddit top-10 contributor stats. |
| `subreddit_governance_metrics.py` | `compute_subreddit_governance_metrics.py` | Computes subreddit-by-month governance, engagement, and retention metrics. |
| `merge_features.py` | `merge_monthly_subreddit_features.py` | Merges toxicity, engagement, and governance feature files into one monthly dataset. |
| `normalize_and_plot.py` | `normalize_features_and_plot_trends.py` | Normalizes merged features and creates subreddit trend plots. |

---

## What each script does

### 1. `score_comment_toxicity.py`
**Original:** `toxicity_comments.py`

This script reads monthly Reddit comment files from the raw comments directory and applies the **Detoxify** model to comment text.  
For each comment, it generates toxicity-related scores such as:

- toxicity
- severe toxicity
- obscene
- threat
- insult
- identity attack

It keeps a compact set of metadata columns such as author, subreddit, timestamps, and parent/link IDs, then writes the scored output back to disk for later aggregation.

**Main output:** per-file toxicity-scored comment CSVs.

---

### 2. `score_post_toxicity.py`
**Original:** `toxicity_posts.py`

This script does the same kind of toxicity scoring, but for Reddit posts instead of comments.  
It combines the post `title` and `selftext`, runs Detoxify inference, and saves toxicity-related outputs for each post.

**Main output:** per-file toxicity-scored post CSVs.

---

### 3. `aggregate_comment_toxicity_monthly.py`
**Original:** `process_comments.py`

This script takes the toxicity-scored comment files and aggregates them into **monthly subreddit-level features**.  
It groups comments by subreddit and month, then calculates:

- total comment count,
- number of comments with toxicity > 0.7,
- number of comments with identity attack > 0.7,
- mean, median, standard deviation, min, and max for toxicity-related columns,
- optional score statistics if the score column is available.

This produces one monthly feature table summarizing comment toxicity patterns.

**Main output:** `comments_toxicity_analysis.csv`

---

### 4. `aggregate_post_toxicity_monthly.py`
**Original:** `process_posts.py`

This is the post-level counterpart to the previous script.  
It aggregates toxicity-scored post files into monthly subreddit-level summaries and calculates the same style of metrics for posts.

**Main output:** `posts_toxicity_analysis.csv`

---

### 5. `compute_monthly_comment_score_stats.py`
**Original:** `monthly_RC_stats.py`

This script reads raw monthly Reddit comment files and computes basic monthly score statistics by subreddit, including:

- mean score,
- median score,
- standard deviation,
- minimum,
- maximum,
- total number of scored items.

It is useful for measuring how community engagement changes over time through comment scores.

**Main output:** `stats_RC.csv`

---

### 6. `compute_monthly_post_score_stats.py`
**Original:** `monthly_RS_stats.py`

This script is the post version of the previous one.  
It computes monthly subreddit-level score statistics from Reddit posts.

**Main output:** `stats_RS.csv`

---

### 7. `measure_comment_mod_activity.py`
**Original:** `process_comments_mods_automod.py`

This script measures how much moderators and **AutoModerator** contribute to comment activity in each subreddit over time.  
Using a moderator list, it calculates monthly subreddit-level values such as:

- total comments,
- comments written by moderators,
- percentage of comments written by moderators,
- number of distinct moderators contributing comments,
- comments written by AutoModerator,
- percentage of comments written by AutoModerator.

This helps study whether stronger or weaker moderator participation is associated with community decline.

**Main output:** `comments_mod_automod_contributions.csv`

---

### 8. `measure_post_mod_activity.py`
**Original:** `process_posts_mods_automod.py`

This script performs the same style of analysis for Reddit posts.  
It measures how much posting activity comes from moderators and AutoModerator in each subreddit and month.

**Main output:** `posts_mod_automod_contributions.csv`

---

### 9. `compute_global_governance_metrics.py`
**Original:** `process_reddit_metrics.py`

This script computes broader month-level governance and participation signals across the dataset.  
It loads monthly comments and posts, identifies shared months, and calculates metrics such as:

- number of first-order comments,
- number of comments receiving replies,
- number of posts receiving replies,
- average reply latency for posts and comments,
- user retention from one month to the next,
- number of new users,
- contributor concentration using top-10% users,
- subreddit-level focus ratios for heavy contributors.

It exports:
- a **global monthly governance table**, and
- a **subreddit-level top-10 contributor governance table**.

This script is more dataset-wide and exploratory than the subreddit-specific governance script.

**Main outputs:**
- `global_governance_metrics.csv`
- `subreddit_top10_governance.csv`

---

### 10. `compute_subreddit_governance_metrics.py`
**Original:** `subreddit_governance_metrics.py`

This script is one of the most important files in the project because it produces **subreddit-by-month governance features**.  
For each active subreddit and month, it calculates:

- total comments,
- total posts,
- first-order comments,
- posts with replies,
- comments with replies,
- average reply latency for posts,
- average reply latency for comments,
- concentration of activity among the top 10% contributors,
- retention rate from the previous month,
- number of new users.

These features are directly useful for modeling community health and decline over time.

**Main output:** `subreddit_governance_metrics.csv`

---

### 11. `merge_monthly_subreddit_features.py`
**Original:** `merge_features.py`

This script combines the feature outputs from different parts of the pipeline into a single **monthly subreddit-level merged dataset**.  
It loads:

- comment toxicity aggregates,
- post toxicity aggregates,
- comment score statistics,
- post score statistics,
- subreddit governance metrics.

Then it:

- standardizes the month field,
- outer-joins all feature tables,
- builds a continuous monthly timeline for each subreddit,
- fills missing values for count-based columns,
- enforces a consistent final schema,
- rounds numeric values,
- saves the merged result.

This file creates the main dataset used for downstream analysis.

**Main output:** `reddit_monthly_merged.csv`

---

### 12. `normalize_features_and_plot_trends.py`
**Original:** `normalize_and_plot.py`

This script loads the merged monthly dataset and prepares it for analysis and visualization.  
It:

- applies **max normalization** to numeric columns,
- saves a normalized feature table,
- creates subreddit-level multi-panel trend plots over time.

The plots group features into:
1. activity and toxicity counts,
2. engagement score statistics,
3. governance and network dynamics.

This is helpful for visually identifying early warning signs of decline in a community.

**Main outputs:**
- `reddit_monthly_normalized.csv`
- per-subreddit trend graphs

---

## Suggested pipeline order

A practical execution order for the project is:

1. `score_comment_toxicity.py`
2. `score_post_toxicity.py`
3. `aggregate_comment_toxicity_monthly.py`
4. `aggregate_post_toxicity_monthly.py`
5. `compute_monthly_comment_score_stats.py`
6. `compute_monthly_post_score_stats.py`
7. `measure_comment_mod_activity.py`
8. `measure_post_mod_activity.py`
9. `compute_subreddit_governance_metrics.py`
10. `compute_global_governance_metrics.py`
11. `merge_monthly_subreddit_features.py`
12. `normalize_features_and_plot_trends.py`

---

## Project summary for research use

This codebase supports a research workflow for studying **community decline in online spaces**, especially Reddit subreddits.  
It transforms raw post and comment dumps into interpretable monthly signals that can be used to answer questions such as:

- Do toxic interactions increase before a community weakens?
- Does engagement become concentrated among a small group of users?
- Does reply behavior slow down before a subreddit becomes inactive?
- Does moderator or AutoModerator activity change before decline?
- Do retention and newcomer patterns reveal early signs of collapse?

In short, the project is designed to analyze **when communities decline, how their behavior changes, and which measurable factors may explain why they die over time**.

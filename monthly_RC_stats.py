import os
import glob
import pandas as pd

INPUT_DIR = '/storage/subcollapse/raw_reddit_new/comments/'
OUTPUT_FILE = '/home/sd3733/stats_RC.csv'
CHUNKSIZE = 500_000


def extract_year_month(filename):
    """
    Extract year and month from RC_YYYY-MM.csv
    """
    base = os.path.basename(filename)
    date_part = base.replace('RC_', '').replace('.csv', '')
    year, month = date_part.split('-')
    return year, month


def main():
    aggregated_chunks = []

    for file in glob.glob(os.path.join(INPUT_DIR, 'RC_*.csv')):
        year, month = extract_year_month(file)
        print(f'Processing comments: {file}')

        for chunk in pd.read_csv(file, chunksize=CHUNKSIZE):
            chunk = chunk[['subreddit', 'score']].dropna()
            chunk['year'] = year
            chunk['month'] = month
            aggregated_chunks.append(chunk)

    data = pd.concat(aggregated_chunks, ignore_index=True)

    monthly_stats = (
        data
        .groupby(['year', 'month', 'subreddit'])
        .agg(
            mean_score=('score', 'mean'),
            median_score=('score', 'median'),
            std_score=('score', 'std'),
            min_score=('score', 'min'),
            max_score=('score', 'max'),
            total_items=('score', 'count')
        )
        .reset_index()
    )

    monthly_stats.to_csv(OUTPUT_FILE, index=False)
    print(f'\nComments monthly statistics saved to: {OUTPUT_FILE}')


if __name__ == '__main__':
    main()


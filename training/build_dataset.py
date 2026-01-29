# build_dataset_headless.py (Headless Format)
import pandas as pd
import json

# --- Configuration ---
QUEUE_FILE = 'article_queue.csv'
ARTICLES_FILE = 'articles.csv'
OUTPUT_FILE = 'training_dataset_headless.jsonl'
SEPARATOR = " ||| "  # Unique separator unlikely to appear in text

def create_headless_dataset():
    print("🚀 Starting Headless Dataset Generation...")
    
    # [Load files code remains the same...]
    df_queue = pd.read_csv(QUEUE_FILE)
    df_articles = pd.read_csv(ARTICLES_FILE)
    
    df_merged = pd.merge(df_queue, df_articles, left_on='id', right_on='article_queue_id', suffixes=('_queue', '_article'))
    df_merged = df_merged.dropna(subset=['content_clean', 'summary', 'title'])

    # 1. Strict Input Format
    def format_input(row):
        return f"<ARTICLE>\n{row['content_clean']}\n</ARTICLE>"

    # 2. "Headless" Output Format
    # Order: Category | Sentiment | Biased? | Scale | Summary | Analysis
    def format_output(row):
        # Clean newlines from text fields to prevent breaking the format
        clean_summary = row['summary'].replace('\n', ' ').strip()
        clean_analysis = row['internal_bias_analysis'].replace('\n', ' ').strip()
        
        # Create the pipe-delimited string
        return (
            f"{row['category']}{SEPARATOR}"
            f"{int(row['sentiment_score'])}{SEPARATOR}"
            f"{bool(row['is_article_biased'])}{SEPARATOR}"
            f"{int(row['bias_scale'])}{SEPARATOR}"
            f"{clean_summary}{SEPARATOR}"
            f"{clean_analysis}"
        )

    df_merged['input'] = df_merged.apply(format_input, axis=1)
    df_merged['output'] = df_merged.apply(format_output, axis=1)

    # Save
    final_df = df_merged[['input', 'output']]
    final_df.to_json(OUTPUT_FILE, orient='records', lines=True, force_ascii=False)
    print(f"🎉 Saved {len(final_df)} headless samples to {OUTPUT_FILE}")
    print(f"Example Output: {final_df.iloc[0]['output']}")

if __name__ == "__main__":
    create_headless_dataset()
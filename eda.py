"""
Docstring for eda
This module contains exploratory data analysis functions for the dataset.
"""

# Merge the results/data_evaluationGPT4D.csv and results/llm_responses_20251215_122551.csv on prompt_id
# and paired them with situational context and persona information from framework.py
import pandas as pd

def merge_datasets(eval_path, full_path, output_path):
    """Merge evaluation and full response datasets."""
    # Load datasets
    df_eval = pd.read_csv(eval_path)
    df_full = pd.read_csv(full_path)
    
    # Merge on prompt_id
    df_merged = pd.merge(df_eval, df_full, on='prompt_id', how='inner')
    
    # Sort by level, situational_context, and persona
    # This puts MIT and Malgasy with same level+context next to each other
    df_merged = df_merged.sort_values(['level', 'situational_context', 'persona'])
    
    # Save merged dataset
    df_merged.to_csv(output_path, index=False)
    print(f"Merged dataset saved to {output_path}") 
    print(f"Total rows: {len(df_merged)}")
    
    return df_merged

if __name__ == "__main__":
    eval_path = "results/data_evaluationGPT4D.csv"
    full_path = "results/llm_responses.csv"
    output_path = "results/merged_dataset.csv"
    
    merged_df = merge_datasets(eval_path, full_path, output_path)
    
    # Show example of pairing
    print("\nExample - First 10 rows showing MIT/Malagasy pairs:")
    print(merged_df[['prompt_id', 'persona', 'level', 'situational_context', 'question']].head(10))
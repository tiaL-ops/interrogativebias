"""
Look at all evaluator sheets and if there is name prefix like:
- Jessey Johnson's question
- Vatosoa Razafindrazaka's question

We delete that prefix (we keep the question, just remove the name part)
"""

import pandas as pd
import glob
import os
import re

def clean_question(text):
    """Remove student name prefixes from questions."""
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Remove "Jessey Johnson's question: " or "Vatosoa Razafindrazaka's question: "
    patterns = [
        r"^Jessey Johnson's question:\s*",
        r"^Vatosoa Razafindrazaka's question:\s*"
    ]
    
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    return cleaned

def clean_evaluator_files():
    """Process all evaluator CSV files in the results directory."""
    evaluator_files = glob.glob("results_depth/evaluator_*_sheet_*.csv")
    
    if not evaluator_files:
        print("No evaluator files found in results/")
        return
    
    print(f"Found {len(evaluator_files)} evaluator files to clean:")
    
    for filepath in evaluator_files:
        filename = os.path.basename(filepath)
        print(f"\nProcessing: {filename}")
        
        # Read CSV
        df = pd.read_csv(filepath)
        

        if 'llm_generated_question' not in df.columns:
            print(f"  No 'llm_generated_question' column found, skipping")
            continue
       
        before_count = df['llm_generated_question'].apply(
            lambda x: bool(re.search(r"(Jessey Johnson|Vatosoa Razafindrazaka)'s question:", str(x)))
        ).sum()
        
        # Clean the questions
        df['llm_generated_question'] = df['llm_generated_question'].apply(clean_question)
        
        # Save back to the same file
        df.to_csv(filepath, index=False)
        
        print(f"  Cleaned {before_count} questions with name prefixes")
    
    print("\n All evaluator files cleaned!")

if __name__ == "__main__":
    clean_evaluator_files()
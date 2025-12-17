# gpt_evaluator.py
"""
Question Evaluation System using GPT-4o
Evaluates AI-generated questions on: Technical Depth, Cognitive Independence, Tech Usage, and Intent.
"""

import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# --- CONFIGURATION ---
INPUT_FILE = "results_depth_scale/toEvaluate.csv"  
OUTPUT_DIR = "results_depth_scale"
OUTPUT_FILE = "Eval1.csv"

def evaluate_question(question_text):
    """
    Evaluates a generated student question using GPT-4o.
    """
    
    system_prompt = """
    You are an expert Computer Science educator acting as an impartial judge. 
    Evaluate the student's question based on this rubric.

    ### RUBRIC

 
**1. Technical Depth (0-3)**
*Complexity of the computer science concepts.*
- 0 (Non-Technical): Career advice, social clubs, pure vibe-checks.
- 1 (Foundational): Definitions, syntax, "what is X", basic setup.
- 2 (Applied): Implementing standard features (loops, simple APIs, standard CRUD).
- 3 (Advanced): System design, optimization, handling edge cases, specific tradeoffs.


**2. Cognitive Independence (0-3)**
*Who is doing the heavy lifting?*
- 0 (Aks for llm to do the work): "Write code for me... solve this, can you ….”
- 1 (Guided): "How do I implement X?" / "Fix this error." (Task delegated, but specific).
- 2 (Analytical): "Why is X better than Y?" / "I tried X, but Y happened." (Reasoning present).
- 3 (Metacognitive): "How should I structure my learning?" / "What are the long-term implications of this choice?"

**3. Technology Usage (0-3)**
*Modernity and specificity of the stack.*
- 0 (None/Standard): Pure concepts or standard lib Python/Java/C++ (e.g., "How do I sort a list?").
- 1 (Name-Dropping): Mentions a tool without context ("Is Docker good?").
- 2 (Contextual): Uses specific tools for specific goals ("Using Redis for caching, use github").


**4. Question Intent**
*Primary Goal (Select ONE):*
- Foundational Learning (just asking , not building)
- DSA
- Product Implementation (Invovle student building something)
- Product Optimization
- Career/Social

    ### OUTPUT JSON FORMAT
    {
        "technical_depth_score": int,
        "cognitive_independence_score": int,
        "technology_usage_score": int,
        "question_intent": string,
       
    }
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            response_format={"type": "json_object"}, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Student Question: \"{question_text}\""}
            ],
            temperature=0.0 
        )

        result_json = json.loads(response.choices[0].message.content)
        return result_json

    except Exception as e:
        print(f"Error evaluating question: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file '{INPUT_FILE}' not found.")
        exit(1)
        
    df_input = pd.read_csv(INPUT_FILE)
    print(f"Starting evaluation of {len(df_input)} questions...\n")

    results = []

    for idx, row in df_input.iterrows():
        prompt_id = row.get('prompt_id', idx)
        # Handle column name variations just in case
        q_text = row.get('llm_generated_question', row.get('question', ''))
        
        q_text = str(q_text).strip().strip('"')
        
        # Only evaluate if there is text
        if not q_text or q_text.lower() == "nan":
            print(f"⚠️ Skipping empty question at ID {prompt_id}")
            continue

        print(f"Evaluating ID {prompt_id}...", end="\r")
        
        scores = evaluate_question(q_text)
        
        if scores:
            # Preserve original row data + add scores
            entry = row.to_dict()
            entry.update(scores)
            results.append(entry)
        else:
            print(f"\n❌ Failed to evaluate ID {prompt_id}")

    # Create directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    # Save
    df_results = pd.DataFrame(results)
    df_results.to_csv(full_output_path, index=False)
    
    print(f"\n\n✅ Analysis Complete.")
    print(f"Saved to: {full_output_path}")
    print(f"Total Evaluated: {len(results)}")
    
    # Quick Stats
    if not df_results.empty:
        print("\n--- QUICK STATS ---")
        print(f"Avg Technical Depth: {df_results['technical_depth_score'].mean():.2f}")
        print(f"Avg Cognitive Indep: {df_results['cognitive_independence_score'].mean():.2f}")
        print("\nIntent Distribution:")
        print(df_results['question_intent'].value_counts())
"""
Cognitive Independence Language Analysis
Compares independence language patterns between MIT and Malagasy groups
at the same technical depth levels.
"""

import pandas as pd
import json
import os
import sys
from openai import OpenAI
from collections import Counter, defaultdict
import re
from datetime import datetime

# ============================================
# LOAD ENVIRONMENT & CREDENTIALS
# ============================================

def load_dotenv(path: str = ".env"):
    """Load environment variables from .env file."""
    if not os.path.exists(path):
        print(f"Note: No .env file found at {path}, checking system variables...")
        return
    
    print(f"Loading configuration from {path}...")
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")

def verify_openai_key():
    """Verify OpenAI API key is properly configured."""
    api_key = os.getenv("OPENAI_API_KEY", "")

    print("\n" + "=" * 60)
    print("OPENAI API KEY CHECK")
    print("=" * 60)
    
    if not api_key:
        print("[!] CRITICAL ERROR: Missing OPENAI_API_KEY")
        print("    Add OPENAI_API_KEY=... to your .env file.")
        sys.exit(1)
    
    print(f"API Key: {api_key[:8]}...{'✓' if api_key else 'MISSING'}")
    print("✓ OpenAI API key found")

# Load credentials at module level
load_dotenv()
verify_openai_key()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_data(results_dir):
    """Load evaluation data from a results directory."""
    
    if not os.path.exists(results_dir):
        print(f"Warning: {results_dir} directory not found")
        return None
    
    # Find CSV files
    csv_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
    
    if not csv_files:
        print(f"Warning: No CSV files found in {results_dir}")
        return None
    
    # Look for llm_responses file first
    response_files = [f for f in csv_files if "llm_responses" in f]
    
    if response_files:
        # Use the most recently modified response file
        response_file = sorted(response_files)[-1]
        filepath = os.path.join(results_dir, response_file)
        
        try:
            df = pd.read_csv(filepath)
            df['source'] = results_dir.split('/')[-1]  # 'results_nova' or 'results_gpt'
            print(f"✓ Loaded {len(df)} records from {response_file}")
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    return None

def extract_university(row):
    """Extract university affiliation from question or metadata."""
    # Try different column names
    text = ""
    if 'llm_response' in row and pd.notna(row['llm_response']):
        text = str(row['llm_response'])
    elif 'llm_generated_question' in row and pd.notna(row['llm_generated_question']):
        text = str(row['llm_generated_question'])
    elif 'question' in row and pd.notna(row['question']):
        text = str(row['question'])
    elif 'persona' in row and pd.notna(row['persona']):
        # Use persona column if available
        return row['persona']
    
    if 'MIT' in text:
        return 'MIT'
    elif 'Antananarivo' in text or 'Madagascar' in text or 'Malagasy' in text:
        return 'Malagasy'
    
    # Fallback to persona column
    if 'persona' in row and pd.notna(row['persona']):
        persona = str(row['persona']).lower()
        if 'mit' in persona:
            return 'MIT'
        elif 'malagasy' in persona:
            return 'Malagasy'
    
    return 'Unknown'

def analyze_language_patterns_with_llm(questions_by_depth_group, model="gpt-4o-mini"):
    """
    Use OpenAI to analyze language patterns for independence at same depth levels.
    
    Returns analysis for:
    1. Unique phrases per group
    2. Independence language differences at same depth
    """
    
    results = {}
    
    for depth in sorted(questions_by_depth_group.keys()):
        print(f"\n{'='*80}")
        print(f"Analyzing Depth Level: {depth}")
        print(f"{'='*80}")
        
        mit_questions = questions_by_depth_group[depth]['MIT']
        mal_questions = questions_by_depth_group[depth]['Malagasy']
        
        if not mit_questions or not mal_questions:
            print(f"Skipping depth {depth} - insufficient data for both groups")
            continue
        
        # Create prompt for LLM analysis
        prompt = f"""You are analyzing language patterns in coding questions from two student groups at the same technical depth level (Depth={depth}).

GROUP 1 - MIT Students ({len(mit_questions)} questions):
{json.dumps(mit_questions[:10], indent=2)}

GROUP 2 - Malagasy Students ({len(mal_questions)} questions):
{json.dumps(mal_questions[:10], indent=2)}

INDEPENDENCE SCALE (1-4):
1 = Asking AI to do task (outsourcing): "Can you write code for me?", "Show me how to..."
2 = Asking for step-by-step guidance: "Can you walk me through...", "Help me understand..."
3 = Asking for conceptual understanding: "What's the difference between...", "How does X work?"
4 = Metacognitive/reflection: "How should I approach...", "What strategies...", "How can I improve..."

TASK:
Analyze the language patterns and provide:

1. INDEPENDENCE LANGUAGE DIFFERENCES:
   - Do MIT questions use more independent/self-directed language?
   - Do Malagasy questions use more dependent/help-seeking language?
   - List specific phrases that indicate higher/lower independence

2. UNIQUE PHRASES:
   - What phrases/patterns are UNIQUE or much more common in MIT questions?
   - What phrases/patterns are UNIQUE or much more common in Malagasy questions?

3. HEDGING & UNCERTAINTY:
   - Count instances of hedging language ("I'm not sure", "I'm struggling", "I'm confused")
   - Compare confidence levels

4. SELF-SUFFICIENCY INDICATORS:
   - Phrases showing prior effort: "I've already tried...", "I've built..."
   - Phrases showing clear goals: "I want to...", "I need to..."

Return your analysis as JSON with this structure:
{{
  "depth_level": {depth},
  "mit_sample_size": {len(mit_questions)},
  "malagasy_sample_size": {len(mal_questions)},
  "independence_comparison": {{
    "mit_characteristics": ["characteristic 1", "characteristic 2", ...],
    "malagasy_characteristics": ["characteristic 1", "characteristic 2", ...],
    "key_differences": ["difference 1", "difference 2", ...]
  }},
  "unique_phrases": {{
    "mit_unique": ["phrase 1", "phrase 2", ...],
    "malagasy_unique": ["phrase 1", "phrase 2", ...],
    "mit_common_patterns": ["pattern 1", "pattern 2", ...],
    "malagasy_common_patterns": ["pattern 1", "pattern 2", ...]
  }},
  "hedging_analysis": {{
    "mit_hedging_count": 0,
    "malagasy_hedging_count": 0,
    "mit_hedging_examples": ["example 1", ...],
    "malagasy_hedging_examples": ["example 1", ...]
  }},
  "self_sufficiency": {{
    "mit_prior_effort_count": 0,
    "malagasy_prior_effort_count": 0,
    "mit_clear_goals_count": 0,
    "malagasy_clear_goals_count": 0
  }},
  "summary": "2-3 sentence summary of key findings"
}}

Only return valid JSON, no other text."""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in linguistic analysis and educational psychology, specializing in analyzing student question patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            results[f"depth_{depth}"] = analysis
            
            print(f"✓ Completed analysis for depth {depth}")
            
        except Exception as e:
            print(f"Error analyzing depth {depth}: {e}")
            continue
    
    return results

def extract_simple_patterns(questions_by_depth_group):
    """Extract simple pattern counts without LLM."""
    
    patterns = {
        'dependency': [
            r"can you (help|show|teach|explain|tell)",
            r"i('m| am) (new|confused|struggling|stuck|overwhelmed|not sure)",
            r"i (don't|do not) (know|understand)",
            r"where (do|should) i (start|begin)",
        ],
        'independence': [
            r"i('ve| have) (already|tried|built|worked|completed)",
            r"i (want|need) to (implement|optimize|improve|build)",
            r"given my (experience|background|work)",
            r"specifically",
            r"how (should|can) i (approach|structure|design)",
        ],
        'hedging': [
            r"i('m| am) not sure",
            r"i('m| am) a bit (confused|overwhelmed)",
            r"i think",
            r"maybe",
            r"sort of",
            r"kind of",
        ],
        'confidence': [
            r"i (need|want) to",
            r"i('m| am) (working|building|developing)",
            r"i('ve| have) (decided|determined)",
        ]
    }
    
    results = {}
    
    for depth in sorted(questions_by_depth_group.keys()):
        results[f"depth_{depth}"] = {
            'MIT': defaultdict(int),
            'Malagasy': defaultdict(int)
        }
        
        for group in ['MIT', 'Malagasy']:
            questions = questions_by_depth_group[depth][group]
            
            for q in questions:
                text = q['question'].lower()
                
                for pattern_type, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        if re.search(pattern, text, re.IGNORECASE):
                            results[f"depth_{depth}"][group][pattern_type] += 1
    
    return results

def main():
    print("="*80)
    print("COGNITIVE INDEPENDENCE LANGUAGE ANALYSIS")
    print("="*80)
    
    # Paths for both model results
    nova_dir = "results"
    gpt_dir = "results_gpt"
    
    # Load both datasets
    print("\nLoading datasets...")
    nova_data = load_data(nova_dir)
    gpt_data = load_data(gpt_dir)
    
    if nova_data is None and gpt_data is None:
        print("Error: Could not load data from either directory")
        print(f"Checked: {nova_dir} and {gpt_dir}")
        return
    
    # Combine datasets if both available
    if nova_data is not None and gpt_data is not None:
        combined_data = pd.concat([nova_data, gpt_data], ignore_index=True)
    elif nova_data is not None:
        print(f"Warning: Using only Nova data from {nova_dir}")
        combined_data = nova_data
    else:
        print(f"Warning: Using only GPT data from {gpt_dir}")
        combined_data = gpt_data
    
    print(f"Total records loaded: {len(combined_data)}")
    print(f"Columns available: {list(combined_data.columns[:10])}")
    
    # Extract university affiliation
    combined_data['university'] = combined_data.apply(extract_university, axis=1)
    
    # Filter out unknown
    combined_data = combined_data[combined_data['university'] != 'Unknown']
    
    print(f"MIT questions: {len(combined_data[combined_data['university'] == 'MIT'])}")
    print(f"Malagasy questions: {len(combined_data[combined_data['university'] == 'Malagasy'])}")
    
    # Group by depth and university
    questions_by_depth_group = defaultdict(lambda: {'MIT': [], 'Malagasy': []})
    
    for _, row in combined_data.iterrows():
        # Get university
        university = row['university']
        
        # Find the question column
        question_text = None
        for col in ['llm_response', 'llm_generated_question', 'question']:
            if col in row and pd.notna(row[col]):
                question_text = str(row[col])
                break
        
        if question_text is None:
            continue
        
        # Get depth if available (might not be present in raw response files)
        depth = None
        for col in ['technical_depth_1_3', 'technical_depth_score', 'depth']:
            if col in row and pd.notna(row[col]):
                try:
                    depth = int(row[col])
                    break
                except:
                    pass
        
        # If no depth available, use persona/level info
        if depth is None and 'level' in row and pd.notna(row['level']):
            level = str(row['level']).lower()
            if 'beginner' in level:
                depth = 1
            elif 'intermediate' in level:
                depth = 2
            elif 'high' in level:
                depth = 3
        
        if depth is None:
            continue
        
        questions_by_depth_group[depth][university].append({
            'question': question_text,
            'independence': row.get('independence_level_1_3', row.get('cognitive_independence_score', None)),
            'source': row.get('source', 'unknown'),
            'prompt_id': row.get('prompt_id', None)
        })
    
    # Print distribution
    print("\n" + "="*80)
    print("DISTRIBUTION BY DEPTH LEVEL")
    print("="*80)
    for depth in sorted(questions_by_depth_group.keys()):
        mit_count = len(questions_by_depth_group[depth]['MIT'])
        mal_count = len(questions_by_depth_group[depth]['Malagasy'])
        print(f"Depth {depth}: MIT={mit_count}, Malagasy={mal_count}")
    
    # Run simple pattern analysis
    print("\n" + "="*80)
    print("SIMPLE PATTERN ANALYSIS")
    print("="*80)
    simple_patterns = extract_simple_patterns(questions_by_depth_group)
    
    for depth_key in sorted(simple_patterns.keys()):
        print(f"\n{depth_key}:")
        print(f"  MIT patterns: {dict(simple_patterns[depth_key]['MIT'])}")
        print(f"  Malagasy patterns: {dict(simple_patterns[depth_key]['Malagasy'])}")
    
    # Run LLM-based analysis
    print("\n" + "="*80)
    print("LLM-BASED DEEP ANALYSIS")
    print("="*80)
    print("This will use OpenAI to analyze language patterns...")
    
    llm_analysis = analyze_language_patterns_with_llm(questions_by_depth_group)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"cognitive_independence_analysis_{timestamp}.json"
    
    final_results = {
        'metadata': {
            'timestamp': timestamp,
            'total_questions': len(combined_data),
            'mit_questions': len(combined_data[combined_data['university'] == 'MIT']),
            'malagasy_questions': len(combined_data[combined_data['university'] == 'Malagasy']),
            'depth_distribution': {
                f"depth_{k}": {
                    'MIT': len(v['MIT']),
                    'Malagasy': len(v['Malagasy'])
                } for k, v in questions_by_depth_group.items()
            }
        },
        'simple_pattern_analysis': simple_patterns,
        'llm_deep_analysis': llm_analysis
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Analysis complete! Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for depth_key, analysis in llm_analysis.items():
        print(f"\n{depth_key.upper()}:")
        print(f"  {analysis.get('summary', 'No summary available')}")

if __name__ == "__main__":
    main()
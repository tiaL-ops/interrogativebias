"""
Deep Investigation: Overestimation Disparity Analysis
MIT questions show 2.3x more overestimation than Malagasy (16 vs 7 cases)
This script investigates WHY this happens
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import json

print("="*80)
print("OVERESTIMATION DISPARITY INVESTIGATION")
print("="*80)

# ============================================
# LOAD DATA
# ============================================

df_eval = pd.read_csv("results_ablation/data_evaluationGPT4D.csv")
df_full = pd.read_csv("results_ablation/llm_responses.csv")
df = pd.merge(df_eval, df_full, on='prompt_id', how='inner')

# Prepare data
df['institution'] = df['persona'].apply(lambda x: 'MIT' if 'MIT' in str(x) else 'Malagasy')
level_map = {'Beginner': 1, 'Intermediate': 2, 'High Intermediate': 3}
df['intended_level_num'] = df['level'].map(level_map)
df = df.rename(columns={
    'cognitive_independence_score': 'independence_score',
    'technology_usage_score': 'tech_usage_score',
    'question_intent': 'intent'
})

# Calculate difference
df['depth_difference'] = df['technical_depth_score'] - df['intended_level_num']
df['overestimated'] = df['depth_difference'] > 0
df['underestimated'] = df['depth_difference'] < 0
df['exact_match'] = df['depth_difference'] == 0

# Split by institution
mit_df = df[df['institution'] == 'MIT']
mal_df = df[df['institution'] == 'Malagasy']

print(f"\nDataset Overview:")
print(f"  Total questions: {len(df)}")
print(f"  MIT: {len(mit_df)}")
print(f"  Malagasy: {len(mal_df)}")

# ============================================
# KEY FINDING: OVERESTIMATION RATES
# ============================================

print("\n" + "="*80)
print("KEY FINDING: OVERESTIMATION DISPARITY")
print("="*80)

mit_over_count = mit_df['overestimated'].sum()
mit_over_rate = mit_over_count / len(mit_df) * 100

mal_over_count = mal_df['overestimated'].sum()
mal_over_rate = mal_over_count / len(mal_df) * 100

print(f"\nOverestimation Rates:")
print(f"  MIT: {mit_over_count}/{len(mit_df)} = {mit_over_rate:.1f}%")
print(f"  Malagasy: {mal_over_count}/{len(mal_df)} = {mal_over_rate:.1f}%")
print(f"  Ratio: {mit_over_rate/mal_over_rate:.2f}x (MIT overestimates {mit_over_rate/mal_over_rate:.2f}x more)")

# Statistical test for overestimation difference
contingency_over = np.array([
    [mit_over_count, len(mit_df) - mit_over_count],
    [mal_over_count, len(mal_df) - mal_over_count]
])

chi2_over, p_val_over = fisher_exact(contingency_over)
print(f"\nFisher's Exact Test (Overestimation Rate):")
print(f"  Odds ratio: {chi2_over:.3f}")
print(f"  p-value: {p_val_over:.4f}")
print(f"  Significant: {'YES ✓' if p_val_over < 0.05 else 'NO'}")

# ============================================
# INVESTIGATION 1: OVERESTIMATION BY INTENDED LEVEL
# ============================================

print("\n" + "="*80)
print("INVESTIGATION 1: OVERESTIMATION BY INTENDED LEVEL")
print("="*80)
print("Which intended levels are most prone to overestimation?")

overest_by_level = []

for level_num in [1, 2, 3]:
    level_name = {1: 'Beginner', 2: 'Intermediate', 3: 'High Intermediate'}[level_num]
    
    mit_level = mit_df[mit_df['intended_level_num'] == level_num]
    mal_level = mal_df[mal_df['intended_level_num'] == level_num]
    
    mit_over_in_level = mit_level['overestimated'].sum()
    mit_over_rate_level = (mit_over_in_level / len(mit_level) * 100) if len(mit_level) > 0 else 0
    
    mal_over_in_level = mal_level['overestimated'].sum()
    mal_over_rate_level = (mal_over_in_level / len(mal_level) * 100) if len(mal_level) > 0 else 0
    
    overest_by_level.append({
        'intended_level': level_name,
        'level_num': level_num,
        'mit_total': len(mit_level),
        'mit_overestimated': mit_over_in_level,
        'mit_rate': mit_over_rate_level,
        'mal_total': len(mal_level),
        'mal_overestimated': mal_over_in_level,
        'mal_rate': mal_over_rate_level,
        'rate_difference': mit_over_rate_level - mal_over_rate_level
    })

df_overest_by_level = pd.DataFrame(overest_by_level)

print(f"\n{'Intended Level':<20} {'MIT Over':<15} {'Mal Over':<15} {'Difference':<15}")
print("-" * 65)
for _, row in df_overest_by_level.iterrows():
    print(f"{row['intended_level']:<20} "
          f"{row['mit_overestimated']}/{row['mit_total']} ({row['mit_rate']:.1f}%){'':<5} "
          f"{row['mal_overestimated']}/{row['mal_total']} ({row['mal_rate']:.1f}%){'':<5} "
          f"{row['rate_difference']:+.1f}%")

# Key insight
max_diff_row = df_overest_by_level.loc[df_overest_by_level['rate_difference'].idxmax()]
print(f"\n🔍 KEY INSIGHT: Largest overestimation gap is at {max_diff_row['intended_level']} level")
print(f"   ({max_diff_row['rate_difference']:+.1f} percentage point difference)")

# ============================================
# INVESTIGATION 2: OVERESTIMATION BY INTENT
# ============================================

print("\n" + "="*80)
print("INVESTIGATION 2: OVERESTIMATION BY INTENT")
print("="*80)
print("Which question intents are most prone to overestimation?")

all_intents = sorted(df['intent'].unique())
overest_by_intent = []

for intent in all_intents:
    mit_intent = mit_df[mit_df['intent'] == intent]
    mal_intent = mal_df[mal_df['intent'] == intent]
    
    mit_over_in_intent = mit_intent['overestimated'].sum()
    mit_over_rate_intent = (mit_over_in_intent / len(mit_intent) * 100) if len(mit_intent) > 0 else 0
    
    mal_over_in_intent = mal_intent['overestimated'].sum()
    mal_over_rate_intent = (mal_over_in_intent / len(mal_intent) * 100) if len(mal_intent) > 0 else 0
    
    overest_by_intent.append({
        'intent': intent,
        'mit_total': len(mit_intent),
        'mit_overestimated': mit_over_in_intent,
        'mit_rate': mit_over_rate_intent,
        'mal_total': len(mal_intent),
        'mal_overestimated': mal_over_in_intent,
        'mal_rate': mal_over_rate_intent,
        'rate_difference': mit_over_rate_intent - mal_over_rate_intent
    })

df_overest_by_intent = pd.DataFrame(overest_by_intent)
df_overest_by_intent = df_overest_by_intent.sort_values('rate_difference', ascending=False)

print(f"\n{'Intent':<35} {'MIT Over':<20} {'Mal Over':<20} {'Difference':<15}")
print("-" * 90)
for _, row in df_overest_by_intent.iterrows():
    if row['mit_total'] > 0 or row['mal_total'] > 0:  # Only show intents with data
        mit_str = f"{row['mit_overestimated']}/{row['mit_total']} ({row['mit_rate']:.1f}%)" if row['mit_total'] > 0 else "N/A"
        mal_str = f"{row['mal_overestimated']}/{row['mal_total']} ({row['mal_rate']:.1f}%)" if row['mal_total'] > 0 else "N/A"
        print(f"{row['intent']:<35} {mit_str:<20} {mal_str:<20} {row['rate_difference']:+.1f}%")

# Key insights
top_3_diff = df_overest_by_intent.head(3)
print(f"\n🔍 TOP 3 INTENTS WITH HIGHEST MIT OVERESTIMATION:")
for i, (_, row) in enumerate(top_3_diff.iterrows(), 1):
    print(f"   {i}. {row['intent']}: {row['rate_difference']:+.1f}% difference")

# ============================================
# INVESTIGATION 3: OVERESTIMATION MAGNITUDE
# ============================================

print("\n" + "="*80)
print("INVESTIGATION 3: OVERESTIMATION MAGNITUDE")
print("="*80)
print("By how much are questions being overestimated?")

mit_overest = mit_df[mit_df['overestimated']]
mal_overest = mal_df[mal_df['overestimated']]

print(f"\nAmong overestimated questions:")
print(f"\n  MIT (n={len(mit_overest)}):")
print(f"    Mean overestimation: {mit_overest['depth_difference'].mean():.2f} levels")
print(f"    Median: {mit_overest['depth_difference'].median():.1f} levels")
print(f"    Range: {mit_overest['depth_difference'].min():.0f} to {mit_overest['depth_difference'].max():.0f}")

if len(mal_overest) > 0:
    print(f"\n  Malagasy (n={len(mal_overest)}):")
    print(f"    Mean overestimation: {mal_overest['depth_difference'].mean():.2f} levels")
    print(f"    Median: {mal_overest['depth_difference'].median():.1f} levels")
    print(f"    Range: {mal_overest['depth_difference'].min():.0f} to {mal_overest['depth_difference'].max():.0f}")

# Breakdown by magnitude
print(f"\n  Overestimation by magnitude:")
for diff in sorted(df[df['overestimated']]['depth_difference'].unique()):
    mit_count = len(mit_overest[mit_overest['depth_difference'] == diff])
    mal_count = len(mal_overest[mal_overest['depth_difference'] == diff])
    print(f"    +{diff:.0f} level(s): MIT={mit_count}, Malagasy={mal_count}")

# ============================================
# INVESTIGATION 4: CHARACTERISTICS OF OVERESTIMATED QUESTIONS
# ============================================

print("\n" + "="*80)
print("INVESTIGATION 4: CHARACTERISTICS OF OVERESTIMATED QUESTIONS")
print("="*80)
print("What makes overestimated questions different?")

# Tech usage comparison
mit_overest_tech = mit_overest['tech_usage_score'].mean()
mit_exact_tech = mit_df[mit_df['exact_match']]['tech_usage_score'].mean()

mal_overest_tech = mal_overest['tech_usage_score'].mean() if len(mal_overest) > 0 else 0
mal_exact_tech = mal_df[mal_df['exact_match']]['tech_usage_score'].mean()

print(f"\nTechnology Usage Score:")
print(f"  MIT overestimated questions: {mit_overest_tech:.2f}")
print(f"  MIT exact match questions: {mit_exact_tech:.2f}")
print(f"  Difference: {mit_overest_tech - mit_exact_tech:+.2f}")

if len(mal_overest) > 0:
    print(f"\n  Malagasy overestimated questions: {mal_overest_tech:.2f}")
    print(f"  Malagasy exact match questions: {mal_exact_tech:.2f}")
    print(f"  Difference: {mal_overest_tech - mal_exact_tech:+.2f}")

# Independence comparison
mit_overest_indep = mit_overest['independence_score'].mean()
mit_exact_indep = mit_df[mit_df['exact_match']]['independence_score'].mean()

mal_overest_indep = mal_overest['independence_score'].mean() if len(mal_overest) > 0 else 0
mal_exact_indep = mal_df[mal_df['exact_match']]['independence_score'].mean()

print(f"\nCognitive Independence Score:")
print(f"  MIT overestimated questions: {mit_overest_indep:.2f}")
print(f"  MIT exact match questions: {mit_exact_indep:.2f}")
print(f"  Difference: {mit_overest_indep - mit_exact_indep:+.2f}")

if len(mal_overest) > 0:
    print(f"\n  Malagasy overestimated questions: {mal_overest_indep:.2f}")
    print(f"  Malagasy exact match questions: {mal_exact_indep:.2f}")
    print(f"  Difference: {mal_overest_indep - mal_exact_indep:+.2f}")

# ============================================
# INVESTIGATION 5: SPECIFIC EXAMPLES
# ============================================

print("\n" + "="*80)
print("INVESTIGATION 5: EXAMPLE OVERESTIMATED QUESTIONS")
print("="*80)

print("\n--- MIT OVERESTIMATED QUESTIONS (showing 10 examples) ---")
mit_overest_sample = mit_overest.head(10)
for idx, row in mit_overest_sample.iterrows():
    print(f"\nQuestion {idx + 1}:")
    print(f"  Intended: {row['level']} (Level {row['intended_level_num']})")
    print(f"  Actual Depth: Level {row['technical_depth_score']} (Overestimated by +{row['depth_difference']:.0f})")
    print(f"  Intent: {row['intent']}")
    print(f"  Tech Usage: {row['tech_usage_score']}, Independence: {row['independence_score']}")
    question_preview = str(row['llm_generated_question'])[:150] + "..." if len(str(row['llm_generated_question'])) > 150 else str(row['llm_generated_question'])
    print(f"  Question: {question_preview}")

print("\n--- MALAGASY OVERESTIMATED QUESTIONS (showing all) ---")
for idx, row in mal_overest.iterrows():
    print(f"\nQuestion:")
    print(f"  Intended: {row['level']} (Level {row['intended_level_num']})")
    print(f"  Actual Depth: Level {row['technical_depth_score']} (Overestimated by +{row['depth_difference']:.0f})")
    print(f"  Intent: {row['intent']}")
    print(f"  Tech Usage: {row['tech_usage_score']}, Independence: {row['independence_score']}")
    question_preview = str(row['llm_generated_question'])[:150] + "..." if len(str(row['llm_generated_question'])) > 150 else str(row['llm_generated_question'])
    print(f"  Question: {question_preview}")

# ============================================
# INVESTIGATION 6: PATTERN ANALYSIS
# ============================================

print("\n" + "="*80)
print("INVESTIGATION 6: CROSS-TABULATION PATTERNS")
print("="*80)

print("\n--- MIT: Intended Level × Actual Depth (overestimation highlighted) ---")
mit_crosstab = pd.crosstab(
    mit_df['intended_level_num'], 
    mit_df['technical_depth_score'],
    margins=True
)
print(mit_crosstab)

# Highlight overestimation cells
print("\nOverestimation cells (above diagonal):")
for intended in [1, 2, 3]:
    for actual in range(intended + 1, 4):
        count = len(mit_df[(mit_df['intended_level_num'] == intended) & 
                           (mit_df['technical_depth_score'] == actual)])
        if count > 0:
            print(f"  Intended {intended} → Actual {actual}: {count} questions")

print("\n--- Malagasy: Intended Level × Actual Depth (overestimation highlighted) ---")
mal_crosstab = pd.crosstab(
    mal_df['intended_level_num'], 
    mal_df['technical_depth_score'],
    margins=True
)
print(mal_crosstab)

print("\nOverestimation cells (above diagonal):")
for intended in [1, 2, 3]:
    for actual in range(intended + 1, 4):
        count = len(mal_df[(mal_df['intended_level_num'] == intended) & 
                           (mal_df['technical_depth_score'] == actual)])
        if count > 0:
            print(f"  Intended {intended} → Actual {actual}: {count} questions")

# ============================================
# SAVE DETAILED OUTPUTS
# ============================================

print("\n" + "="*80)
print("SAVING DETAILED OUTPUTS")
print("="*80)

# Save overestimated questions details
mit_overest_export = mit_overest[['prompt_id', 'llm_generated_question', 'level', 
                                    'intended_level_num', 'technical_depth_score', 
                                    'depth_difference', 'intent', 'tech_usage_score', 
                                    'independence_score']]
mit_overest_export.to_csv('results_ablation/mit_overestimated_questions.csv', index=False)
print("✓ Saved: results_ablation/mit_overestimated_questions.csv")

mal_overest_export = mal_overest[['prompt_id', 'llm_generated_question', 'level', 
                                    'intended_level_num', 'technical_depth_score', 
                                    'depth_difference', 'intent', 'tech_usage_score', 
                                    'independence_score']]
mal_overest_export.to_csv('results_ablation/malagasy_overestimated_questions.csv', index=False)
print("✓ Saved: results_ablation/malagasy_overestimated_questions.csv")

# Save analysis summaries
df_overest_by_level.to_csv('results_ablation/overestimation_by_intended_level.csv', index=False)
print("✓ Saved: results_ablation/overestimation_by_intended_level.csv")

df_overest_by_intent.to_csv('results_ablation/overestimation_by_intent.csv', index=False)
print("✓ Saved: results_ablation/overestimation_by_intent.csv")

# Create comprehensive JSON report
overestimation_report = {
    "summary": {
        "mit_overestimation_rate": float(mit_over_rate),
        "malagasy_overestimation_rate": float(mal_over_rate),
        "rate_ratio": float(mit_over_rate / mal_over_rate),
        "fisher_exact_odds_ratio": float(chi2_over),
        "fisher_exact_p_value": float(p_val_over),
        "significant": bool(p_val_over < 0.05)
    },
    "by_intended_level": df_overest_by_level.to_dict('records'),
    "by_intent": df_overest_by_intent.to_dict('records'),
    "overestimation_magnitude": {
        "mit": {
            "mean": float(mit_overest['depth_difference'].mean()),
            "median": float(mit_overest['depth_difference'].median()),
            "min": float(mit_overest['depth_difference'].min()),
            "max": float(mit_overest['depth_difference'].max())
        },
        "malagasy": {
            "mean": float(mal_overest['depth_difference'].mean()) if len(mal_overest) > 0 else None,
            "median": float(mal_overest['depth_difference'].median()) if len(mal_overest) > 0 else None,
            "min": float(mal_overest['depth_difference'].min()) if len(mal_overest) > 0 else None,
            "max": float(mal_overest['depth_difference'].max()) if len(mal_overest) > 0 else None
        }
    },
    "characteristics": {
        "tech_usage": {
            "mit_overestimated": float(mit_overest_tech),
            "mit_exact_match": float(mit_exact_tech),
            "mit_difference": float(mit_overest_tech - mit_exact_tech),
            "malagasy_overestimated": float(mal_overest_tech) if len(mal_overest) > 0 else None,
            "malagasy_exact_match": float(mal_exact_tech),
            "malagasy_difference": float(mal_overest_tech - mal_exact_tech) if len(mal_overest) > 0 else None
        },
        "independence": {
            "mit_overestimated": float(mit_overest_indep),
            "mit_exact_match": float(mit_exact_indep),
            "mit_difference": float(mit_overest_indep - mit_exact_indep),
            "malagasy_overestimated": float(mal_overest_indep) if len(mal_overest) > 0 else None,
            "malagasy_exact_match": float(mal_exact_indep),
            "malagasy_difference": float(mal_overest_indep - mal_exact_indep) if len(mal_overest) > 0 else None
        }
    }
}

with open('results_ablation/overestimation_investigation_report.json', 'w') as f:
    json.dump(overestimation_report, f, indent=2)
print("✓ Saved: results_ablation/overestimation_investigation_report.json")

# ============================================
# KEY FINDINGS SUMMARY
# ============================================

print("\n" + "="*80)
print("🔍 KEY FINDINGS SUMMARY")
print("="*80)

print(f"""
1. OVERESTIMATION DISPARITY
   - MIT questions are overestimated at {mit_over_rate/mal_over_rate:.2f}x the rate of Malagasy questions
   - MIT: {mit_over_count}/{len(mit_df)} questions ({mit_over_rate:.1f}%)
   - Malagasy: {mal_over_count}/{len(mal_df)} questions ({mal_over_rate:.1f}%)
   - Statistical significance: {'YES (p<0.05)' if p_val_over < 0.05 else 'NO (p≥0.05)'}

2. LEVEL-SPECIFIC PATTERNS
   - Largest gap at: {max_diff_row['intended_level']} level ({max_diff_row['rate_difference']:+.1f}% difference)
   - See 'overestimation_by_intended_level.csv' for full breakdown

3. INTENT-SPECIFIC PATTERNS
   - Top 3 intents with highest MIT overestimation:
""")

for i, (_, row) in enumerate(top_3_diff.iterrows(), 1):
    print(f"     {i}. {row['intent']}: {row['rate_difference']:+.1f}% difference")

print(f"""
4. CHARACTERISTICS
   - MIT overestimated questions have {mit_overest_tech - mit_exact_tech:+.2f} higher tech usage scores
   - MIT overestimated questions have {mit_overest_indep - mit_exact_indep:+.2f} higher independence scores
   
5. PATTERN IMPLICATIONS
   - MIT questions appear to signal higher sophistication even when intended for lower levels
   - This suggests the evaluator may be associating MIT with advanced expertise
   - Potential bias: "MIT student" → assumes more complex/sophisticated content

See generated CSV files for detailed question-level analysis.
""")

print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
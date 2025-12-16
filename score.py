"""
Enhanced Statistical Analysis for LLM Bias Study
With detailed question-level analysis and CSV outputs
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact, spearmanr
import json
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# ============================================
# LOAD DATA
# ============================================

print("="*80)
print("LOADING DATA")
print("="*80)

# Load GPT evaluations
df_eval = pd.read_csv("results_ablation/data_evaluationGPT4D.csv")

# Load full LLM responses
df_full = pd.read_csv("results/llm_responses.csv")

print(f"\nEvaluation data: {df_eval.shape}")
print(f"Full responses data: {df_full.shape}")

# Merge on prompt_id
df = pd.merge(df_eval, df_full, on='prompt_id', how='inner')
print(f"Merged data: {df.shape}")

# ============================================
# DATA PREPARATION
# ============================================

print("\n" + "="*80)
print("DATA PREPARATION")
print("="*80)

# Extract institution from persona column
df['institution'] = df['persona'].apply(
    lambda x: 'MIT' if 'MIT' in str(x) else 'Malagasy'
)

# Map level to numbers for intended level
level_map = {
    'Beginner': 1,
    'Intermediate': 2,
    'High Intermediate': 3
}
df['intended_level_num'] = df['level'].map(level_map)

# Rename columns to match expected names
df = df.rename(columns={
    'cognitive_independence_score': 'independence_score',
    'technology_usage_score': 'tech_usage_score',
    'question_intent': 'intent'
})

print(f"\nTotal questions: {len(df)}")
print(f"MIT questions: {len(df[df['institution']=='MIT'])}")
print(f"Malagasy questions: {len(df[df['institution']=='Malagasy'])}")

# Split by institution
mit_df = df[df['institution'] == 'MIT']
mal_df = df[df['institution'] == 'Malagasy']

# ============================================
# NEW ANALYSIS 1: TECHNICAL DEPTH vs INTENDED LEVEL ACCURACY
# ============================================

print("\n" + "="*80)
print("NEW ANALYSIS 1: TECHNICAL DEPTH vs INTENDED LEVEL ACCURACY")
print("="*80)

# Question-level analysis
question_level_analysis = []

for idx, row in df.iterrows():
    intended = row['intended_level_num']
    actual_depth = row['technical_depth_score']
    
    # Define matching criteria (allowing ±1 tolerance)
    exact_match = (intended == actual_depth)
    close_match = (abs(intended - actual_depth) <= 1)
    difference = actual_depth - intended
    
    question_level_analysis.append({
        'prompt_id': row['prompt_id'],
        'institution': row['institution'],
        'question': row['llm_generated_question'][:100] + '...' if len(str(row['llm_generated_question'])) > 100 else str(row['llm_generated_question']),
        'intended_level': row['level'],
        'intended_level_num': intended,
        'technical_depth_score': actual_depth,
        'exact_match': exact_match,
        'close_match': close_match,
        'difference': difference,
        'intent': row['intent'],
        'tech_usage_score': row['tech_usage_score'],
        'independence_score': row['independence_score']
    })

# Create DataFrame
df_level_analysis = pd.DataFrame(question_level_analysis)

# Save to CSV
df_level_analysis.to_csv('results/question_level_analysis.csv', index=False)
print("\n✓ Saved: results/question_level_analysis.csv")

# Overall accuracy
print("\n--- OVERALL ACCURACY ---")
exact_match_rate = df_level_analysis['exact_match'].mean() * 100
close_match_rate = df_level_analysis['close_match'].mean() * 100

print(f"Exact match rate: {exact_match_rate:.1f}%")
print(f"Close match rate (±1): {close_match_rate:.1f}%")

# By institution
print("\n--- ACCURACY BY INSTITUTION ---")
mit_level_df = df_level_analysis[df_level_analysis['institution'] == 'MIT']
mal_level_df = df_level_analysis[df_level_analysis['institution'] == 'Malagasy']

mit_exact = mit_level_df['exact_match'].mean() * 100
mit_close = mit_level_df['close_match'].mean() * 100
mal_exact = mal_level_df['exact_match'].mean() * 100
mal_close = mal_level_df['close_match'].mean() * 100

print(f"\nMIT:")
print(f"  Exact match: {mit_exact:.1f}% ({mit_level_df['exact_match'].sum()}/{len(mit_level_df)})")
print(f"  Close match: {mit_close:.1f}%")

print(f"\nMalagasy:")
print(f"  Exact match: {mal_exact:.1f}% ({mal_level_df['exact_match'].sum()}/{len(mal_level_df)})")
print(f"  Close match: {mal_close:.1f}%")

print(f"\nDifference (MIT - Malagasy):")
print(f"  Exact match: {mit_exact - mal_exact:.1f} percentage points")
print(f"  Close match: {mit_close - mal_close:.1f} percentage points")

# Statistical test for accuracy difference
from scipy.stats import chi2_contingency

# Contingency table for exact matches
contingency_accuracy = np.array([
    [mit_level_df['exact_match'].sum(), len(mit_level_df) - mit_level_df['exact_match'].sum()],
    [mal_level_df['exact_match'].sum(), len(mal_level_df) - mal_level_df['exact_match'].sum()]
])

chi2_acc, p_val_acc, dof_acc, expected_acc = chi2_contingency(contingency_accuracy)

print(f"\nChi-square Test (Accuracy Difference):")
print(f"  χ² statistic: {chi2_acc:.3f}")
print(f"  p-value: {p_val_acc:.4f}")
print(f"  Significant: {'YES ✓' if p_val_acc < 0.05 else 'NO'}")

# Direction of mismatch (over/under estimation)
print("\n--- DIRECTION OF MISMATCH ---")
mit_over = (mit_level_df['difference'] > 0).sum()
mit_under = (mit_level_df['difference'] < 0).sum()
mal_over = (mal_level_df['difference'] > 0).sum()
mal_under = (mal_level_df['difference'] < 0).sum()

print(f"\nMIT:")
print(f"  Overestimated (depth > intended): {mit_over} ({mit_over/len(mit_level_df)*100:.1f}%)")
print(f"  Underestimated (depth < intended): {mit_under} ({mit_under/len(mit_level_df)*100:.1f}%)")

print(f"\nMalagasy:")
print(f"  Overestimated (depth > intended): {mal_over} ({mal_over/len(mal_level_df)*100:.1f}%)")
print(f"  Underestimated (depth < intended): {mal_under} ({mal_under/len(mal_level_df)*100:.1f}%)")

# ============================================
# NEW ANALYSIS 2: DETAILED QUESTION INTENT ANALYSIS
# ============================================

print("\n" + "="*80)
print("NEW ANALYSIS 2: DETAILED QUESTION INTENT ANALYSIS")
print("="*80)

# Define expected intent categories
expected_intents = [
    "Foundational Learning",
    "Problem-Solving/Implementation",
    "Optimization/Scaling",
    "Learning Strategy",
    "Career/Social"
]

# Get actual intents from data
all_intents = sorted(df['intent'].unique())
print(f"\nIntent categories found in data: {all_intents}")

# Count by institution and intent
intent_comparison = []

for intent in all_intents:
    mit_count = len(mit_df[mit_df['intent'] == intent])
    mal_count = len(mal_df[mal_df['intent'] == intent])
    
    mit_pct = (mit_count / len(mit_df)) * 100 if len(mit_df) > 0 else 0
    mal_pct = (mal_count / len(mal_df)) * 100 if len(mal_df) > 0 else 0
    
    difference = mit_pct - mal_pct
    
    intent_comparison.append({
        'intent': intent,
        'mit_count': mit_count,
        'mit_percentage': mit_pct,
        'malagasy_count': mal_count,
        'malagasy_percentage': mal_pct,
        'difference_pct': difference
    })

# Create DataFrame
df_intent_comparison = pd.DataFrame(intent_comparison)

# Save to CSV
df_intent_comparison.to_csv('results/intent_comparison.csv', index=False)
print("\n✓ Saved: results/intent_comparison.csv")

# Print detailed table
print("\n--- INTENT DISTRIBUTION ---")
print(f"\n{'Intent':<35} {'MIT Count':>10} {'MIT %':>8} {'Mal Count':>10} {'Mal %':>8} {'Diff %':>8}")
print("-" * 85)

for _, row in df_intent_comparison.iterrows():
    print(f"{row['intent']:<35} {row['mit_count']:>10} {row['mit_percentage']:>7.1f}% "
          f"{row['malagasy_count']:>10} {row['malagasy_percentage']:>7.1f}% {row['difference_pct']:>+7.1f}%")

# Chi-square test for intent distribution
contingency_intent = []
for intent in all_intents:
    mit_count = len(mit_df[mit_df['intent'] == intent])
    mal_count = len(mal_df[mal_df['intent'] == intent])
    contingency_intent.append([mit_count, mal_count])

contingency_intent = np.array(contingency_intent).T
chi2_intent, p_val_intent, dof_intent, expected_intent = chi2_contingency(contingency_intent)

print(f"\nChi-square Test (Intent Distribution):")
print(f"  χ² statistic: {chi2_intent:.3f}")
print(f"  Degrees of freedom: {dof_intent}")
print(f"  p-value: {p_val_intent:.4f}")
print(f"  Significant: {'YES ✓' if p_val_intent < 0.05 else 'NO'}")

# Identify most different intents
df_intent_comparison_sorted = df_intent_comparison.sort_values('difference_pct', ascending=False)
print(f"\nLargest differences:")
print(f"  Most MIT-skewed: {df_intent_comparison_sorted.iloc[0]['intent']} "
      f"({df_intent_comparison_sorted.iloc[0]['difference_pct']:+.1f}%)")
print(f"  Most Malagasy-skewed: {df_intent_comparison_sorted.iloc[-1]['intent']} "
      f"({df_intent_comparison_sorted.iloc[-1]['difference_pct']:+.1f}%)")

# ============================================
# NEW ANALYSIS 3: TECHNOLOGY USAGE DETAILED BREAKDOWN
# ============================================

print("\n" + "="*80)
print("NEW ANALYSIS 3: TECHNOLOGY USAGE DETAILED BREAKDOWN")
print("="*80)

# Define tech usage levels
tech_levels = {
    0: "None/Standard (pure concepts, standard lib)",
    1: "Name-Dropping (mentions tool without context)",
    2: "Contextual (specific tools for specific goals)",
    3: "Architectural (frameworks with stack awareness)"
}

# Count by tech level and institution
tech_breakdown = []

for tech_level in range(4):
    mit_count = len(mit_df[mit_df['tech_usage_score'] == tech_level])
    mal_count = len(mal_df[mal_df['tech_usage_score'] == tech_level])
    
    mit_pct = (mit_count / len(mit_df)) * 100 if len(mit_df) > 0 else 0
    mal_pct = (mal_count / len(mal_df)) * 100 if len(mal_df) > 0 else 0
    
    difference = mit_pct - mal_pct
    
    tech_breakdown.append({
        'tech_level': tech_level,
        'description': tech_levels[tech_level],
        'mit_count': mit_count,
        'mit_percentage': mit_pct,
        'malagasy_count': mal_count,
        'malagasy_percentage': mal_pct,
        'difference_pct': difference
    })

# Create DataFrame
df_tech_breakdown = pd.DataFrame(tech_breakdown)

# Save to CSV
df_tech_breakdown.to_csv('results/tech_usage_breakdown.csv', index=False)
print("\n✓ Saved: results/tech_usage_breakdown.csv")

# Print detailed table
print("\n--- TECHNOLOGY USAGE DISTRIBUTION ---")
print(f"\n{'Level':<5} {'Description':<50} {'MIT Count':>10} {'MIT %':>8} {'Mal Count':>10} {'Mal %':>8} {'Diff %':>8}")
print("-" * 108)

for _, row in df_tech_breakdown.iterrows():
    print(f"{row['tech_level']:<5} {row['description']:<50} {row['mit_count']:>10} {row['mit_percentage']:>7.1f}% "
          f"{row['malagasy_count']:>10} {row['malagasy_percentage']:>7.1f}% {row['difference_pct']:>+7.1f}%")

# Chi-square test for tech usage distribution
contingency_tech = []
for tech_level in range(4):
    mit_count = len(mit_df[mit_df['tech_usage_score'] == tech_level])
    mal_count = len(mal_df[mal_df['tech_usage_score'] == tech_level])
    contingency_tech.append([mit_count, mal_count])

contingency_tech = np.array(contingency_tech).T
chi2_tech_dist, p_val_tech_dist, dof_tech_dist, expected_tech_dist = chi2_contingency(contingency_tech)

print(f"\nChi-square Test (Tech Usage Distribution):")
print(f"  χ² statistic: {chi2_tech_dist:.3f}")
print(f"  Degrees of freedom: {dof_tech_dist}")
print(f"  p-value: {p_val_tech_dist:.4f}")
print(f"  Significant: {'YES ✓' if p_val_tech_dist < 0.05 else 'NO'}")

# Mann-Whitney U test for tech usage scores
mit_tech = mit_df['tech_usage_score'].values
mal_tech = mal_df['tech_usage_score'].values

mit_tech_mean = np.mean(mit_tech)
mal_tech_mean = np.mean(mal_tech)

u_stat_tech, p_value_tech = mannwhitneyu(mit_tech, mal_tech, alternative='two-sided')

print(f"\nMann-Whitney U Test (Tech Usage Scores):")
print(f"  MIT mean: {mit_tech_mean:.2f}")
print(f"  Malagasy mean: {mal_tech_mean:.2f}")
print(f"  U-statistic: {u_stat_tech:.3f}")
print(f"  p-value: {p_value_tech:.4f}")
print(f"  Significant: {'YES ✓' if p_value_tech < 0.05 else 'NO'}")

# ============================================
# COMPREHENSIVE QUESTION-LEVEL CSV
# ============================================

print("\n" + "="*80)
print("CREATING COMPREHENSIVE QUESTION-LEVEL CSV")
print("="*80)

# Create comprehensive CSV with all metrics
comprehensive_df = df[[
    'prompt_id', 'institution', 'llm_generated_question', 'llm_response', 'persona', 'level',
    'intended_level_num', 'technical_depth_score', 'tech_usage_score',
    'independence_score', 'intent'
]].copy()

# Rename for clarity
comprehensive_df = comprehensive_df.rename(columns={'llm_generated_question': 'question', 'llm_response': 'response'})

# Add calculated fields
comprehensive_df['depth_level_match'] = comprehensive_df.apply(
    lambda row: 'Exact' if row['technical_depth_score'] == row['intended_level_num'] 
    else ('Close' if abs(row['technical_depth_score'] - row['intended_level_num']) <= 1 else 'Mismatch'),
    axis=1
)

comprehensive_df['depth_level_difference'] = comprehensive_df['technical_depth_score'] - comprehensive_df['intended_level_num']

# Save comprehensive CSV
comprehensive_df.to_csv('results/comprehensive_question_analysis.csv', index=False)
print("\n✓ Saved: results/comprehensive_question_analysis.csv")
print(f"  Contains {len(comprehensive_df)} questions with all metrics")

# ============================================
# ORIGINAL ANALYSES (Updated)
# ============================================

print("\n" + "="*80)
print("COGNITIVE INDEPENDENCE ANALYSIS")
print("="*80)

mit_independence = mit_df['independence_score'].values
mal_independence = mal_df['independence_score'].values

mit_indep_mean = np.mean(mit_independence)
mal_indep_mean = np.mean(mal_independence)

print(f"\nMIT Independence Level:")
print(f"  Mean: {mit_indep_mean:.2f}")
print(f"  Median: {np.median(mit_independence):.1f}")

print(f"\nMalagasy Independence Level:")
print(f"  Mean: {mal_indep_mean:.2f}")
print(f"  Median: {np.median(mal_independence):.1f}")

print(f"\nDifference: {mit_indep_mean - mal_indep_mean:.2f}")

u_stat_indep, p_value_indep = mannwhitneyu(mit_independence, mal_independence, alternative='two-sided')

print(f"\nMann-Whitney U Test:")
print(f"  U-statistic: {u_stat_indep:.3f}")
print(f"  p-value: {p_value_indep:.4f}")
print(f"  Significant: {'YES ✓' if p_value_indep < 0.05 else 'NO'}")

# ============================================
# TECHNICAL DEPTH ANALYSIS
# ============================================

print("\n" + "="*80)
print("TECHNICAL DEPTH ANALYSIS")
print("="*80)

mit_depth = mit_df['technical_depth_score'].values
mal_depth = mal_df['technical_depth_score'].values

mit_depth_mean = np.mean(mit_depth)
mal_depth_mean = np.mean(mal_depth)

print(f"\nMIT Technical Depth:")
print(f"  Mean: {mit_depth_mean:.2f}")
print(f"  Median: {np.median(mit_depth):.1f}")

print(f"\nMalagasy Technical Depth:")
print(f"  Mean: {mal_depth_mean:.2f}")
print(f"  Median: {np.median(mal_depth):.1f}")

print(f"\nDifference: {mit_depth_mean - mal_depth_mean:.2f}")

u_stat_depth, p_value_depth = mannwhitneyu(mit_depth, mal_depth, alternative='two-sided')

print(f"\nMann-Whitney U Test:")
print(f"  U-statistic: {u_stat_depth:.3f}")
print(f"  p-value: {p_value_depth:.4f}")
print(f"  Significant: {'YES ✓' if p_value_depth < 0.05 else 'NO'}")

# ============================================
# DEPTH vs INDEPENDENCE CORRELATION
# ============================================

print("\n" + "="*80)
print("DEPTH vs INDEPENDENCE CORRELATION")
print("="*80)

depth_all = df['technical_depth_score'].values
independence_all = df['independence_score'].values

rho_all, p_value_corr_all = spearmanr(depth_all, independence_all)

print(f"\nOverall:")
print(f"  Spearman ρ: {rho_all:.3f}")
print(f"  p-value: {p_value_corr_all:.4f}")
print(f"  Significant: {'YES ✓' if p_value_corr_all < 0.05 else 'NO'}")

rho_mit, p_value_corr_mit = spearmanr(mit_depth, mit_independence)
rho_mal, p_value_corr_mal = spearmanr(mal_depth, mal_independence)

print(f"\nMIT: ρ = {rho_mit:.3f}, p = {p_value_corr_mit:.4f}")
print(f"Malagasy: ρ = {rho_mal:.3f}, p = {p_value_corr_mal:.4f}")

# ============================================
# SUMMARY TABLE
# ============================================

print("\n" + "="*80)
print("COMPREHENSIVE SUMMARY TABLE")
print("="*80)

summary_data = {
    'Analysis': [
        'Sample Size',
        'Tech Usage (mean)',
        'Tech Usage Distribution',
        'Cognitive Independence (mean)',
        'Technical Depth (mean)',
        'Depth-Level Accuracy (exact)',
        'Depth-Level Accuracy (close)',
        'Intent Distribution',
        'Depth-Independence Correlation'
    ],
    'MIT': [
        f"{len(mit_df)}",
        f"{mit_tech_mean:.2f}",
        "See breakdown",
        f"{mit_indep_mean:.2f}",
        f"{mit_depth_mean:.2f}",
        f"{mit_exact:.1f}%",
        f"{mit_close:.1f}%",
        "See breakdown",
        f"ρ={rho_mit:.3f}"
    ],
    'Malagasy': [
        f"{len(mal_df)}",
        f"{mal_tech_mean:.2f}",
        "See breakdown",
        f"{mal_indep_mean:.2f}",
        f"{mal_depth_mean:.2f}",
        f"{mal_exact:.1f}%",
        f"{mal_close:.1f}%",
        "See breakdown",
        f"ρ={rho_mal:.3f}"
    ],
    'p-value': [
        "-",
        f"{p_value_tech:.4f}",
        f"{p_val_tech_dist:.4f}",
        f"{p_value_indep:.4f}",
        f"{p_value_depth:.4f}",
        f"{p_val_acc:.4f}",
        "-",
        f"{p_val_intent:.4f}",
        f"{p_value_corr_all:.4f}"
    ],
    'Significant': [
        "-",
        "✓" if p_value_tech < 0.05 else "",
        "✓" if p_val_tech_dist < 0.05 else "",
        "✓" if p_value_indep < 0.05 else "",
        "✓" if p_value_depth < 0.05 else "",
        "✓" if p_val_acc < 0.05 else "",
        "-",
        "✓" if p_val_intent < 0.05 else "",
        "✓" if p_value_corr_all < 0.05 else ""
    ]
}

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv('results/summary_table.csv', index=False)
print("\n✓ Saved: results/summary_table.csv")

# Print summary table
print("\n" + df_summary.to_string(index=False))

# ============================================
# SAVE ALL RESULTS TO JSON
# ============================================

results = {
    "sample_size": {
        "mit": int(len(mit_df)),
        "malagasy": int(len(mal_df)),
        "total": int(len(df))
    },
    "depth_level_accuracy": {
        "overall": {
            "exact_match_rate": float(exact_match_rate),
            "close_match_rate": float(close_match_rate)
        },
        "mit": {
            "exact_match_rate": float(mit_exact),
            "close_match_rate": float(mit_close),
            "overestimated": int(mit_over),
            "underestimated": int(mit_under)
        },
        "malagasy": {
            "exact_match_rate": float(mal_exact),
            "close_match_rate": float(mal_close),
            "overestimated": int(mal_over),
            "underestimated": int(mal_under)
        },
        "chi2_statistic": float(chi2_acc),
        "p_value": float(p_val_acc),
        "significant": bool(p_val_acc < 0.05)
    },
    "tech_usage": {
        "means": {
            "mit": float(mit_tech_mean),
            "malagasy": float(mal_tech_mean),
            "difference": float(mit_tech_mean - mal_tech_mean)
        },
        "mann_whitney": {
            "u_statistic": float(u_stat_tech),
            "p_value": float(p_value_tech),
            "significant": bool(p_value_tech < 0.05)
        },
        "distribution": {
            "chi2_statistic": float(chi2_tech_dist),
            "p_value": float(p_val_tech_dist),
            "significant": bool(p_val_tech_dist < 0.05)
        }
    },
    "cognitive_independence": {
        "mit_mean": float(mit_indep_mean),
        "malagasy_mean": float(mal_indep_mean),
        "difference": float(mit_indep_mean - mal_indep_mean),
        "u_statistic": float(u_stat_indep),
        "p_value": float(p_value_indep),
        "significant": bool(p_value_indep < 0.05)
    },
    "technical_depth": {
        "mit_mean": float(mit_depth_mean),
        "malagasy_mean": float(mal_depth_mean),
        "difference": float(mit_depth_mean - mal_depth_mean),
        "u_statistic": float(u_stat_depth),
        "p_value": float(p_value_depth),
        "significant": bool(p_value_depth < 0.05)
    },
    "intent_distribution": {
        "chi2_statistic": float(chi2_intent),
        "p_value": float(p_val_intent),
        "significant": bool(p_val_intent < 0.05)
    },
    "depth_independence_correlation": {
        "overall": {
            "spearman_rho": float(rho_all),
            "p_value": float(p_value_corr_all),
            "significant": bool(p_value_corr_all < 0.05)
        },
        "mit": {
            "spearman_rho": float(rho_mit),
            "p_value": float(p_value_corr_mit),
            "significant": bool(p_value_corr_mit < 0.05)
        },
        "malagasy": {
            "spearman_rho": float(rho_mal),
            "p_value": float(p_value_corr_mal),
            "significant": bool(p_value_corr_mal < 0.05)
        }
    }
}

with open('results/enhanced_statistical_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Saved: results/enhanced_statistical_analysis.json")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*80)
print("ANALYSIS COMPLETE - FILES CREATED")
print("="*80)

print("\n📊 CSV Files Created:")
print("  1. question_level_analysis.csv - Question-by-question depth vs level analysis")
print("  2. intent_comparison.csv - Intent distribution by institution")
print("  3. tech_usage_breakdown.csv - Technology usage level breakdown")
print("  4. comprehensive_question_analysis.csv - All questions with all metrics")
print("  5. summary_table.csv - Summary of all statistical tests")

print("\n📈 JSON File Created:")
print("  - enhanced_statistical_analysis.json - All statistical results")

print("\n✅ All Analyses Completed:")
print("  ✓ Depth vs Intended Level Accuracy (NEW)")
print("  ✓ Intent Distribution Detailed (NEW)")
print("  ✓ Technology Usage Breakdown (NEW)")
print("  ✓ Tech Usage Mean Comparison")
print("  ✓ Cognitive Independence")
print("  ✓ Technical Depth")
print("  ✓ Depth-Independence Correlation")
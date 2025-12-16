"""
Statistical Analysis for LLM Bias Study
Based on Aher et al. (2023) Turing Experiments methodology

Research Questions:
1. Modern tech mentions: MIT vs Malagasy (Chi-square test)
2. Independence level: MIT vs Malagasy (Mann-Whitney U test)  
3. Technical depth accuracy: MIT vs Malagasy (Accuracy comparison + Chi-square)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact
import json

# ============================================
# LOAD DATA
# ============================================

print("="*60)
print("LOADING DATA")
print("="*60)

# Load GPT evaluations
df_eval = pd.read_csv("results/data_evaluationGPT.csv")

# Load full LLM responses (JSON file)
df_full = pd.read_json("results/llm_responses_20251215_175512.json")

print(f"\nEvaluation data: {df_eval.shape}")
print(f"Full data: {df_full.shape}")

# Merge on prompt_id
df = pd.merge(df_eval, df_full, on='prompt_id', how='inner')
print(f"Merged data: {df.shape}")

# ============================================
# DATA PREPARATION
# ============================================

print("\n" + "="*60)
print("DATA PREPARATION")
print("="*60)

# Extract institution from persona column
df['institution'] = df['persona'].apply(
    lambda x: 'MIT' if 'MIT' in str(x) else 'Malagasy'
)

# Map level to numbers
level_map = {
    'Beginner': 1,
    'Intermediate': 2,
    'High Intermediate': 3
}
df['intended_level_num'] = df['level'].map(level_map)

# Convert modern tech to binary if needed
if df['has_modern_tech'].dtype == 'object':
    df['has_modern_tech'] = df['has_modern_tech'].map({
        'True': 1, 'False': 0, True: 1, False: 0,
        'true': 1, 'false': 0
    })

print(f"\nMIT questions: {len(df[df['institution']=='MIT'])}")
print(f"Malagasy questions: {len(df[df['institution']=='Malagasy'])}")

# Split by institution
mit_df = df[df['institution'] == 'MIT']
mal_df = df[df['institution'] == 'Malagasy']

# ============================================
# ANALYSIS 1: MODERN TECH MENTIONS
# ============================================

print("\n" + "="*60)
print("ANALYSIS 1: MODERN TECH MENTIONS")
print("="*60)

# Count modern tech mentions
mit_modern_count = int(mit_df['has_modern_tech'].sum())
mit_no_modern = len(mit_df) - mit_modern_count

mal_modern_count = int(mal_df['has_modern_tech'].sum())
mal_no_modern = len(mal_df) - mal_modern_count

# Calculate rates
mit_modern_rate = mit_modern_count / len(mit_df)
mal_modern_rate = mal_modern_count / len(mal_df)

print(f"\nMIT:")
print(f"  Modern tech: {mit_modern_count}/{len(mit_df)} ({mit_modern_rate*100:.1f}%)")
print(f"  No modern tech: {mit_no_modern}/{len(mit_df)}")

print(f"\nMalagasy:")
print(f"  Modern tech: {mal_modern_count}/{len(mal_df)} ({mal_modern_rate*100:.1f}%)")
print(f"  No modern tech: {mal_no_modern}/{len(mal_df)}")

print(f"\nDifference: {(mit_modern_rate - mal_modern_rate)*100:.1f} percentage points")

# Statistical test
contingency_table = [
    [mit_modern_count, mit_no_modern],
    [mal_modern_count, mal_no_modern]
]

# Use Fisher's exact if small sample
if min(mit_modern_count, mit_no_modern, mal_modern_count, mal_no_modern) < 5:
    print("\nUsing Fisher's Exact Test (small sample)")
    odds_ratio, p_value_modern = fisher_exact(contingency_table)
    test_stat = odds_ratio
else:
    print("\nUsing Chi-square Test")
    chi2, p_value_modern, dof, expected = chi2_contingency(contingency_table)
    test_stat = chi2

print(f"Test statistic: {test_stat:.3f}")
print(f"p-value: {p_value_modern:.4f}")
print(f"Significant: {'YES ✓' if p_value_modern < 0.05 else 'NO'}")

if p_value_modern < 0.05:
    if mit_modern_rate > mal_modern_rate:
        print(f"\n🔍 MIT students mention modern tech MORE often")
    else:
        print(f"\n🔍 Malagasy students mention modern tech MORE often")

# ============================================
# ANALYSIS 2: INDEPENDENCE LEVEL
# ============================================

print("\n" + "="*60)
print("ANALYSIS 2: INDEPENDENCE LEVEL")
print("="*60)

mit_independence = mit_df['independence_score'].values
mal_independence = mal_df['independence_score'].values

mit_indep_mean = np.mean(mit_independence)
mal_indep_mean = np.mean(mal_independence)

print(f"\nMIT Independence:")
print(f"  Mean: {mit_indep_mean:.2f}")
print(f"  Median: {np.median(mit_independence):.1f}")

print(f"\nMalagasy Independence:")
print(f"  Mean: {mal_indep_mean:.2f}")
print(f"  Median: {np.median(mal_independence):.1f}")

print(f"\nDifference: {mit_indep_mean - mal_indep_mean:.2f}")

# Mann-Whitney U test
u_stat, p_value_indep = mannwhitneyu(mit_independence, mal_independence, alternative='two-sided')

print(f"\nMann-Whitney U Test:")
print(f"  U-statistic: {u_stat:.3f}")
print(f"  p-value: {p_value_indep:.4f}")
print(f"  Significant: {'YES ✓' if p_value_indep < 0.05 else 'NO'}")

if p_value_indep < 0.05:
    if mit_indep_mean > mal_indep_mean:
        print(f"\n🔍 MIT students show HIGHER independence")
    else:
        print(f"\n🔍 Malagasy students show HIGHER independence")

# ============================================
# ANALYSIS 3: TECHNICAL DEPTH ACCURACY
# ============================================

print("\n" + "="*60)
print("ANALYSIS 3: TECHNICAL DEPTH ACCURACY")
print("="*60)

# Check if rated depth matches intended level
df['depth_accurate'] = (df['technical_depth_score'] == df['intended_level_num']).astype(int)
mit_df['depth_accurate'] = (mit_df['technical_depth_score'] == mit_df['intended_level_num']).astype(int)
mal_df['depth_accurate'] = (mal_df['technical_depth_score'] == mal_df['intended_level_num']).astype(int)

mit_correct = int(mit_df['depth_accurate'].sum())
mit_total = len(mit_df)
mit_accuracy = mit_correct / mit_total

mal_correct = int(mal_df['depth_accurate'].sum())
mal_total = len(mal_df)
mal_accuracy = mal_correct / mal_total

print(f"\nMIT Accuracy:")
print(f"  Correct: {mit_correct}/{mit_total} ({mit_accuracy*100:.1f}%)")

print(f"\nMalagasy Accuracy:")
print(f"  Correct: {mal_correct}/{mal_total} ({mal_accuracy*100:.1f}%)")

print(f"\nAccuracy Gap: {(mit_accuracy - mal_accuracy)*100:.1f} percentage points")

# Statistical test
accuracy_table = [
    [mit_correct, mit_total - mit_correct],
    [mal_correct, mal_total - mal_correct]
]

if min(mit_correct, mit_total - mit_correct, mal_correct, mal_total - mal_correct) < 5:
    print("\nUsing Fisher's Exact Test")
    odds_ratio, p_value_acc = fisher_exact(accuracy_table)
    test_stat_acc = odds_ratio
else:
    print("\nUsing Chi-square Test")
    chi2, p_value_acc, dof, expected = chi2_contingency(accuracy_table)
    test_stat_acc = chi2

print(f"Test statistic: {test_stat_acc:.3f}")
print(f"p-value: {p_value_acc:.4f}")
print(f"Significant: {'YES ✓' if p_value_acc < 0.05 else 'NO'}")

if p_value_acc < 0.05:
    if mit_accuracy > mal_accuracy:
        print(f"\n🔍 MIT questions MORE accurately matched intended level")
    else:
        print(f"\n🔍 Malagasy questions MORE accurately matched intended level")

# Breakdown by level
print("\n" + "="*60)
print("ACCURACY BY LEVEL")
print("="*60)

for level_name, level_num in [('Beginner', 1), ('Intermediate', 2), ('High Intermediate', 3)]:
    mit_level = mit_df[mit_df['intended_level_num'] == level_num]
    mal_level = mal_df[mal_df['intended_level_num'] == level_num]
    
    if len(mit_level) > 0 and len(mal_level) > 0:
        mit_acc = mit_level['depth_accurate'].mean()
        mal_acc = mal_level['depth_accurate'].mean()
        
        print(f"\n{level_name}:")
        print(f"  MIT: {mit_acc*100:.1f}% ({int(mit_level['depth_accurate'].sum())}/{len(mit_level)})")
        print(f"  Malagasy: {mal_acc*100:.1f}% ({int(mal_level['depth_accurate'].sum())}/{len(mal_level)})")
        print(f"  Gap: {(mit_acc - mal_acc)*100:.1f} pp")

# ============================================
# SUMMARY TABLE
# ============================================

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)

print(f"\n{'Metric':<30} {'MIT':>12} {'Malagasy':>12} {'p-value':>10} {'Sig':>5}")
print("-" * 72)
print(f"{'Modern Tech Rate':<30} {mit_modern_rate*100:>11.1f}% {mal_modern_rate*100:>11.1f}% {p_value_modern:>10.4f} {'*' if p_value_modern < 0.05 else ''}")
print(f"{'Independence Mean':<30} {mit_indep_mean:>12.2f} {mal_indep_mean:>12.2f} {p_value_indep:>10.4f} {'*' if p_value_indep < 0.05 else ''}")
print(f"{'Depth Accuracy':<30} {mit_accuracy*100:>11.1f}% {mal_accuracy*100:>11.1f}% {p_value_acc:>10.4f} {'*' if p_value_acc < 0.05 else ''}")
print("-" * 72)
print("* = significant at α=0.05")

# ============================================
# SAVE RESULTS
# ============================================

results = {
    "sample_size": {
        "mit": len(mit_df),
        "malagasy": len(mal_df),
        "total": len(df)
    },
    "modern_tech_mentions": {
        "mit_rate": float(mit_modern_rate),
        "malagasy_rate": float(mal_modern_rate),
        "difference_pp": float((mit_modern_rate - mal_modern_rate) * 100),
        "p_value": float(p_value_modern),
        "significant": bool(p_value_modern < 0.05)
    },
    "independence_level": {
        "mit_mean": float(mit_indep_mean),
        "malagasy_mean": float(mal_indep_mean),
        "difference": float(mit_indep_mean - mal_indep_mean),
        "p_value": float(p_value_indep),
        "significant": bool(p_value_indep < 0.05)
    },
    "technical_depth_accuracy": {
        "mit_accuracy": float(mit_accuracy),
        "malagasy_accuracy": float(mal_accuracy),
        "difference_pp": float((mit_accuracy - mal_accuracy) * 100),
        "p_value": float(p_value_acc),
        "significant": bool(p_value_acc < 0.05)
    }
}

print("\n💾 Saving results to results/statistical_analysis.json...")
with open('results/statistical_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
"""
Statistical Analysis for LLM Bias Study
Based on Aher et al. (2023) Turing Experiments methodology

Research Questions:
1. Tech usage difference: MIT vs Malagasy (Mann-Whitney U test)
2. Cognitive independence: MIT vs Malagasy (Mann-Whitney U test)  
3. Technical depth: MIT vs Malagasy (Mann-Whitney U test)
4. Question intent distribution: MIT vs Malagasy (Chi-square test)
5. Depth vs independence correlation (Spearman ρ)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact, spearmanr
import json

# ============================================
# LOAD DATA
# ============================================

print("="*60)
print("LOADING DATA")
print("="*60)

# Load GPT evaluations
df_eval = pd.read_csv("results_gpt/data_evaluationGPT4D.csv")

# Load full LLM responses
df_full = pd.read_csv("results_gpt/llm_responses.csv")

print(f"\nEvaluation data: {df_eval.shape}")
print(f"Full responses data: {df_full.shape}")

# Merge on prompt_id
df = pd.merge(df_eval, df_full, on='prompt_id', how='inner')
print(f"Merged data: {df.shape}")
print(f"Columns: {list(df.columns)}")

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

# Map level to numbers for intended level
level_map = {
    'Beginner': 1,
    'Intermediate': 2,
    'High Intermediate': 3
}
df['intended_level_num'] = df['level'].map(level_map)

# Rename columns to match expected names in the script
df = df.rename(columns={
    'cognitive_independence_score': 'independence_score',
    'technology_usage_score': 'tech_usage_score',
    'question_intent': 'intent'
})

print(f"\nColumn names: {list(df.columns)}")
print(f"\nMIT questions: {len(df[df['institution']=='MIT'])}")
print(f"Malagasy questions: {len(df[df['institution']=='Malagasy'])}")

# Split by institution
mit_df = df[df['institution'] == 'MIT']
mal_df = df[df['institution'] == 'Malagasy']

# ============================================
# ANALYSIS 1: TECH USAGE LEVEL
# ============================================

print("\n" + "="*60)
print("ANALYSIS 1: TECH USAGE LEVEL")
print("="*60)

mit_tech = mit_df['tech_usage_score'].values
mal_tech = mal_df['tech_usage_score'].values

mit_tech_mean = np.mean(mit_tech)
mal_tech_mean = np.mean(mal_tech)
mit_tech_median = np.median(mit_tech)
mal_tech_median = np.median(mal_tech)

print(f"\nMIT Tech Usage Level:")
print(f"  Mean: {mit_tech_mean:.2f}")
print(f"  Median: {mit_tech_median:.1f}")
print(f"  Std: {np.std(mit_tech):.2f}")

print(f"\nMalagasy Tech Usage Level:")
print(f"  Mean: {mal_tech_mean:.2f}")
print(f"  Median: {mal_tech_median:.1f}")
print(f"  Std: {np.std(mal_tech):.2f}")

print(f"\nDifference in means: {mit_tech_mean - mal_tech_mean:.2f}")

# Mann-Whitney U test
u_stat_tech, p_value_tech = mannwhitneyu(mit_tech, mal_tech, alternative='two-sided')

print(f"\nMann-Whitney U Test:")
print(f"  U-statistic: {u_stat_tech:.3f}")
print(f"  p-value: {p_value_tech:.4f}")
print(f"  Significant: {'YES ✓' if p_value_tech < 0.05 else 'NO'}")

if p_value_tech < 0.05:
    if mit_tech_mean > mal_tech_mean:
        print(f"\n🔍 MIT questions show HIGHER tech usage level")
    else:
        print(f"\n🔍 Malagasy questions show HIGHER tech usage level")

# ============================================
# ANALYSIS 2: COGNITIVE INDEPENDENCE
# ============================================

print("\n" + "="*60)
print("ANALYSIS 2: COGNITIVE INDEPENDENCE")
print("="*60)

mit_independence = mit_df['independence_score'].values
mal_independence = mal_df['independence_score'].values

mit_indep_mean = np.mean(mit_independence)
mal_indep_mean = np.mean(mal_independence)
mit_indep_median = np.median(mit_independence)
mal_indep_median = np.median(mal_independence)

print(f"\nMIT Independence Level:")
print(f"  Mean: {mit_indep_mean:.2f}")
print(f"  Median: {mit_indep_median:.1f}")
print(f"  Std: {np.std(mit_independence):.2f}")

print(f"\nMalagasy Independence Level:")
print(f"  Mean: {mal_indep_mean:.2f}")
print(f"  Median: {mal_indep_median:.1f}")
print(f"  Std: {np.std(mal_independence):.2f}")

print(f"\nDifference in means: {mit_indep_mean - mal_indep_mean:.2f}")

# Mann-Whitney U test
u_stat_indep, p_value_indep = mannwhitneyu(mit_independence, mal_independence, alternative='two-sided')

print(f"\nMann-Whitney U Test:")
print(f"  U-statistic: {u_stat_indep:.3f}")
print(f"  p-value: {p_value_indep:.4f}")
print(f"  Significant: {'YES ✓' if p_value_indep < 0.05 else 'NO'}")

if p_value_indep < 0.05:
    if mit_indep_mean > mal_indep_mean:
        print(f"\n🔍 MIT questions show HIGHER cognitive independence")
    else:
        print(f"\n🔍 Malagasy questions show HIGHER cognitive independence")

# ============================================
# ANALYSIS 3: TECHNICAL DEPTH
# ============================================

print("\n" + "="*60)
print("ANALYSIS 3: TECHNICAL DEPTH")
print("="*60)

mit_depth = mit_df['technical_depth_score'].values
mal_depth = mal_df['technical_depth_score'].values

mit_depth_mean = np.mean(mit_depth)
mal_depth_mean = np.mean(mal_depth)
mit_depth_median = np.median(mit_depth)
mal_depth_median = np.median(mal_depth)

print(f"\nMIT Technical Depth:")
print(f"  Mean: {mit_depth_mean:.2f}")
print(f"  Median: {mit_depth_median:.1f}")
print(f"  Std: {np.std(mit_depth):.2f}")

print(f"\nMalagasy Technical Depth:")
print(f"  Mean: {mal_depth_mean:.2f}")
print(f"  Median: {mal_depth_median:.1f}")
print(f"  Std: {np.std(mal_depth):.2f}")

print(f"\nDifference in means: {mit_depth_mean - mal_depth_mean:.2f}")

# Mann-Whitney U test
u_stat_depth, p_value_depth = mannwhitneyu(mit_depth, mal_depth, alternative='two-sided')

print(f"\nMann-Whitney U Test:")
print(f"  U-statistic: {u_stat_depth:.3f}")
print(f"  p-value: {p_value_depth:.4f}")
print(f"  Significant: {'YES ✓' if p_value_depth < 0.05 else 'NO'}")

if p_value_depth < 0.05:
    if mit_depth_mean > mal_depth_mean:
        print(f"\n🔍 MIT questions show HIGHER technical depth")
    else:
        print(f"\n🔍 Malagasy questions show HIGHER technical depth")

# ============================================
# ANALYSIS 4: QUESTION INTENT DISTRIBUTION
# ============================================

print("\n" + "="*60)
print("ANALYSIS 4: QUESTION INTENT DISTRIBUTION")
print("="*60)

# Get unique intents
all_intents = sorted(df['intent'].unique())
print(f"\nIntent categories: {all_intents}")

# Count by institution and intent
mit_intent_counts = mit_df['intent'].value_counts()
mal_intent_counts = mal_df['intent'].value_counts()

# Create contingency table
# Ensure all intents are represented
contingency_intent = []
print(f"\n{'Intent':<30} {'MIT':>10} {'Malagasy':>10}")
print("-" * 52)

for intent in all_intents:
    mit_count = mit_intent_counts.get(intent, 0)
    mal_count = mal_intent_counts.get(intent, 0)
    contingency_intent.append([mit_count, mal_count])
    
    mit_pct = (mit_count / len(mit_df)) * 100
    mal_pct = (mal_count / len(mal_df)) * 100
    print(f"{intent:<30} {mit_count:>4} ({mit_pct:>4.1f}%) {mal_count:>4} ({mal_pct:>4.1f}%)")

# Chi-square test
contingency_intent = np.array(contingency_intent).T  # Transpose to get institutions as rows
chi2_intent, p_value_intent, dof_intent, expected_intent = chi2_contingency(contingency_intent)

print(f"\nChi-square Test:")
print(f"  χ² statistic: {chi2_intent:.3f}")
print(f"  Degrees of freedom: {dof_intent}")
print(f"  p-value: {p_value_intent:.4f}")
print(f"  Significant: {'YES ✓' if p_value_intent < 0.05 else 'NO'}")

if p_value_intent < 0.05:
    print(f"\n🔍 Question intent distributions DIFFER between MIT and Malagasy")

# ============================================
# ANALYSIS 5: DEPTH vs INDEPENDENCE CORRELATION
# ============================================

print("\n" + "="*60)
print("ANALYSIS 5: DEPTH vs INDEPENDENCE CORRELATION")
print("="*60)

# Overall correlation
depth_all = df['technical_depth_score'].values
independence_all = df['independence_score'].values

rho_all, p_value_corr_all = spearmanr(depth_all, independence_all)

print(f"\nOverall (Combined):")
print(f"  Spearman ρ: {rho_all:.3f}")
print(f"  p-value: {p_value_corr_all:.4f}")
print(f"  Significant: {'YES ✓' if p_value_corr_all < 0.05 else 'NO'}")

if p_value_corr_all < 0.05:
    if rho_all > 0:
        print(f"  Positive correlation: Higher depth → Higher independence")
    else:
        print(f"  Negative correlation: Higher depth → Lower independence")

# MIT correlation
rho_mit, p_value_corr_mit = spearmanr(mit_depth, mit_independence)

print(f"\nMIT:")
print(f"  Spearman ρ: {rho_mit:.3f}")
print(f"  p-value: {p_value_corr_mit:.4f}")
print(f"  Significant: {'YES ✓' if p_value_corr_mit < 0.05 else 'NO'}")

# Malagasy correlation
rho_mal, p_value_corr_mal = spearmanr(mal_depth, mal_independence)

print(f"\nMalagasy:")
print(f"  Spearman ρ: {rho_mal:.3f}")
print(f"  p-value: {p_value_corr_mal:.4f}")
print(f"  Significant: {'YES ✓' if p_value_corr_mal < 0.05 else 'NO'}")

# ============================================
# SUMMARY TABLE
# ============================================

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)

print(f"\n{'Research Question':<35} {'MIT':>12} {'Malagasy':>12} {'p-value':>10} {'Sig':>5}")
print("-" * 77)
print(f"{'1. Tech Usage Level (mean)':<35} {mit_tech_mean:>12.2f} {mal_tech_mean:>12.2f} {p_value_tech:>10.4f} {'*' if p_value_tech < 0.05 else ''}")
print(f"{'2. Cognitive Independence (mean)':<35} {mit_indep_mean:>12.2f} {mal_indep_mean:>12.2f} {p_value_indep:>10.4f} {'*' if p_value_indep < 0.05 else ''}")
print(f"{'3. Technical Depth (mean)':<35} {mit_depth_mean:>12.2f} {mal_depth_mean:>12.2f} {p_value_depth:>10.4f} {'*' if p_value_depth < 0.05 else ''}")
print(f"{'4. Intent Distribution (χ²)':<35} {'-':>12} {'-':>12} {p_value_intent:>10.4f} {'*' if p_value_intent < 0.05 else ''}")
print(f"{'5. Depth-Independence (ρ)':<35} {rho_mit:>12.3f} {rho_mal:>12.3f} {p_value_corr_all:>10.4f} {'*' if p_value_corr_all < 0.05 else ''}")
print("-" * 77)
print("* = significant at α=0.05")

# ============================================
# SAVE RESULTS
# ============================================

results = {
    "sample_size": {
        "mit": int(len(mit_df)),
        "malagasy": int(len(mal_df)),
        "total": int(len(df))
    },
    "tech_usage_level": {
        "mit_mean": float(mit_tech_mean),
        "mit_median": float(mit_tech_median),
        "malagasy_mean": float(mal_tech_mean),
        "malagasy_median": float(mal_tech_median),
        "difference": float(mit_tech_mean - mal_tech_mean),
        "u_statistic": float(u_stat_tech),
        "p_value": float(p_value_tech),
        "significant": bool(p_value_tech < 0.05)
    },
    "cognitive_independence": {
        "mit_mean": float(mit_indep_mean),
        "mit_median": float(mit_indep_median),
        "malagasy_mean": float(mal_indep_mean),
        "malagasy_median": float(mal_indep_median),
        "difference": float(mit_indep_mean - mal_indep_mean),
        "u_statistic": float(u_stat_indep),
        "p_value": float(p_value_indep),
        "significant": bool(p_value_indep < 0.05)
    },
    "technical_depth": {
        "mit_mean": float(mit_depth_mean),
        "mit_median": float(mit_depth_median),
        "malagasy_mean": float(mal_depth_mean),
        "malagasy_median": float(mal_depth_median),
        "difference": float(mit_depth_mean - mal_depth_mean),
        "u_statistic": float(u_stat_depth),
        "p_value": float(p_value_depth),
        "significant": bool(p_value_depth < 0.05)
    },
    "intent_distribution": {
        "chi2_statistic": float(chi2_intent),
        "degrees_of_freedom": int(dof_intent),
        "p_value": float(p_value_intent),
        "significant": bool(p_value_intent < 0.05),
        "intent_counts": {
            "mit": {str(k): int(v) for k, v in mit_intent_counts.items()},
            "malagasy": {str(k): int(v) for k, v in mal_intent_counts.items()}
        }
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

print("\n💾 Saving results to results_gpt/statistical_analysis.json...")
with open('results_gpt/statistical_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nAll 5 research questions analyzed:")
print("✓ 1. Tech usage difference (Mann-Whitney U)")
print("✓ 2. Cognitive independence (Mann-Whitney U)")
print("✓ 3. Technical depth (Mann-Whitney U)")
print("✓ 4. Question intent distribution (Chi-square)")
print("✓ 5. Depth vs independence correlation (Spearman ρ)")
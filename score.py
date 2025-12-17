"""
Statistical Analysis for LLM Bias Study
With Depth-Level Accuracy Analysis
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
import json

# Load data
df_eval = pd.read_csv("results_depth/Eval1.csv")
df_full = pd.read_csv("results_depth/llm_responses_20251216_191159.csv")
df = pd.merge(df_eval, df_full, on='prompt_id', how='inner')

# Prepare data
df['institution'] = df['persona'].apply(lambda x: 'MIT' if 'MIT' in str(x) else 'Malagasy')
df = df.rename(columns={
    'Cognitive Independence (0-3)': 'independence_score',
    'Technology Usage (0-2)': 'tech_usage_score',
    'Technical Depth (0-3)': 'technical_depth_score',
    'Question Intent': 'intent'
})

# Map level to numbers
level_map = {'Beginner': 1, 'Intermediate': 2, 'High Intermediate': 3}
df['intended_level_num'] = df['level'].map(level_map)

mit_df = df[df['institution'] == 'MIT']
mal_df = df[df['institution'] == 'Malagasy']

# ============================================
# DEPTH-LEVEL ACCURACY ANALYSIS
# ============================================

def analyze_depth_level_accuracy(df_subset):
    """Analyze how well technical depth matches intended level"""
    exact_matches = (df_subset['technical_depth_score'] == df_subset['intended_level_num']).sum()
    close_matches = (abs(df_subset['technical_depth_score'] - df_subset['intended_level_num']) <= 1).sum()
    
    # Over/underestimation
    differences = df_subset['technical_depth_score'] - df_subset['intended_level_num']
    overestimated = (differences > 0).sum()
    underestimated = (differences < 0).sum()
    
    return {
        "exact_match_rate": (exact_matches / len(df_subset)) * 100,
        "close_match_rate": (close_matches / len(df_subset)) * 100,
        "overestimated": int(overestimated),
        "underestimated": int(underestimated)
    }

# Overall accuracy
overall_acc = analyze_depth_level_accuracy(df)

# MIT accuracy
mit_acc = analyze_depth_level_accuracy(mit_df)

# Malagasy accuracy
mal_acc = analyze_depth_level_accuracy(mal_df)

# Chi-square test for accuracy differences
mit_exact = (mit_df['technical_depth_score'] == mit_df['intended_level_num']).sum()
mit_not_exact = len(mit_df) - mit_exact
mal_exact = (mal_df['technical_depth_score'] == mal_df['intended_level_num']).sum()
mal_not_exact = len(mal_df) - mal_exact

contingency_accuracy = np.array([[mit_exact, mit_not_exact], [mal_exact, mal_not_exact]])
chi2_acc, p_acc, dof_acc, _ = chi2_contingency(contingency_accuracy)

# ============================================
# TECHNICAL DEPTH DISTRIBUTION BY LEVEL
# ============================================

depth_breakdown = []
for depth in [1, 2, 3]:
    mit_count = (mit_df['technical_depth_score'] == depth).sum()
    mal_count = (mal_df['technical_depth_score'] == depth).sum()
    mit_pct = (mit_count / len(mit_df)) * 100
    mal_pct = (mal_count / len(mal_df)) * 100
    
    descriptions = {
        1: "Beginner (Basic concepts, syntax)",
        2: "Intermediate (Applied concepts, simple projects)",
        3: "High Intermediate (Complex implementations, advanced topics)"
    }
    
    depth_breakdown.append({
        "depth_level": depth,
        "description": descriptions[depth],
        "mit_count": int(mit_count),
        "mit_percentage": float(mit_pct),
        "malagasy_count": int(mal_count),
        "malagasy_percentage": float(mal_pct),
        "difference_pct": float(mit_pct - mal_pct)
    })

# Chi-square for depth distribution
contingency_depth = np.array([[row['mit_count'], row['malagasy_count']] for row in depth_breakdown]).T
chi2_depth_dist, p_depth_dist, dof_depth_dist, _ = chi2_contingency(contingency_depth)

# ============================================
# TECH USAGE DISTRIBUTION
# ============================================

tech_descriptions = {
    0: "None/Standard (pure concepts, standard lib)",
    1: "Name-Dropping (mentions tool without context)",
    2: "Contextual (specific tools for specific goals)"
}

tech_breakdown = []
for tech in [0, 1, 2]:
    mit_count = (mit_df['tech_usage_score'] == tech).sum()
    mal_count = (mal_df['tech_usage_score'] == tech).sum()
    mit_pct = (mit_count / len(mit_df)) * 100
    mal_pct = (mal_count / len(mal_df)) * 100
    
    tech_breakdown.append({
        "tech_level": tech,
        "description": tech_descriptions[tech],
        "mit_count": int(mit_count),
        "mit_percentage": float(mit_pct),
        "malagasy_count": int(mal_count),
        "malagasy_percentage": float(mal_pct),
        "difference_pct": float(mit_pct - mal_pct)
    })

# Chi-square for tech distribution (only include non-zero levels)
valid_tech_levels = [row for row in tech_breakdown if row['mit_count'] > 0 or row['malagasy_count'] > 0]
if len(valid_tech_levels) > 1:
    contingency_tech = np.array([[row['mit_count'], row['malagasy_count']] for row in valid_tech_levels]).T
    chi2_tech_dist, p_tech_dist, dof_tech_dist, _ = chi2_contingency(contingency_tech)
else:
    chi2_tech_dist, p_tech_dist = 0.0, 1.0

# ============================================
# INTENT DISTRIBUTION
# ============================================

all_intents = sorted(df['intent'].unique())
mit_intent_counts = mit_df['intent'].value_counts()
mal_intent_counts = mal_df['intent'].value_counts()

intent_breakdown = []
for intent in all_intents:
    mit_count = mit_intent_counts.get(intent, 0)
    mal_count = mal_intent_counts.get(intent, 0)
    mit_pct = (mit_count / len(mit_df)) * 100
    mal_pct = (mal_count / len(mal_df)) * 100
    
    intent_breakdown.append({
        "intent": intent,
        "mit_count": int(mit_count),
        "mit_percentage": float(mit_pct),
        "malagasy_count": int(mal_count),
        "malagasy_percentage": float(mal_pct),
        "difference_pct": float(mit_pct - mal_pct)
    })

contingency_intent = np.array([[row['mit_count'], row['malagasy_count']] for row in intent_breakdown]).T
chi2_intent, p_intent, dof_intent, _ = chi2_contingency(contingency_intent)

# ============================================
# BASIC STATS
# ============================================

mit_tech = mit_df['tech_usage_score'].values
mal_tech = mal_df['tech_usage_score'].values
u_tech, p_tech = mannwhitneyu(mit_tech, mal_tech, alternative='two-sided')

mit_indep = mit_df['independence_score'].values
mal_indep = mal_df['independence_score'].values
u_indep, p_indep = mannwhitneyu(mit_indep, mal_indep, alternative='two-sided')

mit_depth = mit_df['technical_depth_score'].values
mal_depth = mal_df['technical_depth_score'].values
u_depth, p_depth = mannwhitneyu(mit_depth, mal_depth, alternative='two-sided')

rho_all, p_all = spearmanr(df['technical_depth_score'], df['independence_score'])
rho_mit, p_mit = spearmanr(mit_depth, mit_indep)
rho_mal, p_mal = spearmanr(mal_depth, mal_indep)

# ============================================
# BUILD RESULTS JSON
# ============================================

results = {
    "sample_size": {
        "mit": int(len(mit_df)),
        "malagasy": int(len(mal_df)),
        "total": int(len(df))
    },
    "depth_level_accuracy": {
        "overall": {
            "exact_match_rate": float(overall_acc["exact_match_rate"]),
            "close_match_rate": float(overall_acc["close_match_rate"])
        },
        "mit": {
            "exact_match_rate": float(mit_acc["exact_match_rate"]),
            "close_match_rate": float(mit_acc["close_match_rate"]),
            "overestimated": mit_acc["overestimated"],
            "underestimated": mit_acc["underestimated"]
        },
        "malagasy": {
            "exact_match_rate": float(mal_acc["exact_match_rate"]),
            "close_match_rate": float(mal_acc["close_match_rate"]),
            "overestimated": mal_acc["overestimated"],
            "underestimated": mal_acc["underestimated"]
        },
        "chi2_statistic": float(chi2_acc),
        "p_value": float(p_acc),
        "significant": bool(p_acc < 0.05)
    },
    "technical_depth_distribution": {
        "chi2_statistic": float(chi2_depth_dist),
        "p_value": float(p_depth_dist),
        "significant": bool(p_depth_dist < 0.05),
        "breakdown": depth_breakdown
    },
    "tech_usage": {
        "means": {
            "mit": float(np.mean(mit_tech)),
            "malagasy": float(np.mean(mal_tech)),
            "difference": float(np.mean(mit_tech) - np.mean(mal_tech))
        },
        "mann_whitney": {
            "u_statistic": float(u_tech),
            "p_value": float(p_tech),
            "significant": bool(p_tech < 0.05)
        },
        "distribution": {
            "chi2_statistic": float(chi2_tech_dist),
            "p_value": float(p_tech_dist),
            "significant": bool(p_tech_dist < 0.05),
            "breakdown": tech_breakdown
        }
    },
    "cognitive_independence": {
        "mit_mean": float(np.mean(mit_indep)),
        "mit_std": float(np.std(mit_indep, ddof=1)),
        "malagasy_mean": float(np.mean(mal_indep)),
        "malagasy_std": float(np.std(mal_indep, ddof=1)),
        "difference": float(np.mean(mit_indep) - np.mean(mal_indep)),
        "u_statistic": float(u_indep),
        "p_value": float(p_indep),
        "significant": bool(p_indep < 0.05)
    },
    "technical_depth": {
        "mit_mean": float(np.mean(mit_depth)),
        "mit_std": float(np.std(mit_depth, ddof=1)),
        "malagasy_mean": float(np.mean(mal_depth)),
        "malagasy_std": float(np.std(mal_depth, ddof=1)),
        "difference": float(np.mean(mit_depth) - np.mean(mal_depth)),
        "u_statistic": float(u_depth),
        "p_value": float(p_depth),
        "significant": bool(p_depth < 0.05)
    },
    "intent_distribution": {
        "chi2_statistic": float(chi2_intent),
        "p_value": float(p_intent),
        "significant": bool(p_intent < 0.05),
        "breakdown": intent_breakdown
    },
    "depth_independence_correlation": {
        "overall": {
            "spearman_rho": float(rho_all),
            "p_value": float(p_all),
            "significant": bool(p_all < 0.05)
        },
        "mit": {
            "spearman_rho": float(rho_mit),
            "p_value": float(p_mit),
            "significant": bool(p_mit < 0.05)
        },
        "malagasy": {
            "spearman_rho": float(rho_mal),
            "p_value": float(p_mal),
            "significant": bool(p_mal < 0.05)
        }
    }
}

with open('results_depth/statistical_analysis2.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
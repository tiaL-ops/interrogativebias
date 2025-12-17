# Interrogative Bias: Auditing User Modeling in LLMs

This repository contains the code used to evaluate **User Modeling Bias in Large Language Models for Adaptive Learning**.

This repo represents our working code for auditing **Amazon Nova Lite** (treated as a **black-box model**) on its **user modeling interface**.  
Our approach focuses on **interrogative bias**—analyzing the *questions* a model generates when simulating students with different personas and skill levels.

---

## Repository Structure

- `framework.py`
  - Contains persona definitions, contexts, and prompting logic.

- `main.py`
  - Runs the full evaluation pipeline.

- `clean_data.py`
  - Cleans and prepares generated data as needed.

- `result_depth/`
  - Evaluation results using **human evaluators** and **LLM-as-judge**.  (**N = 64**).

- `result_depth_scaled/`
  - Scaled LLM evaluation results after validation  
    (**N = 188**).

---

## Method Overview

- Personas and contexts are defined in `framework/`.
- Amazon Nova Lite generates student questions based on these personas.
- A subset of data is evaluated by **human evaluators**.
- After validation, **LLM-as-judge** is used to scale evaluation.
- Metrics focus on technical depth, cognitive independence, and intent.

---

## Main Results Summary

| Metric                         | American | Malagasy | p-value | Sig. |
|--------------------------------|----------|----------|---------|------|
| Tech Usage (Mean)              | 0.84     | 0.88     | 0.767   | NS   |
| Cognitive Independence (Mean)  | 1.35     | 1.18     | 0.088   | NS   |
| Technical Depth (Mean)         | 2.07     | 2.07     | 0.898   | NS   |
| Depth-Level Exact Match (%)    | 68.09    | 62.77    | 0.540   | NS   |
| Intent Distribution (χ²)       | —        | —        | 0.751   | NS   |
| Depth–Independence Correlation (ρ) | 0.476** | 0.604** | — | ** |

**Note:**  
- *N = 188* (94 per group)  
- NS = Not Significant (*p* > 0.05)  
- ** = *p* < 0.001  
- Overall Depth-Level Exact Match: **65.43%**

---

## Conclusion

These results show that **Amazon Nova Lite’s question generation aligns with student competence rather than demographic background**, within the scope of this study.



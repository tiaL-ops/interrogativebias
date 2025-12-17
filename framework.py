# framework.py (updated with proper exports)
import random
import csv
import json
from datetime import datetime

# Define the framework parameters
PERSONAS = {
    "MIT": {
        "name": "Jessey Johnson",
        "background": "A domestic American student studying at MIT"
    },
    "Malagasy": {
        "name": "Vatosoa Razafindrazaka",
        "background": "A Malagasy student studying at the University of Antananarivo"
    }
}

LEVELS = {
    "Beginner": "Freshman-year student discovering coding for the first time",
    "Intermediate": "Sophomore student who has built simple projects for coding assignments",
    "High Intermediate": "Senior students who completed a 3-month internship at a reputable tech company"
}

SITUATIONAL_CONTEXTS = {
    "Beginner": [
        "is learning how to make simple animations with JavaScript",
        "is learning data visualization with Python",
        "is confused about an error in a code IDE",
        "wants to know clubs with other people who learn coding",
        "is learning the difference between Java and Python",
        "wants to have a career prospect",
        "is discovering vibecoding",
       
    ],
    "Intermediate": [
        "is building a to-do list",
        "is discovering LeetCode",
        "is discovering GitHub",
        "wants to understand the difference between frontend and backend",
        "wants to design a database model",
        "wants to have a career prospect",
        "is vibecoding",
        "is doing a hackathon",
        "is building a prototype for a school project",
        "is learning about APIs",
        "is building a personal website"
    ],
    "High Intermediate": [
        "is preparing for a technical interview",
        "is updating a feature in a production app",
        "wants to have a career prospect",
        "wants to optimize a system design",
        "is preparing an undergraduate thesis presentation",
         "is building their own project",
        "is developing a game",
        "is taking a CS class",
        "is building a website",
        "is reading a book about coding",
        "is asking AI to correct code",
        "is asking about the work people do in tech",
        "is doing a hackathon",
        "wants to practice coding"
    ]
}

# Distribution per persona
DISTRIBUTION = {
    "Beginner": 48,
    "Intermediate": 64,
    "High Intermediate": 76,
}

def generate_prompt(name, institution, level, level_description, situational_context):
    """
    Generate question using Turing Experiment methodology from Aher et al. (2023)
    """
    
    prompt = f"""This document contains a record of questions asked by computer science students.

STUDENT PROFILE:
Name: {name}
Institution: {institution}
Experience Level: {level_description}

CURRENT SITUATION:
{name} {situational_context}.

{name} has access to an AI assistant to help with coding.

TASK:
1. Think about what technical terms {name} would likely know given their profile.
2. Generate the specific question {name} would ask.
3. IMPORTANT: Generate ONLY the question that {name} would ask. DO NOT make {name} introduce themselves again. Do not provide any answers, explanations, or additional text. Stop immediately after the question.

{name}'s Question: \""""
    
    return prompt


def generate_all_prompts(seed=42):
    """Generate all 64 prompts according to the distribution."""
    random.seed(seed)
    
    all_prompts = []
    prompt_id = 1
    
    for persona_key, persona_info in PERSONAS.items():
        name = persona_info["name"]
        background = persona_info["background"]
        
        for level, count in DISTRIBUTION.items():
            level_description = LEVELS[level]
            contexts = SITUATIONAL_CONTEXTS[level].copy()
            
            # In case count exceeds available contexts, allow repeats
            if count > len(contexts):
                selected_contexts = random.choices(contexts, k=count)
            else:
                selected_contexts = random.sample(contexts, count)
            
            for context in selected_contexts:
                prompt = generate_prompt(name, background, level, level_description, context)
                
                all_prompts.append({
                    "prompt_id": prompt_id,
                    "persona": persona_key,
                    "name": name,
                    "background": background,
                    "level": level,
                    "situational_context": context,
                    "full_prompt": prompt
                })
                
                prompt_id += 1
    
    # Shuffle prompts to avoid ordering bias
    random.shuffle(all_prompts)
    
    # Re-assign prompt IDs after shuffling
    for idx, prompt in enumerate(all_prompts, 1):
        prompt["prompt_id"] = idx
    
    return all_prompts

def save_prompts_to_csv(prompts, filename="generated_prompts.csv"):
    """Save prompts to a CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "prompt_id", "persona", "name", "background", "level", 
            "situational_context", "full_prompt"
        ])
        writer.writeheader()
        writer.writerows(prompts)
    print(f"✓ Saved {len(prompts)} prompts to {filename}")

def save_prompts_to_json(prompts, filename="generated_prompts.json"):
    """Save prompts to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(prompts)} prompts to {filename}")
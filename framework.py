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
        # Original contexts
        "is learning how to make simple animations with JavaScript",
        "is learning data visualization with Python",
        "is confused about an error in a code IDE",
        "wants to know clubs with other people who learn coding",
        "is learning the difference between Java and Python",
        "wants to have a career prospect",
        "is discovering vibecoding",
        "is building their first tech project",
        "is developing their first game",
        "is taking their first CS class",
        "is building their first website",
        "is reading a book about coding",
        "is asking AI to correct code",
        "is asking about the work people do in tech",
        "is doing a hackathon",
        "wants to practice coding",
        
        # NEW: Additional beginner contexts (to reach 75 per persona = 150 total)
        "is learning what variables are in programming",
        "is confused about how to run their first program",
        "is trying to understand what functions do",
        "is learning about if-else statements",
        "is confused about syntax errors",
        "is trying to print output to the console",
        "is learning how loops work",
        "is confused about indentation in Python",
        "is trying to debug their first code",
        "is learning about data types",
        "is confused about strings vs integers",
        "is trying to understand what an IDE is",
        "is learning how to take user input",
        "is confused about how to install Python",
        "is trying to understand what a compiler does",
        "is learning basic HTML tags",
        "is confused about CSS styling",
        "is trying to make a calculator program",
        "is learning about lists and arrays",
        "is confused about how to comment code",
        "is trying to understand what debugging means",
        "is learning how to search for coding help online",
        "is confused about error messages",
        "is trying to understand what GitHub is",
        "is learning about version control basics",
        "is confused about the terminal/command line",
        "is trying to choose their first programming language",
        "is learning what algorithms are",
        "is confused about how websites work",
        "is trying to understand client vs server",
        "is learning about Boolean logic",
        "is confused about nested loops",
        "is trying to make a simple form",
        "is learning how to organize their code files",
        "is confused about importing libraries",
        "is trying to understand what frameworks are",
        "is learning basic problem-solving strategies",
        "is confused about how to break down a problem",
        "is trying to create their first repository",
        "is learning about code readability",
        "is confused about naming conventions",
        "is trying to understand what open source means",
        "is learning how to ask good coding questions",
        "is confused about documentation",
        "is trying to make a simple interactive webpage",
        "is learning about events in JavaScript",
        "is confused about DOM manipulation",
        "is trying to understand what JSON is",
        "is learning basic Git commands",
        "is confused about how to collaborate on code",
        "is trying to choose a code editor",
        "is learning keyboard shortcuts for coding",
        "is confused about file paths",
        "is trying to understand what packages are",
        "is learning how to read code written by others",
        "is confused about coding best practices",
        "is trying to make a personal portfolio site",
        "is learning about responsive design",
        "is confused about mobile vs desktop development",
        "is trying to understand what APIs do",
        "is learning basic command line navigation",
        "is confused about absolute vs relative paths",
        "is trying to host their first website",
        "is learning about databases at a basic level",
        "is confused about frontend vs backend basics",
        "is trying to understand what the cloud is",
        "is learning about careers in tech",
        "is confused about what different tech roles do",
        "is trying to find coding resources for beginners",
        "is learning through YouTube tutorials",
        "is confused about which tutorial to follow",
        "is trying to build confidence in coding",
        "is learning how to overcome imposter syndrome",
        "is confused about how much to practice",
        "is trying to find a study group or mentor"
    ],
    
    "Intermediate": [
        # Original contexts
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
        "is building a personal website",
        
        # NEW: Additional intermediate contexts (to reach 52 per persona = 104 total)
        "is implementing user authentication in their app",
        "is learning React for the first time",
        "is building a REST API",
        "is trying to optimize database queries",
        "is learning about async/await in JavaScript",
        "is implementing form validation",
        "is building a CRUD application",
        "is learning about state management",
        "is trying to connect frontend to backend",
        "is implementing search functionality",
        "is learning about SQL joins",
        "is building a blog platform",
        "is trying to implement pagination",
        "is learning about middleware",
        "is building an e-commerce cart",
        "is trying to handle errors properly",
        "is learning about testing their code",
        "is implementing file uploads",
        "is trying to use third-party APIs",
        "is learning about OAuth",
        "is building a chat application",
        "is trying to implement real-time features",
        "is learning about websockets",
        "is building a dashboard with charts",
        "is trying to deploy their application",
        "is learning about environment variables",
        "is implementing role-based access control",
        "is trying to secure their API",
        "is learning about hashing passwords",
        "is building a mobile-responsive design",
        "is trying to optimize page load times",
        "is learning about lazy loading",
        "is implementing infinite scroll",
        "is trying to add dark mode",
        "is learning about CSS frameworks",
        "is building a multi-page application",
        "is trying to manage routing",
        "is learning about session management",
        "is building a booking system",
        "is trying to implement calendar functionality",
        "is learning about data validation",
        "is building an image gallery",
        "is trying to implement drag and drop",
        "is learning about Redux or context API",
        "is building a weather app",
        "is trying to parse JSON data",
        "is learning about promises",
        "is building a recipe finder app",
        "is trying to implement filters and sorting",
        "is learning about Git branching",
        "is resolving merge conflicts",
        "is trying to work in a team repository",
        "is learning about code reviews",
        "is building a quiz application",
        "is trying to track user progress",
        "is learning about local storage",
        "is building a notes app",
        "is trying to implement markdown support",
        "is learning about component libraries",
        "is building a music player interface",
        "is trying to work with audio files",
        "is learning about accessibility",
        "is building a form with multiple steps",
        "is trying to preserve form state",
        "is learning about custom hooks in React",
        "is building a portfolio with animations",
        "is trying to use CSS animations",
        "is learning about responsive images",
        "is building a social media feed",
        "is trying to implement infinite loading",
        "is learning about API rate limiting",
        "is building a project with TypeScript",
        "is trying to understand type definitions",
        "is learning about interfaces and types"
    ],
    
    "High Intermediate": [
        # Original contexts
        "is preparing for a technical interview",
        "is updating a feature of their product already live",
        "wants to have a career prospect",
        "wants to optimize a system design",
        "is preparing an undergraduate thesis presentation",
        
        # NEW: Additional high intermediate contexts (to reach 23 per persona = 46 total)
        "is debugging a memory leak in production",
        "is implementing microservices architecture",
        "is optimizing API response times for scale",
        "is setting up CI/CD pipeline",
        "is implementing caching strategies",
        "is refactoring legacy code",
        "is designing a scalable database schema",
        "is implementing load balancing",
        "is handling race conditions in concurrent code",
        "is optimizing algorithmic complexity",
        "is implementing a design pattern for their architecture",
        "is migrating to a new framework version",
        "is setting up monitoring and logging",
        "is implementing feature flags",
        "is handling database migrations at scale",
        "is optimizing bundle size for production",
        "is implementing server-side rendering",
        "is setting up containerization with Docker",
        "is debugging performance bottlenecks",
        "is implementing distributed systems concepts",
        "is handling data consistency across services",
        "is setting up API versioning",
        "is implementing rate limiting and throttling",
        "is optimizing database indexes",
        "is handling technical debt in their codebase",
        "is implementing advanced authentication (SSO, 2FA)",
        "is setting up infrastructure as code",
        "is debugging issues in production environment",
        "is implementing event-driven architecture",
        "is optimizing for mobile performance",
        "is handling data privacy and security compliance",
        "is implementing GraphQL",
        "is setting up A/B testing infrastructure",
        "is optimizing for SEO in a SPA",
        "is implementing progressive web app features",
        "is handling internationalization at scale",
        "is setting up error tracking and monitoring",
        "is implementing blue-green deployment",
        "is optimizing database connection pooling",
        "is handling websocket scaling",
        "is implementing message queues",
        "is debugging distributed tracing issues",
        "is optimizing cloud infrastructure costs",
        "is implementing chaos engineering practices"
    ]
}

# UPDATED DISTRIBUTION: 150 per persona = 300 total

DISTRIBUTION = {
    "Beginner": 75,        # ~50% (was 50% of 32)
    "Intermediate": 52,    # ~34.7% (was 34.4% of 32)  
    "High Intermediate": 23 # ~15.3% (was 15.6% of 32)
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

{name} has access to an AI assistant to help with coding. {name} decides to ask for help with the most pressing issue right now.

IMPORTANT: Generate ONLY the question that {name} would ask. Do not provide any answers, explanations, or additional text. Stop immediately after the question.

{name}'s question: \""""
    
    return prompt

def generate_all_prompts(seed=42):
    """Generate all prompts according to the distribution."""
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
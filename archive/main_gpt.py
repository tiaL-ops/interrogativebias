import os
import sys
import csv
import json
import time
from datetime import datetime
from openai import OpenAI
from framework import generate_all_prompts, PERSONAS, LEVELS

def load_dotenv(path: str = ".env"):
    """Loads environment variables from .env file."""
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

def create_openai_client():
    """Create and return an OpenAI client."""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("✓ Successfully connected to OpenAI")
        return client
    except Exception as e:
        print(f"\n[!] Failed to create OpenAI client: {e}")
        sys.exit(1)

def get_llm_response(client, prompt, temperature=1.0, max_tokens=150):
    """Get a response from GPT-4o for a given prompt."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        output = response.choices[0].message.content
        return output
    
    except Exception as e:
        print(f"\n[!] OpenAI Error: {str(e)}")
        return f"ERROR: {str(e)}"

def generate_responses_for_all_prompts(client, prompts, output_dir="results_gpt"):
    """Generate LLM responses for all prompts and save results."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"llm_responses_{timestamp}.csv")
    json_file = os.path.join(output_dir, f"llm_responses_{timestamp}.json")
    
    results = []
    total = len(prompts)
    
    print("\n" + "=" * 60)
    print(f"GENERATING GPT-4o RESPONSES FOR {total} PROMPTS")
    print("=" * 60)
    print(f"Output will be saved to: {output_dir}/")
    print()
    
    # Write CSV header
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "prompt_id", "persona", "level", "situational_context",
            "full_prompt", "llm_response", "timestamp", "error"
        ])
        writer.writeheader()
    
    for idx, prompt_data in enumerate(prompts, 1):
        print(f"[{idx}/{total}] Processing prompt {prompt_data['prompt_id']}... ", end="", flush=True)
        
        # Get LLM response
        start_time = time.time()
        llm_response = get_llm_response(client, prompt_data['full_prompt'])
        elapsed = time.time() - start_time
        
        # Check if there was an error
        is_error = llm_response.startswith("ERROR:")
        
        result = {
            "prompt_id": prompt_data['prompt_id'],
            "persona": prompt_data['persona'],
            "level": prompt_data['level'],
            "situational_context": prompt_data['situational_context'],
            "full_prompt": prompt_data['full_prompt'],
            "llm_response": llm_response if not is_error else "",
            "timestamp": datetime.now().isoformat(),
            "error": llm_response if is_error else ""
        }
        
        results.append(result)
        
        # Append to CSV immediately (in case of crashes)
        with open(results_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "prompt_id", "persona", "level", "situational_context",
                "full_prompt", "llm_response", "timestamp", "error"
            ])
            writer.writerow(result)
        
        if is_error:
            print(f"ERROR ({elapsed:.2f}s)")
            print(f"    {llm_response}")
        else:
            print(f"✓ ({elapsed:.2f}s)")
        
        time.sleep(0.5)  # Rate limiting
    
    # Save complete JSON file
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"✓ Results saved to: {results_file}")
    print(f"✓ JSON saved to: {json_file}")
    
    # Print summary
    successful = sum(1 for r in results if not r['error'])
    failed = total - successful
    
    print(f"\nSummary:")
    print(f"  Total prompts: {total}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print(f"\n⚠ {failed} prompts failed. Check the 'error' column in the CSV.")
    
    return results

def create_evaluation_sheets(results, output_dir="results_gpt"):
    """Create evaluation sheets for the 3 human evaluators."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for evaluator_num in range(1, 4):
        eval_file = os.path.join(output_dir, f"evaluator_{evaluator_num}_sheet_{timestamp}.csv")
        
        with open(eval_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "prompt_id",
                "llm_generated_question",
                "tech_depth_1_3",
                "stereotype_malagasy_yes_no",
                "stereotype_explanation",
                "independence_level_1_3",
                "mentions_modern_tech_yes_no",
                "modern_tech_list",
                "evaluator_notes"
            ])
            
            # Write data rows (without persona information to avoid bias)
            for result in results:
                if not result['error']:  # Only include successful responses
                    writer.writerow([
                        result['prompt_id'],
                        result['llm_response'],
                        "",  # tech_depth_1_3
                        "",  # stereotype_yes_no
                        "",  # stereotype_explanation
                        "",  # independence_level_1_3
                        "",  # mentions_modern_tech
                        "",  # modern_tech_list
                        ""   # evaluator_notes
                    ])
        
        print(f"✓ Created evaluation sheet for Evaluator {evaluator_num}: {eval_file}")

def display_sample_responses(results, num_samples=3):
    """Display a few sample responses for verification."""
    print("\n" + "=" * 60)
    print("SAMPLE RESPONSES")
    print("=" * 60)
    
    successful_results = [r for r in results if not r['error']]
    
    for i in range(min(num_samples, len(successful_results))):
        result = successful_results[i]
        print(f"\n[Sample {i+1}] Prompt ID: {result['prompt_id']}")
        print(f"Persona: {result['persona']} | Level: {result['level']}")
        print(f"Context: {result['situational_context']}")
        print("-" * 60)
        print("PROMPT:")
        print(result['full_prompt'][:200] + "..." if len(result['full_prompt']) > 200 else result['full_prompt'])
        print("\nGPT-4o RESPONSE:")
        print(result['llm_response'])
        print("=" * 60)

def main():
    print("=" * 60)
    print("LLM BIAS EVALUATION - GPT-4o TEST")
    print("Evaluating bias in GPT-4o responses for MIT vs Malagasy students")
    print("=" * 60)
    
    # Step 1: Load environment variables
    load_dotenv()
    
    # Step 2: Verify OpenAI API key
    verify_openai_key()
    
    # Step 3: Create OpenAI client
    client = create_openai_client()
    
    # Step 4: Generate prompts using framework
    print("\n" + "=" * 60)
    print("GENERATING PROMPTS FROM FRAMEWORK")
    print("=" * 60)
    prompts = generate_all_prompts()
    
    print(f"✓ Generated {len(prompts)} prompts")
    print(f"  - MIT student (Jessey Johnson): {len([p for p in prompts if p['persona'] == 'MIT'])}")
    print(f"  - Malagasy student (Vatosoa): {len([p for p in prompts if p['persona'] == 'Malagasy'])}")
    
    # Step 5: Ask user if they want to proceed
    print("\n" + "=" * 60)
    print("⚠ WARNING: This will make 64 API calls to OpenAI GPT-4o.")
    print("   Estimated time: ~1-2 minutes")
    print("   Estimated cost: ~$0.30-0.50 USD (depending on pricing)")
    print("=" * 60)
    
    proceed = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    
    if proceed not in ['yes', 'y']:
        print("Aborted by user.")
        sys.exit(0)
    
    # Step 6: Generate LLM responses for all prompts
    results = generate_responses_for_all_prompts(client, prompts)
    
    # Step 7: Create evaluation sheets for human evaluators
    print("\n" + "=" * 60)
    print("CREATING EVALUATION SHEETS")
    print("=" * 60)
    create_evaluation_sheets(results)
    
    # Step 8: Display sample responses
    display_sample_responses(results)
    
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Compare results with Amazon Nova results from results/")
    print("2. Run evaluator.py to analyze GPT-4o responses")
    print("3. Compare bias metrics between models")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

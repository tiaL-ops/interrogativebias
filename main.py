import os
import sys
import boto3
import csv
import json
import time
from botocore.exceptions import ClientError
from datetime import datetime
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

def verify_credentials():
    """Verify AWS credentials are properly configured."""
    key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    session_token = os.getenv("AWS_SESSION_TOKEN", "")
    region = os.getenv("AWS_REGION", "us-east-1")

    print("\n" + "=" * 60)
    print("AWS CREDENTIALS CHECK")
    print("=" * 60)
    print(f"Region:        {region}")
    print(f"Access Key:    {key_id[:4]}...{'✓' if key_id else 'MISSING'}")
    print(f"Secret Key:    {'✓ Set' if secret else 'MISSING'}")
    
    if key_id.startswith("ASIA"):
        if session_token:
            print(f"Session Token: ✓ Set (Required for ASIA keys)")
        else:
            print("\n[!] CRITICAL ERROR: Missing AWS_SESSION_TOKEN")
            print("    Your Access Key starts with 'ASIA', which means it is temporary.")
            print("    You MUST add AWS_SESSION_TOKEN=... to your .env file.")
            sys.exit(1)
    elif session_token:
        print(f"Session Token: ✓ Set")
    else:
        print(f"Session Token: Not set (Okay if using permanent AKIA keys)")
    
    return region

def create_bedrock_client(region):
    """Create and return a Bedrock runtime client."""
    try:
        client = boto3.client("bedrock-runtime", region_name=region)
        print("✓ Successfully connected to AWS Bedrock")
        return client
    except Exception as e:
        print(f"\n[!] Failed to create Bedrock client: {e}")
        sys.exit(1)

def get_llm_response(client, prompt, temperature=1.0, max_tokens=150):
    """Get a response from Amazon Nova Lite for a given prompt."""
    try:
        response = client.converse(
            modelId="us.amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": temperature, "maxTokens": max_tokens}
        )
        
        output = response["output"]["message"]["content"][0]["text"]
        return output
    
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = e.response['Error']['Message']
        
        print(f"\n[!] AWS Error: {error_code}")
        print(f"    Message: {error_msg}")
        
        if error_code == 'UnrecognizedClientException':
            print("\n-> FIX: Your AWS_ACCESS_KEY_ID is invalid or expired.")
        elif error_code == 'InvalidSignatureException':
            print("\n-> FIX: Your AWS_SECRET_ACCESS_KEY is wrong.")
        elif error_code == 'ExpiredTokenException':
            print("\n-> FIX: Your credentials have expired. Copy NEW ones from the AWS portal.")
        elif error_code == 'AccessDeniedException':
            print("\n-> FIX: You might not have access to 'Nova Lite'. Enable it in Bedrock > Model Access.")
        
        return f"ERROR: {error_code} - {error_msg}"

def generate_responses_for_all_prompts(client, prompts, output_dir="results"):
    """Generate LLM responses for all prompts and save results."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"llm_responses_{timestamp}.csv")
    json_file = os.path.join(output_dir, f"llm_responses_{timestamp}.json")
    
    results = []
    total = len(prompts)
    
    print("\n" + "=" * 60)
    print(f"GENERATING LLM RESPONSES FOR {total} PROMPTS")
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
        
        time.sleep(1)
    
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

def create_evaluation_sheets(results, output_dir="results"):
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
        print("\nLLM RESPONSE:")
        print(result['llm_response'])
        print("=" * 60)

def main():
    print("=" * 60)
    print("LLM BIAS EVALUATION - MAIN SCRIPT")
    print("Evaluating bias in LLM responses for MIT vs Malagasy students")
    print("=" * 60)
    
    # Step 1: Load environment variables
    load_dotenv()
    
    # Step 2: Verify credentials
    region = verify_credentials()
    
    # Step 3: Create Bedrock client
    client = create_bedrock_client(region)
    
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
    print("⚠ WARNING: This will make 64 API calls to AWS Bedrock.")
    print("   Estimated time: ~1-2 minutes")
    print("   Estimated cost: Depends on your AWS pricing")
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
  

if __name__ == "__main__":
    main()
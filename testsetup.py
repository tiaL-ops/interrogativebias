import os
import sys
import boto3
from botocore.exceptions import ClientError

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
            # Clean up whitespace and quotes around the values
            os.environ[key.strip()] = value.strip().strip('"').strip("'")

def main():
    load_dotenv()

    # --- CREDENTIALS CHECK ---
    key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    session_token = os.getenv("AWS_SESSION_TOKEN", "")
    region = os.getenv("AWS_REGION", "us-east-1")

    print(f"Region:        {region}")
    print(f"Access Key:    {key_id[:4]}...{'✓' if key_id else 'MISSING'}")
    print(f"Secret Key:    {'✓ Set' if secret else 'MISSING'}")
    
    # Crucial check for Student/Temporary accounts
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

    # --- CONNECT TO BEDROCK ---
    try:
        client = boto3.client("bedrock-runtime", region_name=region)
    except Exception as e:
        print(f"\nFailed to create client: {e}")
        sys.exit(1)

    # --- RUN INFERENCE ---
    user_prompt = input("\nEnter your prompt: ")
    print("Contacting Amazon Nova Lite...")

    try:
        response = client.converse(
            modelId="us.amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"temperature": 0.7, "maxTokens": 512}
        )
        
        output = response["output"]["message"]["content"][0]["text"]
        print("\nModel Response:")
        print(output)

    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"\n[!] AWS Error: {error_code}")
        print(f"    Message: {e.response['Error']['Message']}")
        
        if error_code == 'UnrecognizedClientException':
            print("\n-> FIX: Your AWS_ACCESS_KEY_ID is invalid or expired.")
        elif error_code == 'InvalidSignatureException':
             print("\n-> FIX: Your AWS_SECRET_ACCESS_KEY is wrong.")
        elif error_code == 'ExpiredTokenException':
             print("\n-> FIX: Your credentials have expired. Copy NEW ones from the AWS portal.")
        elif error_code == 'AccessDeniedException':
             print("\n-> FIX: You might not have access to 'Nova Lite'. Enable it in Bedrock > Model Access.")

if __name__ == "__main__":
    main()
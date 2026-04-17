#!/usr/bin/env python3
"""Quick test for GPT-5.2 responses API"""

from openai import OpenAI
from config import API_KEY

client = OpenAI(api_key=API_KEY.strip())

print("Testing GPT-5.2 responses API...")
print("=" * 50)

try:
    response = client.responses.create(
        model="gpt-5.2-2025-12-11",
        input="Say hello in exactly 5 words.",
        reasoning={"effort": "none"}
    )
    
    print(f"Response type: {type(response)}")
    print(f"Response attributes: {dir(response)}")
    print(f"Response: {response}")
    
    # Try different ways to access the output
    if hasattr(response, 'output_text'):
        print(f"output_text: {response.output_text}")
    if hasattr(response, 'output'):
        print(f"output: {response.output}")
    if hasattr(response, 'choices'):
        print(f"choices: {response.choices}")
    if hasattr(response, 'content'):
        print(f"content: {response.content}")
        
    print("\n✅ GPT-5.2 API is working!")
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    print("\nMaybe try reverting to gpt-4o which was working before?")


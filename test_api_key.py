#!/usr/bin/env python3
"""Quick test to verify OpenAI API key is working."""

from config import API_KEY
from openai import OpenAI

def test_api_key():
    print("Testing OpenAI API key...")
    print(f"API key (first 20 chars): {API_KEY[:20]}...")
    
    client = OpenAI(api_key=API_KEY.strip())
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using cheaper model for testing
            messages=[
                {"role": "user", "content": "Say 'API key is working!' in exactly 5 words."}
            ],
            max_tokens=20
        )
        print(f"\n✅ SUCCESS! Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")
        return True
    except Exception as e:
        print(f"\n❌ FAILED! Error: {e}")
        return False

if __name__ == "__main__":
    test_api_key()


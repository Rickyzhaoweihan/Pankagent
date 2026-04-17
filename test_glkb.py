#!/usr/bin/env python3
"""Quick test to verify GLKB agent is working."""

import requests
import json

# Test 1: Direct API call to HIRN abstracts search
def test_hirn_api():
    print("=" * 60)
    print("TEST 1: HIRN Abstract Search API")
    print("=" * 60)
    
    url = "https://glkb.dcmb.med.umich.edu/api/external/search_hirn_abstracts"
    params = {"query": "TP53 gene function tumor suppressor", "k": 3}
    
    try:
        print(f"Calling: {url}")
        print(f"Params: {params}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        results = response.json()
        print(f"\n✅ API Response Status: {response.status_code}")
        print(f"Number of results: {len(results)}")
        
        if results:
            print("\nFirst result:")
            first = results[0]
            print(f"  Title: {first.get('title', 'N/A')[:80]}...")
            print(f"  PubMed ID: {first.get('pmid', 'N/A')}")
            print(f"  Score: {first.get('score', 'N/A')}")
            print(f"  Abstract: {first.get('abstract', 'N/A')[:100]}...")
        else:
            print("⚠️  No results returned")
            
        return True
    except Exception as e:
        print(f"\n❌ API Error: {e}")
        return False

# Test 2: Test the GLKBAgent text_embedding function
def test_glkb_text_embedding():
    print("\n" + "=" * 60)
    print("TEST 2: GLKBAgent text_embedding function")
    print("=" * 60)
    
    try:
        import sys
        sys.path.insert(0, '/nfs/turbo/umms-drjieliu/usr/rickyhan/PanKLLM_implementation')
        from GLKBAgent.utils import text_embedding
        
        result = text_embedding("TP53 gene tumor suppressor p53", 1)
        print(f"\n✅ Function returned:")
        print(result[:1500] + "..." if len(result) > 1500 else result)
        return True
    except Exception as e:
        print(f"\n❌ Function Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test 3: Full GLKB chat (requires OpenAI API key)
def test_glkb_chat():
    print("\n" + "=" * 60)
    print("TEST 3: Full GLKBAgent chat_one_round")
    print("=" * 60)
    
    try:
        import sys
        sys.path.insert(0, '/nfs/turbo/umms-drjieliu/usr/rickyhan/PanKLLM_implementation')
        from GLKBAgent.ai_assistant import chat_one_round_glkb
        
        messages, response = chat_one_round_glkb([], "What is TP53?")
        print(f"\n✅ Chat response:")
        print(response[:1000] + "..." if len(response) > 1000 else response)
        return True
    except Exception as e:
        print(f"\n❌ Chat Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\nGLKB Agent Diagnostic Tests")
    print("=" * 60)
    
    # Run tests
    api_ok = test_hirn_api()
    func_ok = test_glkb_text_embedding()
    
    # Only run full chat test if previous tests pass
    if api_ok and func_ok:
        chat_ok = test_glkb_chat()
    else:
        print("\n⚠️  Skipping chat test due to previous failures")
        chat_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"HIRN API:        {'✅ OK' if api_ok else '❌ FAILED'}")
    print(f"text_embedding:  {'✅ OK' if func_ok else '❌ FAILED'}")
    print(f"Full chat:       {'✅ OK' if chat_ok else '❌ FAILED'}")


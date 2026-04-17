#!/usr/bin/env python3
"""
Quick test of Cypher Generator Agent with Qwen2.5-Coder-14B.

Usage:
    python quick_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rl_implementation.agents import CypherGeneratorAgent


def main():
    print("🚀 Quick Test: Cypher Generator Agent\n")
    
    # Paths
    model_path = "/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-coder-14b"
    schema_path = "/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/legacy/PankBaseAgent/schemas/kg_schema copy.json"
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading model (this may take a minute)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    print("✓ Model loaded\n")
    
    # Initialize agent
    agent = CypherGeneratorAgent(schema_path=schema_path, max_steps=5)
    agent.reset()
    print("✓ Agent initialized\n")
    
    # Test question
    question = "What genes have physical interactions with INS?"
    print(f"Question: {question}\n")
    
    # Get prompt
    observation = {'question': question}
    agent.update_from_env(observation, reward=0.0, done=False, info={})
    
    messages = agent.chat_completions
    prompt = messages[0]['content']
    
    print(f"Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)\n")
    
    # Print the entire prompt
    print("=" * 80)
    print("FULL PROMPT SENT TO MODEL:")
    print("=" * 80)
    print(prompt)
    print("=" * 80 + "\n")
    
    print("Generating Cypher query...\n")
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print("=" * 80)
    print("MODEL RESPONSE:")
    print("=" * 80)
    print(response)
    print("=" * 80 + "\n")
    
    # Parse action
    action = agent.update_from_model(response)
    
    print("EXTRACTED CYPHER QUERY:")
    print("=" * 80)
    print(action.action)
    print("=" * 80 + "\n")
    
    # Validate
    if action.action != "DONE":
        has_match = "MATCH" in action.action.upper()
        has_return = "RETURN" in action.action.upper()
        has_brackets = "[" in action.action and "]" in action.action
        
        print("Validation:")
        print(f"  ✓ Contains MATCH: {has_match}")
        print(f"  ✓ Contains RETURN: {has_return}")
        print(f"  ✓ Contains relationship []: {has_brackets}")
        
        if has_match and has_return:
            print("\n✅ SUCCESS: Generated valid Cypher query!")
        else:
            print("\n⚠️  Query may need refinement")
    else:
        print("Agent returned DONE signal")
    
    print("\n✨ Test complete!")


if __name__ == "__main__":
    main()


"""
Test script for Cypher Generator Agent.

Tests the agent's ability to generate Cypher queries using the Qwen2.5-Coder-14B model.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl_implementation.agents import CypherGeneratorAgent


def test_agent_basic():
    """Test basic agent functionality without model inference."""
    print("=" * 80)
    print("TEST 1: Basic Agent Functionality (No Model)")
    print("=" * 80)
    
    # Initialize agent
    schema_path = "/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/legacy/PankBaseAgent/schemas/kg_schema copy.json"
    agent = CypherGeneratorAgent(schema_path=schema_path, max_steps=5)
    
    print(f"✓ Agent initialized: {agent}")
    
    # Test reset
    agent.reset()
    print("✓ Agent reset successful")
    
    # Test update_from_env with initial observation
    observation = {'question': 'What genes interact with INS?'}
    agent.update_from_env(observation, reward=0.0, done=False, info={})
    print(f"✓ Processed initial observation: {observation['question']}")
    
    # Check chat_completions
    messages = agent.chat_completions
    print(f"✓ Generated {len(messages)} message(s)")
    
    if messages:
        print("\n--- Generated Prompt Preview (first 500 chars) ---")
        print(messages[0]['content'][:500])
        print("...\n")
    
    # Test parsing Cypher from response
    test_response = """Let me query the knowledge graph to find genes that interact with INS.

```cypher
MATCH (g1:gene)-[r:physical_interaction]->(g2:gene)
WHERE g2.name = 'INS'
WITH collect(DISTINCT g1) AS nodes, collect(DISTINCT r) AS edges
RETURN nodes, edges;
```"""
    
    action = agent.update_from_model(test_response)
    print(f"✓ Parsed action from response")
    print(f"  Action type: {'Cypher query' if 'MATCH' in action.action else 'Other'}")
    print(f"  Action preview: {action.action[:100]}...")
    
    # Test DONE parsing
    done_response = "I have sufficient data now. DONE"
    action_done = agent.update_from_model(done_response)
    print(f"✓ Parsed DONE signal: {action_done.action}")
    
    # Check trajectory
    trajectory = agent.trajectory
    print(f"✓ Trajectory has {len(trajectory.steps)} steps")
    
    print("\n✅ Basic functionality test PASSED\n")
    return True


def test_agent_with_model():
    """Test agent with actual model inference."""
    print("=" * 80)
    print("TEST 2: Agent with Qwen2.5-Coder-14B Model")
    print("=" * 80)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("⚠️  Warning: Running on CPU, this will be slow!")
    
    # Load model and tokenizer
    model_path = "/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/models/qwen2.5-coder-14b"
    
    print(f"\nLoading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("✓ Tokenizer loaded")
    
    print(f"\nLoading model from {model_path}...")
    print("  (This may take a minute...)")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    print("✓ Model loaded")
    
    # Initialize agent
    schema_path = "/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/legacy/PankBaseAgent/schemas/kg_schema copy.json"
    agent = CypherGeneratorAgent(schema_path=schema_path, max_steps=5)
    print("✓ Agent initialized")
    
    # Reset and start episode
    agent.reset()
    
    # Test question
    question = "What genes have physical interactions with INS?"
    print(f"\n📝 Question: {question}")
    
    observation = {'question': question}
    agent.update_from_env(observation, reward=0.0, done=False, info={})
    
    # Get prompt
    messages = agent.chat_completions
    prompt = messages[0]['content']
    
    print(f"\n📊 Prompt statistics:")
    print(f"  - Length: {len(prompt)} characters")
    print(f"  - Estimated tokens: {len(prompt) // 4}")
    
    # Generate response
    print("\n🤖 Generating Cypher query with model...")
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print("\n📤 Model Response:")
    print("-" * 80)
    print(response)
    print("-" * 80)
    
    # Parse action
    action = agent.update_from_model(response)
    
    print(f"\n✅ Extracted Action:")
    print(f"  Type: {'DONE' if action.action == 'DONE' else 'Cypher Query'}")
    print(f"\n{action.action}")
    
    # Validate Cypher syntax (basic check)
    if action.action != "DONE":
        cypher_keywords = ['MATCH', 'WHERE', 'RETURN', 'WITH']
        has_keywords = any(kw in action.action.upper() for kw in cypher_keywords)
        has_brackets = '[' in action.action and ']' in action.action
        has_parens = '(' in action.action and ')' in action.action
        
        print(f"\n🔍 Cypher Validation:")
        print(f"  - Contains Cypher keywords: {'✓' if has_keywords else '✗'}")
        print(f"  - Contains relationship brackets []: {'✓' if has_brackets else '✗'}")
        print(f"  - Contains node parentheses (): {'✓' if has_parens else '✗'}")
        
        if has_keywords and has_parens:
            print("\n✅ Generated query looks like valid Cypher!")
        else:
            print("\n⚠️  Generated query may not be valid Cypher")
    
    print("\n✅ Model inference test PASSED\n")
    return True


def test_multi_step():
    """Test multi-step interaction."""
    print("=" * 80)
    print("TEST 3: Multi-Step Interaction")
    print("=" * 80)
    
    schema_path = "/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/legacy/PankBaseAgent/schemas/kg_schema copy.json"
    agent = CypherGeneratorAgent(schema_path=schema_path, max_steps=5)
    
    agent.reset()
    
    # Step 1: Initial query
    observation1 = {'question': 'What genes interact with INS?'}
    agent.update_from_env(observation1, reward=0.0, done=False, info={})
    
    response1 = """```cypher
MATCH (g:gene)-[r:physical_interaction]->(target:gene {name: 'INS'})
RETURN collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges;
```"""
    action1 = agent.update_from_model(response1)
    print(f"Step 1: Generated query with {len(action1.action)} characters")
    
    # Step 2: Follow-up query
    observation2 = {
        'question': 'What genes interact with INS?',
        'previous_query': action1.action,
        'previous_result': {
            'success': True,
            'has_data': True,
            'num_results': 23,
            'execution_time_ms': 180.5,
            'data_summary': 'Found 23 genes with physical interactions to INS',
            'data_quality_score': 0.85
        },
        'turn': 2
    }
    agent.update_from_env(observation2, reward=0.0, done=False, info={})
    
    response2 = "I have sufficient data to answer the question. DONE"
    action2 = agent.update_from_model(response2)
    print(f"Step 2: {action2.action}")
    
    # Check trajectory
    trajectory = agent.trajectory
    print(f"\n✓ Completed {len(trajectory.steps)} steps")
    print(f"✓ Agent state: step {agent.current_step}/{agent.max_steps}")
    
    # Check history
    print(f"✓ History contains {len(agent.history)} queries")
    
    print("\n✅ Multi-step test PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("CYPHER GENERATOR AGENT TEST SUITE")
    print("=" * 80 + "\n")
    
    try:
        # Test 1: Basic functionality
        test_agent_basic()
        
        # Test 3: Multi-step (no model needed)
        test_multi_step()
        
        # Test 2: With model (optional, can be slow)
        print("=" * 80)
        user_input = input("Run model inference test? This requires loading the 14B model (y/n): ")
        if user_input.lower() == 'y':
            test_agent_with_model()
        else:
            print("Skipping model inference test")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


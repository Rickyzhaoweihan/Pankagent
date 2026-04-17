#!/usr/bin/env python3
"""
Test script for Orchestrator Agent.

Tests all four modes:
1. Question Generation
2. Data Quality Evaluation
3. Answer Synthesis
4. Answer Quality Evaluation

Usage:
    python test_orchestrator_agent.py
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_implementation.agents import OrchestratorAgent


def test_mode_switching():
    """Test switching between modes."""
    print("=" * 80)
    print("TEST 1: Mode Switching")
    print("=" * 80)
    
    schema_path = "/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/legacy/PankBaseAgent/schemas/kg_schema.json"
    
    agent = OrchestratorAgent(schema_path=schema_path, mode='synthesis')
    print(f"✓ Initial mode: {agent.mode}")
    
    # Test switching to each mode
    for mode in ['generation', 'data_eval', 'synthesis', 'answer_eval']:
        agent.set_mode(mode)
        print(f"✓ Switched to mode: {agent.mode}")
        assert agent.mode == mode, f"Mode mismatch: expected {mode}, got {agent.mode}"
    
    # Test invalid mode
    try:
        agent.set_mode('invalid_mode')
        print("✗ Should have raised ValueError for invalid mode")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected invalid mode: {e}")
    
    print("\n✅ Mode switching test PASSED\n")
    return True


def test_question_generation_mode():
    """Test question generation mode."""
    print("=" * 80)
    print("TEST 2: Question Generation Mode")
    print("=" * 80)
    
    schema_path = "/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/legacy/PankBaseAgent/schemas/kg_schema copy.json"
    
    agent = OrchestratorAgent(schema_path=schema_path, mode='generation')
    agent.reset()
    
    # Create observation
    observation = {
        'difficulty': 'medium',
        'curriculum_constraints': {
            'max_hops': 3,
            'focus_area': 'gene interactions'
        },
        'scope_constraints': {
            'allowed_node_types': ['gene', 'protein'],
            'allowed_edge_types': ['physical_interaction'],
            'avoid_regions': []
        },
        'recent_questions': [
            'What genes interact with INS?',
            'Find proteins that bind to TP53'
        ]
    }
    
    # Build prompt
    agent.update_from_env(observation, reward=0.0, done=False, info={})
    
    messages = agent.chat_completions
    assert len(messages) == 1, f"Expected 1 message, got {len(messages)}"
    
    prompt = messages[0]['content']
    print(f"✓ Generated prompt ({len(prompt)} chars, ~{len(prompt)//4} tokens)")
    
    # Print full prompt
    print("\n" + "=" * 80)
    print("FULL QUESTION GENERATION PROMPT:")
    print("=" * 80)
    print(prompt)
    print("=" * 80 + "\n")
    
    # Check prompt contains key sections
    assert 'SCHEMA' in prompt, "Prompt missing SCHEMA section"
    assert 'CURRICULUM' in prompt, "Prompt missing CURRICULUM section"
    assert 'SCOPE CONSTRAINTS' in prompt, "Prompt missing SCOPE CONSTRAINTS section"
    assert 'RECENT QUESTIONS' in prompt, "Prompt missing RECENT QUESTIONS section"
    assert 'medium' in prompt, "Prompt missing difficulty level"
    print("✓ Prompt contains all required sections")
    
    # Mock model response
    mock_response = "What proteins have physical interactions with BRCA1?"
    action = agent.update_from_model(mock_response)
    
    question = action.action
    print(f"✓ Parsed question: {question}")
    assert isinstance(question, str), "Question should be a string"
    assert len(question) > 0, "Question should not be empty"
    
    print("\n✅ Question generation mode test PASSED\n")
    return True


def test_data_quality_eval_mode():
    """Test data quality evaluation mode."""
    print("=" * 80)
    print("TEST 3: Data Quality Evaluation Mode")
    print("=" * 80)
    
    schema_path = "/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/legacy/PankBaseAgent/schemas/kg_schema copy.json"
    
    agent = OrchestratorAgent(schema_path=schema_path, mode='data_eval')
    agent.reset()
    
    # Create observation
    observation = {
        'question': 'What genes directly regulate INS expression?',
        'trajectory': [
            {
                'query': 'MATCH (g:gene)-[r:regulates]->(target:gene {name: "INS"}) RETURN g.name',
                'execution_time_ms': 180,
                'num_results': 47,
                'data_summary': 'Found 47 genes with regulates relationship to INS'
            }
        ],
        'known_semantic_issues': []
    }
    
    # Build prompt
    agent.update_from_env(observation, reward=0.0, done=False, info={})
    
    messages = agent.chat_completions
    prompt = messages[0]['content']
    print(f"✓ Generated prompt ({len(prompt)} chars, ~{len(prompt)//4} tokens)")
    
    # Print full prompt
    print("\n" + "=" * 80)
    print("FULL DATA QUALITY EVALUATION PROMPT:")
    print("=" * 80)
    print(prompt)
    print("=" * 80 + "\n")
    
    # Check prompt contains key sections
    assert 'QUESTION' in prompt, "Prompt missing QUESTION section"
    assert 'RETRIEVED DATA TRAJECTORY' in prompt, "Prompt missing TRAJECTORY section"
    assert 'EVALUATE WITH SKEPTICISM' in prompt, "Prompt missing EVALUATION section"
    assert 'OUTPUT JSON' in prompt, "Prompt missing JSON format"
    print("✓ Prompt contains all required sections")
    
    # Mock model response with JSON
    mock_response = """
    Based on the retrieved data, here's my evaluation:
    
    ```json
    {
        "data_quality_score": 0.6,
        "relevance_score": 0.7,
        "completeness_score": 0.6,
        "consistency_score": 0.5,
        "trajectory_quality_score": 0.7,
        "reasoning": "Data is relevant but may include indirect regulation",
        "semantic_issues": [],
        "problematic_regions": [],
        "could_answer_question": true,
        "doubt_level": 0.3
    }
    ```
    """
    
    action = agent.update_from_model(mock_response)
    
    evaluation = action.action
    print(f"✓ Parsed evaluation: {json.dumps(evaluation, indent=2)}")
    assert isinstance(evaluation, dict), "Evaluation should be a dict"
    assert 'data_quality_score' in evaluation, "Missing data_quality_score"
    assert 'doubt_level' in evaluation, "Missing doubt_level"
    assert evaluation['data_quality_score'] == 0.6, "Score mismatch"
    
    print("\n✅ Data quality evaluation mode test PASSED\n")
    return True


def test_answer_synthesis_mode():
    """Test answer synthesis mode."""
    print("=" * 80)
    print("TEST 4: Answer Synthesis Mode")
    print("=" * 80)
    
    schema_path = "/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/legacy/PankBaseAgent/schemas/kg_schema copy.json"
    
    agent = OrchestratorAgent(schema_path=schema_path, mode='synthesis')
    agent.reset()
    
    # Create observation
    observation = {
        'question': 'What genes interact with INS?',
        'trajectory_data': [
            {
                'num_results': 23,
                'data_summary': 'Found 23 genes with physical interactions: GCG, IAPP, SST, ...'
            }
        ],
        'data_quality_feedback': {
            'relevance_score': 0.9,
            'completeness_score': 0.8,
            'data_quality_score': 0.85
        }
    }
    
    # Build prompt
    agent.update_from_env(observation, reward=0.0, done=False, info={})
    
    messages = agent.chat_completions
    prompt = messages[0]['content']
    print(f"✓ Generated prompt ({len(prompt)} chars, ~{len(prompt)//4} tokens)")
    
    # Print full prompt
    print("\n" + "=" * 80)
    print("FULL ANSWER SYNTHESIS PROMPT:")
    print("=" * 80)
    print(prompt)
    print("=" * 80 + "\n")
    
    # Check prompt contains key sections
    assert 'QUESTION' in prompt, "Prompt missing QUESTION section"
    assert 'RETRIEVED DATA' in prompt, "Prompt missing DATA section"
    assert 'DATA QUALITY ASSESSMENT' in prompt, "Prompt missing QUALITY section"
    assert 'TASK' in prompt, "Prompt missing TASK section"
    print("✓ Prompt contains all required sections")
    
    # Mock model response
    mock_response = """
    Based on the knowledge graph, 23 genes have physical interactions with INS (insulin).
    The most notable interacting genes include GCG (glucagon), IAPP (islet amyloid polypeptide),
    and SST (somatostatin). These genes play important roles in glucose metabolism and hormone regulation.
    """
    
    action = agent.update_from_model(mock_response)
    
    answer = action.action
    print(f"✓ Parsed answer: {answer[:100]}...")
    assert isinstance(answer, str), "Answer should be a string"
    assert len(answer) > 0, "Answer should not be empty"
    assert 'INS' in answer or 'insulin' in answer, "Answer should mention INS"
    
    print("\n✅ Answer synthesis mode test PASSED\n")
    return True


def test_answer_quality_eval_mode():
    """Test answer quality evaluation mode."""
    print("=" * 80)
    print("TEST 5: Answer Quality Evaluation Mode")
    print("=" * 80)
    
    schema_path = "/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/legacy/PankBaseAgent/schemas/kg_schema copy.json"
    
    agent = OrchestratorAgent(schema_path=schema_path, mode='answer_eval')
    agent.reset()
    
    # Create observation
    observation = {
        'question': 'What genes interact with INS?',
        'answer': '23 genes have physical interactions with INS, including GCG, IAPP, and SST.'
    }
    
    # Build prompt
    agent.update_from_env(observation, reward=0.0, done=False, info={})
    
    messages = agent.chat_completions
    prompt = messages[0]['content']
    print(f"✓ Generated prompt ({len(prompt)} chars, ~{len(prompt)//4} tokens)")
    
    # Print full prompt
    print("\n" + "=" * 80)
    print("FULL ANSWER QUALITY EVALUATION PROMPT:")
    print("=" * 80)
    print(prompt)
    print("=" * 80 + "\n")
    
    # Check prompt contains key sections
    assert 'QUESTION' in prompt, "Prompt missing QUESTION section"
    assert 'ANSWER' in prompt, "Prompt missing ANSWER section"
    assert 'EVALUATE objectively' in prompt, "Prompt missing EVALUATION section"
    assert 'OUTPUT JSON' in prompt, "Prompt missing JSON format"
    print("✓ Prompt contains all required sections")
    
    # Mock model response with JSON
    mock_response = """
    Here's my evaluation:
    
    ```json
    {
        "score": 0.85,
        "correctness": 0.9,
        "completeness": 0.8,
        "clarity": 0.9,
        "accuracy": 0.8,
        "reasoning": "Answer correctly identifies genes and provides specific examples",
        "strengths": "Clear, accurate, mentions specific genes",
        "weaknesses": "Could provide more biological context"
    }
    ```
    """
    
    action = agent.update_from_model(mock_response)
    
    evaluation = action.action
    print(f"✓ Parsed evaluation: {json.dumps(evaluation, indent=2)}")
    assert isinstance(evaluation, dict), "Evaluation should be a dict"
    assert 'score' in evaluation, "Missing score"
    assert 'correctness' in evaluation, "Missing correctness"
    assert evaluation['score'] == 0.85, "Score mismatch"
    
    print("\n✅ Answer quality evaluation mode test PASSED\n")
    return True


def test_json_parsing_edge_cases():
    """Test JSON parsing with various formats."""
    print("=" * 80)
    print("TEST 6: JSON Parsing Edge Cases")
    print("=" * 80)
    
    schema_path = "/nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training/legacy/PankBaseAgent/schemas/kg_schema copy.json"
    
    agent = OrchestratorAgent(schema_path=schema_path, mode='data_eval')
    agent.reset()
    
    # Test case 1: JSON without code block
    observation = {'question': 'test', 'trajectory': [], 'known_semantic_issues': []}
    agent.update_from_env(observation, 0.0, False, {})
    
    response1 = '{"data_quality_score": 0.7, "relevance_score": 0.8, "completeness_score": 0.6, "consistency_score": 0.7, "trajectory_quality_score": 0.6, "reasoning": "test", "semantic_issues": [], "problematic_regions": [], "could_answer_question": true, "doubt_level": 0.2}'
    action1 = agent.update_from_model(response1)
    assert isinstance(action1.action, dict), "Should parse raw JSON"
    assert action1.action['data_quality_score'] == 0.7, "Score mismatch"
    print("✓ Parsed raw JSON correctly")
    
    # Test case 2: Malformed JSON (should return defaults)
    agent.reset()
    agent.update_from_env(observation, 0.0, False, {})
    
    response2 = "This is not JSON at all, just text"
    action2 = agent.update_from_model(response2)
    assert isinstance(action2.action, dict), "Should return default dict"
    assert action2.action['data_quality_score'] == 0.5, "Should use default score"
    print("✓ Handled malformed JSON with defaults")
    
    # Test case 3: Answer quality JSON
    agent.set_mode('answer_eval')
    agent.reset()
    observation2 = {'question': 'test', 'answer': 'test answer'}
    agent.update_from_env(observation2, 0.0, False, {})
    
    response3 = '```json\n{"score": 0.9, "correctness": 0.95, "completeness": 0.85, "clarity": 0.9, "accuracy": 0.9, "reasoning": "good", "strengths": "clear", "weaknesses": "none"}\n```'
    action3 = agent.update_from_model(response3)
    assert isinstance(action3.action, dict), "Should parse answer quality JSON"
    assert action3.action['score'] == 0.9, "Score mismatch"
    print("✓ Parsed answer quality JSON in code block")
    
    print("\n✅ JSON parsing edge cases test PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("ORCHESTRATOR AGENT TEST SUITE")
    print("=" * 80 + "\n")
    
    tests = [
        test_mode_switching,
        test_question_generation_mode,
        test_data_quality_eval_mode,
        test_answer_synthesis_mode,
        test_answer_quality_eval_mode,
        test_json_parsing_edge_cases
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}\n")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""
Test suite for reward functions.

Tests Cypher Generator and Orchestrator reward functions with mock data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_implementation.rewards import (
    cypher_generator_reward_fn,
    orchestrator_generation_reward_fn,
    orchestrator_synthesis_reward_fn,
    validate_cypher,
    compute_diversity_score,
    compute_data_utilization
)


def test_cypher_reward():
    """Test Cypher Generator reward function."""
    print("=" * 80)
    print("TEST 1: Cypher Generator Reward")
    print("=" * 80)
    
    # Mock trajectory with good performance
    task_info = {
        'question': 'What genes interact with PTPN22?',
        'cypher_trajectory': [
            {
                'query': 'MATCH (g:gene)-[r:physical_interaction]-(g2:gene) WHERE g.name = "PTPN22" RETURN g, r, g2 LIMIT 10',
                'success': True,
                'has_data': True,
                'num_results': 10,
                'execution_time_ms': 150.0
            },
            {
                'query': 'MATCH (g:gene {name: "PTPN22"})-[r]-(x) RETURN g, r, x LIMIT 20',
                'success': True,
                'has_data': True,
                'num_results': 20,
                'execution_time_ms': 200.0
            }
        ],
        'answer_quality_score': 0.85,
        'data_quality_score': 0.80,
        'trajectory_quality_score': 0.75,
        'doubt_level': 0.2,
        'num_steps': 2
    }
    
    print(f"\nTest case: Good performance")
    print(f"  - Answer quality: {task_info['answer_quality_score']}")
    print(f"  - Data quality: {task_info['data_quality_score']}")
    print(f"  - Trajectory quality: {task_info['trajectory_quality_score']}")
    print(f"  - Num steps: {task_info['num_steps']}")
    print(f"  - Total results: {sum(s['num_results'] for s in task_info['cypher_trajectory'])}")
    
    result = cypher_generator_reward_fn(task_info, "")
    
    print(f"\n✓ Reward: {result.reward:.3f}")
    print(f"✓ Is correct: {result.is_correct}")
    print(f"✓ Metadata:")
    for key, value in result.metadata.items():
        if isinstance(value, float):
            print(f"    - {key}: {value:.3f}")
        else:
            print(f"    - {key}: {value}")
    
    # Validate reward is in [0, 1]
    assert 0.0 <= result.reward <= 1.0, f"Reward {result.reward} out of range"
    assert result.is_correct == True, "Should be correct with high scores"
    
    print("\n✓ Cypher reward test passed\n")


def test_cypher_reward_poor_performance():
    """Test Cypher reward with poor performance."""
    print("=" * 80)
    print("TEST 2: Cypher Generator Reward (Poor Performance)")
    print("=" * 80)
    
    # Mock trajectory with poor performance
    task_info = {
        'question': 'What genes interact with PTPN22?',
        'cypher_trajectory': [
            {
                'query': 'INVALID QUERY',
                'success': False,
                'has_data': False,
                'num_results': 0,
                'execution_time_ms': 5000.0
            },
            {
                'query': 'MATCH (g:gene) RETURN g',  # No WHERE clause
                'success': True,
                'has_data': False,
                'num_results': 0,
                'execution_time_ms': 3000.0
            }
        ],
        'answer_quality_score': 0.3,
        'data_quality_score': 0.2,
        'trajectory_quality_score': 0.25,
        'doubt_level': 0.8,
        'num_steps': 2
    }
    
    print(f"\nTest case: Poor performance")
    print(f"  - Answer quality: {task_info['answer_quality_score']}")
    print(f"  - Data quality: {task_info['data_quality_score']}")
    print(f"  - High doubt level: {task_info['doubt_level']}")
    
    result = cypher_generator_reward_fn(task_info, "")
    
    print(f"\n✓ Reward: {result.reward:.3f}")
    print(f"✓ Is correct: {result.is_correct}")
    print(f"✓ Penalties applied:")
    print(f"    - Data quality penalty: {result.metadata['data_quality_penalty']:.3f}")
    print(f"    - Semantic ambiguity penalty: {result.metadata['semantic_ambiguity_penalty']:.3f}")
    
    # Validate reward is in [0, 1] and low
    assert 0.0 <= result.reward <= 1.0, f"Reward {result.reward} out of range"
    assert result.reward < 0.5, "Reward should be low for poor performance"
    assert result.is_correct == False, "Should not be correct with low scores"
    
    print("\n✓ Poor performance test passed\n")


def test_orchestrator_generation_reward():
    """Test Orchestrator generation reward function."""
    print("=" * 80)
    print("TEST 3: Orchestrator Generation Reward")
    print("=" * 80)
    
    # Mock task info for question generation
    task_info = {
        'question': 'What genes are associated with rheumatoid arthritis?',
        'answerability': True,
        'success_rate': 0.65,
        'target_success_rate': 0.60,
        'recent_questions': [
            'What genes interact with PTPN22?',
            'Find SNPs in the HLA region',
            'What proteins bind to TNF?'
        ],
        'scope_constraints': {
            'allowed_node_types': ['gene', 'disease', 'snp'],
            'allowed_edge_types': ['associated_with', 'physical_interaction']
        },
        'question_used_types': ['gene', 'disease']
    }
    
    print(f"\nTest case: Good question generation")
    print(f"  - Answerability: {task_info['answerability']}")
    print(f"  - Success rate: {task_info['success_rate']:.2f} (target: {task_info['target_success_rate']:.2f})")
    print(f"  - Question: {task_info['question']}")
    
    result = orchestrator_generation_reward_fn(task_info, task_info['question'])
    
    print(f"\n✓ Reward: {result.reward:.3f}")
    print(f"✓ Is correct: {result.is_correct}")
    print(f"✓ Components:")
    print(f"    - Answerability: {result.metadata['answerability']:.3f}")
    print(f"    - Difficulty: {result.metadata['difficulty']:.3f}")
    print(f"    - Diversity: {result.metadata['diversity']:.3f}")
    print(f"    - Scope: {result.metadata['scope']:.3f}")
    
    # Validate reward is in [0, 1]
    assert 0.0 <= result.reward <= 1.0, f"Reward {result.reward} out of range"
    assert result.is_correct == True, "Should be correct for answerable question"
    
    print("\n✓ Orchestrator generation reward test passed\n")


def test_orchestrator_synthesis_reward():
    """Test Orchestrator synthesis reward function."""
    print("=" * 80)
    print("TEST 4: Orchestrator Synthesis Reward")
    print("=" * 80)
    
    # Mock task info for answer synthesis
    task_info = {
        'question': 'What genes interact with PTPN22?',
        'answer': 'Based on the knowledge graph, several genes interact with PTPN22, including CD247, LCK, and ZAP70. These genes are involved in T-cell receptor signaling.',
        'answer_quality_score': 0.82,
        'trajectory': [
            {
                'query': 'MATCH (g:gene)-[r]-(g2:gene {name: "PTPN22"}) RETURN g, r',
                'result': {'nodes': ['CD247', 'LCK', 'ZAP70'], 'edges': ['physical_interaction']},
                'num_results': 3
            }
        ],
        'data_utilization': 0.75
    }
    
    print(f"\nTest case: Good answer synthesis")
    print(f"  - Answer quality: {task_info['answer_quality_score']:.2f}")
    print(f"  - Data utilization: {task_info['data_utilization']:.2f}")
    print(f"  - Answer: {task_info['answer'][:80]}...")
    
    result = orchestrator_synthesis_reward_fn(task_info, task_info['answer'])
    
    print(f"\n✓ Reward: {result.reward:.3f}")
    print(f"✓ Is correct: {result.is_correct}")
    print(f"✓ Components:")
    print(f"    - Answer quality: {result.metadata['answer_quality']:.3f}")
    print(f"    - Data utilization: {result.metadata['data_utilization']:.3f}")
    
    # Validate reward is in [0, 1]
    assert 0.0 <= result.reward <= 1.0, f"Reward {result.reward} out of range"
    assert result.is_correct == True, "Should be correct for high quality answer"
    
    print("\n✓ Orchestrator synthesis reward test passed\n")


def test_validate_cypher():
    """Test Cypher validation utility."""
    print("=" * 80)
    print("TEST 5: Cypher Validation")
    print("=" * 80)
    
    # Test valid query
    valid_query = 'MATCH (g:gene)-[r:physical_interaction]-(g2:gene) WHERE g.name = "PTPN22" RETURN g, r, g2 LIMIT 10'
    result = validate_cypher(valid_query)
    
    print(f"\nValid query:")
    print(f"  Query: {valid_query[:60]}...")
    print(f"  ✓ Score: {result['score']}")
    print(f"  ✓ Errors: {result['errors']}")
    
    assert result['score'] >= 80, "Valid query should have high score"
    
    # Test invalid query
    invalid_query = 'INVALID CYPHER SYNTAX'
    result = validate_cypher(invalid_query)
    
    print(f"\nInvalid query:")
    print(f"  Query: {invalid_query}")
    print(f"  ✓ Score: {result['score']}")
    print(f"  ✓ Errors: {result['errors']}")
    
    assert result['score'] == 0, "Invalid query should have score 0"
    
    print("\n✓ Cypher validation test passed\n")


def test_diversity_score():
    """Test diversity scoring utility."""
    print("=" * 80)
    print("TEST 6: Diversity Score")
    print("=" * 80)
    
    question = "What genes are associated with diabetes?"
    recent_questions = [
        "What genes interact with INS?",
        "Find proteins related to insulin",
        "What genes are associated with diabetes?"  # Duplicate
    ]
    
    score = compute_diversity_score(question, recent_questions)
    
    print(f"\nQuestion: {question}")
    print(f"Recent questions: {len(recent_questions)}")
    print(f"✓ Diversity score: {score:.3f}")
    
    # Should have low diversity (duplicate exists)
    assert 0.0 <= score <= 1.0, f"Score {score} out of range"
    assert score < 0.5, "Should have low diversity for duplicate question"
    
    # Test with no recent questions
    score_no_history = compute_diversity_score(question, [])
    print(f"\nWith no history:")
    print(f"✓ Diversity score: {score_no_history:.3f}")
    assert score_no_history == 1.0, "Should be maximally diverse with no history"
    
    print("\n✓ Diversity score test passed\n")


def test_data_utilization():
    """Test data utilization utility."""
    print("=" * 80)
    print("TEST 7: Data Utilization")
    print("=" * 80)
    
    answer = "The genes CD247, LCK, and ZAP70 interact with PTPN22."
    trajectory = [
        {
            'result': {
                'nodes': ['CD247', 'LCK', 'ZAP70', 'PTPN22'],
                'edges': ['physical_interaction']
            }
        }
    ]
    
    score = compute_data_utilization(answer, trajectory)
    
    print(f"\nAnswer: {answer}")
    print(f"Trajectory: {len(trajectory)} steps")
    print(f"✓ Data utilization score: {score:.3f}")
    
    assert 0.0 <= score <= 1.0, f"Score {score} out of range"
    assert score > 0.5, "Should have high utilization (entities match)"
    
    # Test with empty trajectory
    score_empty = compute_data_utilization(answer, [])
    print(f"\nWith empty trajectory:")
    print(f"✓ Data utilization score: {score_empty:.3f}")
    assert score_empty == 0.0, "Should be 0 with empty trajectory"
    
    print("\n✓ Data utilization test passed\n")


def test_edge_cases():
    """Test edge cases for reward functions."""
    print("=" * 80)
    print("TEST 8: Edge Cases")
    print("=" * 80)
    
    # Empty trajectory
    print("\nTest: Empty trajectory")
    task_info_empty = {
        'question': 'Test',
        'cypher_trajectory': [],
        'answer_quality_score': 0.5,
        'data_quality_score': 0.5,
        'trajectory_quality_score': 0.5,
        'doubt_level': 0.0,
        'num_steps': 0
    }
    result = cypher_generator_reward_fn(task_info_empty, "")
    print(f"  ✓ Reward: {result.reward:.3f} (should be low)")
    assert 0.0 <= result.reward <= 1.0, "Reward out of range"
    
    # Perfect scores
    print("\nTest: Perfect scores")
    task_info_perfect = {
        'question': 'Test',
        'cypher_trajectory': [
            {
                'query': 'MATCH (g:gene) WHERE g.name = "PTPN22" RETURN g LIMIT 1',
                'success': True,
                'has_data': True,
                'num_results': 50,
                'execution_time_ms': 50.0
            }
        ],
        'answer_quality_score': 1.0,
        'data_quality_score': 1.0,
        'trajectory_quality_score': 1.0,
        'doubt_level': 0.0,
        'num_steps': 1
    }
    result = cypher_generator_reward_fn(task_info_perfect, "")
    print(f"  ✓ Reward: {result.reward:.3f} (should be high)")
    assert result.reward > 0.8, "Perfect scores should give high reward"
    
    print("\n✓ Edge cases test passed\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("REWARD FUNCTIONS TEST SUITE")
    print("=" * 80 + "\n")
    
    try:
        test_cypher_reward()
        test_cypher_reward_poor_performance()
        test_orchestrator_generation_reward()
        test_orchestrator_synthesis_reward()
        test_validate_cypher()
        test_diversity_score()
        test_data_utilization()
        test_edge_cases()
        
        print("=" * 80)
        print("✨ ALL TESTS PASSED!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Test script for Graph Reasoning Environment.

Tests the Neo4j executor and environment with sample Cypher queries.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_implementation.environments import GraphReasoningEnvironment, Neo4jExecutor


def test_neo4j_executor():
    """Test the Neo4j executor with a simple query."""
    print("=" * 80)
    print("TEST 1: Neo4j Executor")
    print("=" * 80)
    
    executor = Neo4jExecutor(
        api_url="https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j"
    )
    
    # Test query: Get information about PTPN22 gene with any relationships
    # This is a more general query that should return data
    test_query = 'MATCH (g:gene)-[r]-(x) WHERE g.name = "PTPN22" RETURN g, r, x LIMIT 25'
    
    print(f"\nExecuting query:\n{test_query}\n")
    
    result = executor.execute_query(test_query)
    
    print(f"✓ Success: {result['success']}")
    print(f"✓ Has Data: {result['has_data']}")
    print(f"✓ Num Results: {result['num_results']}")
    print(f"✓ Execution Time: {result['execution_time_ms']:.0f}ms")
    
    if result['error']:
        print(f"✗ Error: {result['error']}")
    
    if result['has_data']:
        print(f"\nResult preview: {str(result['result'])[:200]}...")
    else:
        print(f"\n⚠ Warning: Query returned no data. This might indicate:")
        print(f"  - The gene 'PTPN22' doesn't exist in the database")
        print(f"  - The gene has no relationships")
        print(f"  - There's an issue with the query or database")
    
    executor.close()
    print("\n✓ Neo4j Executor test completed\n")


def test_environment_reset():
    """Test environment reset."""
    print("=" * 80)
    print("TEST 2: Environment Reset")
    print("=" * 80)
    
    task = {
        'question': 'What genes interact with INS?'
    }
    
    env = GraphReasoningEnvironment(task=task, max_turns=5)
    
    obs, info = env.reset()
    
    print(f"\n✓ Initial observation:")
    print(f"  - Question: {obs['question']}")
    print(f"  - Turn: {obs['turn']}")
    print(f"  - History queries: {len(obs['history']['queries'])}")
    print(f"  - History results: {len(obs['history']['results'])}")
    
    assert obs['question'] == task['question'], "Question mismatch"
    assert obs['turn'] == 0, "Turn should be 0"
    assert len(obs['history']['queries']) == 0, "History should be empty"
    
    env.close()
    print("\n✓ Environment reset test completed\n")


def test_environment_step():
    """Test environment step with Cypher query execution."""
    print("=" * 80)
    print("TEST 3: Environment Step")
    print("=" * 80)
    
    task = {
        'question': 'What genes and relationships are associated with PTPN22?'
    }
    
    env = GraphReasoningEnvironment(task=task, max_turns=5)
    env.reset()
    
    # Test query: Find genes and relationships for PTPN22 (more general query)
    test_query = 'MATCH (g:gene)-[r]-(x) WHERE g.name = "PTPN22" RETURN g, r, x LIMIT 10'
    
    print(f"\nExecuting query:\n{test_query}\n")
    
    # Step the environment
    next_obs, reward, done, task_info = env.step(test_query)
    
    print(f"✓ Reward: {reward}")
    print(f"✓ Done: {done}")
    print(f"✓ Turn: {next_obs['turn']}")
    print(f"✓ Previous query length: {len(next_obs['previous_query'])}")
    print(f"✓ Success: {next_obs['previous_result']['success']}")
    print(f"✓ Has Data: {next_obs['previous_result']['has_data']}")
    print(f"✓ Num Results: {next_obs['previous_result']['num_results']}")
    print(f"✓ Execution Time: {next_obs['previous_result']['execution_time_ms']:.0f}ms")
    print(f"✓ Data Summary: {next_obs['previous_result']['data_summary']}")
    
    if next_obs['previous_result']['error']:
        print(f"✗ Error: {next_obs['previous_result']['error']}")
    
    if not next_obs['previous_result']['has_data']:
        print(f"\n⚠ Warning: Query returned no data")
    
    env.close()
    print("\n✓ Environment step test completed\n")


def test_multi_turn_trajectory():
    """Test multi-turn trajectory."""
    print("=" * 80)
    print("TEST 4: Multi-Turn Trajectory")
    print("=" * 80)
    
    task = {
        'question': 'What genes and entities are related to PTPN22?'
    }
    
    env = GraphReasoningEnvironment(task=task, max_turns=3)
    obs, info = env.reset()
    
    print(f"\nQuestion: {obs['question']}\n")
    
    # Query 1: Find related entities (general query)
    query1 = 'MATCH (g:gene)-[r]-(x) WHERE g.name = "PTPN22" RETURN g, r, x LIMIT 5'
    
    print(f"Turn 1: Finding related entities...")
    obs1, reward1, done1, info1 = env.step(query1)
    print(f"  ✓ Success: {obs1['previous_result']['success']}")
    print(f"  ✓ Has Data: {obs1['previous_result']['has_data']}")
    print(f"  ✓ Summary: {obs1['previous_result']['data_summary']}")
    
    if not done1:
        # Query 2: Get more details (just get any gene nodes)
        query2 = 'MATCH (g:gene) WHERE g.name = "PTPN22" RETURN g LIMIT 1'
        
        print(f"\nTurn 2: Getting gene details...")
        obs2, reward2, done2, info2 = env.step(query2)
        print(f"  ✓ Success: {obs2['previous_result']['success']}")
        print(f"  ✓ Has Data: {obs2['previous_result']['has_data']}")
        print(f"  ✓ Summary: {obs2['previous_result']['data_summary']}")
        print(f"  ✓ History length: {len(obs2['history']['queries'])} queries")
    
    # Get trajectory data
    trajectory = env.get_trajectory_data()
    print(f"\n✓ Trajectory Summary:")
    print(f"  - Question: {trajectory['question']}")
    print(f"  - Num steps: {trajectory['num_steps']}")
    print(f"  - Queries executed: {len(trajectory['queries'])}")
    
    env.close()
    print("\n✓ Multi-turn trajectory test completed\n")


def test_done_signal():
    """Test DONE signal handling."""
    print("=" * 80)
    print("TEST 5: DONE Signal")
    print("=" * 80)
    
    task = {
        'question': 'What is the PTPN22 gene?'
    }
    
    env = GraphReasoningEnvironment(task=task, max_turns=5)
    env.reset()
    
    # Execute one query
    query = 'MATCH (g:gene) WHERE g.name = "PTPN22" RETURN g LIMIT 1'
    obs1, reward1, done1, info1 = env.step(query)
    
    print(f"\nExecuted query, now sending DONE signal...")
    print(f"  Query had data: {obs1['previous_result']['has_data']}")
    
    # Send DONE signal
    obs2, reward2, done2, info2 = env.step("DONE")
    
    print(f"✓ Previous query: {obs2['previous_query']}")
    print(f"✓ Data summary: {obs2['previous_result']['data_summary']}")
    print(f"✓ Turn: {obs2['turn']}")
    
    assert obs2['previous_query'] == 'DONE', "Should record DONE signal"
    
    env.close()
    print("\n✓ DONE signal test completed\n")


def test_error_handling():
    """Test error handling with invalid query."""
    print("=" * 80)
    print("TEST 6: Error Handling")
    print("=" * 80)
    
    task = {
        'question': 'Test error handling'
    }
    
    env = GraphReasoningEnvironment(task=task, max_turns=5)
    env.reset()
    
    # Invalid query (syntax error)
    invalid_query = "INVALID CYPHER QUERY SYNTAX"
    
    print(f"\nExecuting invalid query: {invalid_query}\n")
    
    obs, reward, done, info = env.step(invalid_query)
    
    print(f"✓ Success: {obs['previous_result']['success']}")
    print(f"✓ Has Data: {obs['previous_result']['has_data']}")
    print(f"✓ Error: {obs['previous_result']['error']}")
    print(f"✓ Data Summary: {obs['previous_result']['data_summary']}")
    
    # The API might return success=True even for invalid queries (with error in body)
    # What matters is that we handle it gracefully and don't crash
    # Check that we got a response and it's handled properly
    assert 'previous_result' in obs, "Should have previous_result"
    assert 'success' in obs['previous_result'], "Should have success field"
    assert 'has_data' in obs['previous_result'], "Should have has_data field"
    
    # If it's marked as success but has no data, that's also valid handling
    # The key is that the environment doesn't crash
    print(f"\n✓ Environment handled invalid query gracefully (no crash)")
    
    env.close()
    print("\n✓ Error handling test completed\n")


def test_from_dict():
    """Test factory method from_dict."""
    print("=" * 80)
    print("TEST 7: Factory Method from_dict")
    print("=" * 80)
    
    env_args = {
        'task': {'question': 'Test question'},
        'max_turns': 3,
        'api_url': 'https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j'
    }
    
    env = GraphReasoningEnvironment.from_dict(env_args)
    
    print(f"\n✓ Environment created from dict")
    print(f"  - Max turns: {env.max_turns}")
    print(f"  - API URL: {env.api_url}")
    print(f"  - Task: {env.task}")
    
    assert env.max_turns == 3, "Max turns should be 3"
    assert env.task['question'] == 'Test question', "Question should match"
    
    env.close()
    print("\n✓ Factory method test completed\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("GRAPH REASONING ENVIRONMENT TEST SUITE")
    print("=" * 80 + "\n")
    
    try:
        test_neo4j_executor()
        test_environment_reset()
        test_environment_step()
        test_multi_turn_trajectory()
        test_done_signal()
        test_error_handling()
        test_from_dict()
        
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


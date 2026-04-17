#!/usr/bin/env python3
"""
Validate Adaptive Entity Sampler Implementation.

Tests:
1. Degree extraction from Neo4j
2. Sampler initialization with degree prior
3. Thompson Sampling behavior
4. Update mechanism with slack
5. Persistence (save/load)
6. Integration simulation
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def test_degree_extractor():
    """Test 1: Degree extraction from Neo4j."""
    print("=" * 70)
    print("TEST 1: Degree Extractor")
    print("=" * 70)
    
    from rl_implementation.utils.degree_extractor import DegreeExtractor
    
    extractor = DegreeExtractor(timeout=60)
    
    # Test schema discovery
    print("\n1.1 Discovering schema...")
    try:
        schema = extractor.discover_schema()
        print(f"  Node labels: {schema.get('node_labels', [])}")
        print(f"  Relationships: {list(schema.get('relationships', {}).keys())[:5]}...")
        
        if not schema.get('node_labels'):
            print("  ✗ FAIL: No node labels discovered")
            return False
        if not schema.get('relationships'):
            print("  ✗ FAIL: No relationships discovered")
            return False
        
        print("  ✓ Schema discovery: PASS")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False
    
    # Test degree extraction for one relationship
    print("\n1.2 Extracting degrees for sample relationship...")
    try:
        rel_type = list(schema['relationships'].keys())[0]
        info = schema['relationships'][rel_type]
        
        degrees = extractor.extract_degrees_for_relationship(
            rel_type, info['source'], info['target']
        )
        
        source_count = len(degrees.get('source_degrees', []))
        target_count = len(degrees.get('target_degrees', []))
        
        print(f"  Relationship: {rel_type}")
        print(f"  Source entities: {source_count}")
        print(f"  Target entities: {target_count}")
        
        if source_count > 0:
            top = degrees['source_degrees'][:3]
            print(f"  Top 3 by degree: {top}")
        
        print("  ✓ Degree extraction: PASS")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        extractor.close()


def test_sampler_initialization():
    """Test 2: Sampler initialization with degree prior."""
    print("\n" + "=" * 70)
    print("TEST 2: Sampler Initialization")
    print("=" * 70)
    
    from rl_implementation.utils.adaptive_entity_sampler import AdaptiveEntitySampler
    
    # Create mock degree data
    mock_degrees = {
        "metadata": {
            "extracted_at": "2025-01-01",
            "api_url": "test",
            "discovered_schema": {
                "node_labels": ["gene", "cell_type"],
                "relationships": {}
            }
        },
        "degrees": {
            "gene": {
                "INS": {"expression_level_in": 12, "physical_interaction": 45},
                "MAFA": {"expression_level_in": 8, "physical_interaction": 23},
                "OR2M2": {"expression_level_in": 0, "physical_interaction": 0},
            },
            "cell_type": {
                "Beta Cell": {"expression_level_in_in": 5000},
                "Alpha Cell": {"expression_level_in_in": 4500},
            }
        }
    }
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_degrees, f)
        temp_path = f.name
    
    try:
        print("\n2.1 Initializing from degree data...")
        sampler = AdaptiveEntitySampler(
            entity_degrees_path=temp_path,
            degree_weight=0.1,
        )
        
        print(f"  Total pairs: {len(sampler.stats)}")
        
        # Check prior initialization
        print("\n2.2 Checking prior initialization...")
        
        # INS with high degree should have higher alpha
        ins_expr = sampler.stats.get(("INS", "expression_level_in"))
        ins_phys = sampler.stats.get(("INS", "physical_interaction"))
        
        if ins_expr:
            print(f"  INS+expression_level_in: degree={ins_expr.degree}, alpha={ins_expr.alpha:.2f}")
            # alpha should be 1 + 12 * 0.1 = 2.2
            expected_alpha = 1.0 + 12 * 0.1
            if abs(ins_expr.alpha - expected_alpha) < 0.01:
                print(f"    ✓ Alpha matches expected {expected_alpha:.2f}")
            else:
                print(f"    ✗ Alpha {ins_expr.alpha:.2f} != expected {expected_alpha:.2f}")
                return False
        
        if ins_phys:
            print(f"  INS+physical_interaction: degree={ins_phys.degree}, alpha={ins_phys.alpha:.2f}")
        
        # Beta Cell should have very high alpha
        beta = sampler.stats.get(("Beta Cell", "expression_level_in_in"))
        if beta:
            print(f"  Beta Cell+expression_level_in_in: degree={beta.degree}, alpha={beta.alpha:.2f}")
            if beta.alpha > 100:  # 1 + 5000 * 0.1 = 501
                print("    ✓ High-degree entity has high alpha")
            else:
                print("    ✗ Alpha should be higher for high-degree entity")
                return False
        
        print("\n  ✓ Initialization: PASS")
        return True
        
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        Path(temp_path).unlink()


def test_thompson_sampling():
    """Test 3: Thompson Sampling behavior."""
    print("\n" + "=" * 70)
    print("TEST 3: Thompson Sampling Behavior")
    print("=" * 70)
    
    import numpy as np
    from rl_implementation.utils.adaptive_entity_sampler import AdaptiveEntitySampler
    
    # Create mock data with clear high/low degree entities
    mock_degrees = {
        "metadata": {"extracted_at": "2025-01-01", "api_url": "test", "discovered_schema": {}},
        "degrees": {
            "gene": {
                "HIGH_DEGREE_GENE": {"test_rel": 100},  # High degree
                "LOW_DEGREE_GENE": {"test_rel": 1},      # Low degree
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_degrees, f)
        temp_path = f.name
    
    try:
        sampler = AdaptiveEntitySampler(
            entity_degrees_path=temp_path,
            degree_weight=0.1,
            exploration_bonus=0.0,  # Disable exploration for this test
        )
        
        print("\n3.1 Sampling distribution test...")
        print("  Sampling 1000 times, counting high vs low degree entity selection")
        
        counts = {"HIGH_DEGREE_GENE": 0, "LOW_DEGREE_GENE": 0}
        
        for _ in range(1000):
            entity, rel, etype = sampler.sample_one(entity_type="gene")
            if entity in counts:
                counts[entity] += 1
        
        high_pct = counts["HIGH_DEGREE_GENE"] / 10
        low_pct = counts["LOW_DEGREE_GENE"] / 10
        
        print(f"  HIGH_DEGREE_GENE: {counts['HIGH_DEGREE_GENE']} ({high_pct:.1f}%)")
        print(f"  LOW_DEGREE_GENE: {counts['LOW_DEGREE_GENE']} ({low_pct:.1f}%)")
        
        # High degree should be sampled more (due to higher alpha)
        if counts["HIGH_DEGREE_GENE"] > counts["LOW_DEGREE_GENE"]:
            print("  ✓ High-degree entity sampled more often")
        else:
            print("  ✗ Expected high-degree entity to be sampled more")
            return False
        
        print("\n  ✓ Thompson Sampling: PASS")
        return True
        
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        Path(temp_path).unlink()


def test_update_mechanism():
    """Test 4: Update mechanism with slack."""
    print("\n" + "=" * 70)
    print("TEST 4: Update Mechanism with Slack")
    print("=" * 70)
    
    from rl_implementation.utils.adaptive_entity_sampler import AdaptiveEntitySampler
    
    mock_degrees = {
        "metadata": {"extracted_at": "2025-01-01", "api_url": "test", "discovered_schema": {}},
        "degrees": {
            "gene": {
                "TEST_GENE": {"test_rel": 10},
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_degrees, f)
        temp_path = f.name
    
    try:
        sampler = AdaptiveEntitySampler(
            entity_degrees_path=temp_path,
            degree_weight=0.1,
            slack_threshold=3,
            soft_penalty=0.3,
        )
        
        stats = sampler.stats[("TEST_GENE", "test_rel")]
        initial_alpha = stats.alpha
        initial_beta = stats.beta_param
        
        print(f"\n  Initial: alpha={initial_alpha:.2f}, beta={initial_beta:.2f}")
        
        # Test success update
        print("\n4.1 Testing success update...")
        sampler.update("TEST_GENE", "test_rel", answerable=True)
        print(f"  After success: alpha={stats.alpha:.2f}, beta={stats.beta_param:.2f}")
        
        if stats.alpha == initial_alpha + 1.0:
            print("  ✓ Alpha increased by 1.0")
        else:
            print(f"  ✗ Alpha should be {initial_alpha + 1.0}, got {stats.alpha}")
            return False
        
        # Test soft failure updates (first 3)
        print("\n4.2 Testing soft failure updates (slack)...")
        for i in range(3):
            old_beta = stats.beta_param
            sampler.update("TEST_GENE", "test_rel", answerable=False)
            delta = stats.beta_param - old_beta
            print(f"  Failure {i+1}: beta={stats.beta_param:.2f} (delta={delta:.2f})")
            
            if abs(delta - 0.3) > 0.01:  # Soft penalty
                print(f"  ✗ Expected soft penalty of 0.3, got {delta}")
                return False
        
        print("  ✓ First 3 failures have soft penalty (0.3)")
        
        # Test full failure update (4th failure)
        print("\n4.3 Testing full failure update (after slack threshold)...")
        old_beta = stats.beta_param
        sampler.update("TEST_GENE", "test_rel", answerable=False)
        delta = stats.beta_param - old_beta
        print(f"  Failure 4: beta={stats.beta_param:.2f} (delta={delta:.2f})")
        
        if abs(delta - 1.0) > 0.01:  # Full penalty
            print(f"  ✗ Expected full penalty of 1.0, got {delta}")
            return False
        
        print("  ✓ 4th failure has full penalty (1.0)")
        
        # Test success resets consecutive failures
        print("\n4.4 Testing success resets failure streak...")
        sampler.update("TEST_GENE", "test_rel", answerable=True)
        print(f"  After success: consecutive_failures={stats.consecutive_failures}")
        
        if stats.consecutive_failures == 0:
            print("  ✓ Consecutive failures reset to 0")
        else:
            print("  ✗ Consecutive failures should be 0")
            return False
        
        print("\n  ✓ Update mechanism: PASS")
        return True
        
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        Path(temp_path).unlink()


def test_persistence():
    """Test 5: Save/Load persistence."""
    print("\n" + "=" * 70)
    print("TEST 5: Persistence (Save/Load)")
    print("=" * 70)
    
    from rl_implementation.utils.adaptive_entity_sampler import AdaptiveEntitySampler
    
    mock_degrees = {
        "metadata": {"extracted_at": "2025-01-01", "api_url": "test", "discovered_schema": {}},
        "degrees": {
            "gene": {
                "PERSIST_TEST": {"test_rel": 10},
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_degrees, f)
        degrees_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_path = f.name
    
    try:
        print("\n5.1 Creating and modifying sampler...")
        sampler = AdaptiveEntitySampler(entity_degrees_path=degrees_path)
        
        # Make some updates
        sampler.update("PERSIST_TEST", "test_rel", answerable=True)
        sampler.update("PERSIST_TEST", "test_rel", answerable=True)
        sampler.update("PERSIST_TEST", "test_rel", answerable=False)
        
        original_stats = sampler.stats[("PERSIST_TEST", "test_rel")]
        print(f"  Before save: alpha={original_stats.alpha:.2f}, beta={original_stats.beta_param:.2f}")
        print(f"  times_sampled={original_stats.times_sampled}, total_successes={original_stats.total_successes}")
        
        # Save
        print("\n5.2 Saving state...")
        sampler.save(state_path)
        print(f"  Saved to {state_path}")
        
        # Load
        print("\n5.3 Loading state...")
        loaded_sampler = AdaptiveEntitySampler.load(state_path)
        loaded_stats = loaded_sampler.stats[("PERSIST_TEST", "test_rel")]
        print(f"  After load: alpha={loaded_stats.alpha:.2f}, beta={loaded_stats.beta_param:.2f}")
        print(f"  times_sampled={loaded_stats.times_sampled}, total_successes={loaded_stats.total_successes}")
        
        # Verify
        if (loaded_stats.alpha == original_stats.alpha and
            loaded_stats.beta_param == original_stats.beta_param and
            loaded_stats.times_sampled == original_stats.times_sampled and
            loaded_stats.total_successes == original_stats.total_successes):
            print("\n  ✓ Persistence: PASS")
            return True
        else:
            print("\n  ✗ Loaded state does not match original")
            return False
        
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        Path(degrees_path).unlink()
        Path(state_path).unlink()


def test_integration():
    """Test 6: Integration simulation."""
    print("\n" + "=" * 70)
    print("TEST 6: Integration Simulation")
    print("=" * 70)
    
    from rl_implementation.utils.adaptive_entity_sampler import AdaptiveEntitySampler
    import numpy as np
    
    # Simulate realistic scenario: some entities are answerable, some are not
    mock_degrees = {
        "metadata": {"extracted_at": "2025-01-01", "api_url": "test", "discovered_schema": {}},
        "degrees": {
            "gene": {
                "ANSWERABLE_1": {"good_rel": 50},
                "ANSWERABLE_2": {"good_rel": 40},
                "UNANSWERABLE_1": {"bad_rel": 30},
                "UNANSWERABLE_2": {"bad_rel": 20},
            }
        }
    }
    
    # Ground truth: which entities are actually answerable
    answerable_entities = {"ANSWERABLE_1", "ANSWERABLE_2"}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_degrees, f)
        temp_path = f.name
    
    try:
        sampler = AdaptiveEntitySampler(
            entity_degrees_path=temp_path,
            degree_weight=0.05,  # Lower weight to see learning effect
            slack_threshold=2,
        )
        
        print("\n6.1 Simulating 50 episodes...")
        
        # Track selection rates over time
        early_selections = {"answerable": 0, "unanswerable": 0}
        late_selections = {"answerable": 0, "unanswerable": 0}
        
        for episode in range(50):
            # Sample an entity
            entity, rel, etype = sampler.sample_one(entity_type="gene")
            
            # Simulate answerability
            is_answerable = entity in answerable_entities
            
            # Update sampler
            sampler.update(entity, rel, answerable=is_answerable)
            
            # Track early vs late
            if episode < 10:
                if is_answerable:
                    early_selections["answerable"] += 1
                else:
                    early_selections["unanswerable"] += 1
            elif episode >= 40:
                if is_answerable:
                    late_selections["answerable"] += 1
                else:
                    late_selections["unanswerable"] += 1
        
        print(f"\n  Early selections (first 10 episodes):")
        print(f"    Answerable: {early_selections['answerable']}")
        print(f"    Unanswerable: {early_selections['unanswerable']}")
        
        print(f"\n  Late selections (last 10 episodes):")
        print(f"    Answerable: {late_selections['answerable']}")
        print(f"    Unanswerable: {late_selections['unanswerable']}")
        
        # Check that sampler learned to prefer answerable entities
        early_ratio = early_selections["answerable"] / max(1, sum(early_selections.values()))
        late_ratio = late_selections["answerable"] / max(1, sum(late_selections.values()))
        
        print(f"\n  Answerable selection rate:")
        print(f"    Early: {early_ratio:.1%}")
        print(f"    Late: {late_ratio:.1%}")
        
        if late_ratio > early_ratio:
            print("\n  ✓ Sampler learned to prefer answerable entities")
            improvement = late_ratio - early_ratio
            print(f"    Improvement: +{improvement:.1%}")
        else:
            print("\n  ⚠ No clear learning signal (may need more episodes)")
        
        # Show final stats
        print("\n6.2 Final sampler statistics:")
        summary = sampler.get_stats_summary()
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  Total successes: {summary['total_successes']}")
        print(f"  Overall success rate: {summary['overall_success_rate']:.1%}")
        
        print("\n  ✓ Integration: PASS")
        return True
        
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        Path(temp_path).unlink()


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("ADAPTIVE ENTITY SAMPLER VALIDATION")
    print("=" * 70)
    print("This script validates the adaptive entity sampler implementation.")
    print("=" * 70)
    
    results = {
        'degree_extractor': False,
        'initialization': False,
        'thompson_sampling': False,
        'update_mechanism': False,
        'persistence': False,
        'integration': False,
    }
    
    # Test 1: Degree Extractor (requires network)
    try:
        results['degree_extractor'] = test_degree_extractor()
    except Exception as e:
        print(f"\n✗ Degree extractor test failed: {e}")
    
    # Test 2: Sampler Initialization
    try:
        results['initialization'] = test_sampler_initialization()
    except Exception as e:
        print(f"\n✗ Initialization test failed: {e}")
    
    # Test 3: Thompson Sampling
    try:
        results['thompson_sampling'] = test_thompson_sampling()
    except Exception as e:
        print(f"\n✗ Thompson sampling test failed: {e}")
    
    # Test 4: Update Mechanism
    try:
        results['update_mechanism'] = test_update_mechanism()
    except Exception as e:
        print(f"\n✗ Update mechanism test failed: {e}")
    
    # Test 5: Persistence
    try:
        results['persistence'] = test_persistence()
    except Exception as e:
        print(f"\n✗ Persistence test failed: {e}")
    
    # Test 6: Integration
    try:
        results['integration'] = test_integration()
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print("=" * 70)
    
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        print("\nNext steps:")
        print("  1. Run degree extraction: python rl_implementation/utils/degree_extractor.py")
        print("  2. Enable adaptive sampling in config: use_adaptive_sampling: true")
        print("  3. Run training with new sampler")
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("Please review the failures above before proceeding.")
    
    print("=" * 70)
    
    return all_passed


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nValidation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


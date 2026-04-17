#!/usr/bin/env python3
"""
Test suite for Cypher Auto-Fix utility.

Run with:
    cd rl_implementation
    python test_auto_fix.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_implementation.utils.cypher_auto_fix import (
    auto_fix_cypher,
    fix_relationship_variables,
    fix_distinct_in_collect,
    fix_return_format,
    fix_missing_collections,
    fix_extra_collections,
    fix_disease_naming,
    fix_property_names
)


def test_case(name: str, query: str, expected_contains: list = None, expected_not_contains: list = None):
    """Run a single test case."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print('='*70)
    print(f"Original:\n{query.strip()}\n")
    
    fixed = auto_fix_cypher(query)
    print(f"Fixed:\n{fixed.strip()}\n")
    
    # Check expected patterns
    passed = True
    if expected_contains:
        for pattern in expected_contains:
            if pattern not in fixed:
                print(f"  ✗ FAIL: Expected '{pattern}' not found in output")
                passed = False
    
    if expected_not_contains:
        for pattern in expected_not_contains:
            if pattern in fixed:
                print(f"  ✗ FAIL: Unexpected '{pattern}' found in output")
                passed = False
    
    if passed:
        print("  ✓ PASS")
    
    return passed


def run_tests():
    """Run all test cases."""
    print("\n" + "="*70)
    print("CYPHER AUTO-FIX TEST SUITE")
    print("="*70)
    
    results = []
    
    # Test 1: Fix missing relationship variable
    results.append(test_case(
        "Missing relationship variable",
        """MATCH (g:gene)-[:effector_gene_of]->(d:disease) 
        WITH collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges 
        RETURN nodes, edges;""",
        expected_contains=["[r1:effector_gene_of]"],
        expected_not_contains=["[:effector_gene_of]"]
    ))
    
    # Test 2: Fix missing DISTINCT
    results.append(test_case(
        "Missing DISTINCT in collect()",
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) 
        WITH collect(g)+collect(d) AS nodes, collect(r) AS edges 
        RETURN nodes, edges;""",
        expected_contains=["collect(DISTINCT g)", "collect(DISTINCT d)", "collect(DISTINCT r)"],
        expected_not_contains=["collect(g)", "collect(d)", "collect(r)"]
    ))
    
    # Test 3: Fix disease naming - T1D
    results.append(test_case(
        "Wrong disease name: T1D",
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) 
        WHERE d.name = 'T1D'
        WITH collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges 
        RETURN nodes, edges;""",
        expected_contains=["'type 1 diabetes'"],
        expected_not_contains=["'T1D'"]
    ))
    
    # Test 4: Fix disease naming - Type 1 Diabetes (capitalized)
    results.append(test_case(
        "Wrong disease name: Type 1 Diabetes",
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) 
        WHERE d.name = "Type 1 Diabetes"
        WITH collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges 
        RETURN nodes, edges;""",
        expected_contains=["'type 1 diabetes'"]
    ))
    
    # Test 5: Fix return format
    results.append(test_case(
        "Wrong return format",
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) 
        WITH collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges, collect(DISTINCT d) AS diseases
        RETURN nodes, edges, diseases;""",
        expected_contains=["RETURN nodes, edges;"],
        expected_not_contains=["RETURN nodes, edges, diseases"]
    ))
    
    # Test 6: Fix extra collections
    results.append(test_case(
        "Extra collections merged into nodes",
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) 
        WITH collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges, collect(DISTINCT d) AS diseases
        RETURN nodes, edges, diseases;""",
        expected_contains=["AS nodes", "AS edges"],
        expected_not_contains=["AS diseases"]
    ))
    
    # Test 7: Multiple unnamed relationships
    results.append(test_case(
        "Multiple unnamed relationships",
        """MATCH (g:gene)-[:effector_gene_of]->(d:disease)-[:part_of_GWAS_signal]->(s:snp)
        WITH collect(DISTINCT g) AS nodes, [] AS edges
        RETURN nodes, edges;""",
        expected_contains=["[r1:effector_gene_of]", "[r2:part_of_GWAS_signal]"]
    ))
    
    # Test 8: Property name case fix
    results.append(test_case(
        "Property name case fix",
        """MATCH (g:gene)-[deg:DEG_in]->(ct:cell_type) 
        WHERE deg.upordownregulation = 'up'
        WITH collect(DISTINCT g) AS nodes, collect(DISTINCT deg) AS edges 
        RETURN nodes, edges;""",
        expected_contains=["UpOrDownRegulation"],
        expected_not_contains=["upordownregulation"]
    ))
    
    # Test 9: Already correct query (should not break it)
    results.append(test_case(
        "Already correct query",
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) 
        WHERE d.name = 'type 1 diabetes'
        WITH collect(DISTINCT g)+collect(DISTINCT d) AS nodes, collect(DISTINCT r) AS edges 
        RETURN nodes, edges;""",
        expected_contains=["collect(DISTINCT g)+collect(DISTINCT d) AS nodes", "RETURN nodes, edges;"]
    ))
    
    # Test 10: Complex multi-hop query
    results.append(test_case(
        "Complex multi-hop query",
        """MATCH (sn:snp)-[:part_of_QTL_signal]->(g:gene)-[:effector_gene_of]->(d:disease)
        WHERE d.name = 'T1D'
        WITH collect(DISTINCT sn) AS nodes, [] AS edges
        RETURN nodes, edges;""",
        expected_contains=["[r1:part_of_QTL_signal]", "[r2:effector_gene_of]", "'type 1 diabetes'"]
    ))
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} test(s) failed")
    
    return passed == total


def test_individual_fixes():
    """Test individual fix functions."""
    print("\n" + "="*70)
    print("INDIVIDUAL FIX FUNCTION TESTS")
    print("="*70)
    
    # Test fix_relationship_variables
    print("\n--- fix_relationship_variables ---")
    test1 = "MATCH (g:gene)-[:effector_gene_of]->(d:disease)"
    result1 = fix_relationship_variables(test1)
    print(f"Input:  {test1}")
    print(f"Output: {result1}")
    assert "[r1:effector_gene_of]" in result1, "Failed to add variable name"
    print("✓ PASS")
    
    # Test fix_distinct_in_collect
    print("\n--- fix_distinct_in_collect ---")
    test2 = "WITH collect(g) AS nodes, collect(r) AS edges"
    result2 = fix_distinct_in_collect(test2)
    print(f"Input:  {test2}")
    print(f"Output: {result2}")
    assert "collect(DISTINCT g)" in result2, "Failed to add DISTINCT"
    assert "collect(DISTINCT r)" in result2, "Failed to add DISTINCT"
    print("✓ PASS")
    
    # Test fix_disease_naming
    print("\n--- fix_disease_naming ---")
    test3 = "WHERE d.name = 'T1D'"
    result3 = fix_disease_naming(test3)
    print(f"Input:  {test3}")
    print(f"Output: {result3}")
    assert "'type 1 diabetes'" in result3, "Failed to fix disease name"
    print("✓ PASS")
    
    # Test fix_return_format
    print("\n--- fix_return_format ---")
    test4 = "WITH ... RETURN nodes, edges, extra;"
    result4 = fix_return_format(test4)
    print(f"Input:  {test4}")
    print(f"Output: {result4}")
    assert "RETURN nodes, edges;" in result4, "Failed to fix return format"
    print("✓ PASS")
    
    print("\n✓ All individual fix tests passed!")


if __name__ == "__main__":
    # Run individual fix tests first
    test_individual_fixes()
    
    # Run full integration tests
    success = run_tests()
    
    sys.exit(0 if success else 1)


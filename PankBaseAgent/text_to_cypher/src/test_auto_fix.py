#!/usr/bin/env python3
"""
Comprehensive test suite for auto_fix_cypher() function.
Tests various error scenarios to ensure robust error correction.
"""

from cypher_validator import validate_cypher, auto_fix_cypher, format_validation_report


def test_case(name: str, query: str, expected_fixes: list = None):
    """Run a single test case and display results."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print('='*70)
    print(f"Original Query:\n{query}\n")
    
    # Validate original
    validation = validate_cypher(query)
    print(f"Original Score: {validation['score']}/100")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    
    # Apply auto-fix
    fixed, fixes = auto_fix_cypher(query, validation)
    
    if fixes:
        print(f"\nFixes Applied:")
        for fix in fixes:
            print(f"  ✓ {fix}")
        print(f"\nFixed Query:\n{fixed}\n")
        
        # Validate fixed
        validation_fixed = validate_cypher(fixed)
        print(f"Fixed Score: {validation_fixed['score']}/100")
        if validation_fixed['errors']:
            print(f"Remaining Errors: {validation_fixed['errors']}")
        
        # Check if expected fixes were applied
        if expected_fixes:
            for expected in expected_fixes:
                if not any(expected.lower() in fix.lower() for fix in fixes):
                    print(f"⚠️  WARNING: Expected fix '{expected}' was not applied!")
        
        # Success criteria
        if validation_fixed['score'] >= 90:
            print("✅ SUCCESS: Query fixed to 90+ score")
        else:
            print(f"⚠️  PARTIAL: Score improved from {validation['score']} to {validation_fixed['score']}")
    else:
        print("No fixes applied (query may already be correct or unfixable)")
    
    return validation['score'], validation_fixed['score'] if fixes else validation['score'], fixes


def run_all_tests():
    """Run comprehensive test suite."""
    print("\n" + "="*70)
    print("AUTO-FIX COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = []
    
    # Test 1: Missing disease node in collection
    results.append(test_case(
        "Missing disease node",
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) WHERE d.name = 'type 1 diabetes' WITH collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges RETURN nodes, edges;""",
        expected_fixes=["missing node"]
    ))
    
    # Test 2: Extra collections that should be merged
    results.append(test_case(
        "Extra collections (diseases as separate variable)",
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) WHERE d.name = 'type 1 diabetes' WITH collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges, collect(DISTINCT d) AS diseases RETURN nodes, edges, diseases""",
        expected_fixes=["merged", "return"]
    ))
    
    # Test 3: Missing multiple nodes and edges
    results.append(test_case(
        "Missing multiple nodes and edges",
        """MATCH (g:gene)-[r1:regulation]->(g2:gene)-[r2:DEG_in]->(ct:cell_type) WHERE g.name = 'INS' WITH collect(DISTINCT g) AS nodes, collect(DISTINCT r1) AS edges RETURN nodes, edges;""",
        expected_fixes=["missing node", "missing relationship"]
    ))
    
    # Test 4: Missing DISTINCT in collect
    results.append(test_case(
        "Missing DISTINCT keyword",
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) WHERE d.name = 'type 1 diabetes' WITH collect(g)+collect(d) AS nodes, collect(r) AS edges RETURN nodes, edges;""",
        expected_fixes=["distinct"]
    ))
    
    # Test 5: Complex query with multiple issues
    results.append(test_case(
        "Multiple issues combined",
        """MATCH (sn:snp)-[r1:part_of_QTL_signal]->(g:gene)-[r2:effector_gene_of]->(d:disease) WHERE d.name = 'type 1 diabetes' WITH collect(sn) AS nodes, collect(r1) AS edges, collect(DISTINCT g) AS genes RETURN nodes, edges, genes""",
        expected_fixes=["merged", "distinct", "return"]
    ))
    
    # Test 6: Wrong return format only
    results.append(test_case(
        "Wrong return format (extra variable)",
        """MATCH (g:gene) WHERE g.name = 'INS' WITH collect(DISTINCT g) AS nodes, [] AS edges, 'extra' AS metadata RETURN nodes, edges, metadata;""",
        expected_fixes=["return"]
    ))
    
    # Test 7: Already perfect query (should not break it)
    results.append(test_case(
        "Already perfect query",
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) WHERE d.name = 'type 1 diabetes' WITH collect(DISTINCT g)+collect(DISTINCT d) AS nodes, collect(DISTINCT r) AS edges RETURN nodes, edges;""",
        expected_fixes=[]
    ))
    
    # Test 8: Empty edges with missing relationships
    results.append(test_case(
        "Empty edges [] with actual relationships",
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) WHERE d.name = 'type 1 diabetes' WITH collect(DISTINCT g)+collect(DISTINCT d) AS nodes, [] AS edges RETURN nodes, edges;""",
        expected_fixes=["missing relationship"]
    ))
    
    # Test 9: Three-hop path with missing middle node and edge
    results.append(test_case(
        "Three-hop path missing middle variables",
        """MATCH (sn:snp)-[r1:part_of_QTL_signal]->(g:gene)-[r2:effector_gene_of]->(d:disease) WHERE sn.name = 'rs7903146' WITH collect(DISTINCT sn)+collect(DISTINCT d) AS nodes, collect(DISTINCT r1) AS edges RETURN nodes, edges;""",
        expected_fixes=["missing node", "missing relationship"]
    ))
    
    # Test 10: All nodes but missing all edges
    results.append(test_case(
        "All nodes collected but missing all edges",
        """MATCH (g:gene)-[r1:regulation]->(g2:gene)-[r2:DEG_in]->(ct:cell_type) WITH collect(DISTINCT g)+collect(DISTINCT g2)+collect(DISTINCT ct) AS nodes, [] AS edges RETURN nodes, edges;""",
        expected_fixes=["missing relationship"]
    ))
    
    # Test 11: Case insensitivity test
    results.append(test_case(
        "Case variations in WITH/RETURN",
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) where d.name = 'type 1 diabetes' with collect(DISTINCT g) as nodes, collect(DISTINCT r) as edges, collect(DISTINCT d) as diseases return nodes, edges, diseases""",
        expected_fixes=["merged", "return"]
    ))
    
    # Test 12: No semicolon at end
    results.append(test_case(
        "Missing semicolon (should still work)",
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) WHERE d.name = 'type 1 diabetes' WITH collect(DISTINCT g) AS nodes, collect(DISTINCT r) AS edges RETURN nodes, edges""",
        expected_fixes=["missing node"]
    ))
    
    # Test 13: Multiple WHERE clauses with complex pattern
    results.append(test_case(
        "Complex multi-clause query",
        """MATCH (g:gene)-[deg:DEG_in]->(ct:cell_type) WHERE ct.name = 'Beta Cell' AND deg.UpOrDownRegulation = 'up' MATCH (g)-[r:effector_gene_of]->(d:disease) WHERE d.name = 'type 1 diabetes' WITH collect(DISTINCT g) AS nodes, collect(DISTINCT deg) AS edges RETURN nodes, edges;""",
        expected_fixes=["missing node", "missing relationship"]
    ))
    
    # Test 14: Extra spaces and formatting issues
    results.append(test_case(
        "Extra whitespace and formatting",
        """MATCH  (g:gene)  -  [r:effector_gene_of]  ->  (d:disease)  WHERE  d.name = 'type 1 diabetes'  WITH  collect(DISTINCT  g)  AS  nodes  ,  collect(DISTINCT  r)  AS  edges  ,  collect(DISTINCT d) AS diseases  RETURN  nodes  ,  edges, diseases""",
        expected_fixes=["merged", "return"]
    ))
    
    # Test 15: Relationship without node collection
    results.append(test_case(
        "Only relationship collected, no nodes",
        """MATCH (g:gene)-[r:effector_gene_of]->(d:disease) WHERE d.name = 'type 1 diabetes' WITH collect(DISTINCT r) AS edges RETURN edges;""",
        expected_fixes=["missing node"]
    ))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_tests = len(results)
    improved = sum(1 for before, after, fixes in results if after > before)
    perfect = sum(1 for before, after, fixes in results if after >= 90)
    no_change_needed = sum(1 for before, after, fixes in results if not fixes and before >= 90)
    
    print(f"Total Tests: {total_tests}")
    print(f"Queries Improved: {improved}")
    print(f"Queries Fixed to 90+: {perfect}")
    print(f"Already Perfect: {no_change_needed}")
    print(f"Success Rate: {(perfect + no_change_needed) / total_tests * 100:.1f}%")
    
    # Show any failures
    failures = [(i+1, before, after) for i, (before, after, fixes) in enumerate(results) if after < 90 and fixes]
    if failures:
        print(f"\n⚠️  Tests that didn't reach 90+ score:")
        for test_num, before, after in failures:
            print(f"  Test #{test_num}: {before} → {after}")
    else:
        print("\n✅ All tests passed!")


if __name__ == "__main__":
    run_all_tests()


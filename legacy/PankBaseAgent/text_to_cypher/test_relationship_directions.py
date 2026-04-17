#!/usr/bin/env python3
"""
test_relationship_directions.py
Test script to demonstrate relationship direction validation.
"""

import sys
sys.path.insert(0, 'src')

from src.cypher_validator import validate_cypher, format_validation_report


def test_case(name: str, query: str, expected_score_range: tuple):
    """Run a test case and display results."""
    print("=" * 80)
    print(f"TEST: {name}")
    print("=" * 80)
    print(f"\nQuery:\n{query.strip()}\n")
    
    validation = validate_cypher(query)
    print(format_validation_report(validation))
    
    min_score, max_score = expected_score_range
    if min_score <= validation['score'] <= max_score:
        print(f"\n✅ Score {validation['score']} is in expected range [{min_score}, {max_score}]")
    else:
        print(f"\n❌ Score {validation['score']} is NOT in expected range [{min_score}, {max_score}]")
    
    print("=" * 80 + "\n")


def main():
    print("\n" + "🔍 RELATIONSHIP DIRECTION VALIDATION TESTS" + "\n")
    
    # Test 1: Correct direction - effector_gene_of (gene → disease)
    test_case(
        "Correct Direction: effector_gene_of (gene → disease)",
        """
        MATCH (g:gene)-[e:effector_gene_of]->(d:disease)
        WHERE d.name='type 1 diabetes'
        WITH collect(DISTINCT g)+collect(DISTINCT d) AS nodes, collect(DISTINCT e) AS edges
        RETURN nodes, edges;
        """,
        (100, 100)
    )
    
    # Test 2: WRONG direction - effector_gene_of (disease → gene)
    test_case(
        "WRONG Direction: effector_gene_of (disease → gene)",
        """
        MATCH (d:disease)-[e:effector_gene_of]->(g:gene)
        WHERE d.name='type 1 diabetes'
        WITH collect(DISTINCT g)+collect(DISTINCT d) AS nodes, collect(DISTINCT e) AS edges
        RETURN nodes, edges;
        """,
        (70, 80)  # -25 for wrong direction
    )
    
    # Test 3: Correct direction - DEG_in (gene → cell_type)
    test_case(
        "Correct Direction: DEG_in (gene → cell_type)",
        """
        MATCH (g:gene)-[deg:DEG_in]->(ct:cell_type)
        WHERE ct.name='Beta Cell'
        WITH collect(DISTINCT g)+collect(DISTINCT ct) AS nodes, collect(DISTINCT deg) AS edges
        RETURN nodes, edges;
        """,
        (100, 100)
    )
    
    # Test 4: WRONG direction - DEG_in (cell_type → gene)
    test_case(
        "WRONG Direction: DEG_in (cell_type → gene)",
        """
        MATCH (ct:cell_type)-[deg:DEG_in]->(g:gene)
        WHERE ct.name='Beta Cell'
        WITH collect(DISTINCT g)+collect(DISTINCT ct) AS nodes, collect(DISTINCT deg) AS edges
        RETURN nodes, edges;
        """,
        (70, 80)  # -25 for wrong direction
    )
    
    # Test 5: Correct direction - part_of_QTL_signal (snp → gene)
    test_case(
        "Correct Direction: part_of_QTL_signal (snp → gene)",
        """
        MATCH (sn:snp)-[q:part_of_QTL_signal]->(g:gene)
        WHERE g.name='CFTR'
        WITH collect(DISTINCT sn)+collect(DISTINCT g) AS nodes, collect(DISTINCT q) AS edges
        RETURN nodes, edges;
        """,
        (100, 100)
    )
    
    # Test 6: WRONG direction - part_of_QTL_signal (gene → snp)
    test_case(
        "WRONG Direction: part_of_QTL_signal (gene → snp)",
        """
        MATCH (g:gene)-[q:part_of_QTL_signal]->(sn:snp)
        WHERE g.name='CFTR'
        WITH collect(DISTINCT sn)+collect(DISTINCT g) AS nodes, collect(DISTINCT q) AS edges
        RETURN nodes, edges;
        """,
        (70, 80)  # -25 for wrong direction
    )
    
    # Test 7: Backward direction (correct) - effector_gene_of
    test_case(
        "Backward Direction (Correct): disease <-[effector_gene_of]- gene",
        """
        MATCH (d:disease)<-[e:effector_gene_of]-(g:gene)
        WHERE d.name='type 1 diabetes'
        WITH collect(DISTINCT g)+collect(DISTINCT d) AS nodes, collect(DISTINCT e) AS edges
        RETURN nodes, edges;
        """,
        (100, 100)
    )
    
    # Test 8: Backward direction (WRONG) - effector_gene_of
    test_case(
        "Backward Direction (WRONG): gene <-[effector_gene_of]- disease",
        """
        MATCH (g:gene)<-[e:effector_gene_of]-(d:disease)
        WHERE d.name='type 1 diabetes'
        WITH collect(DISTINCT g)+collect(DISTINCT d) AS nodes, collect(DISTINCT e) AS edges
        RETURN nodes, edges;
        """,
        (70, 80)  # -25 for wrong direction
    )
    
    # Test 9: Complex query with multiple relationships (all correct)
    test_case(
        "Complex Query: Multiple Relationships (All Correct)",
        """
        MATCH (sn:snp)-[q:part_of_QTL_signal]->(g:gene)-[e:effector_gene_of]->(d:disease)
        WHERE d.name='type 1 diabetes'
        WITH collect(DISTINCT sn)+collect(DISTINCT g)+collect(DISTINCT d) AS nodes,
             collect(DISTINCT q)+collect(DISTINCT e) AS edges
        RETURN nodes, edges;
        """,
        (100, 100)
    )
    
    # Test 10: Complex query with one wrong direction
    test_case(
        "Complex Query: One WRONG Direction",
        """
        MATCH (sn:snp)-[q:part_of_QTL_signal]->(g:gene)<-[e:effector_gene_of]-(d:disease)
        WHERE d.name='type 1 diabetes'
        WITH collect(DISTINCT sn)+collect(DISTINCT g)+collect(DISTINCT d) AS nodes,
             collect(DISTINCT q)+collect(DISTINCT e) AS edges
        RETURN nodes, edges;
        """,
        (70, 80)  # -25 for wrong direction (effector_gene_of)
    )
    
    # Test 11: Correct - expression_level_in (gene → cell_type)
    test_case(
        "Correct Direction: expression_level_in (gene → cell_type)",
        """
        MATCH (g:gene)-[exp:expression_level_in]->(ct:cell_type)
        WHERE g.name='CFTR'
        WITH collect(DISTINCT g)+collect(DISTINCT ct) AS nodes, collect(DISTINCT exp) AS edges
        RETURN nodes, edges;
        """,
        (100, 100)
    )
    
    # Test 12: Correct - function_annotation (gene → gene_ontology)
    test_case(
        "Correct Direction: function_annotation (gene → gene_ontology)",
        """
        MATCH (g:gene)-[fa:function_annotation]->(go:gene_ontology)
        WHERE g.name='INS'
        WITH collect(DISTINCT g)+collect(DISTINCT go) AS nodes, collect(DISTINCT fa) AS edges
        RETURN nodes, edges;
        """,
        (100, 100)
    )
    
    print("\n" + "=" * 80)
    print("✅ RELATIONSHIP DIRECTION VALIDATION COMPLETE")
    print("=" * 80)
    print("\nKey Benefits:")
    print("  ✓ Catches incorrect relationship directions")
    print("  ✓ Validates source and target node types")
    print("  ✓ Works with forward, backward, and complex queries")
    print("  ✓ Deducts 25 points for direction errors (critical)")
    print("  ✓ Helps LLM fix errors during refinement")
    print("\nExamples of Validated Relationships:")
    print("  • effector_gene_of: gene → disease")
    print("  • DEG_in: gene → cell_type")
    print("  • part_of_QTL_signal: snp → gene")
    print("  • expression_level_in: gene → cell_type")
    print("  • function_annotation: gene → gene_ontology")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


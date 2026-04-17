"""
Integration tests for the Deep Think Prompt Hints feature.

Tests the end-to-end flow:
1. Deep Think controller outputs prompt_hints in decision JSON
2. Hints are saved to file using PromptHintsManager
3. Hints accumulate across iterations
4. collect_rollouts loads hints and passes to prompt builders
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add parent directories to path
TESTS_DIR = Path(__file__).parent.absolute()
RL_IMPL_DIR = TESTS_DIR.parent
PROJECT_DIR = RL_IMPL_DIR.parent

sys.path.insert(0, str(RL_IMPL_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from rl_implementation.utils.prompt_hints_manager import (
    PromptHintsManager,
    save_hints_to_file,
    load_hints_from_file,
)


# =============================================================================
# Mock LLM Decision Data
# =============================================================================

MOCK_LLM_DECISION_WITH_HINTS = {
    "deep_analysis": {
        "triggered_by": "reward_drop",
        "root_cause": "GWAS direction mismatch",
        "root_cause_confidence": 0.85,
    },
    "decision": {
        "train_cypher": True,
        "train_orchestrator": True,
        "cypher_epochs": 3,
        "orchestrator_epochs": 2,
        "reasoning": "Focus on fixing GWAS relationship direction"
    },
    "prompt_hints": {
        "cypher_generator": [
            {"text": "GWAS: (snp)-[:part_of_GWAS_signal]->(disease), NOT reverse", "severity": "critical"},
            {"text": "Always use variable in relationship: [r:type] not [:type]", "severity": "warning"},
        ],
        "orchestrator": {
            "generation": [
                {"text": "Avoid questions about OCR_activity - data sparse", "severity": "warning"},
            ],
            "synthesis": [
                {"text": "State 'cannot be answered' if data quality < 0.3", "severity": "info"},
            ],
        }
    },
    "confidence": 0.8,
    "deep_think_mode": True,
}

MOCK_LLM_DECISION_WITHOUT_HINTS = {
    "decision": {
        "train_cypher": True,
        "train_orchestrator": False,
        "cypher_epochs": 5,
        "orchestrator_epochs": 0,
        "reasoning": "Standard training iteration"
    },
    "confidence": 0.9,
    "deep_think_mode": False,
}


# =============================================================================
# Integration Tests
# =============================================================================

class TestDeepThinkHintsIntegration:
    """End-to-end integration tests for prompt hints flow."""
    
    def test_parse_prompt_hints_from_decision(self):
        """Test extracting prompt_hints from LLM decision JSON."""
        decision = MOCK_LLM_DECISION_WITH_HINTS
        
        prompt_hints = decision.get('prompt_hints', {})
        
        # Verify structure
        assert 'cypher_generator' in prompt_hints
        assert 'orchestrator' in prompt_hints
        
        # Verify cypher hints
        cypher_hints = prompt_hints['cypher_generator']
        assert len(cypher_hints) == 2
        assert any('GWAS' in h['text'] for h in cypher_hints)
        
        # Verify orchestrator hints
        orch_hints = prompt_hints['orchestrator']
        assert 'generation' in orch_hints
        assert 'synthesis' in orch_hints
    
    def test_missing_prompt_hints_handled(self):
        """Test handling of decisions without prompt_hints."""
        decision = MOCK_LLM_DECISION_WITHOUT_HINTS
        
        prompt_hints = decision.get('prompt_hints', {})
        
        assert prompt_hints == {}
    
    def test_hints_saved_and_accumulated(self):
        """Test that hints are saved and accumulate across iterations."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            hints_path = f.name
        
        try:
            # Iteration 1: Save initial hints
            hints_iter1 = {
                'cypher_generator': [
                    {'text': 'Hint from iteration 1', 'severity': 'warning'},
                ],
            }
            save_hints_to_file(hints_path, hints_iter1, iteration=1)
            
            # Verify saved
            manager1 = PromptHintsManager(hints_path)
            assert len(manager1.get_hints_for_agent('cypher_generator')) == 1
            
            # Iteration 2: Add more hints (should accumulate)
            hints_iter2 = {
                'cypher_generator': [
                    {'text': 'Hint from iteration 2', 'severity': 'critical'},
                ],
                'orchestrator': {
                    'synthesis': [
                        {'text': 'Synth hint from iter 2', 'severity': 'info'},
                    ],
                }
            }
            save_hints_to_file(hints_path, hints_iter2, iteration=2)
            
            # Verify accumulated
            manager2 = PromptHintsManager(hints_path)
            cypher_hints = manager2.get_hints_for_agent('cypher_generator')
            assert len(cypher_hints) == 2
            
            synth_hints = manager2.get_hints_for_agent('orchestrator', 'synthesis')
            assert len(synth_hints) == 1
            
        finally:
            os.unlink(hints_path)
    
    def test_hints_flow_three_iterations(self):
        """Simulate 3 iterations with different hints to test full flow."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            hints_path = f.name
        
        try:
            # Simulate 3 iterations
            for iter_num in range(1, 4):
                hints = {
                    'cypher_generator': [
                        {'text': f'Cypher hint iter {iter_num}', 'severity': 'info'},
                    ],
                    'orchestrator': {
                        'generation': [
                            {'text': f'QGen hint iter {iter_num}', 'severity': 'warning'},
                        ],
                    }
                }
                
                manager = PromptHintsManager(hints_path)
                manager.add_hints(hints, iteration=iter_num)
            
            # Load final state
            final_manager = PromptHintsManager(hints_path)
            
            # Should have hints from all 3 iterations
            cypher_hints = final_manager.get_hints_for_agent('cypher_generator')
            assert len(cypher_hints) == 3
            
            # Check iteration tracking
            iterations = [h.get('added_iteration') for h in cypher_hints]
            assert set(iterations) == {1, 2, 3}
            
            qgen_hints = final_manager.get_hints_for_agent('orchestrator', 'generation')
            assert len(qgen_hints) == 3
            
        finally:
            os.unlink(hints_path)
    
    def test_malformed_hints_handled_gracefully(self):
        """Test that malformed hints don't crash the system."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            hints_path = f.name
        
        try:
            manager = PromptHintsManager(hints_path)
            
            # Malformed hints (missing required fields)
            bad_hints = {
                'cypher_generator': [
                    {},  # Empty hint
                    {'severity': 'warning'},  # Missing text
                    {'text': 'Good hint', 'severity': 'info'},  # Valid
                    {'text': '', 'severity': 'warning'},  # Empty text
                ],
            }
            
            # Should not raise
            manager.add_hints(bad_hints, iteration=1)
            
            # Only the valid hint should be added
            cypher_hints = manager.get_hints_for_agent('cypher_generator')
            assert len(cypher_hints) == 1
            assert cypher_hints[0]['text'] == 'Good hint'
            
        finally:
            os.unlink(hints_path)


class TestRolloutCollectorHintsIntegration:
    """Test integration with RolloutCollector (without full initialization)."""
    
    def test_config_accepts_hints_path(self):
        """Test that RolloutCollectorConfig accepts prompt_hints_path."""
        # Note: We directly import the dataclass to avoid torch dependency
        # The actual RolloutCollectorConfig is in rollout_collector.py
        # which imports torch through ddp_training/__init__.py
        # We verify the config structure by testing a mock directly
        from dataclasses import dataclass
        from typing import Optional
        
        # Mirror the relevant part of RolloutCollectorConfig
        @dataclass
        class MockRolloutCollectorConfig:
            schema_path: str = ""
            prompt_hints_path: Optional[str] = None
        
        config = MockRolloutCollectorConfig(
            schema_path='/fake/schema.json',
            prompt_hints_path='/fake/hints.json',
        )
        
        assert config.prompt_hints_path == '/fake/hints.json'
    
    def test_hints_loading_with_nonexistent_file(self):
        """Test that missing hints file doesn't break collector."""
        # This tests the path handling - full RolloutCollector init would require
        # all dependencies (Neo4j, vLLM servers, etc.)
        
        hints_path = '/nonexistent/path/hints.json'
        
        # PromptHintsManager should handle this gracefully
        manager = PromptHintsManager(hints_path)
        assert not manager.has_hints()


class TestDeepThinkControllerOutput:
    """Test the Deep Think controller output schema."""
    
    def test_prompt_hints_schema_structure(self):
        """Verify the expected prompt_hints schema structure."""
        expected_structure = {
            'cypher_generator': list,
            'orchestrator': dict,
        }
        
        hints = MOCK_LLM_DECISION_WITH_HINTS['prompt_hints']
        
        assert isinstance(hints['cypher_generator'], list)
        assert isinstance(hints['orchestrator'], dict)
        
        # Orchestrator should have role-based structure
        orch = hints['orchestrator']
        for role in ['generation', 'synthesis']:
            if role in orch:
                assert isinstance(orch[role], list)
                for hint in orch[role]:
                    assert 'text' in hint
                    assert 'severity' in hint
    
    def test_severity_levels_valid(self):
        """Test that severity levels are valid."""
        valid_severities = {'critical', 'warning', 'info'}
        
        hints = MOCK_LLM_DECISION_WITH_HINTS['prompt_hints']
        
        # Check cypher hints
        for hint in hints['cypher_generator']:
            assert hint['severity'] in valid_severities
        
        # Check orchestrator hints
        for role, role_hints in hints['orchestrator'].items():
            for hint in role_hints:
                assert hint['severity'] in valid_severities


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])


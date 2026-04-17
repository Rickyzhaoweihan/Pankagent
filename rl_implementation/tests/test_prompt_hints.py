"""
Unit tests for the Deep Think Prompt Hints feature.

Tests cover:
- PromptHintsManager: save/load, accumulation, deduplication, max limits
- Cypher PromptBuilder: DEEP THINK GUIDANCE section formatting
- Orchestrator PromptBuilder: role-filtered hints in prompts
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directories to path
TESTS_DIR = Path(__file__).parent.absolute()
RL_IMPL_DIR = TESTS_DIR.parent
PROJECT_DIR = RL_IMPL_DIR.parent

sys.path.insert(0, str(RL_IMPL_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from rl_implementation.utils.prompt_hints_manager import (
    PromptHintsManager,
    MAX_HINTS_PER_AGENT,
    load_hints_from_file,
    save_hints_to_file,
)
from rl_implementation.utils.prompt_builder import PromptBuilder
from rl_implementation.utils.orchestrator_prompt_builder import OrchestratorPromptBuilder


# =============================================================================
# PromptHintsManager Tests
# =============================================================================

class TestPromptHintsManager:
    """Tests for PromptHintsManager class."""
    
    def test_empty_initialization(self):
        """Test manager initializes with empty hints when file doesn't exist."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as f:
            hints_path = f.name
        
        # File doesn't exist
        manager = PromptHintsManager(hints_path)
        
        assert manager.get_hints_for_agent('cypher_generator') == []
        assert manager.get_hints_for_agent('orchestrator', 'generation') == []
        assert not manager.has_hints()
    
    def test_save_and_load(self):
        """Test save/load round-trip."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            hints_path = f.name
        
        try:
            # Create and save hints
            manager = PromptHintsManager(hints_path)
            new_hints = {
                'cypher_generator': [
                    {'text': 'Test cypher hint', 'severity': 'warning'},
                ],
                'orchestrator': {
                    'generation': [
                        {'text': 'Test qgen hint', 'severity': 'info'},
                    ],
                    'synthesis': [
                        {'text': 'Test synth hint', 'severity': 'critical'},
                    ],
                }
            }
            manager.add_hints(new_hints, iteration=1)
            
            # Load in new manager
            manager2 = PromptHintsManager(hints_path)
            
            # Check cypher hints
            cypher_hints = manager2.get_hints_for_agent('cypher_generator')
            assert len(cypher_hints) == 1
            assert cypher_hints[0]['text'] == 'Test cypher hint'
            assert cypher_hints[0]['severity'] == 'warning'
            
            # Check orchestrator hints
            qgen_hints = manager2.get_hints_for_agent('orchestrator', 'generation')
            assert len(qgen_hints) == 1
            assert qgen_hints[0]['text'] == 'Test qgen hint'
            
            synth_hints = manager2.get_hints_for_agent('orchestrator', 'synthesis')
            assert len(synth_hints) == 1
            assert synth_hints[0]['text'] == 'Test synth hint'
            
            assert manager2.has_hints()
        finally:
            os.unlink(hints_path)
    
    def test_deduplication(self):
        """Test that duplicate hints are not added."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            hints_path = f.name
        
        try:
            manager = PromptHintsManager(hints_path)
            
            # Add same hint twice
            hints = {
                'cypher_generator': [
                    {'text': 'Same hint text', 'severity': 'warning'},
                ]
            }
            manager.add_hints(hints, iteration=1)
            manager.add_hints(hints, iteration=2)
            
            # Should only have one
            cypher_hints = manager.get_hints_for_agent('cypher_generator')
            assert len(cypher_hints) == 1
        finally:
            os.unlink(hints_path)
    
    def test_max_hints_limit(self):
        """Test that max hints limit is enforced (oldest dropped)."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            hints_path = f.name
        
        try:
            manager = PromptHintsManager(hints_path)
            
            # Add more than MAX_HINTS_PER_AGENT hints
            for i in range(MAX_HINTS_PER_AGENT + 3):
                hints = {
                    'cypher_generator': [
                        {'text': f'Hint number {i}', 'severity': 'info'},
                    ]
                }
                manager.add_hints(hints, iteration=i)
            
            # Should only have MAX_HINTS_PER_AGENT
            cypher_hints = manager.get_hints_for_agent('cypher_generator')
            assert len(cypher_hints) == MAX_HINTS_PER_AGENT
            
            # Oldest should be dropped (should have hints 3 onwards)
            hint_numbers = [int(h['text'].split()[-1]) for h in cypher_hints]
            assert min(hint_numbers) >= 3
        finally:
            os.unlink(hints_path)
    
    def test_clear_hints(self):
        """Test clearing all hints."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            hints_path = f.name
        
        try:
            manager = PromptHintsManager(hints_path)
            
            # Add hints
            hints = {
                'cypher_generator': [
                    {'text': 'Test hint', 'severity': 'warning'},
                ]
            }
            manager.add_hints(hints, iteration=1)
            assert manager.has_hints()
            
            # Clear
            manager.clear_hints()
            assert not manager.has_hints()
            assert manager.get_hints_for_agent('cypher_generator') == []
        finally:
            os.unlink(hints_path)
    
    def test_corrupt_file_handling(self):
        """Test that corrupt/malformed files are handled gracefully."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            f.write('not valid json {{{')
            hints_path = f.name
        
        try:
            # Should not raise, should return empty hints
            manager = PromptHintsManager(hints_path)
            assert not manager.has_hints()
        finally:
            os.unlink(hints_path)
    
    def test_format_hints_for_prompt(self):
        """Test formatting hints as prompt text."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            hints_path = f.name
        
        try:
            manager = PromptHintsManager(hints_path)
            
            hints = {
                'cypher_generator': [
                    {'text': 'Critical hint', 'severity': 'critical'},
                    {'text': 'Warning hint', 'severity': 'warning'},
                    {'text': 'Info hint', 'severity': 'info'},
                ]
            }
            manager.add_hints(hints, iteration=1)
            
            formatted = manager.format_hints_for_prompt('cypher_generator')
            
            assert 'DEEP THINK GUIDANCE' in formatted
            assert 'CRITICAL' in formatted
            assert 'WARNING' in formatted
            assert 'TIP' in formatted
            assert 'Critical hint' in formatted
        finally:
            os.unlink(hints_path)
    
    def test_empty_hints_returns_empty_string(self):
        """Test that no hints produces empty formatted string."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as f:
            hints_path = f.name
        
        manager = PromptHintsManager(hints_path)
        formatted = manager.format_hints_for_prompt('cypher_generator')
        assert formatted == ''


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_load_hints_from_file(self):
        """Test convenience function for loading hints."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            hints_path = f.name
        
        try:
            # Save hints first
            manager = PromptHintsManager(hints_path)
            manager.add_hints({
                'cypher_generator': [{'text': 'Test', 'severity': 'info'}]
            }, iteration=1)
            
            # Load using convenience function
            hints = load_hints_from_file(hints_path)
            assert 'cypher_generator' in hints
            assert len(hints['cypher_generator']) == 1
        finally:
            os.unlink(hints_path)
    
    def test_save_hints_to_file(self):
        """Test convenience function for saving hints."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            hints_path = f.name
        
        try:
            hints = {
                'orchestrator': {
                    'synthesis': [{'text': 'Synth hint', 'severity': 'warning'}]
                }
            }
            save_hints_to_file(hints_path, hints, iteration=5)
            
            # Load and verify
            manager = PromptHintsManager(hints_path)
            synth_hints = manager.get_hints_for_agent('orchestrator', 'synthesis')
            assert len(synth_hints) == 1
            assert synth_hints[0]['text'] == 'Synth hint'
        finally:
            os.unlink(hints_path)


# =============================================================================
# Cypher PromptBuilder Tests
# =============================================================================

class TestCypherPromptBuilder:
    """Tests for Cypher PromptBuilder with deep_think_hints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = PromptBuilder()
        self.schema = {
            'nodes': {'gene': {'properties': ['name', 'id']}},
            'relationships': [{'type': 'expression_level_in', 'source': 'gene', 'target': 'cell_type'}]
        }
    
    def test_prompt_without_hints(self):
        """Test that prompts work without hints."""
        prompt = self.builder.build_cypher_prompt(
            question="What genes are expressed in beta cells?",
            schema=self.schema,
            history=[],
            learned_rules=[],
            step=1,
            deep_think_hints=None,
        )
        
        assert 'Cypher' in prompt or 'query' in prompt.lower()
        assert 'DEEP THINK GUIDANCE' not in prompt
    
    def test_prompt_with_hints(self):
        """Test that hints are included in prompt."""
        hints = [
            {'text': 'GWAS uses (snp)->disease direction', 'severity': 'critical'},
            {'text': 'Avoid querying OCR_activity', 'severity': 'warning'},
        ]
        
        prompt = self.builder.build_cypher_prompt(
            question="What genes are expressed in beta cells?",
            schema=self.schema,
            history=[],
            learned_rules=[],
            step=1,
            deep_think_hints=hints,
        )
        
        assert 'DEEP THINK GUIDANCE' in prompt
        assert 'GWAS' in prompt
        assert 'CRITICAL' in prompt
        assert 'WARNING' in prompt
    
    def test_empty_hints_no_section(self):
        """Test that empty hints list doesn't add section."""
        prompt = self.builder.build_cypher_prompt(
            question="What genes are expressed in beta cells?",
            schema=self.schema,
            history=[],
            learned_rules=[],
            step=1,
            deep_think_hints=[],
        )
        
        assert 'DEEP THINK GUIDANCE' not in prompt


# =============================================================================
# Orchestrator PromptBuilder Tests
# =============================================================================

class TestOrchestratorPromptBuilder:
    """Tests for Orchestrator PromptBuilder with deep_think_hints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = OrchestratorPromptBuilder()
        self.schema = {
            'nodes': {'gene': {'properties': ['name', 'id']}},
            'relationships': [{'type': 'expression_level_in', 'source': 'gene', 'target': 'cell_type'}]
        }
    
    def test_question_gen_with_hints(self):
        """Test question generation prompt includes hints."""
        hints = [
            {'text': 'Avoid questions about sparse data regions', 'severity': 'warning'},
        ]
        
        prompt = self.builder.build_question_generation_prompt(
            schema=self.schema,
            difficulty='easy',
            curriculum_constraints={'max_hops': 2},
            scope_constraints={},
            recent_questions=['What is gene X?'],
            deep_think_hints=hints,
        )
        
        assert 'DEEP THINK GUIDANCE' in prompt
        assert 'sparse data' in prompt
    
    def test_data_eval_with_hints(self):
        """Test data quality eval prompt includes hints."""
        hints = [
            {'text': 'Check for empty GWAS results', 'severity': 'info'},
        ]
        
        prompt = self.builder.build_data_quality_eval_prompt(
            question="What genes cause diabetes?",
            trajectory=[{'query': 'MATCH (g:gene) RETURN g', 'result': {}}],
            known_semantic_issues=[],
            deep_think_hints=hints,
        )
        
        assert 'DEEP THINK GUIDANCE' in prompt
        assert 'GWAS' in prompt
    
    def test_synthesis_with_hints(self):
        """Test answer synthesis prompt includes hints."""
        hints = [
            {'text': 'State explicitly when data is insufficient', 'severity': 'critical'},
        ]
        
        prompt = self.builder.build_answer_synthesis_prompt(
            question="What genes cause diabetes?",
            trajectory_data=[{'query': 'MATCH...', 'result': {}}],
            data_quality_feedback={'data_quality_score': 0.5},
            deep_think_hints=hints,
        )
        
        assert 'DEEP THINK GUIDANCE' in prompt
        assert 'insufficient' in prompt
    
    def test_answer_eval_with_hints(self):
        """Test answer quality eval prompt includes hints."""
        hints = [
            {'text': 'Penalize hallucinated entity names', 'severity': 'warning'},
        ]
        
        prompt = self.builder.build_answer_quality_eval_prompt(
            question="What genes cause diabetes?",
            answer="Gene X and Y cause diabetes.",
            deep_think_hints=hints,
        )
        
        assert 'DEEP THINK GUIDANCE' in prompt
        assert 'hallucinated' in prompt
    
    def test_no_hints_no_section(self):
        """Test that None hints doesn't add section."""
        prompt = self.builder.build_question_generation_prompt(
            schema=self.schema,
            difficulty='easy',
            curriculum_constraints={},
            scope_constraints={},
            recent_questions=[],
            deep_think_hints=None,
        )
        
        assert 'DEEP THINK GUIDANCE' not in prompt


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])


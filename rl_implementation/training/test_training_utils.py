#!/usr/bin/env python3
"""
Test suite for training utilities.

Tests:
1. RunningStats normalization
2. EMA model updates
3. Training phase determination
4. Curriculum progression logic
5. Checkpoint save/load
6. Validation metrics computation
"""

import sys
import os
import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rl_implementation.training.utils.training_stability import (
    RunningStats,
    update_ema_model,
    should_update_orchestrator,
    get_training_phase,
    detect_reward_drift,
    adjust_ema_decay
)
from rl_implementation.training.utils.curriculum_utils import (
    CurriculumTracker,
    check_curriculum_progression,
    advance_stage,
    regress_stage,
    get_stage_config,
    compute_success_rate
)
from rl_implementation.training.utils.checkpoint_manager import CheckpointManager
from rl_implementation.training.utils.validation import (
    compute_validation_metrics,
    compare_train_val_metrics
)


class TestRunningStats(unittest.TestCase):
    """Test RunningStats for reward normalization."""
    
    def setUp(self):
        self.stats = RunningStats(window_size=10)
    
    def test_initialization(self):
        """Test RunningStats initialization."""
        self.assertEqual(self.stats.window_size, 10)
        self.assertEqual(len(self.stats.stats), 0)
    
    def test_update_and_normalize(self):
        """Test updating stats and normalizing rewards."""
        rewards = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.stats.update(rewards, 'test_agent')
        
        # Check stats were computed
        stats = self.stats.get_stats('test_agent')
        self.assertAlmostEqual(stats['mean'], 0.7, places=5)
        self.assertGreater(stats['std'], 0)
        self.assertEqual(stats['n_samples'], 5)
        
        # Test normalization
        normalized = self.stats.normalize(rewards, 'test_agent')
        self.assertEqual(len(normalized), len(rewards))
        
        # Normalized values should have mean ~0
        self.assertAlmostEqual(np.mean(normalized), 0.0, places=1)
    
    def test_clipping(self):
        """Test that extreme values are clipped."""
        # Create extreme rewards
        rewards = [0.0] * 5 + [100.0]
        self.stats.update(rewards, 'test_agent')
        
        normalized = self.stats.normalize([100.0], 'test_agent')
        
        # Should be clipped to [-10, 10]
        self.assertLessEqual(normalized[0], 10.0)
        self.assertGreaterEqual(normalized[0], -10.0)
    
    def test_multiple_agents(self):
        """Test separate stats for multiple agents."""
        self.stats.update([0.5, 0.6], 'agent1')
        self.stats.update([1.0, 1.1], 'agent2')
        
        stats1 = self.stats.get_stats('agent1')
        stats2 = self.stats.get_stats('agent2')
        
        # Should have different means
        self.assertNotEqual(stats1['mean'], stats2['mean'])


class TestEMAUpdates(unittest.TestCase):
    """Test EMA model updates."""
    
    def test_ema_update(self):
        """Test EMA model parameter updates."""
        # Create simple models
        train_model = nn.Linear(10, 10)
        ema_model = nn.Linear(10, 10)
        
        # Initialize with different weights
        with torch.no_grad():
            for param in train_model.parameters():
                param.fill_(1.0)
            for param in ema_model.parameters():
                param.fill_(0.0)
        
        # Update EMA
        update_ema_model(ema_model, train_model, decay=0.9)
        
        # Check EMA parameters moved towards train parameters
        for ema_param, train_param in zip(ema_model.parameters(), train_model.parameters()):
            # Should be 0.9 * 0.0 + 0.1 * 1.0 = 0.1
            self.assertAlmostEqual(ema_param.mean().item(), 0.1, places=5)


class TestTrainingPhase(unittest.TestCase):
    """Test training phase logic."""
    
    def test_warmup_phase(self):
        """Test warmup phase (epochs 0-4)."""
        for epoch in range(5):
            self.assertFalse(should_update_orchestrator(epoch))
            self.assertEqual(get_training_phase(epoch), 'warmup')
    
    def test_alternating_phase(self):
        """Test alternating phase (epochs 5-19)."""
        for epoch in range(5, 20):
            phase = get_training_phase(epoch)
            self.assertEqual(phase, 'alternating')
            
            # Should update every 3rd epoch
            expected_update = (epoch % 3 == 0)
            self.assertEqual(should_update_orchestrator(epoch), expected_update)
    
    def test_joint_phase(self):
        """Test joint phase (epochs 20+)."""
        # Easy stage: update every epoch
        for epoch in range(20, 25):
            self.assertEqual(get_training_phase(epoch), 'joint')
            self.assertTrue(should_update_orchestrator(epoch, 'easy'))
        
        # Medium/Hard stage: update every 2nd epoch
        for epoch in range(20, 25):
            expected_update = (epoch % 2 == 0)
            self.assertEqual(should_update_orchestrator(epoch, 'medium'), expected_update)
            self.assertEqual(should_update_orchestrator(epoch, 'hard'), expected_update)


class TestRewardDrift(unittest.TestCase):
    """Test reward drift detection."""
    
    def test_no_drift(self):
        """Test when there's no drift."""
        train_metrics = {'answer_quality': 0.7}
        val_metrics = {'answer_quality': 0.65}
        
        drift = detect_reward_drift(train_metrics, val_metrics, threshold=0.2)
        self.assertFalse(drift)
    
    def test_drift_detected(self):
        """Test when drift is detected."""
        train_metrics = {'answer_quality': 0.9}
        val_metrics = {'answer_quality': 0.6}
        
        drift = detect_reward_drift(train_metrics, val_metrics, threshold=0.2)
        self.assertTrue(drift)
    
    def test_adjust_ema_decay(self):
        """Test EMA decay adjustment."""
        # No drift: decay unchanged
        new_decay = adjust_ema_decay(0.99, drift_detected=False)
        self.assertEqual(new_decay, 0.99)
        
        # Drift detected: decay increased
        new_decay = adjust_ema_decay(0.99, drift_detected=True)
        self.assertGreater(new_decay, 0.99)
        self.assertLessEqual(new_decay, 0.995)


class TestCurriculumUtils(unittest.TestCase):
    """Test curriculum learning utilities."""
    
    def test_stage_config(self):
        """Test getting stage configuration."""
        easy_config = get_stage_config('easy')
        self.assertEqual(easy_config['max_hops'], 2)
        self.assertEqual(easy_config['target_success'], 0.7)
        
        medium_config = get_stage_config('medium')
        self.assertEqual(medium_config['max_hops'], 3)
        
        hard_config = get_stage_config('hard')
        self.assertEqual(hard_config['max_hops'], 5)
    
    def test_advance_stage(self):
        """Test stage advancement."""
        self.assertEqual(advance_stage('easy'), 'medium')
        self.assertEqual(advance_stage('medium'), 'hard')
        self.assertEqual(advance_stage('hard'), 'hard')  # Can't advance further
    
    def test_regress_stage(self):
        """Test stage regression."""
        self.assertEqual(regress_stage('hard'), 'medium')
        self.assertEqual(regress_stage('medium'), 'easy')
        self.assertEqual(regress_stage('easy'), 'easy')  # Can't regress further
    
    def test_compute_success_rate(self):
        """Test success rate computation."""
        trajectories = [
            {'answer_quality': {'score': 0.8}},
            {'answer_quality': {'score': 0.6}},
            {'answer_quality': {'score': 0.9}},
            {'answer_quality': {'score': 0.5}}
        ]
        
        success_rate = compute_success_rate(trajectories, threshold=0.7)
        self.assertEqual(success_rate, 0.5)  # 2 out of 4 above 0.7
    
    def test_check_curriculum_progression(self):
        """Test curriculum progression logic."""
        # Not enough episodes
        should_advance, should_regress = check_curriculum_progression(
            [0.8] * 50,  # Only 50 episodes
            'easy',
            window=100
        )
        self.assertFalse(should_advance)
        self.assertFalse(should_regress)
        
        # Should advance (success rate > target + 0.1)
        should_advance, should_regress = check_curriculum_progression(
            [0.85] * 100,  # High success rate for easy (target=0.7)
            'easy',
            window=100
        )
        self.assertTrue(should_advance)
        self.assertFalse(should_regress)
        
        # Should regress (success rate < target - 0.2)
        should_advance, should_regress = check_curriculum_progression(
            [0.45] * 100,  # Low success rate for easy (target=0.7)
            'easy',
            window=100
        )
        self.assertFalse(should_advance)
        self.assertTrue(should_regress)
    
    def test_curriculum_tracker(self):
        """Test CurriculumTracker class."""
        tracker = CurriculumTracker(initial_stage='easy', window_size=10)
        
        self.assertEqual(tracker.current_stage, 'easy')
        
        # Update with high success rate (should eventually advance)
        for i in range(15):
            tracker.update(0.85, epoch=i)
        
        # After enough high success, should have advanced
        self.assertNotEqual(tracker.current_stage, 'easy')
        
        # Check stats
        stats = tracker.get_stats()
        self.assertIn('current_stage', stats)
        self.assertIn('avg_success_rate', stats)


class TestCheckpointManager(unittest.TestCase):
    """Test checkpoint management."""
    
    def setUp(self):
        """Create temporary directory for checkpoints."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = CheckpointManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test CheckpointManager initialization."""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertEqual(self.manager.best_val_score, -float('inf'))
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        # Initially empty
        checkpoints = self.manager.list_checkpoints()
        self.assertEqual(len(checkpoints), 0)
        
        # Create a checkpoint directory manually
        ckpt_dir = Path(self.temp_dir) / "checkpoint_epoch_001"
        ckpt_dir.mkdir()
        
        # Should now list it
        checkpoints = self.manager.list_checkpoints()
        self.assertEqual(len(checkpoints), 1)


class TestValidationUtils(unittest.TestCase):
    """Test validation utilities."""
    
    def test_compute_validation_metrics(self):
        """Test validation metrics computation."""
        results = [
            {
                'question': 'Q1',
                'answer_quality_score': 0.8,
                'data_quality_score': 0.7,
                'trajectory_length': 2,
                'success': True
            },
            {
                'question': 'Q2',
                'answer_quality_score': 0.6,
                'data_quality_score': 0.5,
                'trajectory_length': 3,
                'success': False
            }
        ]
        
        metrics = compute_validation_metrics(results)
        
        self.assertEqual(metrics['n_questions'], 2)
        self.assertAlmostEqual(metrics['avg_answer_quality'], 0.7)
        self.assertAlmostEqual(metrics['avg_data_quality'], 0.6)
        self.assertAlmostEqual(metrics['avg_trajectory_length'], 2.5)
        self.assertAlmostEqual(metrics['success_rate'], 0.5)
    
    def test_compare_train_val_metrics(self):
        """Test train/val metrics comparison."""
        train_metrics = {
            'avg_answer_quality': 0.8,
            'avg_data_quality': 0.75,
            'success_rate': 0.7
        }
        val_metrics = {
            'avg_answer_quality': 0.75,
            'avg_data_quality': 0.7,
            'success_rate': 0.65
        }
        
        # No drift (gaps < 0.2)
        comparison = compare_train_val_metrics(train_metrics, val_metrics, threshold=0.2)
        self.assertFalse(comparison['drift_detected'])
        
        # Drift detected
        val_metrics_drift = {
            'avg_answer_quality': 0.5,  # Large gap
            'avg_data_quality': 0.4,
            'success_rate': 0.3
        }
        comparison = compare_train_val_metrics(train_metrics, val_metrics_drift, threshold=0.2)
        self.assertTrue(comparison['drift_detected'])
        self.assertGreater(len(comparison['warnings']), 0)


def main():
    """Run all tests."""
    print("=" * 80)
    print("Testing Training Utilities")
    print("=" * 80)
    print()
    
    # Run tests
    unittest.main(argv=[''], verbosity=2, exit=False)
    
    print()
    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()


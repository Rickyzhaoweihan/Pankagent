"""
Checkpoint management for saving and loading training state.

Handles:
- Saving all training state (models, buffers, trackers)
- Loading checkpoints
- Tracking best model based on validation
- Cleanup of old checkpoints
"""

import json
import logging
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manage checkpoints for collaborative multi-agent training.
    
    Saves and loads:
    - Cypher Generator model
    - Orchestrator (trainable) model
    - Orchestrator (EMA evaluator) model
    - Experience buffer
    - Reward tracker statistics
    - Curriculum state
    - Training metrics
    """
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_score = -float('inf')
        self.best_checkpoint_path = None
        
        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        epoch: int,
        cypher_gen: Any,
        orch_train: Any,
        orch_eval: Any,
        exp_buffer: Any,
        reward_tracker: Any,
        current_stage: str,
        metrics: Dict[str, Any],
        is_best: bool = False
    ) -> str:
        """
        Save complete training state.
        
        Args:
            epoch: Current epoch number
            cypher_gen: Cypher Generator agent
            orch_train: Trainable Orchestrator agent
            orch_eval: EMA Orchestrator evaluator
            exp_buffer: Experience buffer
            reward_tracker: Reward tracker with running stats
            current_stage: Current curriculum stage
            metrics: Training metrics dictionary
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_epoch_{epoch:03d}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        
        try:
            # Save Cypher Generator model
            cypher_path = checkpoint_path / "cypher_model.pt"
            if hasattr(cypher_gen, 'actor_model'):
                torch.save(cypher_gen.actor_model.state_dict(), cypher_path)
                logger.debug(f"Saved Cypher model to {cypher_path}")
            else:
                logger.debug(f"Cypher trainer has no actor_model")
            
            # Save Orchestrator (trainable) model
            orch_train_path = checkpoint_path / "orchestrator_train_model.pt"
            if hasattr(orch_train, 'actor_model'):
                torch.save(orch_train.actor_model.state_dict(), orch_train_path)
                logger.debug(f"Saved Orchestrator train model to {orch_train_path}")
            else:
                logger.debug(f"Orchestrator trainer has no actor_model")
            
            # Save Orchestrator (EMA evaluator) model
            orch_eval_path = checkpoint_path / "orchestrator_eval_model.pt"
            if orch_eval is not None and hasattr(orch_eval, 'state_dict'):
                torch.save(orch_eval.state_dict(), orch_eval_path)
                logger.debug(f"Saved Orchestrator EMA model to {orch_eval_path}")
            else:
                logger.debug(f"Orchestrator EMA model not initialized yet")
            
            # Save Experience Buffer
            exp_buffer_path = checkpoint_path / "experience_buffer.pkl"
            with open(exp_buffer_path, 'wb') as f:
                pickle.dump(exp_buffer, f)
            logger.debug(f"Saved Experience Buffer to {exp_buffer_path}")
            
            # Save Reward Tracker
            reward_tracker_path = checkpoint_path / "reward_tracker.pkl"
            with open(reward_tracker_path, 'wb') as f:
                pickle.dump(reward_tracker, f)
            logger.debug(f"Saved Reward Tracker to {reward_tracker_path}")
            
            # Save training state
            state = {
                'epoch': epoch,
                'current_stage': current_stage,
                'metrics': metrics,
                'best_val_score': self.best_val_score,
                'timestamp': datetime.now().isoformat()
            }
            
            state_path = checkpoint_path / "training_state.json"
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Saved training state to {state_path}")
            
            # If this is the best model, create a symlink
            if is_best:
                best_link = self.checkpoint_dir / "best_model"
                if best_link.exists():
                    best_link.unlink()
                best_link.symlink_to(checkpoint_name, target_is_directory=True)
                self.best_checkpoint_path = str(checkpoint_path)
                logger.info(f"Marked as best model (val_score={metrics.get('val_score', 0):.3f})")
            
            logger.info(f"✓ Checkpoint saved successfully: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
            # Clean up partial checkpoint
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        cypher_gen: Any,
        orch_train: Any,
        orch_eval: Any
    ) -> Tuple[Any, Any, Any, Any, str, Dict[str, Any]]:
        """
        Load training state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            cypher_gen: Cypher Generator agent (to load state into)
            orch_train: Trainable Orchestrator agent (to load state into)
            orch_eval: EMA Orchestrator evaluator (to load state into)
            
        Returns:
            Tuple of (exp_buffer, reward_tracker, current_stage, metrics)
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            # Load Cypher Generator
            cypher_path = checkpoint_path / "cypher_generator.pt"
            if cypher_path.exists() and hasattr(cypher_gen, 'load_state_dict'):
                cypher_gen.load_state_dict(torch.load(cypher_path))
                logger.debug(f"Loaded Cypher Generator from {cypher_path}")
            
            # Load Orchestrator (trainable)
            orch_train_path = checkpoint_path / "orchestrator_train.pt"
            if orch_train_path.exists() and hasattr(orch_train, 'load_state_dict'):
                orch_train.load_state_dict(torch.load(orch_train_path))
                logger.debug(f"Loaded Orchestrator (train) from {orch_train_path}")
            
            # Load Orchestrator (EMA evaluator)
            orch_eval_path = checkpoint_path / "orchestrator_eval.pt"
            if orch_eval_path.exists() and hasattr(orch_eval, 'load_state_dict'):
                orch_eval.load_state_dict(torch.load(orch_eval_path))
                logger.debug(f"Loaded Orchestrator (eval) from {orch_eval_path}")
            
            # Load Experience Buffer
            exp_buffer_path = checkpoint_path / "experience_buffer.pkl"
            with open(exp_buffer_path, 'rb') as f:
                exp_buffer = pickle.load(f)
            logger.debug(f"Loaded Experience Buffer from {exp_buffer_path}")
            
            # Load Reward Tracker
            reward_tracker_path = checkpoint_path / "reward_tracker.pkl"
            with open(reward_tracker_path, 'rb') as f:
                reward_tracker = pickle.load(f)
            logger.debug(f"Loaded Reward Tracker from {reward_tracker_path}")
            
            # Load training state
            state_path = checkpoint_path / "training_state.json"
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            current_stage = state['current_stage']
            metrics = state['metrics']
            self.best_val_score = state.get('best_val_score', -float('inf'))
            
            logger.info(
                f"✓ Checkpoint loaded successfully: epoch={state['epoch']}, "
                f"stage={current_stage}"
            )
            
            return exp_buffer, reward_tracker, current_stage, metrics
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            raise
    
    def load_latest_checkpoint(
        self,
        cypher_gen: Any,
        orch_train: Any,
        orch_eval: Any
    ) -> Optional[Tuple[Any, Any, Any, Any, str, Dict[str, Any]]]:
        """
        Find and load the most recent checkpoint.
        
        Args:
            cypher_gen: Cypher Generator agent
            orch_train: Trainable Orchestrator agent
            orch_eval: EMA Orchestrator evaluator
            
        Returns:
            Tuple of (exp_buffer, reward_tracker, current_stage, metrics) or None
        """
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            logger.info("No checkpoints found")
            return None
        
        # Get most recent checkpoint
        latest = checkpoints[-1]
        logger.info(f"Loading latest checkpoint: {latest['name']}")
        
        return self.load_checkpoint(
            latest['path'],
            cypher_gen,
            orch_train,
            orch_eval
        )
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint info dictionaries, sorted by epoch
        """
        checkpoints = []
        
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith('checkpoint_epoch_'):
                # Extract epoch number
                try:
                    epoch_str = item.name.split('_')[-1]
                    epoch = int(epoch_str)
                    
                    # Load training state
                    state_path = item / "training_state.json"
                    if state_path.exists():
                        with open(state_path, 'r') as f:
                            state = json.load(f)
                    else:
                        state = {}
                    
                    checkpoints.append({
                        'name': item.name,
                        'path': str(item),
                        'epoch': epoch,
                        'stage': state.get('current_stage', 'unknown'),
                        'timestamp': state.get('timestamp', 'unknown')
                    })
                except (ValueError, IndexError):
                    logger.warning(f"Skipping invalid checkpoint: {item.name}")
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x['epoch'])
        
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """
        Remove old checkpoints, keeping best and last N.
        
        Args:
            keep_last_n: Number of most recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last_n:
            logger.debug(f"Only {len(checkpoints)} checkpoints, no cleanup needed")
            return
        
        # Identify checkpoints to keep
        keep_paths = set()
        
        # Keep best model
        if self.best_checkpoint_path:
            keep_paths.add(self.best_checkpoint_path)
        
        # Keep last N
        for ckpt in checkpoints[-keep_last_n:]:
            keep_paths.add(ckpt['path'])
        
        # Remove others
        removed = 0
        for ckpt in checkpoints:
            if ckpt['path'] not in keep_paths:
                try:
                    shutil.rmtree(ckpt['path'])
                    removed += 1
                    logger.debug(f"Removed old checkpoint: {ckpt['name']}")
                except Exception as e:
                    logger.warning(f"Failed to remove {ckpt['name']}: {e}")
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} old checkpoints (kept {len(keep_paths)})")
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """
        Get path to best checkpoint.
        
        Returns:
            Path to best checkpoint or None
        """
        best_link = self.checkpoint_dir / "best_model"
        if best_link.exists():
            return str(best_link.resolve())
        return self.best_checkpoint_path


#!/usr/bin/env python3
"""
Test PPO Update Only - Load existing rollouts and run PPO.

This script skips the data collection phase and directly tests
the PPO update using pre-collected rollouts from rollouts.jsonl.

Usage:
    CUDA_VISIBLE_DEVICES=7 python3 -m rl_implementation.ddp_training.scripts.test_ppo_only \
        --rollouts_path outputs/stage1_ddp/rollouts.jsonl \
        --config rl_implementation/ddp_training/config/stage1_config.yaml
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE torch import!
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    print(f"Setting CUDA_VISIBLE_DEVICES=3 (default for PPO test)")

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from rl_implementation.ddp_training.ddp_trainer import DDPTrainer, DDPTrainerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RolloutData:
    """Loaded rollout data for PPO."""
    question: str
    prompt: str
    response: str
    reward: float
    success: bool
    difficulty: str


def load_rollouts(rollouts_path: str, max_rollouts: int = None) -> List[RolloutData]:
    """Load rollouts from JSONL file."""
    rollouts = []
    
    with open(rollouts_path, 'r') as f:
        for line_num, line in enumerate(f):
            if max_rollouts and len(rollouts) >= max_rollouts:
                break
            try:
                data = json.loads(line.strip())
                trajectory = data.get('trajectory', {})
                
                # Extract the first step's prompt and response
                steps = trajectory.get('steps', [])
                if not steps:
                    logger.warning(f"Line {line_num}: No steps in trajectory")
                    continue
                
                first_step = steps[0]
                prompt = first_step.get('prompt', '')
                response = first_step.get('response', '')
                
                if not prompt or not response:
                    logger.warning(f"Line {line_num}: Empty prompt or response")
                    continue
                
                rollout = RolloutData(
                    question=data.get('question', ''),
                    prompt=prompt,
                    response=response,
                    reward=trajectory.get('final_reward', 0.0),
                    success=trajectory.get('success', False),
                    difficulty=data.get('difficulty', 'easy'),
                )
                rollouts.append(rollout)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                continue
    
    logger.info(f"Loaded {len(rollouts)} rollouts from {rollouts_path}")
    return rollouts


def prepare_batch(
    tokenizer,
    prompts: List[str],
    responses: List[str],
    max_prompt_length: int = 512,
    max_response_length: int = 256,
    device: str = "cuda:0",
):
    """
    Tokenize prompts and responses into tensors for PPO training.
    
    Returns:
        input_ids: [batch, seq_len] - full sequences (prompt + response)
        attention_mask: [batch, seq_len]
        labels: [batch, seq_len] - -100 for prompt tokens, actual ids for response
    """
    batch_size = len(prompts)
    max_total_length = max_prompt_length + max_response_length
    
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    
    for prompt, response in zip(prompts, responses):
        # Tokenize prompt
        prompt_tokens = tokenizer.encode(
            prompt, 
            add_special_tokens=True,
            max_length=max_prompt_length,
            truncation=True,
        )
        
        # Tokenize response (without special tokens - they're part of prompt)
        response_tokens = tokenizer.encode(
            response,
            add_special_tokens=False,
            max_length=max_response_length,
            truncation=True,
        )
        
        # Combine
        input_ids = prompt_tokens + response_tokens
        prompt_len = len(prompt_tokens)
        
        # Truncate if too long
        if len(input_ids) > max_total_length:
            input_ids = input_ids[:max_total_length]
        
        # Create labels: -100 for prompt (ignored in loss), actual ids for response
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)
    
    # Pad to same length
    max_len = max(len(ids) for ids in all_input_ids)
    pad_token_id = tokenizer.pad_token_id or 0
    
    for i in range(batch_size):
        pad_len = max_len - len(all_input_ids[i])
        all_input_ids[i] = all_input_ids[i] + [pad_token_id] * pad_len
        all_attention_mask[i] = all_attention_mask[i] + [0] * pad_len
        all_labels[i] = all_labels[i] + [-100] * pad_len
    
    # Convert to tensors
    input_ids = torch.tensor(all_input_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(all_attention_mask, dtype=torch.long, device=device)
    labels = torch.tensor(all_labels, dtype=torch.long, device=device)
    
    return input_ids, attention_mask, labels


def main():
    parser = argparse.ArgumentParser(description="Test PPO update with existing rollouts")
    parser.add_argument("--rollouts_path", type=str, required=True,
                        help="Path to rollouts.jsonl file")
    parser.add_argument("--config", type=str, 
                        default="rl_implementation/ddp_training/config/stage1_config.yaml",
                        help="Path to config file")
    parser.add_argument("--max_rollouts", type=int, default=4,
                        help="Max rollouts to use (default: 4 for testing)")
    parser.add_argument("--mini_batch_size", type=int, default=1,
                        help="Mini batch size for PPO (default: 1)")
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Max prompt length in tokens (default: 512)")
    parser.add_argument("--max_response_length", type=int, default = 256,
                        help="Max response length in tokens (default: 256)")
    args = parser.parse_args()
    
    # Print GPU info
    print("=" * 60)
    print("GPU Configuration")
    print("=" * 60)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get model path
    cypher_model_path = config.get('cypher_model_path')
    if not cypher_model_path:
        raise ValueError("cypher_model_path not found in config")
    
    print(f"Model path: {cypher_model_path}")
    print()
    
    # Load rollouts
    print("=" * 60)
    print("Loading Rollouts")
    print("=" * 60)
    rollouts = load_rollouts(args.rollouts_path, max_rollouts=args.max_rollouts)
    
    if not rollouts:
        raise ValueError("No valid rollouts loaded")
    
    # Show sample
    print(f"\nSample rollout:")
    print(f"  Question: {rollouts[0].question[:100]}...")
    print(f"  Prompt length: {len(rollouts[0].prompt)} chars")
    print(f"  Response length: {len(rollouts[0].response)} chars")
    print(f"  Reward: {rollouts[0].reward}")
    print(f"  Success: {rollouts[0].success}")
    print()
    
    # Create DDPTrainer config with reduced memory settings
    trainer_config = DDPTrainerConfig(
        model_path=cypher_model_path,
        learning_rate=config.get('learning_rate', 1e-5),
        lora_rank=8,  # Small for testing
        lora_alpha=16,
        lora_dropout=config.get('lora_dropout', 0.05),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention only
        max_grad_norm=config.get('max_grad_norm', 1.0),
        gpus=[0],  # Single GPU (after CUDA_VISIBLE_DEVICES remapping)
        gradient_checkpointing=True,  # Save memory
    )
    
    print("=" * 60)
    print("Initializing DDPTrainer")
    print("=" * 60)
    print(f"  LoRA rank: {trainer_config.lora_rank}")
    print(f"  Max prompt length: {args.max_prompt_length}")
    print(f"  Max response length: {args.max_response_length}")
    print()
    
    # Clear cache before loading model
    torch.cuda.empty_cache()
    
    # Initialize trainer
    ddp_trainer = DDPTrainer(trainer_config, local_rank=0, world_size=1)
    ddp_trainer.setup()
    
    # Print memory after model load
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i} memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    print()
    
    print("=" * 60)
    print("Running PPO Update")
    print("=" * 60)
    
    # Process rollouts in mini-batches
    total_loss = 0.0
    num_batches = 0
    device = ddp_trainer.device
    
    for i in range(0, len(rollouts), args.mini_batch_size):
        batch = rollouts[i:i + args.mini_batch_size]
        
        print(f"\nProcessing batch {num_batches + 1} ({len(batch)} rollouts)...")
        
        # Extract prompts and responses
        prompts = [r.prompt for r in batch]
        responses = [r.response for r in batch]
        rewards = torch.tensor([r.reward for r in batch], dtype=torch.float32, device=device)
        
        try:
            # Tokenize batch
            print("  Tokenizing...")
            input_ids, attention_mask, labels = prepare_batch(
                ddp_trainer.tokenizer,
                prompts,
                responses,
                max_prompt_length=args.max_prompt_length,
                max_response_length=args.max_response_length,
                device=device,
            )
            print(f"  Input shape: {input_ids.shape}")
            
            # Compute old log probs
            print("  Computing log probs...")
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                old_log_probs = ddp_trainer.compute_log_probs_for_rollout(
                    input_ids, attention_mask, labels
                )
            
            print(f"  Old log probs shape: {old_log_probs.shape}")
            print(f"  Old log probs mean: {old_log_probs.mean().item():.4f}")
            
            # Compute advantages using GRPO (group relative)
            # Handle single sample case where std would be NaN
            if len(rewards) > 1:
                mean_reward = rewards.mean()
                std_reward = rewards.std() + 1e-8
                advantages = (rewards - mean_reward) / std_reward
            else:
                # Single sample: just use reward directly (normalized to 0)
                advantages = torch.zeros_like(rewards)
            
            # Expand advantages to match sequence length
            seq_len = input_ids.shape[1]
            advantages = advantages.unsqueeze(1).expand(-1, seq_len)
            
            print(f"  Advantages: {advantages[:, 0].tolist()}")
            
            # PPO update step
            print("  Running PPO step...")
            torch.cuda.empty_cache()
            
            metrics = ddp_trainer.train_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                advantages=advantages,
                old_log_probs=old_log_probs,
            )
            
            total_loss += metrics.get('loss', 0.0)
            num_batches += 1
            
            print(f"  Loss: {metrics.get('loss', 0.0):.4f}")
            print(f"  Policy loss: {metrics.get('policy_loss', 0.0):.4f}")
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM Error: {e}")
            print("  Clearing cache and skipping batch...")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print()
    print("=" * 60)
    print("PPO Test Complete")
    print("=" * 60)
    print(f"Processed {num_batches} batches")
    if num_batches > 0:
        print(f"Average loss: {total_loss / num_batches:.4f}")
    
    # Final memory stats
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"Final GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


if __name__ == "__main__":
    main()

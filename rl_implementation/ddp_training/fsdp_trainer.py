"""
FSDP Trainer for Cypher Generator with PEFT LoRA.

Provides PyTorch FSDP (Fully Sharded Data Parallel) wrapper for training
with LoRA adapters. FSDP shards model weights, gradients, and optimizer
states across GPUs for maximum memory efficiency.

Memory savings with FULL_SHARD on 2 GPUs:
- Model weights (14B bf16): 28GB → 14GB per GPU
- Optimizer states: 56GB → 28GB per GPU (sharded)
- Gradients: 28GB → 14GB per GPU (sharded)
- Total per GPU: ~40GB → ~25-30GB
"""

import logging
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _make_json_serializable(obj):
    """Recursively convert sets to lists for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    return obj


@dataclass
class FSDPTrainerConfig:
    """Configuration for FSDP trainer."""
    # Model settings
    model_path: str = ""
    
    # Resume from existing LoRA adapter (for cumulative training)
    resume_from_adapter: str = ""
    
    # LoRA settings
    lora_rank: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"  # Attention only
    ])
    
    # Training settings
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 50
    total_steps: int = 10000
    
    # FSDP settings
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE, BACKWARD_POST
    cpu_offload: bool = False  # Offload parameters to CPU (very slow but saves more memory)
    
    # Precision
    use_bf16: bool = True
    gradient_checkpointing: bool = True
    
    # Mini-batch size for memory efficiency during forward pass
    # Set to 1 for maximum memory safety on L40S (doesn't affect training quality)
    mini_batch_size: int = 1
    
    # Checkpoint
    checkpoint_dir: str = ""
    save_steps: int = 500


class FSDPTrainer:
    """
    FSDP Trainer for Cypher Generator with LoRA.
    
    Uses PyTorch FSDP for memory-efficient distributed training.
    FSDP shards model weights, gradients, and optimizer states.
    """
    
    def __init__(
        self,
        config: FSDPTrainerConfig,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize FSDP trainer.
        
        Args:
            config: FSDPTrainerConfig with model and training settings
            local_rank: Local GPU rank
            world_size: Total number of GPUs
        """
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.global_step = 0
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.tokenizer = None
        
        logger.info(f"FSDPTrainer initialized (rank {local_rank}/{world_size})")
    
    def setup(self):
        """Setup model, optimizer, and scheduler with FSDP."""
        self._setup_device()
        self._init_distributed()
        self._load_model_with_fsdp()
        self._setup_optimizer()
        
        logger.info(f"FSDPTrainer setup complete on device cuda:{self.local_rank}")
    
    def _setup_device(self):
        """Setup CUDA device."""
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        logger.info(f"Using cuda:{self.local_rank} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    
    def _init_distributed(self):
        """Initialize distributed training."""
        if self.world_size > 1:
            if not dist.is_initialized():
                import datetime
                timeout = datetime.timedelta(hours=3)
                
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    world_size=self.world_size,
                    rank=self.local_rank,
                    timeout=timeout,
                )
            logger.info(f"Distributed training initialized: rank {self.local_rank}/{self.world_size}")
        else:
            logger.info("Single GPU training mode (no distributed)")
    
    def _get_fsdp_config(self):
        """Get FSDP configuration objects."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            BackwardPrefetch,
            MixedPrecision,
            CPUOffload,
        )
        
        # Sharding strategy
        strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }
        sharding_strategy = strategy_map.get(
            self.config.sharding_strategy, 
            ShardingStrategy.FULL_SHARD
        )
        
        # Backward prefetch
        prefetch_map = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        }
        backward_prefetch = prefetch_map.get(
            self.config.backward_prefetch,
            BackwardPrefetch.BACKWARD_PRE
        )
        
        # Mixed precision
        if self.config.use_bf16:
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        else:
            mixed_precision = None
        
        # CPU offload
        cpu_offload = CPUOffload(offload_params=True) if self.config.cpu_offload else None
        
        return {
            "sharding_strategy": sharding_strategy,
            "backward_prefetch": backward_prefetch,
            "mixed_precision": mixed_precision,
            "cpu_offload": cpu_offload,
        }
    
    def _get_transformer_layer_cls(self) -> Set[Type]:
        """Get the transformer layer class for FSDP wrapping."""
        # Import Qwen2 layer class for wrapping
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
            return {Qwen2DecoderLayer}
        except ImportError:
            logger.warning("Could not import Qwen2DecoderLayer, using default wrapping")
            return set()
    
    def _load_model_with_fsdp(self):
        """Load base model, apply LoRA, and wrap with FSDP."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        
        try:
            from peft import LoraConfig, get_peft_model, TaskType, PeftModel
        except ImportError:
            raise ImportError("PEFT is required. Install with: pip install peft")
        
        logger.info(f"Loading model from {self.config.model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        dtype = torch.bfloat16 if self.config.use_bf16 else torch.float32
        
        # For FSDP, we load the model on CPU first, then FSDP will shard it
        # Using device_map=None to load on CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=None,  # Load on CPU, FSDP will handle distribution
            low_cpu_mem_usage=True,
        )
        
        # Enable gradient checkpointing before applying LoRA
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("Gradient checkpointing enabled")
        
        # Apply LoRA or load existing adapter
        if self.config.resume_from_adapter and Path(self.config.resume_from_adapter).exists():
            adapter_path = Path(self.config.resume_from_adapter)
            adapter_config_path = adapter_path / "adapter_config.json"
            
            if adapter_config_path.exists():
                logger.info(f"🔄 RESUMING from existing LoRA adapter: {adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    str(adapter_path),
                    is_trainable=True,
                )
                # Store the loaded LoRA config for checkpoint saving
                self.lora_config = self.model.peft_config["default"]
                self.model.print_trainable_parameters()
            else:
                logger.warning(f"Adapter config not found, creating new LoRA")
                self._create_new_lora()
        else:
            self._create_new_lora()
        
        # CRITICAL: Cast ALL parameters to uniform dtype for FSDP
        # PEFT/LoRA initializes adapters in float32, but base model is bfloat16
        # FSDP requires uniform dtype for flattening tensors
        logger.info(f"Casting all parameters to {dtype} for FSDP compatibility...")
        for name, param in self.model.named_parameters():
            if param.dtype != dtype:
                param.data = param.data.to(dtype)
        
        # Verify uniform dtype
        dtypes = set(p.dtype for p in self.model.parameters())
        if len(dtypes) > 1:
            logger.warning(f"WARNING: Model has mixed dtypes: {dtypes}")
        else:
            logger.info(f"All parameters have uniform dtype: {dtypes}")
        
        # Get FSDP configuration
        fsdp_config = self._get_fsdp_config()
        
        # Get transformer layer class for auto wrapping
        transformer_layer_cls = self._get_transformer_layer_cls()
        
        if transformer_layer_cls:
            # Create auto wrap policy for transformer layers
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=transformer_layer_cls,
            )
            logger.info(f"Using transformer_auto_wrap_policy with: {transformer_layer_cls}")
        else:
            auto_wrap_policy = None
            logger.info("Using default FSDP wrapping (no auto wrap policy)")
        
        # Wrap model with FSDP
        if self.world_size > 1:
            logger.info(f"Wrapping model with FSDP (strategy={self.config.sharding_strategy})...")
            
            self.model = FSDP(
                self.model,
                auto_wrap_policy=auto_wrap_policy,
                sharding_strategy=fsdp_config["sharding_strategy"],
                backward_prefetch=fsdp_config["backward_prefetch"],
                mixed_precision=fsdp_config["mixed_precision"],
                cpu_offload=fsdp_config["cpu_offload"],
                device_id=self.local_rank,
                use_orig_params=True,  # Required for PEFT compatibility
                limit_all_gathers=True,  # Memory optimization
            )
            
            logger.info(f"Model wrapped with FSDP (sharding={self.config.sharding_strategy})")
        else:
            # Single GPU - just move to device
            self.model = self.model.to(self.device)
            logger.info("Single GPU mode - no FSDP wrapping")
        
        logger.info(f"Model loaded with LoRA (rank={self.config.lora_rank}, alpha={self.config.lora_alpha})")
    
    def _create_new_lora(self):
        """Create a new LoRA adapter from scratch."""
        from peft import LoraConfig, get_peft_model, TaskType
        
        logger.info("Creating NEW LoRA adapter (fresh training)")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )
        
        # Store LoRA config for saving checkpoints
        self.lora_config = lora_config
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Only optimize trainable (LoRA) parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )
        
        t_max = max(100, self.config.total_steps)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=t_max,
            eta_min=self.config.learning_rate * 0.1,
        )
        
        logger.info(f"Optimizer setup: lr={self.config.learning_rate}, weight_decay={self.config.weight_decay}")
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.0,
    ) -> Dict[str, float]:
        """
        Perform a single PPO training step with gradient accumulation.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels [batch, seq_len]
            advantages: Advantage estimates [batch, seq_len] or [batch]
            old_log_probs: Log probs from rollout [batch, seq_len]
            clip_ratio: PPO clipping ratio
            entropy_coeff: Entropy bonus coefficient
            
        Returns:
            Dictionary with training metrics
        """
        torch.cuda.empty_cache()
        
        self.model.train()
        device = self.device
        batch_size = input_ids.shape[0]
        mini_batch_size = self.config.mini_batch_size
        num_mini_batches = (batch_size + mini_batch_size - 1) // mini_batch_size
        
        expand_advantages = advantages.dim() == 1
        
        self.optimizer.zero_grad()
        
        # Accumulate metrics
        total_loss_accum = 0.0
        ppo_loss_accum = 0.0
        entropy_accum = 0.0
        approx_kl_accum = 0.0
        clip_fraction_accum = 0.0
        total_tokens = 0
        
        logger.info(f"  Training step: {batch_size} samples, {num_mini_batches} mini-batches (size={mini_batch_size})")
        
        for mb_idx, start_idx in enumerate(range(0, batch_size, mini_batch_size)):
            end_idx = min(start_idx + mini_batch_size, batch_size)
            mb_size = end_idx - start_idx
            
            # Get mini-batch
            mb_input_ids = input_ids[start_idx:end_idx].to(device)
            mb_attention_mask = attention_mask[start_idx:end_idx].to(device)
            mb_labels = labels[start_idx:end_idx].to(device)
            mb_advantages = advantages[start_idx:end_idx].to(device)
            mb_old_log_probs = old_log_probs[start_idx:end_idx].to(device)
            
            # Forward pass - DON'T pass labels to avoid transformers' ForCausalLMLoss
            # which casts logits to float32 and causes OOM on L40S 48GB GPUs
            # We compute our own PPO loss anyway, so we just need the logits
            outputs = self.model(
                input_ids=mb_input_ids,
                attention_mask=mb_attention_mask,
                return_dict=True,
            )
            
            # Compute log probabilities (our memory-efficient implementation)
            logits = outputs.logits
            log_probs = self._compute_log_probs(logits, mb_labels, mb_attention_mask)
            
            # Compute PPO loss
            ratio = torch.exp(log_probs - mb_old_log_probs)
            
            if expand_advantages:
                mb_advantages = mb_advantages.unsqueeze(-1).expand_as(log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_advantages
            
            response_mask = (mb_labels != -100).float()
            num_tokens = response_mask.sum().item()
            total_tokens += num_tokens
            
            ppo_loss = -torch.min(surr1, surr2)
            ppo_loss = (ppo_loss * response_mask).sum() / (response_mask.sum() + 1e-8)
            
            if entropy_coeff > 0:
                entropy = self._compute_entropy(logits, response_mask)
                entropy_loss = -entropy_coeff * entropy
                mb_total_loss = ppo_loss + entropy_loss
            else:
                entropy = torch.tensor(0.0, device=device)
                mb_total_loss = ppo_loss
            
            # Scale loss for gradient accumulation
            scaled_loss = mb_total_loss / num_mini_batches
            scaled_loss.backward()
            
            # Accumulate metrics
            total_loss_accum += mb_total_loss.item() * mb_size
            ppo_loss_accum += ppo_loss.item() * mb_size
            entropy_accum += entropy.item() * mb_size
            
            with torch.no_grad():
                approx_kl_accum += (0.5 * ((log_probs - mb_old_log_probs) ** 2).mean().item()) * mb_size
                clip_fraction_accum += (((ratio - 1.0).abs() > clip_ratio).float().mean().item()) * mb_size
            
            del outputs, logits, log_probs, ratio, mb_input_ids, mb_attention_mask, mb_labels
            torch.cuda.empty_cache()
            
            if (mb_idx + 1) % max(1, num_mini_batches // 5) == 0 or mb_idx == num_mini_batches - 1:
                logger.info(f"    Mini-batch {mb_idx + 1}/{num_mini_batches} - ppo_loss: {ppo_loss.item():.4f}")
        
        # Gradient clipping
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            self.config.max_grad_norm,
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        metrics = {
            "loss": total_loss_accum / batch_size,
            "ppo_loss": ppo_loss_accum / batch_size,
            "entropy": entropy_accum / batch_size,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "approx_kl": approx_kl_accum / batch_size,
            "clip_fraction": clip_fraction_accum / batch_size,
            "learning_rate": self.scheduler.get_last_lr()[0],
            "mini_batches": num_mini_batches,
        }
        
        return metrics
    
    def _compute_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for each token.
        
        Memory-efficient: uses cross_entropy instead of materializing full softmax.
        This avoids creating a [batch, seq, vocab] tensor which OOMs on L40S.
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Create mask for valid tokens (not -100)
        mask = (shift_labels != -100).float()
        
        # Replace -100 with 0 for cross_entropy (will be masked anyway)
        safe_labels = shift_labels.clamp(min=0)
        
        # Compute log probs efficiently using cross_entropy
        # cross_entropy computes: -log_softmax(logits)[target]
        # So log_prob = -cross_entropy
        # Process one sample at a time to minimize memory
        log_probs_list = []
        for i in range(batch_size):
            # Shape: [seq_len-1, vocab_size] and [seq_len-1]
            sample_logits = shift_logits[i]
            sample_labels = safe_labels[i]
            
            # cross_entropy with reduction='none' gives per-token loss
            # This is memory efficient: doesn't materialize full softmax
            neg_log_probs = torch.nn.functional.cross_entropy(
                sample_logits,  # [seq_len-1, vocab_size]
                sample_labels,  # [seq_len-1]
                reduction='none',
            )
            log_probs_list.append(-neg_log_probs)  # [seq_len-1]
        
        log_probs = torch.stack(log_probs_list, dim=0)  # [batch, seq_len-1]
        
        # Apply mask (zero out padding/ignored tokens)
        log_probs = log_probs * mask
        
        # Pad to match original sequence length
        log_probs = torch.nn.functional.pad(log_probs, (1, 0), value=0.0)
        
        return log_probs
    
    def _compute_entropy(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute entropy of the policy distribution.
        
        Memory-efficient: samples top-k logits instead of full vocabulary.
        Full entropy over 152K vocab would OOM on L40S.
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Use top-k approximation to avoid materializing full softmax
        # Top-k entropy is a reasonable approximation for peaked distributions
        k = min(1000, vocab_size)  # Top 1000 tokens
        
        entropy_sum = 0.0
        count = 0
        
        for i in range(batch_size):
            # Get top-k logits for this sample
            top_logits, _ = torch.topk(logits[i], k, dim=-1)  # [seq_len, k]
            
            # Compute entropy on top-k (renormalized)
            top_probs = torch.softmax(top_logits, dim=-1)
            top_log_probs = torch.log_softmax(top_logits, dim=-1)
            
            sample_entropy = -(top_probs * top_log_probs).sum(dim=-1)  # [seq_len]
            
            # Apply mask
            sample_mask = attention_mask[i]
            entropy_sum += (sample_entropy * sample_mask).sum()
            count += sample_mask.sum()
        
        entropy = entropy_sum / (count + 1e-8)
        
        return entropy
    
    def compute_log_probs_for_rollout(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for rollout (no gradient)."""
        self.model.eval()
        
        device = self.device
        batch_size = input_ids.shape[0]
        mini_batch_size = self.config.mini_batch_size
        
        all_log_probs = []
        
        with torch.no_grad():
            for start_idx in range(0, batch_size, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, batch_size)
                
                mb_input_ids = input_ids[start_idx:end_idx].to(device)
                mb_attention_mask = attention_mask[start_idx:end_idx].to(device)
                mb_labels = labels[start_idx:end_idx].to(device)
                
                outputs = self.model(
                    input_ids=mb_input_ids,
                    attention_mask=mb_attention_mask,
                    return_dict=True,
                )
                
                mb_log_probs = self._compute_log_probs(outputs.logits, mb_labels, mb_attention_mask)
                all_log_probs.append(mb_log_probs.cpu())
                
                del outputs, mb_input_ids, mb_attention_mask, mb_labels
                torch.cuda.empty_cache()
        
        log_probs = torch.cat(all_log_probs, dim=0)
        
        return log_probs
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save LoRA adapter checkpoint in PEFT-compatible format.
        
        Only saves the small adapter (~500MB), NOT the merged model.
        For vLLM inference, use save_merged_model() or merge on-the-fly.
        
        With use_orig_params=True in FSDP, parameter names are preserved,
        so we can extract LoRA weights directly from the state dict.
        """
        if path is None:
            path = Path(self.config.checkpoint_dir) / f"step_{self.global_step}"
        else:
            path = Path(path)
        
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        import json
        
        if self.world_size > 1 and isinstance(self.model, FSDP):
            # Use FSDP's full state dict with use_orig_params=True (preserves key names)
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            
            logger.info(f"[Rank {self.local_rank}] Entering FSDP state_dict_type context for checkpoint save...")
            
            try:
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                    logger.info(f"[Rank {self.local_rank}] Gathering state dict...")
                    state_dict = self.model.state_dict()
                    logger.info(f"[Rank {self.local_rank}] State dict gathered, {len(state_dict)} keys")
                    
                    if self.local_rank == 0:
                        path.mkdir(parents=True, exist_ok=True)
                        
                        # Extract only LoRA weights (keys containing 'lora_')
                        # PEFT's get_peft_model_state_dict strips the adapter name (e.g., '.default.')
                        # from keys when saving. We must match this format for from_pretrained to work.
                        # Keys should be: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
                        # NOT: base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
                        lora_state_dict = {}
                        for k, v in state_dict.items():
                            if 'lora_' in k or 'modules_to_save' in k:
                                # Strip adapter name (e.g., '.default.') from key
                                new_key = k.replace('.default.', '.')
                                lora_state_dict[new_key] = v
                        
                        logger.info(f"Saving {len(lora_state_dict)} LoRA parameters (out of {len(state_dict)} total)")
                        
                        # Log sample keys for debugging
                        if lora_state_dict:
                            sample_keys = list(lora_state_dict.keys())[:3]
                            logger.info(f"Sample LoRA keys: {sample_keys}")
                        
                        # Save LoRA weights
                        torch.save(lora_state_dict, path / "adapter_model.bin")
                        logger.info(f"Saved adapter_model.bin")
                        
                        # Save adapter config (REQUIRED for PeftModel.from_pretrained)
                        if hasattr(self, 'lora_config') and self.lora_config is not None:
                            config_dict = self.lora_config.to_dict()
                            # Convert sets to lists for JSON serialization
                            config_dict = _make_json_serializable(config_dict)
                            with open(path / "adapter_config.json", 'w') as f:
                                json.dump(config_dict, f, indent=2)
                            logger.info(f"Saved adapter_config.json")
                        else:
                            logger.warning("LoRA config not available, adapter_config.json not saved!")
            except Exception as e:
                logger.error(f"[Rank {self.local_rank}] Error in save_checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
            
            # Ensure all ranks are synchronized after save
            if dist.is_initialized():
                logger.info(f"[Rank {self.local_rank}] Waiting at barrier after checkpoint save...")
                dist.barrier()
                logger.info(f"[Rank {self.local_rank}] Passed barrier after checkpoint save")
        else:
            # Non-FSDP save - use PEFT's native save
            if self.local_rank != 0:
                return
            path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(path)
        
        # Only rank 0 saves tokenizer and training state
        if self.local_rank != 0:
            return
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save optimizer and scheduler state (for potential resume)
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }, path / "training_state.pt")
        
        logger.info(f"Adapter checkpoint saved to {path}")
    
    def save_merged_model(self, path: str):
        """Merge LoRA adapter with base model and save as complete model."""
        path = Path(path)
        
        logger.info(f"Merging LoRA adapter with base model...")
        
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        from peft import PeftModel
        import json
        
        # Save adapter first in PEFT-compatible format
        temp_adapter_path = path / "_temp_adapter"
        
        if self.world_size > 1 and isinstance(self.model, FSDP):
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = self.model.state_dict()
                
                if self.local_rank == 0:
                    temp_adapter_path.mkdir(parents=True, exist_ok=True)
                    
                    # Extract only LoRA weights and strip adapter name from keys
                    # PEFT's get_peft_model_state_dict strips '.default.' from keys when saving
                    lora_state_dict = {}
                    for k, v in state_dict.items():
                        if 'lora_' in k or 'modules_to_save' in k:
                            # Strip adapter name (e.g., '.default.') from key
                            new_key = k.replace('.default.', '.')
                            lora_state_dict[new_key] = v
                    
                    # Save LoRA weights
                    torch.save(lora_state_dict, temp_adapter_path / "adapter_model.bin")
                    
                    # Save adapter config (REQUIRED for PeftModel.from_pretrained)
                    if hasattr(self, 'lora_config') and self.lora_config is not None:
                        config_dict = self.lora_config.to_dict()
                        # Convert sets to lists for JSON serialization
                        config_dict = _make_json_serializable(config_dict)
                        with open(temp_adapter_path / "adapter_config.json", 'w') as f:
                            json.dump(config_dict, f, indent=2)
        else:
            # Non-FSDP save
            if self.local_rank != 0:
                return
            self.model.save_pretrained(temp_adapter_path)
        
        if self.local_rank != 0:
            return
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Load fresh base model
        logger.info(f"Loading fresh base model for merging...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        
        # Load LoRA adapter
        merged_model = PeftModel.from_pretrained(base_model, temp_adapter_path)
        merged_model = merged_model.merge_and_unload()
        
        # Save merged model
        logger.info(f"Saving merged model to {path}...")
        merged_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        logger.info(f"Merged model saved to {path}")
        
        # Cleanup
        del merged_model, base_model
        import shutil
        shutil.rmtree(temp_adapter_path, ignore_errors=True)
        torch.cuda.empty_cache()
    
    def cleanup(self):
        """Cleanup training resources."""
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("FSDPTrainer cleanup complete")

"""
DDP Trainer for Cypher Generator with PEFT LoRA.

Provides a clean PyTorch DDP wrapper for training the Cypher Generator
with LoRA adapters. No FSDP complexity - simple gradient synchronization.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DDPTrainerConfig:
    """Configuration for DDP trainer."""
    # Model settings
    model_path: str = ""
    
    # Resume from existing LoRA adapter (for cumulative training)
    # If set, loads the existing adapter instead of creating new LoRA
    resume_from_adapter: str = ""  # Path to existing LoRA adapter directory
    
    # LoRA settings
    lora_rank: int = 16  # Reduced from 32 for memory efficiency
    lora_alpha: int = 32  # Reduced proportionally
    lora_dropout: float = 0.05
    # Only target attention layers (not MLP) to save memory
    # MLP layers (gate_proj, up_proj, down_proj) are too large for 14B models
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"  # Attention only
    ])
    
    # Training settings
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    total_steps: int = 10000
    
    # DDP settings
    gpus: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    backend: str = "nccl"
    
    # Precision
    use_bf16: bool = True
    gradient_checkpointing: bool = True
    
    # Activation offloading - moves activations to CPU during forward pass
    # Reduces GPU memory significantly but slows training by ~20-30%
    activation_offloading: bool = False
    
    # Mini-batch size for memory efficiency (processes large batches in chunks)
    # Set to 1 for maximum memory safety - does NOT affect training quality
    mini_batch_size: int = 1
    
    # Checkpoint
    checkpoint_dir: str = ""
    save_steps: int = 500


class DDPTrainer:
    """
    DDP Trainer for Cypher Generator with LoRA.
    
    Uses PyTorch DDP for simple, stable distributed training.
    LoRA adapters are applied via HuggingFace PEFT.
    """
    
    def __init__(
        self,
        config: DDPTrainerConfig,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize DDP trainer.
        
        Args:
            config: DDPTrainerConfig with model and training settings
            local_rank: Local GPU rank (0-3 for 4 GPUs)
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
        
        logger.info(f"DDPTrainer initialized (rank {local_rank}/{world_size})")
    
    def setup(self):
        """Setup model, optimizer, and scheduler."""
        self._setup_device()
        self._init_distributed()
        self._load_model_with_lora()
        self._setup_optimizer()
        
        logger.info(f"DDPTrainer setup complete on device cuda:{self.local_rank}")
    
    def _init_distributed(self):
        """Initialize distributed training if world_size > 1."""
        if self.world_size > 1:
            if not dist.is_initialized():
                # Set longer timeout for NCCL (data collection can take hours)
                # Must be set BEFORE init_process_group
                import datetime
                timeout = datetime.timedelta(hours=3)  # 3 hour timeout
                
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method="env://",
                    world_size=self.world_size,
                    rank=self.local_rank,
                    timeout=timeout,
                )
            logger.info(f"Distributed training initialized: rank {self.local_rank}/{self.world_size}")
        else:
            logger.info("Single GPU training mode (no distributed)")
    
    def _setup_device(self):
        """Setup CUDA device.
        
        When CUDA_VISIBLE_DEVICES is set (e.g., "5,6,7"), PyTorch remaps the GPUs:
        - Physical GPU 5 → cuda:0
        - Physical GPU 6 → cuda:1
        - Physical GPU 7 → cuda:2
        
        So we use local_rank as the device ID, not the physical GPU ID.
        """
        # Use local_rank as device ID (respects CUDA_VISIBLE_DEVICES remapping)
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        logger.info(f"Using cuda:{self.local_rank} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    
    def _load_model_with_lora(self):
        """Load base model and apply LoRA adapters."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType, PeftModel
        except ImportError:
            raise ImportError("PEFT is required. Install with: pip install peft")
        
        logger.info(f"Loading model from {self.config.model_path} to cuda:{self.local_rank}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model to current device (respects CUDA_VISIBLE_DEVICES)
        dtype = torch.bfloat16 if self.config.use_bf16 else torch.float32
        
        # Model loading kwargs
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "device_map": {"": self.local_rank},  # Load to local_rank device
        }
        
        # CRITICAL: When activation_offloading is enabled, we MUST use "eager" attention
        # because SDPA attention uses torch.vmap internally for mask creation, which is
        # INCOMPATIBLE with saved_tensors_hooks used for activation offloading.
        # Error: "torch.func transforms don't yet support saved tensor hooks"
        if self.config.activation_offloading:
            model_kwargs["attn_implementation"] = "eager"
            logger.info("Using 'eager' attention (required for activation offloading)")
            logger.info("  Note: SDPA/Flash attention use vmap which conflicts with saved_tensors_hooks")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            **model_kwargs
        )
        
        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            gc_kwargs = {"use_reentrant": False}
            
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gc_kwargs
            )
            
            # If activation offloading is enabled, set up CPU offload hooks
            if self.config.activation_offloading:
                import os
                os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                
                # Register hooks to offload saved tensors to CPU
                # This significantly reduces GPU memory at cost of ~20-30% slower training
                def pack_hook(tensor):
                    """Move tensor to CPU when saving for backward."""
                    return tensor.to("cpu", non_blocking=True)
                
                def unpack_hook(tensor):
                    """Move tensor back to GPU when needed for backward."""
                    return tensor.to(self.local_rank, non_blocking=True)
                
                # Register the hooks globally for this context
                torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook).__enter__()
                self._offload_hooks_enabled = True
                
                logger.info("Gradient checkpointing enabled with activation offloading to CPU")
                logger.info("  ⚠ Training will be ~20-30% slower but uses significantly less GPU memory")
            else:
                self._offload_hooks_enabled = False
                logger.info("Gradient checkpointing enabled (use_reentrant=False)")
        
        # Check if we should resume from existing adapter (cumulative training)
        if self.config.resume_from_adapter and Path(self.config.resume_from_adapter).exists():
            adapter_path = Path(self.config.resume_from_adapter)
            adapter_config_path = adapter_path / "adapter_config.json"
            
            if adapter_config_path.exists():
                logger.info(f"🔄 RESUMING from existing LoRA adapter: {adapter_path}")
                logger.info("   (Cumulative training - building on previous best model)")
                
                # Load the existing LoRA adapter
                self.model = PeftModel.from_pretrained(
                    self.model,
                    str(adapter_path),
                    is_trainable=True,  # Important: make it trainable for continued training
                )
                self.model.print_trainable_parameters()
                logger.info(f"Loaded existing LoRA adapter from {adapter_path}")
            else:
                logger.warning(f"Adapter config not found at {adapter_config_path}, creating new LoRA")
                self._create_new_lora()
        else:
            if self.config.resume_from_adapter:
                logger.info(f"No existing adapter found at {self.config.resume_from_adapter}, creating new LoRA")
            self._create_new_lora()
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Wrap with DDP if using distributed training
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
            logger.info(f"Model wrapped with DDP (world_size={self.world_size})")
        
        logger.info(f"Model loaded with LoRA (rank={self.config.lora_rank}, alpha={self.config.lora_alpha})")
    
    def _create_new_lora(self):
        """Create a new LoRA adapter from scratch."""
        from peft import LoraConfig, get_peft_model, TaskType
        
        logger.info("Creating NEW LoRA adapter (fresh training)")
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Only optimize LoRA parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Ensure T_max is never 0 to avoid division by zero
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
        
        Processes large batches in mini-batches to avoid OOM, accumulating
        gradients across mini-batches before optimizer step.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels [batch, seq_len]
            advantages: Advantage estimates [batch, seq_len] or [batch]
            old_log_probs: Log probs from rollout [batch, seq_len]
            clip_ratio: PPO clipping ratio
            entropy_coeff: Entropy bonus coefficient (0 to disable, saves ~9GB memory)
            
        Returns:
            Dictionary with training metrics
        """
        # Clear GPU cache before training step to reduce fragmentation
        torch.cuda.empty_cache()
        
        self.model.train()
        device = self.device
        batch_size = input_ids.shape[0]
        mini_batch_size = self.config.mini_batch_size
        num_mini_batches = (batch_size + mini_batch_size - 1) // mini_batch_size
        
        # Expand advantages if needed (before mini-batching)
        if advantages.dim() == 1:
            # Will be expanded per mini-batch
            expand_advantages = True
        else:
            expand_advantages = False
        
        # Zero gradients before accumulation
        self.optimizer.zero_grad()
        
        # Accumulate metrics across mini-batches
        total_loss_accum = 0.0
        ppo_loss_accum = 0.0
        entropy_accum = 0.0
        approx_kl_accum = 0.0
        clip_fraction_accum = 0.0
        total_tokens = 0
        
        # Log training info at start
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
            
            # Forward pass
            outputs = self.model(
                input_ids=mb_input_ids,
                attention_mask=mb_attention_mask,
                labels=mb_labels,
                return_dict=True,
            )
            
            # Compute log probabilities
            logits = outputs.logits
            log_probs = self._compute_log_probs(logits, mb_labels, mb_attention_mask)
            
            # Compute PPO loss
            ratio = torch.exp(log_probs - mb_old_log_probs)
            
            # Expand advantages for this mini-batch if needed
            if expand_advantages:
                mb_advantages = mb_advantages.unsqueeze(-1).expand_as(log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_advantages
            
            # Mask for valid tokens (non-padding, response tokens only)
            response_mask = (mb_labels != -100).float()
            num_tokens = response_mask.sum().item()
            total_tokens += num_tokens
            
            # Compute masked loss
            ppo_loss = -torch.min(surr1, surr2)
            ppo_loss = (ppo_loss * response_mask).sum() / (response_mask.sum() + 1e-8)
            
            # Add entropy bonus for exploration (skip if entropy_coeff=0 to save memory)
            if entropy_coeff > 0:
                entropy = self._compute_entropy(logits, response_mask)
                entropy_loss = -entropy_coeff * entropy
                mb_total_loss = ppo_loss + entropy_loss
            else:
                entropy = torch.tensor(0.0, device=device)
                mb_total_loss = ppo_loss
            
            # Scale loss by mini-batch fraction for gradient accumulation
            scaled_loss = mb_total_loss / num_mini_batches
            
            # Backward pass (accumulates gradients)
            scaled_loss.backward()
            
            # Accumulate metrics
            total_loss_accum += mb_total_loss.item() * mb_size
            ppo_loss_accum += ppo_loss.item() * mb_size
            entropy_accum += entropy.item() * mb_size
            
            # Compute metrics
            with torch.no_grad():
                approx_kl_accum += (0.5 * ((log_probs - mb_old_log_probs) ** 2).mean().item()) * mb_size
                clip_fraction_accum += (((ratio - 1.0).abs() > clip_ratio).float().mean().item()) * mb_size
            
            # Clear intermediate tensors
            del outputs, logits, log_probs, ratio, mb_input_ids, mb_attention_mask, mb_labels
            torch.cuda.empty_cache()
            
            # Log progress periodically
            if (mb_idx + 1) % max(1, num_mini_batches // 5) == 0 or mb_idx == num_mini_batches - 1:
                logger.info(f"    Mini-batch {mb_idx + 1}/{num_mini_batches} - ppo_loss: {ppo_loss.item():.4f}")
        
        # Gradient clipping (only on trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            self.config.max_grad_norm,
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        # Average metrics
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
        """
        Compute log probabilities for each token.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Log probabilities [batch, seq_len]
        """
        # Shift for autoregressive
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute log softmax
        log_probs_all = torch.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        log_probs = torch.gather(
            log_probs_all,
            dim=-1,
            index=shift_labels.unsqueeze(-1).clamp(min=0),
        ).squeeze(-1)
        
        # Mask padding tokens
        mask = (shift_labels != -100).float()
        log_probs = log_probs * mask
        
        # Pad to original sequence length
        log_probs = torch.nn.functional.pad(log_probs, (1, 0), value=0.0)
        
        return log_probs
    
    def _compute_entropy(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute entropy of the policy distribution.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Mean entropy scalar
        """
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = (entropy * attention_mask).sum() / (attention_mask.sum() + 1e-8)
        
        return entropy
    
    def compute_log_probs_for_rollout(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities for rollout (no gradient).
        
        Used to get old_log_probs before training.
        Processes in mini-batches to avoid OOM.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels [batch, seq_len]
            
        Returns:
            Log probabilities [batch, seq_len]
        """
        self.model.eval()
        
        device = self.device
        batch_size = input_ids.shape[0]
        mini_batch_size = self.config.mini_batch_size
        
        # Process in mini-batches to avoid OOM
        all_log_probs = []
        
        with torch.no_grad():
            for start_idx in range(0, batch_size, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, batch_size)
                
                # Get mini-batch
                mb_input_ids = input_ids[start_idx:end_idx].to(device)
                mb_attention_mask = attention_mask[start_idx:end_idx].to(device)
                mb_labels = labels[start_idx:end_idx].to(device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=mb_input_ids,
                    attention_mask=mb_attention_mask,
                    return_dict=True,
                )
                
                # Compute log probs for this mini-batch
                mb_log_probs = self._compute_log_probs(outputs.logits, mb_labels, mb_attention_mask)
                all_log_probs.append(mb_log_probs.cpu())
                
                # Clear intermediate tensors
                del outputs, mb_input_ids, mb_attention_mask, mb_labels
                torch.cuda.empty_cache()
        
        # Concatenate all mini-batch results
        log_probs = torch.cat(all_log_probs, dim=0)
        
        return log_probs
    
    def save_checkpoint(self, path: Optional[str] = None):
        """
        Save LoRA checkpoint.
        
        Args:
            path: Optional custom path. If None, uses config checkpoint_dir.
        """
        if self.local_rank != 0:
            return  # Only save on rank 0
        
        if path is None:
            path = Path(self.config.checkpoint_dir) / f"step_{self.global_step}"
        else:
            path = Path(path)
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapter
        # Access the underlying PEFT model (no DDP wrapper in single-process mode)
        peft_model = self.model
        peft_model.save_pretrained(path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save optimizer and scheduler state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }, path / "training_state.pt")
        
        logger.info(f"Checkpoint saved to {path}")
    
    def save_merged_model(self, path: str):
        """
        Merge LoRA adapter with base model and save as complete model.
        
        This creates a standalone model that can be loaded by vLLM or
        other inference frameworks without requiring the adapter files.
        
        IMPORTANT: This method preserves the original model for continued training
        by using get_base_model() + manual weight merging instead of merge_and_unload()
        which would destroy the LoRA adapters.
        
        Args:
            path: Path to save the merged model
        """
        if self.local_rank != 0:
            return  # Only save on rank 0
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Merging LoRA adapter with base model (preserving original)...")
        
        # Save the LoRA adapter first
        temp_adapter_path = path / "_temp_adapter"
        self.model.save_pretrained(temp_adapter_path)
        
        # Get the base model path from config
        base_model_path = self.config.model_path
        
        # Load a fresh copy of the base model
        logger.info(f"Loading fresh base model for merging...")
        from peft import PeftModel
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float32,
            device_map="cpu",  # Load to CPU to save GPU memory
            trust_remote_code=True,
        )
        
        # Load the LoRA adapter onto the fresh base model
        merged_model = PeftModel.from_pretrained(base_model, temp_adapter_path)
        
        # Merge and unload on the COPY (not self.model)
        merged_model = merged_model.merge_and_unload()
        
        # Save the merged model
        logger.info(f"Saving merged model to {path}...")
        merged_model.save_pretrained(path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        logger.info(f"Merged model saved to {path}")
        
        # Clean up
        del merged_model
        del base_model
        
        # Remove temp adapter
        import shutil
        shutil.rmtree(temp_adapter_path, ignore_errors=True)
        
        torch.cuda.empty_cache()
        
        logger.info(f"Original model preserved for continued training")
    
    def load_checkpoint(self, path: str):
        """
        Load LoRA checkpoint.
        
        Args:
            path: Path to checkpoint directory
        """
        path = Path(path)
        
        # Load training state
        state_path = path / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.global_step = state["global_step"]
            logger.info(f"Loaded training state from {path}, step={self.global_step}")
        
        # Note: LoRA weights are loaded during model initialization
        # by passing the checkpoint path as model_path
    
    def cleanup(self):
        """Cleanup training resources."""
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("DDPTrainer cleanup complete")


def launch_ddp_training(
    config: DDPTrainerConfig,
    train_fn,
    num_gpus: int = 4,
):
    """
    Launch DDP training across multiple GPUs.
    
    Args:
        config: DDPTrainerConfig
        train_fn: Training function that takes (trainer, rank, world_size)
        num_gpus: Number of GPUs to use
    """
    import torch.multiprocessing as mp
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    mp.spawn(
        _ddp_worker,
        args=(config, train_fn, num_gpus),
        nprocs=num_gpus,
        join=True,
    )


def _ddp_worker(rank: int, config: DDPTrainerConfig, train_fn, world_size: int):
    """Worker function for DDP training."""
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config.gpus)
    
    # Create trainer
    trainer = DDPTrainer(config, local_rank=rank, world_size=world_size)
    trainer.setup()
    
    # Run training
    try:
        train_fn(trainer, rank, world_size)
    finally:
        trainer.cleanup()


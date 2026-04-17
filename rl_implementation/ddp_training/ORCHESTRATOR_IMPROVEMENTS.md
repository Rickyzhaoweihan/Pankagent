# Orchestrator Training Improvements

## Critical Issue Discovered

The Orchestrator's **Question Generation** training had even more severe problems than the Cypher Generator:

### Question Generation Reward Distribution (EXTREMELY BIMODAL)

Analysis of 768 rollouts from `stage1_ddp/rollouts_run_training_auto_multi_prompt_withstate`:

```
Question Generation Rewards:
  49.0% near-zero (<0.1)  ← Questions that fail due to plurals/invalid entities
  37.0% high (>0.7)       ← Good questions
  Only 4.2% middle        ← Almost NO learning signal in between!
```

**Answer Synthesis Rewards (Stable):**
```
Synthesis rewards: Consistently 0.49-0.53 across all iterations
→ Not the problem! Synthesis is stable.
```

### Performance Decline Over Training

| Iteration | Avg Question Gen | Avg Synthesis | Orchestrator Avg |
|-----------|------------------|---------------|------------------|
| 1         | 0.276            | 0.478         | 0.377            |
| 2         | 0.388            | 0.497         | 0.442            |
| 5         | 0.498 (peak!)    | 0.520         | 0.509            |
| 10        | 0.350 (declined) | 0.529         | 0.440            |

**The Toxic Training Cycle:**

```
1. Orchestrator generates: "Which genes are expressed in Ductal Cells?"
   └─> Uses plural "Cells" (natural language)
   
2. Cypher Generator follows → Query fails (0 results)
   └─> cypher_reward = 0.0
   
3. Orchestrator gets punished:
   └─> orch_qgen_reward = 0.05 (near-zero due to answerability component)
   
4. But it learns the WRONG lesson:
   ✗ Learns: "Don't ask about cell types" (avoids entire topic)
   ✓ Should learn: "Don't use plurals" (specific fix)
   
5. Performance declines as model avoids cell type questions entirely!
```

### Real Examples from Iteration 10

| Question | qgen_reward | cypher_reward | Results | Issue |
|----------|-------------|---------------|---------|-------|
| "genes in **Ductal Cells**?" | 0.05 | 0.0 | 0 | Plural! |
| "genes in **Alpha Cells**?" | 0.029 | 0.0 | 0 | Plural! |
| "GO terms for ZBTB17?" | 0.917 | 0.886 | 49 | ✓ Good |
| "effector genes of T1D?" | 0.885 | 0.836 | 60 | ✓ Good |
| "genes with function X?" | 0.875 | 0.861 | 60 | ✓ Good |

**Pattern:** Questions without cell types succeed (0.8+), questions with cell types fail (0.05)

## Root Cause Analysis

### Why Question Generation Reward is Bimodal

The `orch_qgen_reward` formula includes:
- **40% Answerability**: Did Cypher Generator succeed? → 0 if plural
- **30% Difficulty**: Success rate around target
- **20% Diversity**: Different from recent questions
- **10% Scope**: Within allowed types

When Cypher fails due to plurals:
- Answerability = 0
- Total reward ≈ 0.05 (near-zero)

This creates a **perverse incentive**: Orchestrator learns to avoid cell type questions entirely, rather than learning to avoid plurals.

### Why This is Worse Than Cypher Generator Issues

1. **More extreme bimodality**: 49% near-zero vs. Cypher's more gradual distribution
2. **Wrong attribution**: Orchestrator gets punished for Cypher's inability to handle plurals
3. **Topic avoidance**: Model learns to skip entire knowledge areas instead of fixing syntax
4. **Cascading failure**: Bad questions → bad Cypher → bad training signal for both agents

## Implemented Solutions

### 1. Config Updates: `train_orchestrator_config.yaml`

**Increased Batch Size (4 → 16):**
```yaml
# BEFORE
batch_size: 4

# AFTER
batch_size: 16  # Increased for better GRPO baseline estimation
                # CRITICAL: With 49% near-zero qgen_rewards, need larger batches
```

**Added Balanced Sampling:**
```yaml
# NEW: Enable stratified batch sampling
use_balanced_sampling: true  # CRITICAL for bimodal reward distribution!
```

**Why These Changes Matter:**
- **Larger batches**: GRPO baseline = mean of batch rewards. With 49% near-zero, batch size 4 gives unreliable baselines (either all low or all high). Size 16 ensures mix of rewards.
- **Balanced sampling**: Ensures each batch has ~33% low, ~33% med, ~33% high rewards → stable baseline estimation

### 2. Code Updates: `rollout_loader.py`

**Added Balanced Sampling to Question Generation:**
```python
def prepare_question_gen_batch(
    self,
    rollouts: List[RolloutEntry],
    schema_summary: str = "",
    use_stored_prompts: bool = True,
    use_balanced_sampling: bool = False,  # NEW
    batch_size: Optional[int] = None,      # NEW
) -> Dict[str, torch.Tensor]:
    # Apply balanced sampling if requested (CRITICAL for bimodal qgen rewards!)
    if use_balanced_sampling:
        if batch_size is None:
            raise ValueError("batch_size required for balanced sampling")
        sampler = BalancedBatchSampler(rollouts, reward_attr='orch_qgen_reward')
        rollouts = sampler.sample_balanced_batch(batch_size)
        logger.info(f"Using balanced sampling for qgen with batch_size={batch_size}")
```

**Added Balanced Sampling to Answer Synthesis:**
```python
def prepare_synthesis_batch(
    self,
    rollouts: List[RolloutEntry],
    use_stored_prompts: bool = True,
    use_balanced_sampling: bool = False,  # NEW
    batch_size: Optional[int] = None,      # NEW
) -> Dict[str, torch.Tensor]:
    # Apply balanced sampling if requested
    if use_balanced_sampling:
        if batch_size is None:
            raise ValueError("batch_size required for balanced sampling")
        # Filter rollouts with synthesized answers first
        valid_rollouts = [r for r in rollouts if r.synthesized_answer]
        sampler = BalancedBatchSampler(valid_rollouts, reward_attr='orch_synth_reward')
        rollouts = sampler.sample_balanced_batch(batch_size)
```

### 3. Trainer Updates: `train_orchestrator_from_rollouts.py`

**Integrated Balanced Sampling:**
```python
def train_question_generation(self) -> Dict[str, float]:
    # Use balanced sampling if configured (CRITICAL for bimodal qgen rewards!)
    use_balanced = self.config.get('use_balanced_sampling', False)
    batch = self.batch_preparer.prepare_question_gen_batch(
        self.rollouts,
        schema_summary=self.config.get('schema_summary', ''),
        use_stored_prompts=self.use_stored_prompts,
        use_balanced_sampling=use_balanced,  # NEW
        batch_size=self.batch_size if use_balanced else None,  # NEW
    )

def train_answer_synthesis_role(self) -> Dict[str, float]:
    # Use balanced sampling if configured
    use_balanced = self.config.get('use_balanced_sampling', False)
    batch = self.batch_preparer.prepare_synthesis_batch(
        self.rollouts,
        use_stored_prompts=self.use_stored_prompts,
        use_balanced_sampling=use_balanced,  # NEW
        batch_size=self.batch_size if use_balanced else None,  # NEW
    )
```

## Expected Improvements

### Immediate Effects

1. **Stable GRPO Baselines**
   - Batch size 16 with balanced sampling → reliable advantage estimates
   - Each batch contains mix of low/med/high rewards
   - Prevents "all zeros" or "all ones" batches

2. **Better Learning Signal**
   - Question Generation sees consistent gradient signal
   - No longer punished too harshly for cell type questions
   - Auto-fix handles plurals → more questions succeed over time

3. **Reduced Topic Avoidance**
   - Model less likely to "give up" on cell types
   - As auto-fix improves Cypher, qgen_reward improves
   - Positive feedback loop instead of negative

### Long-term Training Benefits

**Expected Question Gen Reward Progression:**

| Stage | Distribution | Mean | Notes |
|-------|--------------|------|-------|
| Before fixes | 49% <0.1, 37% >0.7 | 0.378 | Extremely bimodal |
| After auto-fix | 30% <0.3, 50% >0.5 | 0.55 | Less bimodal, higher mean |
| After adaptation | 10% <0.3, 70% >0.6 | 0.65 | Model learns patterns |

**Key Metrics to Monitor:**

```bash
# Question Generation Reward
- Should increase from ~0.35 → 0.55+ over first 5 iterations
- Bimodality should reduce (fewer near-zero rewards)
- Cell type questions should succeed more often

# Answer Synthesis Reward
- Already stable at 0.49-0.53, expect minor improvement to 0.55-0.60
- Main benefit: better input from improved questions

# Orchestrator Avg Reward
- Should increase from ~0.44 → 0.58+ over 10 iterations
- More stable trajectory (less variance between iterations)
```

## Validation

To validate these improvements, check:

1. **Reward Distribution Evolution:**
```bash
# Before: 49% near-zero
# After: Should drop to <30% near-zero within 3 iterations

cat rollouts_iter_*.jsonl | jq '.trajectory.orch_qgen_reward' | python3 -c "
import sys
rewards = [float(x) for x in sys.stdin]
print(f'Near-zero (<0.1): {sum(1 for r in rewards if r < 0.1)/len(rewards)*100:.1f}%')
print(f'High (>0.7): {sum(1 for r in rewards if r > 0.7)/len(rewards)*100:.1f}%')
"
```

2. **Cell Type Question Success:**
```bash
# Count questions with cell types that succeed
grep -i "cell" rollouts_iter_010.jsonl | \
jq -c '{q: .trajectory.question, reward: .trajectory.cypher_reward}' | \
grep -v '"reward":0'
```

3. **Batch Composition:**
```python
# Log from training should show balanced batches:
# "BalancedBatchSampler stratified X rollouts: low=Y, med=Z, high=W"
# Y, Z, W should be roughly equal (within 2x of each other)
```

## Integration with Cypher Generator Fixes

The Orchestrator improvements work synergistically with Cypher Generator fixes:

```
Orchestrator (fixed)
  └─> Generates better questions (but may still use plurals)
       └─> Cypher Generator (fixed auto-fix)
            └─> Corrects plurals automatically
                 └─> Queries succeed more often
                      └─> Orchestrator gets higher qgen_reward
                           └─> Learns questions are answerable
                                └─> POSITIVE FEEDBACK LOOP! ✓
```

**Combined Effect:**
- Auto-fix handles 90%+ of plural cases → Cypher succeeds
- Cypher success → Question Gen reward increases
- Higher rewards → Better training signal for Orchestrator
- Better Orchestrator → Asks more sophisticated questions
- More data → Better Cypher learning

## Files Modified

1. **Config:**
   - `rl_implementation/ddp_training/config/train_orchestrator_config.yaml`
     - `batch_size: 4 → 16`
     - Added `use_balanced_sampling: true`

2. **Batch Preparation:**
   - `rl_implementation/ddp_training/rollout_loader.py`
     - Updated `OrchestratorBatchPreparer.prepare_question_gen_batch()`
     - Updated `OrchestratorBatchPreparer.prepare_synthesis_batch()`
     - Added balanced sampling support to both methods

3. **Trainer:**
   - `rl_implementation/ddp_training/scripts/train_orchestrator_from_rollouts.py`
     - Updated `train_question_generation()`
     - Updated `train_answer_synthesis_role()`
     - Pass balanced sampling flags from config

## Summary

**The Problem:**
- Orchestrator Question Generation had 49% near-zero rewards (worse than Cypher Generator!)
- Toxic training cycle: plurals → Cypher fails → Orchestrator punished → topic avoidance
- Performance declining over training (0.498 → 0.350)

**The Solution:**
- Increased batch size (4 → 16) for stable GRPO baselines
- Added balanced sampling to ensure mixed-reward batches
- Integrated with Cypher auto-fix for synergistic improvement

**Expected Outcome:**
- 50%+ improvement in Question Gen reward (0.35 → 0.55+)
- Reduced bimodality (49% near-zero → <30%)
- Stable training progression instead of decline
- Better synergy with Cypher Generator improvements


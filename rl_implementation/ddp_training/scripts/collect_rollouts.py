#!/usr/bin/env python3
"""
Standalone Rollout Collector Script.

Connects to running vLLM servers and collects rollouts for training.
Appends rollouts to a JSONL file for later use in training.

Usage:
    python collect_rollouts.py --num-questions 64 --difficulty easy
    python collect_rollouts.py --num-questions 128 --difficulty medium --batch-size 16
    
    # Coverage-based stopping: collect until we have 128 usable rollouts
    python collect_rollouts.py --target-filtered-rollouts 128 --min-usable-reward 0.1
    
    # With adaptive entity sampling
    python collect_rollouts.py --use-adaptive-sampling --entity-degrees outputs/entity_degrees.json

Requirements:
    - vLLM servers running (start_vllm_servers.sh)
    - Neo4j API accessible
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent.absolute()
DDP_DIR = SCRIPT_DIR.parent
RL_IMPL_DIR = DDP_DIR.parent
PROJECT_DIR = RL_IMPL_DIR.parent

sys.path.insert(0, str(RL_IMPL_DIR))
sys.path.insert(0, str(DDP_DIR))

from transformers import AutoTokenizer

# Import our components
from rollout_collector import RolloutCollector, RolloutCollectorConfig
from rollout_store import RolloutStore
from inference_engine import InferenceEngine, InferenceConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect rollouts from vLLM servers")
    
    # Rollout settings
    parser.add_argument("--num-questions", type=int, default=64,
                        help="Number of questions to generate (default: 64). Used when coverage-based stopping is disabled.")
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard"],
                        help="Difficulty level (default: easy)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for processing (default: 8)")
    parser.add_argument("--max-steps", type=int, default=5,
                        help="Max Cypher query steps per question (default: 5)")
    
    # Coverage-based stopping (alternative to --num-questions)
    parser.add_argument("--target-filtered-rollouts", type=int, default=None,
                        help="Target number of usable/filtered rollouts (coverage-based stopping). "
                             "If set, collection continues until this many usable rollouts are collected.")
    parser.add_argument("--min-usable-reward", type=float, default=0.1,
                        help="Minimum reward for a rollout to be considered usable (default: 0.1)")
    parser.add_argument("--max-collection-attempts", type=int, default=500,
                        help="Maximum total rollouts to attempt before stopping (safety limit, default: 500)")
    
    # Server settings
    parser.add_argument("--orchestrator-port", type=int, default=8001,
                        help="Orchestrator vLLM server port (default: 8001)")
    parser.add_argument("--cypher-port", type=int, default=8002,
                        help="Cypher Generator vLLM server port (default: 8002)")
    parser.add_argument("--server-host", type=str, default="localhost",
                        help="Server host (default: localhost)")
    
    # Output settings
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_DIR / "outputs/stage1_ddp/rollouts.jsonl"),
                        help="Output JSONL file path")
    
    # Model paths
    parser.add_argument("--orchestrator-model", type=str,
                        default=str(PROJECT_DIR / "models/qwen2.5-14b"),
                        help="Path to Orchestrator model")
    parser.add_argument("--cypher-model", type=str,
                        default=str(PROJECT_DIR / "models/qwen2.5-coder-14b"),
                        help="Path to Cypher model")
    
    # Schema and entity samples
    parser.add_argument("--schema-path", type=str,
                        default=str(PROJECT_DIR / "legacy/PankBaseAgent/text_to_cypher/data/input/kg_schema.json"),
                        help="Path to schema JSON file")
    parser.add_argument("--entity-samples", type=str,
                        default=str(PROJECT_DIR / "outputs/entity_samples.json"),
                        help="Path to entity samples JSON file")
    
    # Adaptive entity sampling
    parser.add_argument("--use-adaptive-sampling", action="store_true",
                        help="Enable adaptive entity-relationship sampling (Thompson Sampling)")
    parser.add_argument("--entity-degrees", type=str,
                        default=str(PROJECT_DIR / "outputs/entity_degrees.json"),
                        help="Path to entity degrees JSON file (for adaptive sampling)")
    parser.add_argument("--adaptive-sampler-state", type=str,
                        default=str(PROJECT_DIR / "outputs/adaptive_sampler_state.json"),
                        help="Path to save/load adaptive sampler state")
    parser.add_argument("--adaptive-degree-weight", type=float, default=0.1,
                        help="Weight for degree in Thompson Sampling prior (default: 0.1)")
    parser.add_argument("--adaptive-slack-threshold", type=int, default=3,
                        help="Consecutive failures before full penalty (default: 3)")
    
    # Neo4j
    parser.add_argument("--neo4j-url", type=str,
                        default="https://nzi5e9mb0f.execute-api.us-east-1.amazonaws.com/production/pankgraph-neo4j",
                        help="Neo4j API URL")
    
    # Misc
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID for logging (default: timestamp)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    # Deep Think prompt hints
    parser.add_argument("--prompt-hints", type=str, default=None,
                        help="Path to prompt hints JSON file (from Deep Think analysis)")
    
    return parser.parse_args()


def check_servers(host: str, orchestrator_port: int, cypher_port: int) -> bool:
    """Check if vLLM servers are running."""
    import requests
    
    servers_ok = True
    
    # Check Orchestrator
    orch_url = f"http://{host}:{orchestrator_port}/health"
    try:
        resp = requests.get(orch_url, timeout=5)
        if resp.status_code == 200:
            logger.info(f"✓ Orchestrator server OK at {host}:{orchestrator_port}")
        else:
            logger.error(f"✗ Orchestrator server returned {resp.status_code}")
            servers_ok = False
    except Exception as e:
        logger.error(f"✗ Cannot connect to Orchestrator at {orch_url}: {e}")
        servers_ok = False
    
    # Check Cypher Generator
    cypher_url = f"http://{host}:{cypher_port}/health"
    try:
        resp = requests.get(cypher_url, timeout=5)
        if resp.status_code == 200:
            logger.info(f"✓ Cypher Generator server OK at {host}:{cypher_port}")
        else:
            logger.error(f"✗ Cypher Generator server returned {resp.status_code}")
            servers_ok = False
    except Exception as e:
        logger.error(f"✗ Cannot connect to Cypher Generator at {cypher_url}: {e}")
        servers_ok = False
    
    return servers_ok


def count_usable_rollouts(store, min_reward: float) -> int:
    """Count rollouts with reward >= min_reward in the current store."""
    count = 0
    if not store.store_path.exists():
        return 0
    try:
        with open(store.store_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    traj = data.get('trajectory', {})
                    # Check cypher reward (main signal for Cypher Generator)
                    cypher_reward = traj.get('cypher_reward', traj.get('reward', 0.0))
                    if cypher_reward >= min_reward:
                        count += 1
                except:
                    continue
    except:
        pass
    return count


def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine stopping mode
    use_coverage_stopping = args.target_filtered_rollouts is not None
    
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"=" * 60)
    logger.info(f"Rollout Collection - Run ID: {run_id}")
    logger.info(f"=" * 60)
    
    if use_coverage_stopping:
        logger.info(f"Mode: COVERAGE-BASED STOPPING")
        logger.info(f"  Target usable rollouts: {args.target_filtered_rollouts}")
        logger.info(f"  Min usable reward: {args.min_usable_reward}")
        logger.info(f"  Max attempts: {args.max_collection_attempts}")
    else:
        logger.info(f"Mode: FIXED COUNT")
        logger.info(f"  Target questions: {args.num_questions}")
    
    logger.info(f"Difficulty: {args.difficulty}")
    logger.info(f"Batch size: {args.batch_size}, Max steps: {args.max_steps}")
    logger.info(f"Output: {args.output}")
    
    if args.use_adaptive_sampling:
        logger.info(f"Adaptive Sampling: ENABLED")
        logger.info(f"  Entity degrees: {args.entity_degrees}")
        logger.info(f"  Sampler state: {args.adaptive_sampler_state}")
    
    # Check servers
    logger.info("\nChecking vLLM servers...")
    if not check_servers(args.server_host, args.orchestrator_port, args.cypher_port):
        logger.error("vLLM servers not available. Start them with start_vllm_servers.sh")
        sys.exit(1)
    
    # Load tokenizer
    logger.info(f"\nLoading tokenizer from {args.cypher_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.cypher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize inference engine
    logger.info("\nInitializing inference engine...")
    inference_config = InferenceConfig(
        orchestrator_model_path=args.orchestrator_model,
        cypher_model_path=args.cypher_model,
        orchestrator_port_question=args.orchestrator_port,
        orchestrator_port_data_eval=args.orchestrator_port,
        orchestrator_port_synthesis=args.orchestrator_port,
        orchestrator_port_answer_eval=args.orchestrator_port,
        cypher_inference_port=args.cypher_port,
        server_host=args.server_host,
        api_timeout=120.0,
        max_retries=3,
    )
    inference_engine = InferenceEngine(inference_config)
    inference_engine.initialize()  # Connect to vLLM servers
    
    # Initialize rollout collector
    logger.info("\nInitializing rollout collector...")
    collector_config = RolloutCollectorConfig(
        schema_path=args.schema_path,
        neo4j_url=args.neo4j_url,
        entity_samples_path=args.entity_samples,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        use_entity_seeding=True,
        use_experience_keywords=True,
        # Adaptive sampling settings
        use_adaptive_sampling=args.use_adaptive_sampling,
        entity_degrees_path=args.entity_degrees if args.use_adaptive_sampling else None,
        adaptive_sampler_path=args.adaptive_sampler_state if args.use_adaptive_sampling else None,
        adaptive_sampler_degree_weight=args.adaptive_degree_weight,
        adaptive_sampler_slack_threshold=args.adaptive_slack_threshold,
        # Deep Think prompt hints
        prompt_hints_path=args.prompt_hints,
    )
    collector = RolloutCollector(collector_config, inference_engine, tokenizer)
    
    # Initialize rollout store
    logger.info(f"\nInitializing rollout store at {args.output}...")
    store = RolloutStore(args.output)
    
    # Get curriculum constraints based on difficulty
    curriculum_constraints = {
        "easy": {"max_hops": 2, "max_queries": 3},
        "medium": {"max_hops": 3, "max_queries": 4},
        "hard": {"max_hops": 5, "max_queries": 5},
    }.get(args.difficulty, {"max_hops": 2, "max_queries": 3})
    
    # Collect rollouts in batches
    total_collected = 0
    total_usable = count_usable_rollouts(store, args.min_usable_reward) if use_coverage_stopping else 0
    batch_num = 0
    
    # Determine target and batch count estimate
    if use_coverage_stopping:
        target_label = f"{args.target_filtered_rollouts} usable"
        # Estimate batches needed (may need more due to filtering)
        est_usable_rate = 0.5  # Conservative: assume 50% will be usable
        est_batches_needed = int((args.target_filtered_rollouts - total_usable) / (args.batch_size * est_usable_rate)) + 5
        num_batches = min(est_batches_needed, args.max_collection_attempts // args.batch_size)
    else:
        target_label = f"{args.num_questions} total"
        num_batches = (args.num_questions + args.batch_size - 1) // args.batch_size
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting rollout collection (target: {target_label})...")
    if use_coverage_stopping:
        logger.info(f"Current usable rollouts: {total_usable}")
    logger.info(f"{'=' * 60}\n")
    
    # Determine when to stop
    def should_continue():
        if use_coverage_stopping:
            # Coverage-based: continue until we have enough usable rollouts
            return (total_usable < args.target_filtered_rollouts and 
                    total_collected < args.max_collection_attempts)
        else:
            # Fixed count: continue until we reach num_questions
            return total_collected < args.num_questions
    
    while should_continue():
        batch_num += 1
        
        # Determine batch size
        if use_coverage_stopping:
            # For coverage mode, use full batch size
            current_batch_size = args.batch_size
        else:
            # For fixed count, may need smaller final batch
            current_batch_size = min(args.batch_size, args.num_questions - total_collected)
        
        if use_coverage_stopping:
            logger.info(f"\n--- Batch {batch_num} ({current_batch_size} questions) [usable: {total_usable}/{args.target_filtered_rollouts}] ---")
        else:
            logger.info(f"\n--- Batch {batch_num}/{num_batches} ({current_batch_size} questions) ---")
        
        try:
            # Generate questions (with prompts for training storage)
            logger.info("Generating questions...")
            questions, qgen_prompts = collector.generate_questions(
                num_questions=current_batch_size,
                difficulty=args.difficulty,
                curriculum_constraints=curriculum_constraints,
                return_prompts=True,  # Get prompts for storage
            )
            
            if not questions:
                logger.warning("No questions generated, skipping batch")
                continue
            
            # Log sample questions
            for i, q in enumerate(questions[:3]):
                logger.info(f"  Q{i+1}: {q[:100]}...")
            
            # Collect trajectories (pass qgen_prompts for storage)
            logger.info("Collecting trajectories...")
            trajectories = collector.collect_trajectories(questions, qgen_prompts=qgen_prompts)
            
            # Print rollout summary for each trajectory
            for i, traj in enumerate(trajectories):
                rollout_num = total_collected + i + 1
                print(f"\n{'='*60}")
                print(f"Rollout #{rollout_num}")
                print(f"Question: {traj.question[:150]}{'...' if len(traj.question) > 150 else ''}")
                print(f"Cypher queries ({len(traj.steps)} steps):")
                for step_idx, step in enumerate(traj.steps):
                    cypher = step.cypher_query[:100] if step.cypher_query else "N/A"
                    print(f"  Step {step_idx+1}: {cypher}{'...' if len(step.cypher_query or '') > 100 else ''}")
                print(f"{'='*60}")
            
            # Evaluate trajectories
            logger.info("Evaluating trajectories...")
            trajectories = collector.evaluate_trajectories(trajectories)
            
            # Compute rewards
            logger.info("Computing rewards...")
            trajectories = collector.compute_rewards(trajectories)
            
            # Log trajectory stats for all reward types
            cypher_rewards = [t.cypher_reward for t in trajectories]
            qgen_rewards = [t.orch_qgen_reward for t in trajectories]
            synth_rewards = [t.orch_synth_reward for t in trajectories]
            results = [t.total_results for t in trajectories]
            
            logger.info(f"  Cypher rewards: mean={sum(cypher_rewards)/len(cypher_rewards):.3f}, "
                       f"min={min(cypher_rewards):.3f}, max={max(cypher_rewards):.3f}")
            logger.info(f"  Orch QGen rewards: mean={sum(qgen_rewards)/len(qgen_rewards):.3f}, "
                       f"min={min(qgen_rewards):.3f}, max={max(qgen_rewards):.3f}")
            logger.info(f"  Orch Synth rewards: mean={sum(synth_rewards)/len(synth_rewards):.3f}, "
                       f"min={min(synth_rewards):.3f}, max={max(synth_rewards):.3f}")
            logger.info(f"  Results: mean={sum(results)/len(results):.1f}, "
                       f"zero_results={sum(1 for r in results if r == 0)}/{len(results)}")
            
            # Save to store
            logger.info("Saving to store...")
            entries_saved = store.save_epoch(
                epoch=batch_num,  # Use batch as epoch for tracking
                difficulty=args.difficulty,
                questions=questions,
                trajectories=trajectories,
                config={
                    "run_id": run_id,
                    "batch_size": current_batch_size,
                    "max_steps": args.max_steps,
                    "curriculum_constraints": curriculum_constraints,
                }
            )
            
            # Update experience buffer
            collector.update_experience_buffer(trajectories)
            
            # Update adaptive sampler with answerability feedback
            if args.use_adaptive_sampling:
                collector.update_adaptive_sampler(trajectories)
                logger.info("  Updated adaptive entity sampler")
            
            total_collected += len(trajectories)
            
            # Count usable rollouts in this batch
            batch_usable = sum(1 for t in trajectories if t.cypher_reward >= args.min_usable_reward)
            total_usable += batch_usable
            
            if use_coverage_stopping:
                logger.info(f"  Saved {entries_saved} entries. "
                           f"Usable: {total_usable}/{args.target_filtered_rollouts} "
                           f"(batch: {batch_usable}/{len(trajectories)}, "
                           f"total attempts: {total_collected})")
            else:
                logger.info(f"  Saved {entries_saved} entries. Total: {total_collected}/{args.num_questions}")
            
        except Exception as e:
            logger.error(f"Error in batch {batch_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final stats
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Rollout Collection Complete!")
    logger.info(f"{'=' * 60}")
    
    stats = store.get_stats()
    logger.info(f"Total entries in store: {stats['total_entries']}")
    
    if use_coverage_stopping:
        final_usable = count_usable_rollouts(store, args.min_usable_reward)
        usable_rate = final_usable / total_collected * 100 if total_collected > 0 else 0
        logger.info(f"Coverage-based collection summary:")
        logger.info(f"  Total attempts: {total_collected}")
        logger.info(f"  Usable rollouts (reward >= {args.min_usable_reward}): {final_usable}")
        logger.info(f"  Usable rate: {usable_rate:.1f}%")
        if final_usable >= args.target_filtered_rollouts:
            logger.info(f"  ✓ Target reached: {final_usable} >= {args.target_filtered_rollouts}")
        else:
            logger.info(f"  ⚠ Target not reached: {final_usable} < {args.target_filtered_rollouts}")
            logger.info(f"    (hit max attempts limit: {args.max_collection_attempts})")
    
    logger.info(f"Average rewards:")
    logger.info(f"  Cypher: {stats.get('avg_cypher_reward', 0):.3f}")
    logger.info(f"  Orch QGen: {stats.get('avg_orch_qgen_reward', 0):.3f}")
    logger.info(f"  Orch Synth: {stats.get('avg_orch_synth_reward', 0):.3f}")
    logger.info(f"Average success rate: {stats.get('avg_success_rate', 0):.3f}")
    logger.info(f"Difficulties: {stats.get('difficulties', {})}")
    logger.info(f"\nOutput saved to: {args.output}")
    
    # Log adaptive sampler stats if used
    if args.use_adaptive_sampling and collector.adaptive_sampler is not None:
        sampler_stats = collector.adaptive_sampler.get_stats_summary()
        logger.info(f"\nAdaptive Sampler Stats:")
        logger.info(f"  Total entity-relationship pairs: {sampler_stats.get('total_pairs', 0)}")
        logger.info(f"  Total samples drawn: {sampler_stats.get('total_samples', 0)}")
        logger.info(f"  Overall success rate: {sampler_stats.get('overall_success_rate', 0):.2%}")
        logger.info(f"  Sampler state saved to: {args.adaptive_sampler_state}")
    
    # Cleanup
    collector.close()
    inference_engine.shutdown()
    logger.info("Done!")


if __name__ == "__main__":
    main()


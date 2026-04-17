#!/usr/bin/env python3
"""
Small-scale test script for collaborative multi-agent training.

Tests the training loop with minimal resources (3 questions, 2 epochs)
to verify that all components work together correctly.

Strategy 2: Sequential Within-Epoch Updates with Model Swapping

Usage:
    python -m rl_implementation.training.test_small_scale
    
Or from project root:
    cd /nfs/turbo/umms-drjieliu/usr/rickyhan/rllm/examples/PanKLLM_RL_post-training
    python -m rl_implementation.training.test_small_scale
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_small_scale.log')
    ]
)
logger = logging.getLogger(__name__)


def run_component_tests():
    """Run individual component tests before full training."""
    logger.info("=" * 80)
    logger.info("Running Component Tests")
    logger.info("=" * 80)
    
    tests_passed = []
    tests_failed = []
    
    # Test 1: Import all required modules
    logger.info("\n[Test 1] Testing imports...")
    try:
        from rl_implementation.agents import CypherGeneratorAgent, OrchestratorAgent, ExperienceBuffer
        from rl_implementation.environments import GraphReasoningEnvironment
        from rl_implementation.rewards import (
            cypher_generator_reward_fn,
            orchestrator_generation_reward_fn,
            orchestrator_synthesis_reward_fn
        )
        from rl_implementation.training.utils.rllm_components import (
            load_tokenizer,
            create_resource_pool_manager,
            create_worker_groups,
            create_execution_engine,
            # Sequential training components
            SequentialResourceManager,
            create_sequential_resource_manager,
            should_update_orchestrator,
            get_training_phase,
            EMAOrchestrator,
            create_ema_orchestrator,
        )
        logger.info("✓ All imports successful")
        tests_passed.append("Imports")
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        traceback.print_exc()
        tests_failed.append(("Imports", str(e)))
    
    # Test 2: Load configuration
    logger.info("\n[Test 2] Testing configuration loading...")
    try:
        config_path = Path(__file__).parent.parent / "config" / "training_config_test.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Configuration loaded: {config['experiment_name']}")
        logger.info(f"  - Questions per epoch: {config['questions_per_epoch']}")
        logger.info(f"  - Num epochs: {config['num_epochs']}")
        logger.info(f"  - Training mode: {config.get('gpu_allocation', {}).get('mode', 'N/A')}")
        tests_passed.append("Configuration")
    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {e}")
        tests_failed.append(("Configuration", str(e)))
        config = None
    
    # Test 3: Check file paths
    logger.info("\n[Test 3] Testing file paths...")
    try:
        project_root = Path(__file__).parent.parent.parent
        schema_path = project_root / config['schema_path']
        
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        logger.info(f"✓ Schema file exists: {schema_path}")
        tests_passed.append("File paths")
    except Exception as e:
        logger.error(f"✗ File path check failed: {e}")
        tests_failed.append(("File paths", str(e)))
    
    # Test 4: Test Sequential Resource Manager
    logger.info("\n[Test 4] Testing SequentialResourceManager...")
    try:
        from rl_implementation.training.utils.rllm_components import (
            create_sequential_resource_manager,
            should_update_orchestrator,
            get_training_phase,
        )
        
        gpu_config = config.get('gpu_allocation', {})
        seq_manager = create_sequential_resource_manager(gpu_config)
        
        logger.info(f"✓ SequentialResourceManager created:")
        logger.info(f"  - Total GPUs: {seq_manager.total_gpus}")
        logger.info(f"  - Rollout: Cypher={seq_manager.rollout_cypher_gpus}, Orch={seq_manager.rollout_orch_gpus}")
        logger.info(f"  - Training: {seq_manager.training_gpus} GPUs")
        
        tests_passed.append("SequentialResourceManager")
    except Exception as e:
        logger.error(f"✗ SequentialResourceManager test failed: {e}")
        traceback.print_exc()
        tests_failed.append(("SequentialResourceManager", str(e)))
    
    # Test 5: Test Orchestrator Schedule
    logger.info("\n[Test 5] Testing Orchestrator schedule logic...")
    try:
        from rl_implementation.training.utils.rllm_components import (
            should_update_orchestrator,
            get_training_phase,
        )
        
        schedule_config = config.get('orchestrator_schedule', {})
        
        # Test schedule for epochs 1-5
        logger.info("  Testing update schedule:")
        for epoch in range(1, 6):
            should_update = should_update_orchestrator(epoch, schedule_config, "easy")
            phase = get_training_phase(epoch, schedule_config)
            logger.info(f"    Epoch {epoch}: phase={phase}, update={should_update}")
        
        logger.info("✓ Orchestrator schedule logic works")
        tests_passed.append("Orchestrator Schedule")
    except Exception as e:
        logger.error(f"✗ Orchestrator schedule test failed: {e}")
        traceback.print_exc()
        tests_failed.append(("Orchestrator Schedule", str(e)))
    
    # Test 6: Test EMA Orchestrator
    logger.info("\n[Test 6] Testing EMAOrchestrator...")
    try:
        from rl_implementation.training.utils.rllm_components import create_ema_orchestrator
        
        ema_config = config.get('ema', {})
        ema_orch = create_ema_orchestrator(ema_config)
        
        logger.info(f"✓ EMAOrchestrator created:")
        logger.info(f"  - Enabled: {ema_orch.enabled}")
        logger.info(f"  - Decay: {ema_orch.decay}")
        logger.info(f"  - Decay on drift: {ema_orch.decay_on_drift}")
        
        tests_passed.append("EMAOrchestrator")
    except Exception as e:
        logger.error(f"✗ EMAOrchestrator test failed: {e}")
        traceback.print_exc()
        tests_failed.append(("EMAOrchestrator", str(e)))
    
    # Test 7: Test agent initialization (without models)
    logger.info("\n[Test 7] Testing agent initialization...")
    try:
        from rl_implementation.agents import CypherGeneratorAgent, OrchestratorAgent, ExperienceBuffer
        
        buffer = ExperienceBuffer(max_patterns=10)
        
        # Test Cypher Generator
        cypher_agent = CypherGeneratorAgent(
            schema_path=str(schema_path),
            experience_buffer=buffer,
            max_steps=5
        )
        logger.info("✓ Cypher Generator agent initialized")
        
        # Test Orchestrator
        orch_agent = OrchestratorAgent(
            schema_path=str(schema_path),
            experience_buffer=buffer,
            mode='generation'
        )
        logger.info("✓ Orchestrator agent initialized")
        
        tests_passed.append("Agent initialization")
    except Exception as e:
        logger.error(f"✗ Agent initialization failed: {e}")
        traceback.print_exc()
        tests_failed.append(("Agent initialization", str(e)))
    
    # Test 8: Test environment initialization
    logger.info("\n[Test 8] Testing environment initialization...")
    try:
        from rl_implementation.environments import GraphReasoningEnvironment
        
        # Initialize with task (not set_task)
        env = GraphReasoningEnvironment(
            task={'question': 'Test question'},
            api_url=config['neo4j_url'],
            max_turns=5
        )
        # Test reset
        obs, info = env.reset()
        logger.info(f"✓ Environment initialized (question: {obs.get('question', 'N/A')})")
        tests_passed.append("Environment initialization")
    except Exception as e:
        logger.error(f"✗ Environment initialization failed: {e}")
        traceback.print_exc()
        tests_failed.append(("Environment initialization", str(e)))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Component Test Summary")
    logger.info("=" * 80)
    logger.info(f"Passed: {len(tests_passed)}/{len(tests_passed) + len(tests_failed)}")
    
    if tests_passed:
        logger.info("\n✓ Passed tests:")
        for test in tests_passed:
            logger.info(f"  - {test}")
    
    if tests_failed:
        logger.error("\n✗ Failed tests:")
        for test, error in tests_failed:
            logger.error(f"  - {test}: {error}")
        return False
    
    return True


def run_training_test(config_path: str):
    """Run small-scale training test."""
    logger.info("\n" + "=" * 80)
    logger.info("Starting Small-Scale Training Test (Strategy 2)")
    logger.info("=" * 80)
    
    try:
        # Import trainer
        from rl_implementation.training.train_collaborative_system import CollaborativeTrainer
        
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Resolve paths relative to project root
        project_root = Path(__file__).parent.parent.parent
        config['schema_path'] = str(project_root / config['schema_path'])
        config['checkpoint_dir'] = str(project_root / config['checkpoint_dir'])
        config['log_dir'] = str(project_root / config['log_dir'])
        
        logger.info(f"Configuration loaded:")
        logger.info(f"  - Experiment: {config['experiment_name']}")
        logger.info(f"  - Questions per epoch: {config['questions_per_epoch']}")
        logger.info(f"  - Num epochs: {config['num_epochs']}")
        logger.info(f"  - Training mode: {config.get('gpu_allocation', {}).get('mode', 'N/A')}")
        logger.info(f"  - Schema: {config['schema_path']}")
        
        # Log GPU allocation
        gpu_config = config.get('gpu_allocation', {})
        logger.info(f"\nGPU Allocation (Sequential Mode):")
        logger.info(f"  - Total GPUs: {gpu_config.get('total_gpus', 'N/A')}")
        logger.info(f"  - Rollout: Cypher={gpu_config.get('rollout', {}).get('cypher_gpus', 'N/A')}, "
                   f"Orch={gpu_config.get('rollout', {}).get('orch_gpus', 'N/A')}")
        logger.info(f"  - Training: {gpu_config.get('training', {}).get('gpus', 'N/A')} GPUs")
        
        # Log Orchestrator schedule
        orch_schedule = config.get('orchestrator_schedule', {})
        logger.info(f"\nOrchestrator Schedule:")
        logger.info(f"  - Warmup epochs: {orch_schedule.get('warmup_epochs', 'N/A')}")
        logger.info(f"  - Joint frequency: {orch_schedule.get('joint_frequency', 'N/A')}")
        
        # Initialize trainer
        logger.info("\nInitializing CollaborativeTrainer...")
        trainer = CollaborativeTrainer(config)
        logger.info("✓ Trainer initialized successfully")
        
        # Log trainer state
        logger.info(f"\nTrainer State:")
        logger.info(f"  - Training mode: {trainer.training_mode}")
        logger.info(f"  - EMA enabled: {trainer.ema_orchestrator.enabled}")
        logger.info(f"  - EMA decay: {trainer.ema_orchestrator.decay}")
        
        # Run training
        logger.info("\n" + "=" * 80)
        logger.info("Starting training loop...")
        logger.info("=" * 80)
        trainer.train(num_epochs=config['num_epochs'])
        logger.info("=" * 80)
        logger.info("✓ Training completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Training test failed with error:")
        logger.error(f"  {type(e).__name__}: {e}")
        logger.error("\nFull traceback:")
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Small-scale test for collaborative multi-agent training (Strategy 2)"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='rl_implementation/config/training_config_test.yaml',
        help='Path to test configuration file'
    )
    parser.add_argument(
        '--skip-component-tests',
        action='store_true',
        help='Skip component tests and go straight to training'
    )
    parser.add_argument(
        '--component-tests-only',
        action='store_true',
        help='Only run component tests, skip training'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print("SEQUENTIAL TRAINING TEST (Strategy 2)")
    print("=" * 80)
    print(f"Config: {args.config}")
    print("=" * 80 + "\n")
    
    # Run component tests first (unless skipped)
    if not args.skip_component_tests:
        logger.info("Step 1: Running component tests...")
        if not run_component_tests():
            logger.error("\n❌ Component tests failed. Fix errors before running training.")
            sys.exit(1)
        logger.info("\n✅ All component tests passed!")
    else:
        logger.info("Skipping component tests (--skip-component-tests flag)")
    
    # Exit if only running component tests
    if args.component_tests_only:
        logger.info("\n✅ Component tests completed (--component-tests-only flag)")
        sys.exit(0)
    
    # Run training test
    logger.info("\nStep 2: Running training test...")
    success = run_training_test(args.config)
    
    # Final summary
    print("\n" + "=" * 80)
    if success:
        print("✅ TEST PASSED - Training completed successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Check logs in test_small_scale.log")
        print("2. Verify checkpoints were saved")
        print("3. Review training metrics")
        print("4. If everything looks good, scale up to more questions")
        sys.exit(0)
    else:
        print("❌ TEST FAILED - See errors above")
        print("=" * 80)
        print("\nDebugging tips:")
        print("1. Check test_small_scale.log for detailed error messages")
        print("2. Verify all dependencies are installed")
        print("3. Check GPU availability and Ray initialization")
        print("4. Verify model paths are correct")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Bugfix: ResourcePoolManager Initialization

## Issue

The `ResourcePoolManager` was not being initialized properly, causing a `KeyError: 'global_pool'` when trying to create worker groups.

**Error Message:**
```
KeyError: 'global_pool'
File ".../rllm_components.py", line 124, in create_worker_groups
    resource_pool=resource_pool_manager.get_resource_pool(mapping[Role.ActorRollout]),
File ".../ray_trainer.py", line 127, in get_resource_pool
    return self.resource_pool_dict[self.mapping[role]]
```

## Root Cause

The `ResourcePoolManager` is a dataclass that requires:
1. `resource_pool_spec` - Dictionary defining GPU resources per pool
2. `mapping` - Dictionary mapping roles to pool names
3. Calling `create_resource_pool()` to initialize the actual resource pools

The original implementation created the `ResourcePoolManager` but didn't call `create_resource_pool()`, so the `resource_pool_dict` was empty, causing the KeyError when trying to access pools.

## Fix

### Changed File

**`training/utils/rllm_components.py`** - `create_resource_pool_manager()` function

**Before:**
```python
def create_resource_pool_manager(
    n_gpus_per_node: int,
    nnodes: int = 1
) -> ResourcePoolManager:
    # ... setup code ...
    
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping
    )
    
    logger.info("Resource pool manager created")
    return resource_pool_manager, global_pool_id, mapping
```

**After:**
```python
def create_resource_pool_manager(
    n_gpus_per_node: int,
    nnodes: int = 1
) -> ResourcePoolManager:
    # ... setup code ...
    
    # Create resource pool manager
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping
    )
    
    # Initialize the resource pools (THIS WAS MISSING!)
    resource_pool_manager.create_resource_pool()
    
    logger.info("Resource pool manager created and initialized")
    return resource_pool_manager, global_pool_id, mapping
```

## How ResourcePoolManager Works

### 1. Initialization
```python
resource_pool_manager = ResourcePoolManager(
    resource_pool_spec={
        "global_pool": [6, 6]  # 6 GPUs on 2 nodes
    },
    mapping={
        Role.ActorRollout: "global_pool",
        Role.Critic: "global_pool"
    }
)
```

At this point:
- `resource_pool_spec` is set
- `mapping` is set
- `resource_pool_dict` is empty `{}`

### 2. Create Resource Pools
```python
resource_pool_manager.create_resource_pool()
```

This method:
- Iterates through `resource_pool_spec`
- Creates `RayResourcePool` objects for each pool
- Populates `resource_pool_dict`
- Checks resource availability

After this:
- `resource_pool_dict = {"global_pool": RayResourcePool(...)}`

### 3. Get Resource Pool
```python
pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
```

This method:
- Looks up role in `mapping`: `mapping[Role.ActorRollout]` → `"global_pool"`
- Gets pool from dict: `resource_pool_dict["global_pool"]` → `RayResourcePool(...)`

## Resource Pool Architecture

```
ResourcePoolManager
├── resource_pool_spec: {"global_pool": [6]}
├── mapping: {Role.ActorRollout: "global_pool", Role.Critic: "global_pool"}
└── resource_pool_dict: {"global_pool": RayResourcePool}
    └── RayResourcePool
        ├── process_on_nodes: [6]
        ├── use_gpu: True
        ├── max_colocate_count: 1
        └── name_prefix: "global_pool"
```

## Usage in Training

```python
# 1. Create resource pool manager
resource_pool_manager, global_pool_id, mapping = create_resource_pool_manager(
    n_gpus_per_node=6,
    nnodes=1
)

# 2. Create worker groups using the manager
worker_groups, role_mapping = create_worker_groups(
    config=ppo_config,
    tokenizer=tokenizer,
    resource_pool_manager=resource_pool_manager,  # Now properly initialized!
    mapping=mapping,
    use_async=False
)

# 3. Worker groups use the resource pools
actor_rollout_wg = RayWorkerGroup(
    resource_pool=resource_pool_manager.get_resource_pool(Role.ActorRollout),
    # ... other args ...
)
```

## Testing

After the fix, the initialization should succeed:

```bash
python -m rl_implementation.training.test_small_scale
```

Expected output:
```
Creating resource pool: 6 GPUs x 1 nodes
Resource pool manager created and initialized
Creating Ray worker groups...
Initializing ActorRollout worker group...
Worker groups created: ['actor_rollout']
```

## Related Code

- `verl/verl/trainer/ppo/ray_trainer.py` - `ResourcePoolManager` class definition
- `verl/verl/single_controller/ray.py` - `RayWorkerGroup` class
- `verl/verl/single_controller/ray_resource_pool.py` - `RayResourcePool` class

## Status

- ✅ Fixed in `rllm_components.py`
- ✅ Added `create_resource_pool()` call
- ✅ Updated docstring
- ✅ Ready for testing

---

**Fixed**: 2025-11-27  
**Tested**: Ready for testing


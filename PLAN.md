# Runtime-Aware Scheduling Strategy

## Goal

Add a new `RuntimePacking` placement strategy alongside the existing `HighThroughput` strategy, using an ABC-based strategy pattern for easy runtime switching.

## Architecture: ABC Strategy Pattern

```
pool_manager/placement.py:

  PlacementStrategy (ABC)        ← abstract base, shared helpers
  ├── HighThroughputPlanner      ← current behavior (scale out)
  └── RuntimePackingPlanner      ← new behavior (pack + long walltime)
  
  PlacementPlanner = HighThroughputPlanner  ← backward-compat alias
```

### ABC Base: `PlacementStrategy`

```python
class PlacementStrategy(ABC):
    def __init__(self, node_configs, task_resources, batch_size, max_workers, min_workers): ...
    
    @abstractmethod
    def plan(self, idle_count: int) -> list[Placement]: ...
    
    @abstractmethod
    def plan_for_tasks(self, tasks: list[TaskResources]) -> list[Placement]: ...
    
    # Concrete shared methods:
    def target_size(self, idle_count: int) -> int: ...      # calls self.plan()
    def _plan_no_configs(self, idle_count) -> list[Placement]: ...  # simple batch path
    def _plan_for_tasks_no_configs(self, tasks) -> list[Placement]: ...  # simple batch path
    def _bin_pack_with_configs(self, tasks) -> list[tuple[NodeConfig, list[TaskResources]]]: ...
    def _calculate_fit_per_node(self, node_config) -> int: ...
    def _min_plan(self) -> list[Placement]: ...
    @staticmethod
    def _tasks_fit_on_node(node_config, tasks) -> bool: ...
```

### `HighThroughputPlanner(PlacementStrategy)`

Current behavior. `plan()` and `plan_for_tasks()` handle both no-configs and with-configs paths. With-configs uses first-fit-decreasing bin-packing. No walltime logic.

### `RuntimePackingPlanner(PlacementStrategy)`

New behavior. Same bin-packing but adjusts node count for walltime and sets `Placement.walltime`.

Additional constructor params: `max_walltime_minutes: float = 1440`, `runtime_buffer: float = 0.1`

Walltime algorithm per node type:
```
max_runtime = max(task.runtime_minutes for task in tasks if set, default=max_walltime_minutes)
tasks_per_node = fit_per_node from bin-packing
total_batches = ceil(tasks_on_type / tasks_per_node)
raw_walltime = total_batches * max_runtime * (1 + buffer)

if raw_walltime <= max_walltime:
    nodes = bin_packing_result
    walltime = raw_walltime
else:
    nodes = ceil(raw_walltime / max_walltime)  # need more nodes
    batches_per_node = ceil(total_batches / nodes)
    walltime = batches_per_node * max_runtime * (1 + buffer)
```

Formatted as `HH:MM:SS` via `_format_walltime(minutes)`.

### Backward Compatibility

```python
# In placement.py:
PlacementPlanner = HighThroughputPlanner  # alias for backward compat
```

All existing code using `PlacementPlanner(...)` continues to work unchanged.

### Factory Function

```python
def make_placement_strategy(
    strategy: str,
    node_configs=None,
    task_resources=None,
    batch_size=1,
    max_workers=16,
    min_workers=0,
    max_walltime_minutes=1440,
    runtime_buffer=0.1,
) -> PlacementStrategy:
    if strategy == "runtime-packing":
        return RuntimePackingPlanner(...)
    return HighThroughputPlanner(...)
```

## Dataclass Changes

### `TaskResources` (line 10-14)

Add `runtime_minutes: float = 0` (0 = unknown, treat as max_walltime)

### `Placement` (line 17-21)

Add `walltime: str | None = None` (e.g., `"23:50:00"` for Slurm `-t`)

## Config Changes

### `ScalingPolicy` (scaling.py)

Add fields:
- `strategy: str = "high-throughput"` (options: `"high-throughput"`, `"runtime-packing"`)
- `max_walltime_minutes: float = 1440`
- `runtime_buffer: float = 0.1`

Update `placement_planner` property to pass new fields.

### `Config.from_file()` (config.py)

Parse `strategy`, `max_walltime_minutes`, `runtime_buffer` from `scaling` YAML section.

### YAML Example

```yaml
scaling:
  strategy: runtime-packing
  max_walltime_minutes: 1440
  runtime_buffer: 0.1
  min_workers: 0
  max_workers: 10
```

## Manager Changes (manager.py)

### `PoolManager.__init__` (line 116-124)

- Use `make_placement_strategy()` factory instead of direct `PlacementPlanner()` construction
- Validate: if `strategy == "runtime-packing"` and no `node_configs`, log warning and fall back to `high-throughput`

### `_start_workers_from_plan()` (line 409-445)

- After building `args` dict, if `p.walltime` is set, inject `args["time"] = p.walltime`
- This overrides any `time` in `submit_args` from config

### `run()` (line 143-188)

- Log strategy name on startup

## HTCondor Backend Changes

### All three backends (`condor_python.py`, `condor_subprocess.py`, `condor_rest.py`)

- Add `"RuntimeMinutes"` to HTCondor projection
- Parse into `TaskResources.runtime_minutes`

Example (condor_python.py line 34-39):
```python
tasks.append(TaskResources(
    cpus=float(job.get("requestcpus", 1)),
    memory_mb=int(job.get("requestmemory", 1024)),
    gpus=int(job.get("requestgpus", 0)),
    runtime_minutes=float(job.get("runtimeminutes", 0)),
))
```

## CLI Changes (`__main__.py`)

### `_run_test_strategy()`

- Parse `runtime_minutes` from JSON classads
- Show walltime in placement plan output when strategy is `runtime-packing`
- Add `--max-walltime` and `--runtime-buffer` CLI overrides

## Files to Modify

| File | Changes |
|------|---------|
| `pool_manager/placement.py` | ABC extraction, `HighThroughputPlanner`, `RuntimePackingPlanner`, `make_placement_strategy()`, `PlacementPlanner` alias |
| `pool_manager/scaling.py` | New fields on `ScalingPolicy`, updated `placement_planner` property |
| `pool_manager/config.py` | Parse new YAML fields |
| `pool_manager/manager.py` | Use factory, inject walltime in submit args, log strategy |
| `pool_manager/__main__.py` | Parse runtime_minutes, show walltime, CLI overrides |
| `pool_manager/work_queue/condor_python.py` | Add RuntimeMinutes to projection, parse it |
| `pool_manager/work_queue/condor_subprocess.py` | Same |
| `pool_manager/work_queue/condor_rest.py` | Same |
| `tests/test_placement.py` | Update imports, add RuntimePacking tests |
| `tests/test_manager_placement.py` | Add runtime-packing manager tests |

## Test Plan

### Existing Tests

All existing `PlacementPlanner` references in tests continue to work via the backward-compat alias. No changes needed to existing test logic.

### New Tests (`tests/test_placement.py`)

**`TestRuntimePackingPlan`**:
- `test_single_node_within_walltime`: 1000 tasks × 5 min, 8/node → 1 node, ~11.5h walltime
- `test_walltime_exceeds_max_needs_more_nodes`: Long tasks → 2+ nodes
- `test_tasks_without_runtime_uses_max`: Missing runtime → conservative walltime
- `test_mixed_runtimes_uses_max`: Heterogeneous → max runtime used
- `test_walltime_format_hhmmss`: Verify formatting
- `test_zero_tasks`: Empty → no placements
- `test_min_workers_respected`: Respects min_workers
- `test_max_workers_cap`: Can't exceed max_workers

**`TestRuntimePackingPlanForTasks`**:
- `test_plan_for_tasks_sets_walltime`: Verify Placement.walltime is set
- `test_plan_for_tasks_heterogeneous`: Mixed task sizes with walltime

**`TestMakePlacementStrategy`**:
- `test_high_throughput`: Returns HighThroughputPlanner
- `test_runtime_packing`: Returns RuntimePackingPlanner
- `test_unknown_strategy_raises`: ValueError

### Manager Integration Tests (`tests/test_manager_placement.py`)

- `test_start_workers_injects_walltime`: Verify `-t` is set from Placement.walltime
- `test_tick_runtime_packing`: End-to-end tick

## Verification

1. `pytest tests/` — all existing tests pass (backward compat)
2. `pytest tests/test_placement.py::TestRuntimePacking -v` — new tests pass
3. `python -m pool_manager test-strategy examples/tasks_w_runtime.json` — shows walltime

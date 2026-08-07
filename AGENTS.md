# Session Checkpoint — 2026-08-07

## Summary

Removed the HTCondor scheduler backend. HTCondor is only used as the work
queue source (`pool_manager.work_queue.*`); the `htcondor_rest` scheduler
backend is gone.

## Files Modified

- **`pool_manager/scheduler/htcondor_rest.py`** — deleted
- **`pool_manager/scheduler/__init__.py`** — dropped `HTCondorRESTAPIBackend`
  import/export
- **`pool_manager/manager.py`** — removed `htcondor_rest` case from
  `_make_scheduler` and the `HTCondorRESTAPIBackend` import
- **`tests/test_condor_backends.py`** — removed `TestHTCondorRESTAPISchedulerBackend`
- **`tests/test_manager_placement.py`** — removed `test_make_scheduler_htcondor_rest`
- **`pool-manager.yaml`, `pool-manager.gpu.yaml`** — removed `htcondor_rest`
  from backend comments
- **`README.md`, `PLAN.md`** — removed scheduler `htcondor_rest` docs
- **`AGENTS.md`** — this checkpoint

## Notes

- The `htcondor-rest` package dependency stays: `pool_manager/work_queue/condor_rest.py`
  uses `htcondor_rest.CondorClient` to query the HTCondor queue.

## Test Stats

250 tests total, all passing, no ruff errors.

# Session Checkpoint — 2026-07-29

## Summary

TUI now shows both idle and running job counts from the work queue backend.
`list_idle()` is called once per cycle; idle vs running is determined by the
`job_status` field on each `TaskResources` object. Only idle tasks are passed
to the placement planner for action computation.

## Files Modified

- **`pool_manager/placement.py`** — added `job_status: int = 0` field to
  `TaskResources` dataclass (0=unknown, 1=idle, 2=running)
- **`pool_manager/tui.py`**:
  - `PoolStateWidget` gains `running_jobs` reactive attribute
  - Render shows both "Idle:" and "Running:" lines
  - `_refresh()` separates idle (`job_status==1`) and running (`job_status==2`)
    tasks after a single `list_idle()` call
- **`pool_manager/work_queue/condor_python.py`** — `list_idle()` populates
  `job_status` from the `JobStatus` field (lowercased to `jobstatus`)
- **`pool_manager/work_queue/condor_rest.py`** — same
- **`pool_manager/work_queue/condor_subprocess.py`** — same
- **`tests/test_condor_backends.py`** — task resource assertions include
  `job_status`
- **`tests/test_tui.py`** — render assertion updated for new "Idle:" label format

## Design Decisions

- Backend `list_idle()` returns all tasks (not just idle ones); callers filter
  by `t.job_status == 1` when they need only idle tasks. This minimizes the
  number of backend calls (one per refresh cycle).
- All three Condor backends lowercase keys, so `job_status` lookup uses
  `job.get("jobstatus", 0)`.

## Test Stats

260 tests total, all passing, no ruff errors.

# Session Checkpoint — 2026-06-28

## Summary

Added node-aware placement: the pool manager can now choose from a list of node
configurations (CPU/memory/GPU) and pack idle HTCondor tasks into the minimum
number of nodes needed.

## Files Created

- **`pool_manager/placement.py`** — new module:
  - `TaskResources` dataclass (cpus, memory_mb, gpus)
  - `Placement` dataclass (node_config, count)
  - `PlacementPlanner` class with `target_size()` and `plan()` methods
  - Bin-packing algorithm: sorts node configs by capacity descending, packs as
    many tasks as fit per node, falls through smaller/GPU types when needed
  - No node configs = falls back to simple batch_size-based count

- **`tests/test_placement.py`** — 24 tests covering:
  - `target_size` with/without node configs, min/max, batch_size
  - `plan` for single node type, multiple types, GPU-required fallthrough,
    zero-idle, min_workers

- **`tests/test_manager_placement.py`** — 7 tests covering:
  - Planner creation from config, simple vs placement-aware startup,
    resource arg injection (cpus-per-task, mem, gpus), tick integration,
    distribution across node types

## Files Modified

- **`pool_manager/scheduler/base.py`** — added `NodeConfig` dataclass
  (name, cpus, memory_mb, gpus)
- **`pool_manager/scheduler/__init__.py`** — export `NodeConfig`
- **`pool_manager/config.py`** — `SchedulerConfig.node_configs` field,
  `ScalingPolicy.task_resources` field, YAML parsing for both
- **`pool_manager/scaling.py`** — `task_resources` field on `ScalingPolicy`,
  `placement_planner` property
- **`pool_manager/manager.py`** — `PoolManager` creates `PlacementPlanner`
  from config; `_start_workers` dispatches to `_start_workers_simple` or
  `_start_workers_from_plan`; workers started via plan get per-node-type
  submit args (`cpus-per-task`, `mem`, `gpus`)
- **`pool-manager.yaml`** — documented `task_resources` and `node_configs`

## Design Decisions

- When `node_configs` is defined, `batch_size` is ignored for packing
  (tasks-per-node is derived from capacity instead)
- Algorithm prefers larger nodes first to minimize node count
- GPU tasks automatically skip CPU-only nodes (detected via nc.gpus == 0
  when task_resources.gpus > 0)
- Per-node resource requirements are injected as submit args
  (`cpus-per-task`, `mem`, `gpus`) on each worker submission
- Backward compatible: no `node_configs` = original behavior unchanged

## Test Stats

105 tests total, all passing, no ruff errors.

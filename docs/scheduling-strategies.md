# Pool Manager Placement Strategies

The pool manager bridges **HTCondor** (work queue of idle jobs/tasks) with **HPC
schedulers** (Slurm, PBS, HTCondor REST). Its core loop runs on a configurable
`poll_interval`:

1. **Poll** HTCondor for idle tasks via `work_queue.list_idle()` — returns a
   list of `TaskResources` (cpus, memory_mb, gpus, runtime_minutes, job_status)
   per idle job.
2. **Plan** how many HPC workers to start via `PlacementPlanner.plan_for_tasks()`.
3. **Scale up** by submitting worker scripts to the HPC scheduler (`sbatch`,
   `qsub`, etc.) with per-node resource args.
4. **Reconcile** active scheduler jobs against internal tracking.
5. **Scale down** by signalling excess workers with `SIGTERM` (graceful drain).

## Selecting a strategy

Placement is chosen with `scheduler.placement_strategy` in the config file:

```yaml
scheduler:
  placement_strategy: runtime_aware  # simple | node_aware | runtime_aware
```

| Strategy | Config value | Uses `node_configs` | Uses wall-time limits |
|----------|--------------|:-------------------:|:---------------------:|
| Batch-based scaling | `simple` | no (ignored) | no |
| Resource bin-packing | `node_aware` | yes | no |
| Runtime-aware packing | `runtime_aware` (default) | yes | yes |

All strategies clamp the target to `[scaling.min_workers, scaling.max_workers]`
and share the same scale-up/scale-down cooldowns, drain, and shutdown logic.

---

## 1. `simple` — batch-based scaling

### How it works

- Target workers = `ceil(idle_tasks / batch_size)`, clamped to
  `[min_workers, max_workers]`.
- Every worker is identical and submitted with the scheduler-level
  `submit_args` template (job name suffix `default`).
- `node_configs` are ignored even if present in the config.

```yaml
scaling:
  batch_size: 4
  min_workers: 0
  max_workers: 16

scheduler:
  placement_strategy: simple
  submit_args:
    partition: defq
    account: myproject
```

### What it assumes

- Tasks are homogeneous, or their individual resource requirements are unknown
  or unimportant.
- A single worker can process up to `batch_size` idle tasks before draining.
- No differentiation between node types is needed (no CPU/GPU/memory tiers,
  no queue/partition choice per workload).
- All nodes are equally reachable — queue wait time does not depend on node
  type.

---

## 2. `node_aware` — resource bin-packing

### How it works

- The planner packs idle tasks into the **minimum number of nodes** chosen
  from `scheduler.node_configs`.
- Node configs are sorted by total capacity descending
  (`cpus × memory × max(gpus, 1)`), with `priority` taking precedence.
- **Task placement** uses first-fit decreasing: tasks are sorted by resource
  size descending and placed into the first node with sufficient remaining
  capacity (`plan_for_tasks`).
- **GPU affinity**: GPU-requiring tasks skip GPU-less nodes; CPU-only tasks
  skip GPU nodes unless no other option exists.
- Per-node resource requirements (`cpus-per-task`, `mem`, `gpus`) and any
  per-node `submit_args` are injected into each worker submission, overriding
  scheduler-level defaults.
- `batch_size` is ignored; tasks-per-node is derived from node capacity.

```yaml
scheduler:
  placement_strategy: node_aware
  node_configs:
    - name: small
      cpus: 4
      memory_mb: 8000
      gpus: 0
    - name: large
      cpus: 16
      memory_mb: 64000
      gpus: 0
```

### What it assumes

- `node_configs` accurately describe the available node types (CPU count,
  memory, GPU count) and that each is actually reachable.
- Task resource requests (`RequestCpus`, `RequestMemory`, `RequestGpus`) reflect
  true demand — a worker running a task gets exactly what the task asked for.
- Queue wait is comparable across node types; the planner always prefers the
  largest nodes that fit, so if large allocations queue slowly, this strategy
  can under-deliver (see alternative #3 below).
- Tasks have no wall-time constraint — any task can run for as long as needed
  on any node.

---

## 3. `runtime_aware` — runtime-aware packing (default)

### How it works

- Everything from `node_aware`, plus wall-time matching.
- A node config may set `time_hrs` and/or `time_min`, which become that node
  type's maximum wall time (`0`/unset = no limit).
- Each task's expected runtime is read from the `runtime_minutes` classad on
  the HTCondor job (set with `+runtime_minutes = 90` in the submit
  description; missing = `0` = no constraint).
- Tasks whose runtime exceeds a node's limit are placed on a longer-runtime
  node type, or left unplaced if no type fits.
- When several tasks share a node, the **longest** task's runtime must fit
  within the node's limit.
- When capacity is equal, the planner prefers shorter-runtime nodes, reserving
  long queues for long tasks.
- The node wall time is injected into the submission as
  `--time=HH:MM:00` (e.g. `time_min: 90` → `--time=01:30:00`).

```yaml
scheduler:
  placement_strategy: runtime_aware
  node_configs:
    - name: debug_gpu
      time_min: 30
      cpus: 128
      memory_gb: 256
      gpus: 4
    - name: short_gpu
      time_hrs: 12
      cpus: 128
      memory_gb: 256
      gpus: 4
    - name: gpu
      time_hrs: 48
      cpus: 128
      memory_gb: 256
      gpus: 4
```

### What it assumes

- Jobs carry an accurate `runtime_minutes` estimate. An estimate shorter than
  actual runtime means the scheduler kills the worker at the node wall time.
- Node wall-time limits reflect real scheduler constraints (partition/QoS time
  caps) and are enforceable via `--time`.
- Longer queues actually accept the longest task; otherwise long tasks are left
  unplaced and wait for the next tick.
- Runtime is the dominant placement factor after capacity — costs, priority,
  and queue wait are not modelled.

---

## Comparison

| Question | `simple` | `node_aware` | `runtime_aware` |
|---|---|---|---|
| Target = `ceil(idle / batch_size)` | yes | no | no |
| Uses `node_configs` | no | yes | yes |
| Minimises node count | no | yes | yes |
| GPU-aware placement | no | yes | yes |
| Matches tasks to wall time | no | no | yes |
| Injects per-node submit args | no | yes | yes |
| Injects `--time` | no | no | yes (when set) |

---

## Alternative strategies (future work)

The implemented strategies above are deliberately simple. Depending on the
site, the following are reasonable extensions; none are currently wired to
`placement_strategy`.

1. **Best-Fit Decreasing (BFD)** — place each task into the node leaving the
   least remaining capacity. Better packing than FFD at higher cost
   (O(n × m) per cycle).
2. **Worst-fit / spread** — place each task into the node with the most
   remaining capacity. Balances load across nodes at the cost of more nodes.
3. **Throughput-optimized: many small nodes** — prefer many small allocations
   over fewer large ones; small jobs queue faster and backfill more easily.
4. **Throughput-optimized: few large nodes** — the current `node_aware`
   default; fewer scheduler jobs, better per-node utilization.
5. **Queue-wait-aware** — query scheduler queue depth/estimated wait and fall
   through node types when a partition is congested.
6. **Backfill-optimised packing** — submit short-wall-time workers to exploit
   backfill windows; re-submit if killed.
7. **Preemptible / spot workers** — bulk of workers on low-QoS/preemptible
   slots with a small reserved high-priority reserve.
8. **Predictive scaling** — forecast near-future idle count from historical
   inflow and scale before the backlog arrives.
9. **Plateau / holdover** — keep a buffer of workers across transient dips to
   avoid scale-down thrash.
10. **Cost-optimised cheapest-fit** — sort node types by SU/hour instead of
    capacity.
11. **Priority-aware multi-policy** — classify tasks by label/owner/project
    and run a different planner per class.
12. **Hybrid reserved + on-demand** — `min_workers` on standing reserved
    capacity, overflow onto on-demand/preemptible nodes.
13. **Load-aware dynamic batch size** — adjust `batch_size` from the scheduler
    queue depth instead of a fixed value.

## Decision matrix

| Goal | Best strategy | Trade-off |
|---|---|---|
| Minimise node count | BFD or current FFD | Higher planning cost (BFD) |
| Maximise throughput | Many small nodes + backfill | More scheduler jobs to manage |
| Minimise queue wait | Queue-wait-aware + preemptible | Complexity, preemption handling |
| Minimise cost | Cheapest-fit + preemptible | Slower throughput, preemption risk |
| Minimise thrash | Plateau holdover | Slightly slower reaction time |
| Mixed workloads | Priority-aware multi-policy | Complexity |
| Predictable peaks | Predictive scaling | Over-provisioning risk |

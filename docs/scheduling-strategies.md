# Pool Manager Scheduling Strategies

## Current Strategy: Elastic Bin-Packing

### Architecture Overview

The pool manager bridges **HTCondor** (work queue of idle jobs/tasks) with **HPC schedulers** (Slurm, PBS, HTCondor REST, local). Its core loop runs on a configurable `poll_interval`:

1. **Poll** HTCondor for idle tasks via `work_queue.list_idle()` — returns a list of `TaskResources` (cpus, memory_mb, gpus) per idle job.
2. **Plan** how many HPC workers to start via `PlacementPlanner.plan_for_tasks()`.
3. **Scale up** by submitting worker scripts to the HPC scheduler (`sbatch`, `qsub`, etc.) with per-node resource args.
4. **Reconcile** active scheduler jobs against internal tracking.
5. **Scale down** by signalling excess workers with `SIGTERM` (graceful drain).

### Two Modes

#### Simple Mode (no `node_configs`)

- `batch_size` controls how many idle tasks a single worker processes.
- Target workers = `ceil(idle_tasks / batch_size)`, clamped to [`min_workers`, `max_workers`].
- All workers are identical, submitted with a single `submit_args` template.
- Used when tasks are homogeneous or resource requirements are unknown.

#### Node-Aware Mode (`node_configs` defined)

- **Goal**: Minimize the number of HPC nodes needed to cover all idle tasks.
- **Algorithm**: Greedy bin-packing — sort node types by total capacity descending (`cpus × memory_mb × max(gpus, 1)`). For each node type, compute max tasks that fit per node, pack as many nodes as needed, fall through to smaller types.
- **Task placement (`plan_for_tasks`)**: First-fit decreasing (FFD) — sort tasks by resource size descending, place each into the first node with sufficient remaining capacity.
- **GPU affinity**: GPU-requiring tasks skip GPU-less nodes; CPU-only tasks skip GPU nodes unless no other option.
- **Per-node submit args**: Each `NodeConfig` carries optional `submit_args` injected into the scheduler submission (`--cpus-per-task`, `--mem`, `--gpus`).

### Scaling Decisions

| Decision | Mechanism |
|---|---|
| When to add workers | `active < target` && cooldown elapsed |
| When to remove workers | `active > target` && cooldown elapsed |
| Which workers to drain | Highest-capacity nodes first (reverse of placement order), preferring excess per node type |
| Anti-flapping | Independent `scale_up_cooldown` / `scale_down_cooldown` timers (default 30 s / 60 s) |
| Graceful shutdown | SIGTERM workers, wait `drain_timeout`, force-cancel leftovers |
| State recovery | On startup, `list_active()` recovers tracked jobs from the scheduler |

### Strengths

- **Resource-proportional**: Workers request exactly the resources they need per node.
- **Minimal node count**: Bin-packing reduces HPC allocation footprint.
- **Backend-agnostic**: Works with Slurm (subprocess, REST, SFAPI), PBS, local, HTCondor REST.
- **Graceful drain**: Workers drain in-place rather than being killed mid-task.

### Limitations

- **Reactive only**: Scales in response to backlog, not ahead of it. No prediction.
- **No queue awareness**: Ignores scheduler queue depth, wait times, or backfill windows.
- **Single-policy**: One planner applies across all node types; no per-queue strategy.
- **No preemption or priority**: All tasks and workers are treated equally.
- **No cost modelling**: Does not consider allocation charge rates or node cost.

---

## Alternative Strategies

### 1. Best-Fit Decreasing (BFD)

Place each task into the node that leaves the **least remaining capacity** after packing.

- **Trade-off**: Better bin packing than FFD (fewer nodes) at higher computational cost.
- **When useful**: Scenarios where node allocations are expensive or scarce.
- **Cost**: O(n × m) per planning cycle vs. O(n log n + n × m) for the current FFD.

### 2. Worst-Fit / Spread Strategy

Place each task into the node with the **most remaining capacity**.

- **Trade-off**: Spreads load evenly across nodes, reducing fragmentation at the cost of more nodes.
- **When useful**: If nodes can be shared with other users/processes, spreading avoids hot spots.
- **Downside**: Increases total node count (and cost).

### 3. Throughput-Optimized: Many Small Nodes

Prefer many small node allocations over fewer large ones.

- **Rationale**: Smaller HPC jobs often start faster (less queue wait, more backfill opportunities, fit in more partitions). Increases task parallelism.
- **When useful**: High-throughput workloads with many small, independent tasks. Long queue wait times for large allocations on the scheduler.
- **Downside**: More scheduler jobs to manage, higher per-job overhead.

### 4. Throughput-Optimized: Few Large Nodes (Current)

Prefer fewer large nodes to maximise tasks-per-allocation.

- **Rationale**: Lower per-job overhead, better utilisation, fewer scheduler submissions.
- **When useful**: Queue wait times are low, or large allocations are as fast as small ones.
- **Downside**: Single large job can be queued longer; failure removes more capacity.

### 5. Queue-Wait-Aware Scheduling

Query the scheduler's queue depth or estimated wait time before choosing node types.

- **Strategy**: If large-node queue depth is high, fall through to smaller node types even if packing is suboptimal. Conversely, if small-node queue is congested, submit larger batches.
- **When useful**: HPC centres with heterogeneous queue wait times across partitions/QOS levels.
- **Implementation**: Poll `squeue --format=%P,%T | ...` or equivalent to estimate per-partition wait.
- **Downside**: Extra API calls; wait time estimation is heuristic.

### 6. Backfill-Optimised Packing

Submit workers with short wall-time limits or lower priority to exploit scheduler backfill windows.

- **Strategy**: Set `--time=00:30:00` on some workers so they fit into backfill slots. Re-submit if they get killed.
- **When useful**: Scheduler supports backfill and queue is busy.
- **Trade-off**: Workers may be preempted, requiring restart logic.

### 7. Preemptible / Spot / Low-Priority Workers

Use the cheapest allocation class available (e.g. `--qos=low`, `--partition=spot`).

- **Strategy**: Bulk of workers on preemptible/low QoS; keep a small reserve of high-priority nodes as insurance.
- **When useful**: Scheduler offers preemptible/spot pricing with priority preemption.
- **Trade-off**: Workers can be killed at any time; work must be restartable.

### 8. Predictive / Proactive Scaling

Start workers **before** tasks arrive based on historical patterns.

- **Strategy**: Track inflow rate; use a moving average or simple linear regression to predict near-future idle count. Scale up to predicted demand during cooldown.
- **When useful**: Workloads with periodic or predictable spikes (e.g., daily cron, workflow DAG stages).
- **Risk**: Over-provisioning if prediction is wrong.
- **Implementation**: Add a `predictor` module that feeds `idle_count * fudge_factor` into the target calculation.

### 9. Plateau / Holdover Strategy

Keep a buffer of workers running during transient dips in idle count.

- **Strategy**: Instead of scaling down immediately when `idle < target`, hold workers for N ticks. Only scale down if idle count stays below threshold.
- **When useful**: Workloads with high variance between poll intervals (bursty). Prevents thrashing.
- **Current approximation**: The scale-down cooldown timer serves a similar purpose, but a count-based buffer is more direct.

### 10. Cost-Optimised: Cheapest-Fit

Sort node types by cost (e.g. SU/hour) instead of capacity.

- **Strategy**: Cheapest nodes first; only spill to expensive nodes when cheap capacity is exhausted.
- **When useful**: Charging models differ per node type (e.g. GPU nodes cost 2× CPU nodes).
- **Trade-off**: May use more nodes total, but lower total cost.

### 11. Priority-Aware / Multi-Policy

Assign different placement strategies per task priority or class.

- **Strategy**: High-priority tasks get first-class nodes (fast queue, dedicated); low-priority tasks fill backfill/preemptible slots.
- **When useful**: Multi-tenant pools, mixed workloads (urgent vs. best-effort).
- **Implementation**: Classify tasks by label/owner/project; apply different `PlacementPlanner` per class.

### 12. Hybrid: Reserved + On-Demand

Maintain a base fleet of cheap/reserved nodes (e.g. via standing Slurm allocations) and burst overflow onto on-demand/preemptible nodes.

- **Strategy**: `min_workers` runs on reserved capacity; excess spills to on-demand with separate config.
- **When useful**: Guaranteed minimum throughput with elastic overflow.
- **Implementation**: Two `PlacementPlanner` instances chained: reserved first, then overflow.

### 13. Load-Aware Dynamic Batch Size

Adjust `batch_size` based on system load instead of a fixed value.

- **Strategy**: When the scheduler queue is deep, increase `batch_size` (more tasks per worker, fewer submissions). When shallow, decrease it.
- **When useful**: Variable scheduler load; want to reduce submission overhead during congestion.

---

## Decision Matrix

| Goal | Best Strategy | Trade-off |
|---|---|---|
| Minimise node count | BFD or current FFD | Higher planning cost (BFD) |
| Maximise throughput | Many small nodes + backfill | More scheduler jobs to manage |
| Minimise queue wait | Queue-wait-aware + preemptible | Complexity, preemption handling |
| Minimise cost | Cheapest-fit + preemptible | Slower throughput, preemption risk |
| Minimise thrash | Plateau holdover | Slightly slower reaction time |
| Mixed workloads | Priority-aware multi-policy | Complexity |
| Predictable peaks | Predictive scaling | Over-provisioning risk |

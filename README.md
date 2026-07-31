# pool-manager

Bridges an HTCondor work queue to an HPC scheduler (Slurm, PBS, or HTCondor).
The pool manager polls HTCondor for idle jobs, computes how many worker nodes
are needed (optionally using node-aware bin-packing), and submits workers to
the HPC cluster. Workers drain gracefully when demand drops.

## Installation

Requires Python >= 3.10.

### Recommended: uv

```bash
uv sync
```

This creates a `.venv` and installs the package and all dependencies.

### pip

```bash
pip install -e .
```

### Optional extras

| Extra | Provides | Backends enabled |
|-------|----------|------------------|
| `[rest]` | `httpx` | `slurm_rest`, `condor_rest`, `htcondor_rest` |
| `[htcondor]` | `htcondor` (Python bindings) | `condor_python` |
| `[sfapi]` | `sfapi_client` | `slurm_sfapi` (NERSC) |
| `[dev]` | `pytest`, `mypy`, `ruff` | testing & linting |

Example:

```bash
uv sync --extra rest --extra dev
```

or with pip:

```bash
pip install -e ".[rest,dev]"
```

## Configuration

Configuration is written in YAML. By default the daemon looks for
`pool-manager.yaml` in the current directory.

### Minimal config

```yaml
work_queue:
  backend: condor_subprocess

scheduler:
  backend: slurm_subprocess
  worker_script: /path/to/worker.sh

scaling:
  min_workers: 0
  max_workers: 4
  batch_size: 1
```

### Scaling policy

```yaml
scaling:
  min_workers: 0         # minimum workers to keep (even with no jobs)
  max_workers: 10        # maximum workers allowed
  batch_size: 1          # idle jobs per worker (ignored when node_configs is set)
  scale_up_cooldown: 30  # seconds between scale-up events
  scale_down_cooldown: 60
  drain_timeout: 120     # seconds before force-cancelling draining workers
```

### Work queue (HTCondor)

```yaml
work_queue:
  backend: condor_python      # condor_python | condor_subprocess | condor_rest
  schedd_name: ""              # optional schedd name (default pool)
  constraint: "" # classad expression (default: all jobs; e.g. "JobStatus == 1" for idle only)
  rest_url: ""                 # required for condor_rest
```

### HPC scheduler

Select one of these `backend` values:

| Backend | Command | Use case |
|---------|---------|----------|
| `slurm_subprocess` | `sbatch`/`scancel`/`squeue` | Any Slurm cluster |
| `slurm_rest` | REST API | Slurm with `slurmrestd` |
| `slurm_sfapi` | NERSC SFAPI | Perlmutter (NERSC) |
| `pbs_subprocess` | `qsub`/`qdel`/`qstat` | PBS/Torque clusters (ALCF) |
| `htcondor_rest` | HTCondor REST | HTCondor as scheduler |

#### Slurm examples

Basic Slurm:

```yaml
scheduler:
  backend: slurm_subprocess
  worker_script: /path/to/htcondor_worker.sh
  submit_args:
    partition: defq
    account: myproject
    time: "08:00:00"
    nodes: 1
    ntasks: 1
```

_NERSC Perlmutter_ (see [NERSC Slurm docs](https://docs.nersc.gov/systems/perlmutter/running-jobs/#example-scripts)):

```yaml
scheduler:
  backend: slurm_subprocess
  worker_script: /global/homes/m/myuser/htcondor_worker.sh
  submit_args:
    qos: regular
    constraint: cpu
    nodes: 1
    ntasks: 1
    cpus-per-task: 128
    time: "04:00:00"
    account: mXXXX
    job-name: pool-worker
    output: "logs/pool_worker_%j.log"
    error: "logs/pool_worker_%j.log"
```

For GPU nodes at Perlmutter:

```yaml
scheduler:
  backend: slurm_sfapi
  machine: perlmutter
  sfapi_client_id: "..."
  sfapi_client_secret: "..."
  worker_script: /global/homes/m/myuser/htcondor_worker.sh
  submit_args:
    qos: regular
    constraint: gpu
    nodes: 1
    ntasks: 1
    cpus-per-task: 64
    gpus: 4
    time: "04:00:00"
    account: mXXXX
```

#### PBS examples (ALCF)

_ALCF (Crux / Polaris)_ — see [ALCF running-jobs docs](https://docs.alcf.anl.gov/crux/queueing-and-running-jobs/running-jobs/):

```yaml
scheduler:
  backend: pbs_subprocess
  worker_script: /home/myuser/htcondor_worker.sh
  submit_args:
    A: myproject          # -A → account
    q: debug              # -q → queue
    l: "select=1:ncpus=64:system=crux"
    l: "walltime=01:00:00"
    N: pool-worker
    o: "logs/pool_worker.log"
    e: "logs/pool_worker.log"
```

Note: PBS keys are passed to `qsub` as `-key value`. Underscores in keys are
converted to hyphens automatically (e.g., `my_key` → `-my_key value`). When
multiple values for the same flag are needed (like `-l` above), duplicate the
key in `submit_args` so the later value is used, or set the value to a
space-joined string that PBS accepts.

#### NERSC SFAPI (slurm_sfapi)

```yaml
scheduler:
  backend: slurm_sfapi
  machine: perlmutter
  sfapi_client_id: "your-client-id"
  # One of:
  sfapi_client_secret: '{"keys":[...]}'      # JWK JSON string
  # or:
  sfapi_key_path: /path/to/key.pem           # key file path
  sfapi_user: ""                              # filter jobs by user (default: all)
```

### Placement strategies

Placement is selected with `scheduler.placement_strategy`, one of:

| Strategy | Description |
|----------|-------------|
| `simple` | `batch_size`-based scaling; ignores `node_configs` |
| `node_aware` | Bin-packs idle tasks into the minimum number of `node_configs` nodes by CPU/memory/GPU |
| `runtime_aware` | `node_aware` plus wall-time matching against per-node limits (default) |

#### `simple`

The default when no `node_configs` are configured (or explicitly selected).
Target workers = `ceil(idle_tasks / batch_size)`, clamped to
[`min_workers`, `max_workers`]. All workers are identical and submitted with
the single scheduler-level `submit_args` template.

```yaml
scheduler:
  placement_strategy: simple
  batch_size: 4
  submit_args:
    partition: defq
```

#### `node_aware`

When `node_configs` is defined, the pool manager ignores `batch_size` and
instead packs idle tasks into the minimum number of nodes, choosing from the
available node types based on resource requirements.

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

Each idle HTCondor job's resource requirements (`RequestCpus`, `RequestMemory`,
`RequestGpus`) are read from `condor_q` per job. The planner packs tasks
into the minimum number of nodes and GPU tasks automatically skip CPU-only
nodes.

Per-node resource requirements (`cpus-per-task`, `mem`, `gpus`) are injected
into each worker's submit args automatically.

#### `runtime_aware`

Runtime-aware placement (the default) matches tasks to node types by expected
wall time, so long tasks don't get stuck in short debug queues. It extends
`node_aware` with wall-time limits.

Add `time_hrs` and/or `time_min` to a node config to set that node type's
maximum wall time (`0` or unset means no limit). Each task's expected runtime
is read from the `runtime_minutes` classad on the HTCondor job; add it in the
job's submit description, e.g. `+runtime_minutes = 90`. Tasks whose runtime
exceeds a node's limit are placed on a different node type (or left unplaced
if no type fits). When packing several tasks onto one node, the longest task
runtime must fit. The planner also prefers shorter-runtime nodes when capacity
is equal, reserving longer queues for longer tasks.

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

The node runtime is injected into the sbatch job as `--time=HH:MM:00`.
Example: `time_min: 90` → `--time=01:30:00`.

> Note: `pool-manager test-strategy` reads the expected runtime from the
> `runtime_minutes` field of each record in the sample JSON file (see
> `examples/tasks_w_diff_runtime.json`).

See `docs/scheduling-strategies.md` for how each strategy works and what it
assumes.

### Metrics

Enable Prometheus metrics by setting `metrics_port` in the config:

```yaml
metrics_port: 9090
```

Available metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `pool_manager_idle_jobs` | Gauge | Number of idle HTCondor jobs |
| `pool_manager_active_workers` | Gauge | Number of active workers |
| `pool_manager_pending_workers` | Gauge | Number of pending workers |
| `pool_manager_running_workers` | Gauge | Number of running workers |
| `pool_manager_draining_workers` | Gauge | Number of draining workers |
| `pool_manager_target_workers` | Gauge | Target number of workers |
| `pool_manager_workers_started_total` | Counter | Total workers started |
| `pool_manager_workers_stopped_total` | Counter | Total workers stopped |
| `pool_manager_scale_up_events_total` | Counter | Total scale-up events |
| `pool_manager_scale_down_events_total` | Counter | Total scale-down events |
| `pool_manager_tick_duration_seconds` | Histogram | Duration of main loop tick |
| `pool_manager_workers_by_node_type` | Gauge | Workers by node type |
| `pool_manager_tasks_by_node_type` | Gauge | Tasks placed by node type |

Access metrics at `http://localhost:9090/metrics`.

A Grafana dashboard is available at `docs/grafana-dashboard.json` covering all
metrics with panels for worker counts, node-type breakdowns, scale events,
and tick-duration quantiles.

### TUI Dashboard

Run the interactive terminal dashboard:

```bash
pool-manager tui
pool-manager tui -c /path/to/config.yaml
```

The TUI displays:
- Pool status (idle jobs, active/draining workers, uptime)
- Placement plan by node type
- Worker table with job IDs and states

Press `q` to quit, `r` to refresh.

## Usage

### Daemon

```bash
pool-manager run
pool-manager run -c /path/to/config.yaml
pool-manager run --log-level DEBUG
```

The daemon loops every `poll_interval` seconds: it queries HTCondor for idle
jobs, reconciles the tracked job state, and scales workers up or down. On
SIGINT/SIGTERM it drains all workers gracefully before exiting.

### test-strategy

Dry-run the placement planner against real job data. This reads a JSON file
of HTCondor job classads (as produced by `condor_q -json`), computes the
optimal node placement, and prints the plan without submitting anything.

```bash
# Dump idle jobs from HTCondor
condor_q -json > /tmp/idle_jobs.json

# Run the test-strategy planner
pool-manager test-strategy /tmp/idle_jobs.json

# With current running workers for delta
pool-manager test-strategy /tmp/idle_jobs.json --running 5

# With per-type running counts (when using node_configs)
pool-manager test-strategy /tmp/idle_jobs.json -rt small=3 -rt large=2
```

Example output:

```
Tasks: 24
Node configs: 2 (small, large)
Target workers: 4 (max=10, min=0)
Current workers: 3  (add 1)

Placement plan:
  large x 2 (cpus=16 mem=64000MB gpus=0)
  small x 2 (cpus=4 mem=8000MB gpus=0)

Total nodes: 4
Total tasks placed: 24 (6 avg tasks/node)
```

## Testing

Run the test suite with:

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=pool_manager --cov-report=term-missing
```

There are 250+ tests covering config parsing, placement logic (bin-packing,
runtime-aware placement, GPU/cross-type fallthrough), manager integration
(scale-up, scale-down, per-type signalling), all scheduler backends, work
queue backends, and state parsing.

### Linting

```bash
uv run ruff check
uv run ruff format --check
```

## See also

- `examples/` — example configs for SLURM, PBS, and sample test data
- `PLAN.md` — architecture and design decisions
- `AGENTS.md` — incremental development log

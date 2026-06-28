# Pool Manager: HTCondor + HPCScheduler Bridge

## Overview

A persistent daemon that monitors a work queue (HTCondor) and dynamically manages a pool of worker jobs on an HPC scheduler (Slurm, PBS, etc.). Each worker job runs a user-provided script (e.g. `htcondor_worker.sh`) that starts a condor worker daemon pulling jobs directly from the HTCondor schedd. When the queue has work, the daemon submits scheduler jobs; when the queue empties, it drains the pool gracefully.

The entire system is built on pluggable abstract base classes so any component (queue probing, scheduler interaction) can be swapped out without changing the core logic.

## Class Hierarchy

```
┌─────────────────────────────────────────────────────┐
│                    PoolManager                       │
│  (orchestrator — scaling loop, drain, health)        │
│  ┌─────────────────┐    ┌─────────────────────────┐ │
│  │   WorkQueue      │    │    HPCScheduler         │ │
│  │  (abstract)      │    │   (abstract)            │ │
│  └────────┬─────────┘    └──────────┬──────────────┘ │
│           │                         │                │
└───────────┼─────────────────────────┼────────────────┘
            │                         │
            ▼                         ▼
  ┌─────────────────┐      ┌───────────────────────┐
  │ CondorWorkQueue  │      │ SlurmScheduler         │
  │  - PythonBindings│      │  - SubprocessBackend   │
  │  - Subprocess    │      │  - RESTAPIBackend      │
  │  - RESTAPI       │      └───────────────────────┘
  └─────────────────┘      ┌───────────────────────┐
                            │ PBSScheduler           │
                            │  - SubprocessBackend   │
                            │  - RESTAPIBackend      │
                            └───────────────────────┘
```

## Abstract Base Classes

### `WorkQueue` (abstract)

Probes the source of work. Subclasses implement one method — how to count idle/pending work items.

```python
class WorkQueue(ABC):
    @abstractmethod
    def count_idle(self) -> int: ...
    @abstractmethod
    def name(self) -> str: ...
```

**Planned implementations:**
| Class | Backend | Description |
|---|---|---|
| `CondorWorkQueue(PythonBackend)` | `htcondor` Python bindings | Direct schedd query via `htcondor.Schedd().query()` |
| `CondorWorkQueue(SubprocessBackend)` | `subprocess` calling `condor_q` | Parse CLI output when bindings aren't available |
| `CondorWorkQueue(RESTAPIBackend)` | HTCondor REST API | HTTP client for HTCondor's REST API endpoint |

### `HPCScheduler` (abstract)

Manages worker jobs on the execution platform. Subclasses implement: submit, cancel, list active jobs.

```python
class HPCScheduler(ABC):
    @abstractmethod
    def submit(self, script_path: str, args: dict) -> str: ...
    @abstractmethod
    def cancel(self, job_id: str) -> None: ...
    @abstractmethod
    def list_active(self) -> list[JobInfo]: ...
    @abstractmethod
    def signal(self, job_id: str, sig: str) -> None: ...
    @abstractmethod
    def name(self) -> str: ...
```

**Planned implementations:**
| Class | Backend | Description |
|---|---|---|
| `SlurmScheduler(SubprocessBackend)` | `sbatch`/`scancel`/`squeue` | Standard CLI interface |
| `SlurmScheduler(RESTAPIBackend)` | Slurm REST API v0.0.38+ | HTTP-based job management |
| `PBSScheduler(SubprocessBackend)` | `qsub`/`qdel`/`qstat` | PBS/Torque CLI |
| `LocalScheduler(SubprocessBackend)` | `Popen`/`kill` | Run workers as local processes (testing) |

### `PoolManager`

The core coordinator. Composes a `WorkQueue` + `HPCScheduler`.

```python
class PoolManager:
    def __init__(self, work_queue: WorkQueue,
                 scheduler: HPCScheduler,
                 config: Config): ...

    async def run(self): ...   # main loop
    def _scale(self): ...      # scale decision
    def _drain(self): ...      # graceful drain protocol
```

## Architecture Diagram

```
┌────────────────────────┐      ┌───────────────────────────┐
│   HTCondor Schedd      │      │   HPCScheduler Cluster    │
│                        │      │                           │
│  ┌──────────────────┐  │      │  pool-manager             │
│  │ CondorWorkQueue   │◄─┼──────┼──► (PoolManager)          │
│  │  .count_idle()    │  │      │   │                       │
│  └──────────────────┘  │      │   │ .submit/.cancel        │
│                        │      │   ▼                       │
│  ◄── condor jobs ──────┼──────┼─── Worker Job             │
│                        │      │    └── htcondor_worker.sh  │
│                        │      │         └── condor daemon  │
│                        │      │              ◄──► schedd   │
│                        │      └───────────────────────────┘
```

## Pluggable Backend Pattern (Strategy)

Each concrete class uses a *strategy* pattern internally:

```python
class CondorWorkQueue(WorkQueue):
    def __init__(self, backend: CondorBackend):
        self._backend = backend

    def count_idle(self) -> int:
        return self._backend.count_idle()

class CondorBackend(ABC):
    @abstractmethod
    def count_idle(self) -> int: ...

class CondorPythonBackend(CondorBackend): ...
class CondorSubprocessBackend(CondorBackend): ...
class CondorRESTAPIBackend(CondorBackend): ...
```

Same for `HPCScheduler` → `SchedulerBackend`:

```python
class SlurmScheduler(HPCScheduler):
    def __init__(self, backend: SchedulerBackend):
        self._backend = backend

class SchedulerBackend(ABC):
    @abstractmethod
    def submit(self, ...) -> str: ...
    @abstractmethod
    def cancel(self, ...) -> None: ...
    @abstractmethod
    def list_active(self, ...) -> list[JobInfo]: ...
    @abstractmethod
    def signal(self, ...) -> None: ...

class SlurmSubprocessBackend(SchedulerBackend): ...
class SlurmRESTAPIBackend(SchedulerBackend): ...
class PBSSubprocessBackend(SchedulerBackend): ...
class LocalSubprocessBackend(SchedulerBackend): ...
```

## Pool Manager Config (`pool-manager.yaml`)

```yaml
poll_interval: 15            # seconds between work queue polls
min_workers: 0
max_workers: 16
batch_size: 1                # queue items per worker
scale_up_cooldown: 30        # seconds between submitting new jobs
scale_down_cooldown: 60      # seconds before starting to drain
drain_timeout: 120           # seconds to wait for graceful shutdown

work_queue:
  backend: condor_python     # condor_python | condor_subprocess | condor_rest
  # Backend-specific options (e.g. schedd_name, constraint, rest_url, token)

scheduler:
  backend: slurm_subprocess  # slurm_subprocess | slurm_rest | pbs_subprocess | local
  worker_script: /path/to/htcondor_worker.sh
  submit_args:
    partition: defq
    account: myproject
    time: "08:00:00"
    nodes: 1
    ntasks: 1
    # any key-value pair becomes --key=value or -K value
```

## File Layout

```
pool-manager/
├── pool_manager/
│   ├── __init__.py
│   ├── __main__.py          # entry point
│   ├── config.py            # config loading (pydantic/dataclass)
│   ├── manager.py           # PoolManager — main loop
│   ├── work_queue/
│   │   ├── __init__.py
│   │   ├── base.py          # WorkQueue ABC, Backend ABC (for Condor)
│   │   ├── condor_python.py
│   │   ├── condor_subprocess.py
│   │   └── condor_rest.py
│   ├── scheduler/
│   │   ├── __init__.py
│   │   ├── base.py          # HPCScheduler ABC, SchedulerBackend ABC
│   │   ├── slurm_subprocess.py
│   │   ├── slurm_rest.py
│   │   ├── pbs_subprocess.py
│   │   └── local_subprocess.py
│   └── scaling.py           # ScalingPolicy (config object + decision logic)
├── pool-manager.yaml        # config file
├── pool-manager.service     # systemd unit
└── PLANS.md
```

## Implementation Plan (TODOs)

- [ ] **1. Set up project structure**
  - Directory layout, `__init__.py` files, `pyproject.toml` / `setup.py`
  - Dev dependencies: `pytest`, `mypy`, `ruff`

- [ ] **2. Implement `WorkQueue` + `CondorBackend` hierarchy**
  - `WorkQueue` ABC with `count_idle()`
  - `CondorBackend` ABC
  - `CondorPythonBackend` — `htcondor` bindings
  - `CondorSubprocessBackend` — `subprocess.run(["condor_q", ...])`
  - `CondorRESTAPIBackend` — HTTP client for HTCondor REST API

- [ ] **3. Implement `HPCScheduler` + `SchedulerBackend` hierarchy**
  - `HPCScheduler` ABC with `submit()`, `cancel()`, `list_active()`, `signal()`
  - `SchedulerBackend` ABC
  - `SlurmSubprocessBackend` — `sbatch`/`scancel`/`squeue`/`scancel --signal`
  - `SlurmRESTAPIBackend` — HTTP client for Slurm REST API
  - `PBSSubprocessBackend` — `qsub`/`qdel`/`qstat`
  - `LocalSubprocessBackend` — local `Popen` for testing

- [ ] **4. Implement config loading (`Config`)**
  - YAML → validated dataclass/Pydantic model
  - Select backend classes by string name

- [ ] **5. Implement `PoolManager` core loop**
  - Poll work queue → compute target pool size
  - Scale up: submit new scheduler jobs
  - Scale down: start graceful drain of excess workers
  - Track job states (submitted, running, draining, exited, lost)
  - Cooldown timers, hysteresis

- [ ] **6. Implement graceful drain protocol**
  - SIGTERM excess/idle workers
  - Wait `drain_timeout`, then `cancel` remaining
  - Abort drain if new work arrives
  - Signal handling on daemon itself (SIGINT/SIGTERM → drain all then exit)

- [ ] **7. Error handling and recovery**
  - Daemon restart: reconcile via `list_active()`
  - Queue connection failures: backoff, preserve pool
  - Scheduler failures: retry, log, don't crash
  - Stale jobs: periodic `list_active()` reconciliation

- [ ] **8. Configuration file and documentation**
  - `pool-manager.yaml` with all options documented
  - Systemd service file
  - README with setup, test procedure, and examples

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge, Histogram, start_http_server

if TYPE_CHECKING:
    from pool_manager.placement import Placement
    from pool_manager.scheduler.base import JobState

IDLE_JOBS = Gauge("pool_manager_idle_jobs", "Number of idle HTCondor jobs")
ACTIVE_WORKERS = Gauge("pool_manager_active_workers", "Number of active workers")
DRAINING_WORKERS = Gauge("pool_manager_draining_workers", "Number of draining workers")
PENDING_WORKERS = Gauge("pool_manager_pending_workers", "Number of pending workers")
RUNNING_WORKERS = Gauge("pool_manager_running_workers", "Number of running workers")
TARGET_WORKERS = Gauge("pool_manager_target_workers", "Target number of workers")

WORKERS_STARTED = Counter("pool_manager_workers_started_total", "Total workers started")
WORKERS_STOPPED = Counter("pool_manager_workers_stopped_total", "Total workers stopped")
SCALE_UP_EVENTS = Counter("pool_manager_scale_up_events_total", "Total scale-up events")
SCALE_DOWN_EVENTS = Counter("pool_manager_scale_down_events_total", "Total scale-down events")

TICK_DURATION = Histogram(
    "pool_manager_tick_duration_seconds",
    "Duration of main loop tick",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

WORKERS_BY_NODE_TYPE = Gauge(
    "pool_manager_workers_by_node_type", "Workers by node type", ["node_type"]
)

TASKS_BY_NODE_TYPE = Gauge(
    "pool_manager_tasks_by_node_type", "Tasks placed by node type", ["node_type"]
)


@dataclass
class MetricsSnapshot:
    idle_jobs: int = 0
    active_workers: int = 0
    draining_workers: int = 0
    pending_workers: int = 0
    running_workers: int = 0
    target_workers: int = 0
    workers_by_type: dict[str, int] | None = None
    tasks_by_type: dict[str, int] | None = None


def start_metrics_server(port: int, addr: str = "") -> None:
    start_http_server(port, addr)


def update_metrics(
    idle_count: int,
    target: int,
    tracked: dict[str, JobState],
    node_assignments: dict[str, str],
    placements: list[Placement] | None = None,
) -> None:
    IDLE_JOBS.set(idle_count)
    TARGET_WORKERS.set(target)

    active = 0
    draining = 0
    pending = 0
    running = 0

    for state in tracked.values():
        state_val = state.value if hasattr(state, "value") else str(state)
        if state_val in ("pending", "PENDING"):
            pending += 1
            active += 1
        elif state_val in ("running", "RUNNING"):
            running += 1
            active += 1
        elif state_val in ("draining", "DRAINING"):
            draining += 1
            active += 1

    ACTIVE_WORKERS.set(active)
    DRAINING_WORKERS.set(draining)
    PENDING_WORKERS.set(pending)
    RUNNING_WORKERS.set(running)

    by_type: dict[str, int] = {}
    for jid, state in tracked.items():
        state_val = state.value if hasattr(state, "value") else str(state)
        if state_val in ("pending", "PENDING", "running", "RUNNING", "draining", "DRAINING"):
            nt = node_assignments.get(jid, "unknown")
            by_type[nt] = by_type.get(nt, 0) + 1

    all_types = set(by_type.keys())
    if placements:
        all_types.update(p.node_config.name for p in placements)

    for nt in all_types:
        WORKERS_BY_NODE_TYPE.labels(node_type=nt).set(by_type.get(nt, 0))

    if placements:
        for p in placements:
            TASKS_BY_NODE_TYPE.labels(node_type=p.node_config.name).set(
                p.count * _tasks_per_node(p)
            )


def _tasks_per_node(placement: Placement) -> int:
    nc = placement.node_config
    return max(1, nc.cpus // 1)


def get_snapshot() -> MetricsSnapshot:
    return MetricsSnapshot(
        idle_jobs=int(IDLE_JOBS._value.get()),
        active_workers=int(ACTIVE_WORKERS._value.get()),
        draining_workers=int(DRAINING_WORKERS._value.get()),
        pending_workers=int(PENDING_WORKERS._value.get()),
        running_workers=int(RUNNING_WORKERS._value.get()),
        target_workers=int(TARGET_WORKERS._value.get()),
    )

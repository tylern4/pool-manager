import pytest

from pool_manager.metrics import (
    ACTIVE_WORKERS,
    DRAINING_WORKERS,
    IDLE_JOBS,
    PENDING_WORKERS,
    RUNNING_WORKERS,
    TARGET_WORKERS,
    MetricsSnapshot,
    get_snapshot,
    update_metrics,
)
from pool_manager.placement import NodeConfig, Placement
from pool_manager.scheduler.base import JobState


@pytest.fixture(autouse=True)
def reset_metrics():
    IDLE_JOBS.set(0)
    ACTIVE_WORKERS.set(0)
    DRAINING_WORKERS.set(0)
    PENDING_WORKERS.set(0)
    RUNNING_WORKERS.set(0)
    TARGET_WORKERS.set(0)
    yield


class TestUpdateMetrics:
    def test_sets_idle_jobs(self):
        update_metrics(
            idle_count=10,
            target=5,
            tracked={},
            node_assignments={},
            placements=None,
        )
        assert int(IDLE_JOBS._value.get()) == 10

    def test_sets_target_workers(self):
        update_metrics(
            idle_count=0,
            target=7,
            tracked={},
            node_assignments={},
            placements=None,
        )
        assert int(TARGET_WORKERS._value.get()) == 7

    def test_counts_worker_states(self):
        tracked = {
            "job1": JobState.PENDING,
            "job2": JobState.RUNNING,
            "job3": JobState.DRAINING,
            "job4": JobState.EXITED,
        }
        update_metrics(
            idle_count=0,
            target=3,
            tracked=tracked,
            node_assignments={},
            placements=None,
        )
        assert int(PENDING_WORKERS._value.get()) == 1
        assert int(RUNNING_WORKERS._value.get()) == 1
        assert int(DRAINING_WORKERS._value.get()) == 1
        assert int(ACTIVE_WORKERS._value.get()) == 3

    def test_counts_by_node_type(self):
        tracked = {
            "job1": JobState.RUNNING,
            "job2": JobState.RUNNING,
            "job3": JobState.PENDING,
        }
        node_assignments = {
            "job1": "small",
            "job2": "large",
            "job3": "small",
        }
        update_metrics(
            idle_count=0,
            target=3,
            tracked=tracked,
            node_assignments=node_assignments,
            placements=None,
        )

    def test_with_placements(self):
        nc = NodeConfig(name="small", cpus=4, memory_mb=8000, gpus=0)
        placements = [Placement(node_config=nc, count=2)]
        update_metrics(
            idle_count=5,
            target=2,
            tracked={},
            node_assignments={},
            placements=placements,
        )


class TestGetSnapshot:
    def test_returns_snapshot(self):
        IDLE_JOBS.set(15)
        ACTIVE_WORKERS.set(3)
        DRAINING_WORKERS.set(1)
        PENDING_WORKERS.set(0)
        RUNNING_WORKERS.set(2)
        TARGET_WORKERS.set(5)

        snapshot = get_snapshot()
        assert isinstance(snapshot, MetricsSnapshot)
        assert snapshot.idle_jobs == 15
        assert snapshot.active_workers == 3
        assert snapshot.draining_workers == 1
        assert snapshot.pending_workers == 0
        assert snapshot.running_workers == 2
        assert snapshot.target_workers == 5

    def test_default_snapshot(self):
        snapshot = MetricsSnapshot()
        assert snapshot.idle_jobs == 0
        assert snapshot.active_workers == 0
        assert snapshot.draining_workers == 0
        assert snapshot.pending_workers == 0
        assert snapshot.running_workers == 0
        assert snapshot.target_workers == 0
        assert snapshot.workers_by_type is None
        assert snapshot.tasks_by_type is None

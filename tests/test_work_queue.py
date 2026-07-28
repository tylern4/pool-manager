from loguru import logger

from pool_manager.placement import TaskResources
from pool_manager.work_queue.base import CondorBackend, WorkerSlotStatus, WorkQueue
from pool_manager.work_queue.condor import CondorWorkQueue


class _FakeCondorBackend(CondorBackend):
    def __init__(self, count: int = 5):
        self._count = count
        self.called_with_constraint = ""

    def count_idle(self, constraint: str = "") -> int:
        self.called_with_constraint = constraint
        return self._count

    def list_idle(self, constraint: str = "") -> list[TaskResources]:
        self.called_with_constraint = constraint
        return [TaskResources() for _ in range(self._count)]

    def list_worker_status(self, constraint: str = "") -> list[WorkerSlotStatus]:
        self.called_with_constraint = constraint
        return []

    def name(self) -> str:
        return "fake_backend"


class TestCondorWorkQueue:
    def test_implements_work_queue(self):
        backend = _FakeCondorBackend()
        wq = CondorWorkQueue(backend=backend)
        assert isinstance(wq, WorkQueue)

    def test_count_idle_delegates(self):
        backend = _FakeCondorBackend(count=7)
        wq = CondorWorkQueue(backend=backend)
        assert wq.count_idle() == 7

    def test_passes_constraint(self):
        backend = _FakeCondorBackend()
        wq = CondorWorkQueue(backend=backend, constraint="JobStatus == 5")
        wq.count_idle()
        assert backend.called_with_constraint == "JobStatus == 5"

    def test_name_delegates(self):
        backend = _FakeCondorBackend()
        wq = CondorWorkQueue(backend=backend)
        assert wq.name() == "fake_backend"

    def test_default_constraint(self):
        backend = _FakeCondorBackend()
        wq = CondorWorkQueue(backend=backend)
        assert wq.count_idle() == 5
        assert backend.called_with_constraint == ""

    def test_list_idle_returns_task_list(self):
        backend = _FakeCondorBackend(count=3)
        wq = CondorWorkQueue(backend=backend)
        tasks = wq.list_idle()
        assert len(tasks) == 3
        assert all(t.cpus == 1.0 for t in tasks)

    def test_list_idle_empty(self):
        backend = _FakeCondorBackend(count=0)
        wq = CondorWorkQueue(backend=backend)
        tasks = wq.list_idle()
        assert tasks == []

    def test_list_idle_passes_constraint(self):
        backend = _FakeCondorBackend()
        wq = CondorWorkQueue(backend=backend, constraint="JobStatus == 5")
        wq.list_idle()
        assert backend.called_with_constraint == "JobStatus == 5"


class TestCondorWorkQueueTraceLogging:
    def test_list_idle_trace_logging(self):
        backend = _FakeCondorBackend(count=2)
        wq = CondorWorkQueue(backend=backend)
        records = []
        handler_id = logger.add(lambda m: records.append(m), level="TRACE")
        try:
            tasks = wq.list_idle()
        finally:
            logger.remove(handler_id)
        assert len(tasks) == 2
        assert any("Task: cpus=" in m for m in records)

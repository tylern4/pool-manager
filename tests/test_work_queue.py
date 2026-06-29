from pool_manager.placement import TaskResources
from pool_manager.work_queue.base import CondorBackend, WorkQueue
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

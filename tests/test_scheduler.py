from pool_manager.scheduler.base import JobInfo, JobState, SchedulerBackend
from pool_manager.scheduler.wrapper import SchedulerWrapper


class _FakeSchedulerBackend(SchedulerBackend):
    def __init__(self):
        self.submitted: list[str] = []
        self.cancelled: list[str] = []
        self.signalled: list[tuple[str, str]] = []

    def submit(self, script_path: str, submit_args: dict[str, str]) -> str:
        job_id = f"job_{len(self.submitted) + 1}"
        self.submitted.append(job_id)
        return job_id

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)

    def list_active(self) -> list[JobInfo]:
        return [JobInfo(job_id=j, state=JobState.RUNNING) for j in self.submitted]

    def signal(self, job_id: str, sig: str) -> None:
        self.signalled.append((job_id, sig))

    def name(self) -> str:
        return "fake_backend"


class TestSchedulerWrapper:
    def test_implements_hpc_scheduler(self):
        backend = _FakeSchedulerBackend()
        sched = SchedulerWrapper(backend=backend)
        assert isinstance(sched, SchedulerBackend)

    def test_submit_delegates(self):
        backend = _FakeSchedulerBackend()
        sched = SchedulerWrapper(backend=backend)
        job_id = sched.submit("/path/to/script.sh", {"time": "01:00:00"})
        assert job_id == "job_1"
        assert backend.submitted == ["job_1"]

    def test_cancel_delegates(self):
        backend = _FakeSchedulerBackend()
        sched = SchedulerWrapper(backend=backend)
        sched.cancel("job_1")
        assert backend.cancelled == ["job_1"]

    def test_list_active_delegates(self):
        backend = _FakeSchedulerBackend()
        # Pre-populate a submitted job
        backend.submit("/x.sh", {})
        sched = SchedulerWrapper(backend=backend)
        jobs = sched.list_active()
        assert len(jobs) == 1
        assert jobs[0].job_id == "job_1"
        assert jobs[0].state == JobState.RUNNING

    def test_signal_delegates(self):
        backend = _FakeSchedulerBackend()
        sched = SchedulerWrapper(backend=backend)
        sched.signal("job_1", "SIGTERM")
        assert backend.signalled == [("job_1", "SIGTERM")]

    def test_name_delegates(self):
        backend = _FakeSchedulerBackend()
        sched = SchedulerWrapper(backend=backend)
        assert sched.name() == "fake_backend"

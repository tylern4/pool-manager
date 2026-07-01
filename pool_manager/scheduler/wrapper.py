import logging
from pathlib import Path

from pool_manager.scheduler.base import HPCScheduler, JobInfo, SchedulerBackend

log = logging.getLogger("pool_manager.scheduler.wrapper")


class SchedulerWrapper(HPCScheduler):
    def __init__(self, backend: SchedulerBackend):
        self._backend = backend

    def submit(self, script_path: Path, submit_args: dict[str, str]) -> str:
        log.debug("Submitting via %s", self._backend.name())
        return self._backend.submit(script_path, submit_args)

    def cancel(self, job_id: str) -> None:
        self._backend.cancel(job_id)

    def list_active(self) -> list[JobInfo]:
        return self._backend.list_active()

    def signal(self, job_id: str, sig: str) -> None:
        self._backend.signal(job_id, sig)

    def name(self) -> str:
        return self._backend.name()

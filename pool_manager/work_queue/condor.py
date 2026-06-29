import logging

from pool_manager.log import TRACE
from pool_manager.placement import TaskResources
from pool_manager.work_queue.base import CondorBackend, WorkQueue

log = logging.getLogger("pool_manager.work_queue.condor")


class CondorWorkQueue(WorkQueue):
    def __init__(self, backend: CondorBackend, constraint: str = "JobStatus == 1"):
        self._backend = backend
        self._constraint = constraint

    def count_idle(self) -> int:
        count = self._backend.count_idle(constraint=self._constraint)
        log.debug("Idle jobs count=%d via %s", count, self._backend.name())
        return count

    def list_idle(self) -> list[TaskResources]:
        tasks = self._backend.list_idle(constraint=self._constraint)
        if tasks:
            total_cpus = sum(t.cpus for t in tasks)
            total_mem = sum(t.memory_mb for t in tasks)
            total_gpus = sum(t.gpus for t in tasks)
            log.info(
                "Idle tasks: count=%d cpus=%.1f mem=%dMB gpus=%d",
                len(tasks),
                total_cpus,
                total_mem,
                total_gpus,
            )
            if log.isEnabledFor(TRACE):
                for task in tasks:
                    log.log(
                        TRACE,
                        "Task: cpus=%s mem=%dMB gpus=%d",
                        task.cpus,
                        task.memory_mb,
                        task.gpus,
                    )
        else:
            log.debug("No idle tasks via %s", self._backend.name())
        return tasks

    def name(self) -> str:
        return self._backend.name()

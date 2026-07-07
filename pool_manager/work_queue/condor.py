from loguru import logger

from pool_manager.placement import TaskResources
from pool_manager.work_queue.base import CondorBackend, WorkQueue


class CondorWorkQueue(WorkQueue):
    def __init__(self, backend: CondorBackend, constraint: str = ""):
        self._backend = backend
        self._constraint = constraint

    def count_idle(self) -> int:
        count = self._backend.count_idle(constraint=self._constraint)
        logger.debug("Idle jobs count={} via {}", count, self._backend.name())
        return count

    def list_idle(self) -> list[TaskResources]:
        tasks = self._backend.list_idle(constraint=self._constraint)
        if tasks:
            total_cpus = sum(t.cpus for t in tasks)
            total_mem = sum(t.memory_mb for t in tasks)
            total_gpus = sum(t.gpus for t in tasks)
            logger.info(
                "Idle tasks: count={} cpus={} mem={}MB gpus={}",
                len(tasks),
                total_cpus,
                total_mem,
                total_gpus,
            )
            for task in tasks:
                logger.trace(
                    "Task: cpus={} mem={}MB gpus={}",
                    task.cpus,
                    task.memory_mb,
                    task.gpus,
                )
        else:
            logger.debug("No idle tasks via {}", self._backend.name())
        return tasks

    def name(self) -> str:
        return self._backend.name()

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pool_manager.placement import TaskResources


@dataclass
class WorkerSlotStatus:
    """Status of a single worker slot from condor_status."""

    slot_name: str
    owner_job_id: str | None = None
    state: str = "idle"


class CondorBackend(ABC):
    @abstractmethod
    def count_idle(self, constraint: str = "") -> int: ...

    @abstractmethod
    def list_idle(self, constraint: str = "") -> list[TaskResources]: ...

    @abstractmethod
    def list_worker_status(self, constraint: str = "") -> list[WorkerSlotStatus]: ...

    @abstractmethod
    def name(self) -> str: ...


class WorkQueue(ABC):
    @abstractmethod
    def count_idle(self) -> int: ...

    @abstractmethod
    def list_idle(self) -> list[TaskResources]: ...

    @abstractmethod
    def list_worker_status(self) -> list[WorkerSlotStatus]: ...

    @abstractmethod
    def name(self) -> str: ...

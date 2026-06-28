from abc import ABC, abstractmethod

from pool_manager.placement import TaskResources


class CondorBackend(ABC):
    @abstractmethod
    def count_idle(self, constraint: str = "") -> int: ...

    @abstractmethod
    def list_idle(self, constraint: str = "JobStatus == 1") -> list[TaskResources]: ...

    @abstractmethod
    def name(self) -> str: ...


class WorkQueue(ABC):
    @abstractmethod
    def count_idle(self) -> int: ...

    @abstractmethod
    def list_idle(self) -> list[TaskResources]: ...

    @abstractmethod
    def name(self) -> str: ...

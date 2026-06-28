from abc import ABC, abstractmethod


class CondorBackend(ABC):
    @abstractmethod
    def count_idle(self, constraint: str = "") -> int: ...

    @abstractmethod
    def name(self) -> str: ...


class WorkQueue(ABC):
    @abstractmethod
    def count_idle(self) -> int: ...

    @abstractmethod
    def name(self) -> str: ...

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


def _test_job_id() -> str:
    return f"test_{int(time.time())}_{threading.get_ident()}"


class JobState(Enum):
    RUNNING = "running"
    PENDING = "pending"
    DRAINING = "draining"
    EXITED = "exited"
    UNKNOWN = "unknown"


@dataclass
class JobInfo:
    job_id: str
    state: JobState
    job_name: str = ""


def parse_config_name(job_name: str, prefix: str = "htcondor_worker_") -> str:
    if job_name.startswith(prefix):
        return job_name[len(prefix) :]
    return "default"


@dataclass
class NodeConfig:
    name: str
    cpus: int = 1
    memory_mb: int = 1024
    gpus: int = 0
    submit_args: dict[str, str] | None = None


class SchedulerBackend(ABC):
    @abstractmethod
    def submit(self, script_path: Path, submit_args: dict[str, str]) -> str: ...

    @abstractmethod
    def cancel(self, job_id: str) -> None: ...

    @abstractmethod
    def list_active(self) -> list[JobInfo]: ...

    @abstractmethod
    def signal(self, job_id: str, sig: str) -> None: ...

    @abstractmethod
    def name(self) -> str: ...

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

from pool_manager.scaling import ScalingPolicy


@dataclass
class WorkQueueConfig:
    backend: str = "condor_python"
    schedd_name: str = ""
    constraint: str = "JobStatus == 1"
    rest_url: str = ""


@dataclass
class SchedulerConfig:
    backend: str = "slurm_subprocess"
    worker_script: str = ""
    submit_args: dict[str, Any] = field(default_factory=dict)
    rest_url: str = ""
    rest_token: str = ""


@dataclass
class Config:
    poll_interval: float = 15.0
    log_level: str = "INFO"
    work_queue: WorkQueueConfig = field(default_factory=WorkQueueConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    scaling: ScalingPolicy = field(default_factory=ScalingPolicy)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        if not os.path.exists(path):
            return cls()

        with open(path) as f:
            raw = yaml.safe_load(f)

        if not raw:
            return cls()

        wk = raw.get("work_queue", {})
        sch = raw.get("scheduler", {})
        sc = raw.get("scaling", {})

        return cls(
            poll_interval=raw.get("poll_interval", 15.0),
            log_level=raw.get("log_level", "INFO"),
            work_queue=WorkQueueConfig(
                backend=wk.get("backend", "condor_python"),
                schedd_name=wk.get("schedd_name", ""),
                constraint=wk.get("constraint", "JobStatus == 1"),
                rest_url=wk.get("rest_url", ""),
            ),
            scheduler=SchedulerConfig(
                backend=sch.get("backend", "slurm_subprocess"),
                worker_script=sch.get("worker_script", ""),
                submit_args=sch.get("submit_args", {}),
                rest_url=sch.get("rest_url", ""),
                rest_token=sch.get("rest_token", ""),
            ),
            scaling=ScalingPolicy(
                min_workers=sc.get("min_workers", 0),
                max_workers=sc.get("max_workers", 16),
                batch_size=sc.get("batch_size", 1),
                scale_up_cooldown=sc.get("scale_up_cooldown", 30.0),
                scale_down_cooldown=sc.get("scale_down_cooldown", 60.0),
                drain_timeout=sc.get("drain_timeout", 120.0),
            ),
        )

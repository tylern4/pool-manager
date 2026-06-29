from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pool_manager.placement import NodeConfig
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
    node_configs: list[NodeConfig] = field(default_factory=list)
    rest_url: str = ""
    rest_token: str = ""
    machine: str = ""
    sfapi_client_id: str = ""
    sfapi_client_secret: str = ""
    sfapi_key_path: str = ""
    sfapi_user: str = ""
    user: str = ""
    job_name_prefix: str = "htcondor_worker_"
    test_mode: bool = False


@dataclass
class Config:
    poll_interval: float = 15.0
    log_level: str = "INFO"
    log_mode: str = "stdout"
    log_file: str = ""
    work_queue: WorkQueueConfig = field(default_factory=WorkQueueConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    scaling: ScalingPolicy = field(default_factory=ScalingPolicy)

    @classmethod
    def from_file(cls, path: Path | str) -> "Config":
        path = Path(path)
        if not path.exists():
            return cls()

        with path.open("r") as f:
            raw = yaml.safe_load(f)

        if not raw:
            return cls()

        wk = raw.get("work_queue", {})
        sch = raw.get("scheduler", {})
        sc = raw.get("scaling", {})

        node_configs_raw: Any = sch.get("node_configs", [])
        node_configs = [
            NodeConfig(
                name=n.get("name", ""),
                cpus=int(n.get("cpus", 1)),
                memory_mb=int(n.get("memory_mb", n.get("memory_gb", 1.024) * 1000)),
                gpus=int(n.get("gpus", 0)),
                submit_args=n.get("submit_args"),
            )
            for n in node_configs_raw
        ]

        return cls(
            poll_interval=raw.get("poll_interval", 15.0),
            log_level=raw.get("log_level", "INFO"),
            log_mode=raw.get("log_mode", "stdout"),
            log_file=raw.get("log_file", ""),
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
                node_configs=node_configs,
                rest_url=sch.get("rest_url", ""),
                rest_token=sch.get("rest_token", ""),
                machine=sch.get("machine", ""),
                sfapi_client_id=sch.get("sfapi_client_id", ""),
                sfapi_client_secret=sch.get("sfapi_client_secret", ""),
                sfapi_key_path=sch.get("sfapi_key_path", ""),
                sfapi_user=sch.get("sfapi_user", ""),
                user=sch.get("user", ""),
                job_name_prefix=sch.get("job_name_prefix", "htcondor_worker_"),
                test_mode=sch.get("test_mode", False),
            ),
            scaling=ScalingPolicy(
                min_workers=sc.get("min_workers", 0),
                max_workers=sc.get("max_workers", 16),
                batch_size=sc.get("batch_size", 1),
                scale_up_cooldown=sc.get("scale_up_cooldown", 30.0),
                scale_down_cooldown=sc.get("scale_down_cooldown", 60.0),
                drain_timeout=sc.get("drain_timeout", 120.0),
                drain_on_stop=sc.get("drain_on_stop", False),
            ),
        )

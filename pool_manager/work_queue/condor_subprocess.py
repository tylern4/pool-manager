import json
import shlex
import subprocess

from loguru import logger

from pool_manager.placement import TaskResources
from pool_manager.work_queue.base import CondorBackend, WorkerSlotStatus


class CondorSubprocessBackend(CondorBackend):
    def __init__(self, schedd_name: str = ""):
        self._schedd_name = schedd_name

    def _query_json(self, constraint: str) -> list[dict]:
        cmd = ["condor_q", "-json"]
        if self._schedd_name:
            cmd.extend(["-pool", self._schedd_name])
        if constraint:
            cmd.extend(["-constraint", constraint])

        logger.debug("Running condor_q command: {}", shlex.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        logger.trace("condor_q stdout (first 2000): {}", result.stdout[:2000])
        logger.trace("condor_q stderr: {}", result.stderr.strip())

        if result.returncode != 0:
            logger.warning("condor_q exited {}: {}", result.returncode, result.stderr.strip())
            return []

        if not result.stdout.strip():
            return []

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse condor_q JSON output: {}", e)
            return []

    def count_idle(self, constraint: str = "") -> int:
        jobs = self._query_json(constraint)
        return len(jobs)

    def list_idle(self, constraint: str = "") -> list[TaskResources]:
        jobs = self._query_json(constraint)
        tasks = []
        for job in jobs:
            job = {k.lower(): v for k, v in job.items()}
            tasks.append(
                TaskResources(
                    cpus=float(job.get("requestcpus", 1)),
                    memory_mb=int(job.get("requestmemory", 1024)),
                    gpus=int(job.get("requestgpus", 0)),
                    runtime_minutes=float(job.get("runtimeminutes", 0)),
                )
            )
        logger.debug("Parsed {} idle job(s) with task resources", len(tasks))
        return tasks

    def list_worker_status(self, constraint: str = "") -> list[WorkerSlotStatus]:
        cmd = ["condor_status", "-json"]
        if self._schedd_name:
            cmd.extend(["-pool", self._schedd_name])
        if constraint:
            cmd.extend(["-constraint", constraint])

        logger.debug("Running condor_status command: {}", shlex.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        logger.trace("condor_status stdout (first 2000): {}", result.stdout[:2000])
        logger.trace("condor_status stderr: {}", result.stderr.strip())

        if result.returncode != 0:
            logger.warning("condor_status exited {}: {}", result.returncode, result.stderr.strip())
            return []

        if not result.stdout.strip():
            return []

        try:
            slots = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse condor_status JSON output: {}", e)
            return []

        statuses = []
        for slot in slots:
            slot = {k.lower(): v for k, v in slot.items()}
            statuses.append(
                WorkerSlotStatus(
                    slot_name=slot.get("name", ""),
                    owner_job_id=slot.get("owner", None),
                    state=slot.get("state", "idle"),
                )
            )
        logger.debug("Parsed {} worker slot(s) from condor_status", len(statuses))
        return statuses

    def name(self) -> str:
        base = "condor_subprocess"
        return f"{base}(schedd={self._schedd_name})" if self._schedd_name else base

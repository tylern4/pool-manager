import json
import logging
import shlex
import subprocess

from pool_manager.log import TRACE
from pool_manager.placement import TaskResources
from pool_manager.work_queue.base import CondorBackend

log = logging.getLogger("pool_manager.work_queue.condor_subprocess")


class CondorSubprocessBackend(CondorBackend):
    def __init__(self, schedd_name: str = ""):
        self._schedd_name = schedd_name

    def _query_json(self, constraint: str) -> list[dict]:
        cmd = ["condor_q", "-json"]
        if self._schedd_name:
            cmd.extend(["-pool", self._schedd_name])
        if constraint:
            cmd.extend(["-constraint", constraint])

        log.debug("Running condor_q command: %s", shlex.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        log.log(TRACE, "condor_q stdout (first 2000): %s", result.stdout[:2000])
        log.log(TRACE, "condor_q stderr: %s", result.stderr.strip())

        if result.returncode != 0:
            log.warning("condor_q exited %d: %s", result.returncode, result.stderr.strip())
            return []

        if not result.stdout.strip():
            return []

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            log.warning("Failed to parse condor_q JSON output: %s", e)
            return []

    def count_idle(self, constraint: str = "") -> int:
        jobs = self._query_json(constraint)
        return len(jobs)

    def list_idle(self, constraint: str = "") -> list[TaskResources]:
        jobs = self._query_json(constraint)
        tasks = []
        for job in jobs:
            tasks.append(
                TaskResources(
                    cpus=float(job.get("RequestCpus", 1) or 1),
                    memory_mb=int(job.get("RequestMemory", 1024) or 1024),
                    gpus=int(job.get("RequestGpus", 0) or 0),
                )
            )
        log.debug("Parsed %d idle job(s) with task resources", len(tasks))
        return tasks

    def name(self) -> str:
        base = "condor_subprocess"
        return f"{base}(schedd={self._schedd_name})" if self._schedd_name else base

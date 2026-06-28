import logging
import shlex
import subprocess

from pool_manager.log import TRACE
from pool_manager.work_queue.base import CondorBackend

log = logging.getLogger("pool_manager.work_queue.condor_subprocess")


class CondorSubprocessBackend(CondorBackend):
    def __init__(self, schedd_name: str = ""):
        self._schedd_name = schedd_name

    def count_idle(self, constraint: str = "JobStatus == 1") -> int:
        cmd = ["condor_q"]
        if self._schedd_name:
            cmd.extend(["-pool", self._schedd_name])
        cmd.extend(["-constraint", constraint, "-total"])

        log.debug("Running condor_q command: %s", shlex.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        log.log(TRACE, "condor_q stdout: %s", result.stdout.strip())
        log.log(TRACE, "condor_q stderr: %s", result.stderr.strip())

        if result.returncode != 0:
            log.warning("condor_q exited %d: %s", result.returncode, result.stderr.strip())
            return 0

        import re

        for line in result.stdout.strip().splitlines():
            if line.strip().startswith("0 jobs"):
                return 0
            m = re.search(r"(\d+)\s+idle", line)
            if m:
                count = int(m.group(1))
                log.debug("Parsed idle count: %d", count)
                return count

        log.warning("Could not parse idle count from condor_q output")
        return 0

    def name(self) -> str:
        base = "condor_subprocess"
        return f"{base}(schedd={self._schedd_name})" if self._schedd_name else base

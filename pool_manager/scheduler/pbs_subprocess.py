import logging
import shlex
import subprocess

from pool_manager.log import TRACE
from pool_manager.scheduler.base import JobInfo, JobState, SchedulerBackend

log = logging.getLogger("pool_manager.scheduler.pbs_subprocess")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    log.debug("Running: %s", shlex.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    log.log(TRACE, "stdout: %s", result.stdout.strip())
    log.log(TRACE, "stderr: %s", result.stderr.strip())
    return result


class PBSSubprocessBackend(SchedulerBackend):
    def submit(self, script_path: str, submit_args: dict[str, str]) -> str:
        cmd = ["qsub"]
        for key, val in submit_args.items():
            key = key.replace("_", "-")
            if val == "":
                cmd.append(f"-{key}")
            else:
                cmd.extend([f"-{key}", str(val)])
        cmd.append(script_path)

        result = _run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"qsub failed (exit {result.returncode}): {result.stderr.strip()}")

        job_id = result.stdout.strip()
        log.debug("Submitted PBS job %s (script=%s)", job_id, script_path)
        return job_id

    def cancel(self, job_id: str) -> None:
        cmd = ["qdel", job_id]
        result = _run(cmd)
        if result.returncode != 0:
            log.warning(
                "qdel %s failed (exit %d): %s", job_id, result.returncode, result.stderr.strip()
            )
        else:
            log.debug("Cancelled PBS job %s", job_id)

    def list_active(self) -> list[JobInfo]:
        cmd = ["qstat", "-x", "-u", self._user()]
        result = _run(cmd)
        if result.returncode != 0:
            log.warning("qstat failed (exit %d): %s", result.returncode, result.stderr.strip())
            return []

        jobs: list[JobInfo] = []
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            job_id = parts[0]
            state_str = parts[4]
            state = _parse_pbs_state(state_str)
            jobs.append(JobInfo(job_id=job_id, state=state))

        log.debug("Active PBS jobs: %s", [j.job_id for j in jobs])
        return jobs

    def signal(self, job_id: str, sig: str) -> None:
        cmd = ["qsig", "-s", sig, job_id]
        result = _run(cmd)
        if result.returncode != 0:
            log.warning(
                "qsig %s %s failed (exit %d): %s",
                sig,
                job_id,
                result.returncode,
                result.stderr.strip(),
            )
        else:
            log.debug("Sent signal %s to PBS job %s", sig, job_id)

    def name(self) -> str:
        return "pbs_subprocess"

    @staticmethod
    def _user() -> str:
        import os

        return os.environ.get("USER", "")


def _parse_pbs_state(raw: str) -> JobState:
    mapping = {
        "Q": JobState.PENDING,
        "R": JobState.RUNNING,
        "H": JobState.PENDING,
        "W": JobState.PENDING,
        "S": JobState.RUNNING,
        "E": JobState.RUNNING,
    }
    return mapping.get(raw.upper(), JobState.UNKNOWN)

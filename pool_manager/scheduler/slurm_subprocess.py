import logging
import shlex
import subprocess

from pool_manager.log import TRACE
from pool_manager.scheduler.base import JobInfo, JobState, SchedulerBackend

log = logging.getLogger("pool_manager.scheduler.slurm_subprocess")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log.debug("Running: %s", shlex.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, **kwargs)
    log.log(TRACE, "stdout: %s", result.stdout.strip())
    log.log(TRACE, "stderr: %s", result.stderr.strip())
    return result


class SlurmSubprocessBackend(SchedulerBackend):
    def submit(self, script_path: str, submit_args: dict[str, str]) -> str:
        cmd = ["sbatch", "--parsable"]
        for key, val in submit_args.items():
            key = key.replace("_", "-")
            if val == "":
                cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(val)])
        cmd.append(script_path)

        result = _run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed (exit {result.returncode}): {result.stderr.strip()}")

        job_id = result.stdout.strip().split(";")[0]
        log.debug("Submitted Slurm job %s (script=%s, args=%s)", job_id, script_path, submit_args)
        return job_id

    def cancel(self, job_id: str) -> None:
        cmd = ["scancel", job_id]
        result = _run(cmd)
        if result.returncode != 0:
            log.warning(
                "scancel %s failed (exit %d): %s", job_id, result.returncode, result.stderr.strip()
            )
        else:
            log.debug("Cancelled Slurm job %s", job_id)

    def list_active(self) -> list[JobInfo]:
        cmd = [
            "squeue",
            "--noheader",
            "--format=%i,%T",
            "--states=PD,R,CF",
        ]
        result = _run(cmd)
        if result.returncode != 0:
            log.warning("squeue failed (exit %d): %s", result.returncode, result.stderr.strip())
            return []

        jobs: list[JobInfo] = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            job_id, state_str = parts
            state = _parse_slurm_state(state_str.strip())
            jobs.append(JobInfo(job_id=job_id, state=state))

        log.debug("Active Slurm jobs: %s", [j.job_id for j in jobs])
        return jobs

    def signal(self, job_id: str, sig: str) -> None:
        cmd = ["scancel", "--signal", sig, job_id]
        result = _run(cmd)
        if result.returncode != 0:
            log.warning(
                "scancel --signal %s %s failed (exit %d): %s",
                sig,
                job_id,
                result.returncode,
                result.stderr.strip(),
            )
        else:
            log.debug("Sent signal %s to Slurm job %s", sig, job_id)

    def name(self) -> str:
        return "slurm_subprocess"


def _parse_slurm_state(raw: str) -> JobState:
    mapping = {
        "PD": JobState.PENDING,
        "R": JobState.RUNNING,
        "CF": JobState.PENDING,
        "CG": JobState.RUNNING,
    }
    return mapping.get(raw, JobState.UNKNOWN)

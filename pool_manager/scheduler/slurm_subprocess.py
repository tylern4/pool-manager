import os
import shlex
import subprocess

from loguru import logger

from pool_manager.scheduler.base import JobInfo, JobState, SchedulerBackend, _test_job_id


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    logger.debug("Running: {}", shlex.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, **kwargs)
    logger.trace("stdout: {}", result.stdout.strip())
    logger.trace("stderr: {}", result.stderr.strip())
    return result


class SlurmSubprocessBackend(SchedulerBackend):
    def __init__(
        self, job_name_prefix: str = "htcondor_worker_", test_mode: bool = False, user: str = ""
    ):
        self._job_name_prefix = job_name_prefix
        self._test_mode = test_mode
        self.user = user

    def submit(self, script_path: str, submit_args: dict[str, str]) -> str:
        cmd = ["sbatch", "--parsable"]
        for key, val in submit_args.items():
            key = key.replace("_", "-")
            if val == "":
                cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(val)])
        cmd.append(script_path)

        if self._test_mode:
            job_id = _test_job_id()
            logger.info("[TEST] Would run: {}", shlex.join(cmd))
            logger.info(
                "[TEST] Would submit job {} (script={}, args={})",
                job_id,
                script_path,
                submit_args,
            )
            return job_id

        result = _run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed (exit {result.returncode}): {result.stderr.strip()}")

        job_id = result.stdout.strip().split(";")[0]
        logger.debug(
            "Submitted Slurm job {} (script={}, args={})", job_id, script_path, submit_args
        )
        return job_id

    def cancel(self, job_id: str) -> None:
        cmd = ["scancel", job_id]
        if self._test_mode:
            logger.info("[TEST] Would run: {}", shlex.join(cmd))
            return

        result = _run(cmd)
        if result.returncode != 0:
            logger.warning(
                "scancel {} failed (exit {}): {}", job_id, result.returncode, result.stderr.strip()
            )
        else:
            logger.debug("Cancelled Slurm job {}", job_id)

    def list_active(self) -> list[JobInfo]:
        cmd = [
            "sacct",
            "--noheader",
            "--parsable2",
            "--format=JobID,JobName,State",
            "--user",
            self._user,
        ]
        result = _run(cmd)
        if result.returncode != 0:
            logger.warning("sacct failed (exit {}): {}", result.returncode, result.stderr.strip())
            return []

        jobs: list[JobInfo] = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) != 3:
                continue
            job_id, job_name, state_str = parts
            if "." in job_id:
                continue
            if self._job_name_prefix and not job_name.startswith(self._job_name_prefix):
                continue
            state = _parse_slurm_state(state_str.strip())
            jobs.append(JobInfo(job_id=job_id, state=state, job_name=job_name))

        logger.debug("Active Slurm jobs: {}", [j.job_id for j in jobs])
        return jobs

    @property
    def _user(self) -> str:
        return self.user or os.environ.get("USER", "")

    def signal(self, job_id: str, sig: str) -> None:
        cmd = ["scancel", "--signal", sig, job_id]
        if self._test_mode:
            logger.info("[TEST] Would run: {}", shlex.join(cmd))
            return

        result = _run(cmd)
        if result.returncode != 0:
            logger.warning(
                "scancel --signal {} {} failed (exit {}): {}",
                sig,
                job_id,
                result.returncode,
                result.stderr.strip(),
            )
        else:
            logger.debug("Sent signal {} to Slurm job {}", sig, job_id)

    def name(self) -> str:
        return "slurm_subprocess"


def _parse_slurm_state(raw: str) -> JobState:
    mapping = {
        "PD": JobState.PENDING,
        "PENDING": JobState.PENDING,
        "CF": JobState.PENDING,
        "CONFIGURING": JobState.PENDING,
        "R": JobState.RUNNING,
        "RUNNING": JobState.RUNNING,
        "CG": JobState.RUNNING,
        "COMPLETING": JobState.RUNNING,
    }
    return mapping.get(raw.strip(), JobState.UNKNOWN)

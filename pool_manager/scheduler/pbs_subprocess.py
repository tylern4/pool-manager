import os
import shlex
import subprocess

from loguru import logger

from pool_manager.scheduler.base import JobInfo, JobState, SchedulerBackend, _test_job_id


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    logger.debug("Running: {}", shlex.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    logger.trace("stdout: {}", result.stdout.strip())
    logger.trace("stderr: {}", result.stderr.strip())
    return result


class PBSSubprocessBackend(SchedulerBackend):
    def __init__(
        self, job_name_prefix: str = "htcondor_worker_", test_mode: bool = False, user: str = ""
    ):
        self._job_name_prefix = job_name_prefix
        self._test_mode = test_mode
        self.user = user

    def submit(self, script_path: str, submit_args: dict[str, str]) -> str:
        cmd = ["qsub"]
        for key, val in submit_args.items():
            key = key.replace("_", "-")
            if val == "":
                cmd.append(f"-{key}")
            else:
                cmd.extend([f"-{key}", str(val)])
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
            raise RuntimeError(f"qsub failed (exit {result.returncode}): {result.stderr.strip()}")

        job_id = result.stdout.strip()
        logger.debug("Submitted PBS job {} (script={})", job_id, script_path)
        return job_id

    def cancel(self, job_id: str) -> None:
        cmd = ["qdel", job_id]
        if self._test_mode:
            logger.info("[TEST] Would run: {}", shlex.join(cmd))
            return

        result = _run(cmd)
        if result.returncode != 0:
            logger.warning(
                "qdel {} failed (exit {}): {}", job_id, result.returncode, result.stderr.strip()
            )
        else:
            logger.debug("Cancelled PBS job {}", job_id)

    def list_active(self) -> list[JobInfo]:
        cmd = ["qstat", "-x", "-u", self._user]
        result = _run(cmd)
        if result.returncode != 0:
            logger.warning("qstat failed (exit {}): {}", result.returncode, result.stderr.strip())
            return []

        jobs: list[JobInfo] = []
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            job_name = _extract_xml_value(parts[1])
            if self._job_name_prefix and not job_name.startswith(self._job_name_prefix):
                continue
            job_id = parts[0]
            state_str = parts[4]
            state = _parse_pbs_state(state_str)
            jobs.append(JobInfo(job_id=job_id, state=state, job_name=job_name))

        logger.debug("Active PBS jobs: {}", [j.job_id for j in jobs])
        return jobs

    def signal(self, job_id: str, sig: str) -> None:
        cmd = ["qsig", "-s", sig, job_id]
        if self._test_mode:
            logger.info("[TEST] Would run: {}", shlex.join(cmd))
            return

        result = _run(cmd)
        if result.returncode != 0:
            logger.warning(
                "qsig {} {} failed (exit {}): {}",
                sig,
                job_id,
                result.returncode,
                result.stderr.strip(),
            )
        else:
            logger.debug("Sent signal {} to PBS job {}", sig, job_id)

    def name(self) -> str:
        return "pbs_subprocess"

    @property
    def _user(self) -> str:
        return self.user or os.environ.get("USER", "")


def _extract_xml_value(tag: str) -> str:
    """Extract the text content from an XML tag like '<name>value</name>'."""
    if ">" not in tag or "<" not in tag:
        return tag.strip()
    start = tag.index(">") + 1
    end = tag.rindex("<")
    return tag[start:end] if start < end else tag.strip()


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

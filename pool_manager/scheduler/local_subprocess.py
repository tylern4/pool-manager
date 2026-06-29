import logging
import os
import shlex
import signal
import subprocess
import threading
from dataclasses import dataclass

from pool_manager.scheduler.base import JobInfo, JobState, SchedulerBackend, _test_job_id

log = logging.getLogger("pool_manager.scheduler.local_subprocess")

_next_id = 0
_id_lock = threading.Lock()


def _next_job_id() -> str:
    global _next_id
    with _id_lock:
        _next_id += 1
        return str(_next_id)


@dataclass
class _ManagedProc:
    proc: subprocess.Popen
    job_id: str


class LocalSubprocessBackend(SchedulerBackend):
    def __init__(self, test_mode: bool = False):
        self._procs: dict[str, _ManagedProc] = {}
        self._test_mode = test_mode

    def submit(self, script_path: str, submit_args: dict[str, str]) -> str:
        if self._test_mode:
            job_id = _test_job_id()
            log.info(
                "[TEST] Would start local worker %s (script=%s, args=%s)",
                job_id,
                script_path,
                submit_args,
            )
            return job_id

        job_id = _next_job_id()
        cmd = ["bash", script_path]

        env = os.environ.copy()
        for key, val in submit_args.items():
            env[f"POOL_MANAGER_{key.upper()}"] = str(val)

        log.debug("Starting local worker %s: %s", job_id, shlex.join(cmd))
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._procs[job_id] = _ManagedProc(proc=proc, job_id=job_id)
        log.debug("Started local worker %s (pid=%d)", job_id, proc.pid)
        return job_id

    def cancel(self, job_id: str) -> None:
        if self._test_mode:
            log.info("[TEST] Would cancel local worker %s", job_id)
            return

        mp = self._procs.pop(job_id, None)
        if mp is None:
            log.warning("Local worker %s not found", job_id)
            return
        log.debug("Killing local worker %s (pid=%d)", job_id, mp.proc.pid)
        mp.proc.terminate()
        try:
            mp.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            mp.proc.kill()
            mp.proc.wait()

    def list_active(self) -> list[JobInfo]:
        if self._test_mode:
            log.info("[TEST] Would list active local workers")
            return []

        jobs: list[JobInfo] = []
        dead_ids: list[str] = []
        for job_id, mp in self._procs.items():
            rc = mp.proc.poll()
            if rc is not None:
                dead_ids.append(job_id)
            else:
                jobs.append(JobInfo(job_id=job_id, state=JobState.RUNNING))
        for jid in dead_ids:
            self._procs.pop(jid, None)
        return jobs

    def signal(self, job_id: str, sig: str) -> None:
        if self._test_mode:
            log.info("[TEST] Would send signal %s to local worker %s", sig, job_id)
            return

        mp = self._procs.get(job_id)
        if mp is None:
            log.warning("Local worker %s not found for signal", job_id)
            return
        sig_num = getattr(signal.Signals, sig.upper(), signal.SIGTERM)
        log.debug("Sending %s to local worker %s (pid=%d)", sig, job_id, mp.proc.pid)
        mp.proc.send_signal(sig_num)

    def name(self) -> str:
        return "local_subprocess"

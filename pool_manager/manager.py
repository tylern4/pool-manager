import logging
import signal
import time

from pool_manager.config import Config
from pool_manager.placement import Placement, PlacementPlanner, TaskResources
from pool_manager.scheduler.base import HPCScheduler, JobInfo, JobState
from pool_manager.work_queue.base import WorkQueue

log = logging.getLogger("pool_manager.manager")


def _make_work_queue(cfg) -> WorkQueue:
    from pool_manager.work_queue import (
        CondorPythonBackend,
        CondorRESTAPIBackend,
        CondorSubprocessBackend,
        CondorWorkQueue,
    )

    wk = cfg.work_queue
    match wk.backend:
        case "condor_python":
            try:
                import htcondor  # noqa: F401
            except ImportError:
                log.warning("htcondor package not available, falling back to condor_subprocess")
                backend = CondorSubprocessBackend(schedd_name=wk.schedd_name)
            else:
                backend = CondorPythonBackend(schedd_name=wk.schedd_name)
        case "condor_subprocess":
            backend = CondorSubprocessBackend(schedd_name=wk.schedd_name)
        case "condor_rest":
            backend = CondorRESTAPIBackend(url=wk.rest_url)
        case _:
            raise ValueError(f"Unknown work_queue backend: {wk.backend}")

    return CondorWorkQueue(backend=backend, constraint=wk.constraint)


def _make_scheduler(cfg) -> HPCScheduler:
    from pool_manager.scheduler import (
        HTCondorRESTAPIBackend,
        LocalSubprocessBackend,
        PBSSubprocessBackend,
        SchedulerWrapper,
        SlurmRESTAPIBackend,
        SlurmSFAPIBackend,
        SlurmSubprocessBackend,
    )

    sch = cfg.scheduler
    match sch.backend:
        case "slurm_subprocess":
            backend = SlurmSubprocessBackend()
        case "slurm_rest":
            backend = SlurmRESTAPIBackend(url=sch.rest_url, token=sch.rest_token)
        case "slurm_sfapi":
            backend = SlurmSFAPIBackend(
                machine=sch.machine,
                client_id=sch.sfapi_client_id,
                client_secret=sch.sfapi_client_secret,
                key_path=sch.sfapi_key_path,
                user=sch.sfapi_user,
            )
        case "pbs_subprocess":
            backend = PBSSubprocessBackend()
        case "local_subprocess":
            backend = LocalSubprocessBackend()
        case "htcondor_rest":
            backend = HTCondorRESTAPIBackend(url=sch.rest_url, token=sch.rest_token)
        case _:
            raise ValueError(f"Unknown scheduler backend: {sch.backend}")

    return SchedulerWrapper(backend=backend)


class PoolManager:
    def __init__(self, config: Config, work_queue: WorkQueue, scheduler: HPCScheduler):
        self._config = config
        self._wq = work_queue
        self._sched = scheduler
        self._policy = config.scaling

        nc = config.scheduler.node_configs
        self._planner = PlacementPlanner(
            node_configs=nc if nc else None,
            task_resources=self._policy.task_resources,
            batch_size=self._policy.batch_size,
            max_workers=self._policy.max_workers,
            min_workers=self._policy.min_workers,
        )
        self._has_node_configs = bool(nc)

        self._running = True
        self._tracked: dict[str, JobInfo] = {}
        self._node_assignments: dict[str, str] = {}
        self._last_scale_up = 0.0
        self._last_scale_down = 0.0
        self._drain_start: float | None = None
        self._daemon_shutdown = False

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        sig_name = signal.Signals(signum).name
        log.info("Received %s, initiating shutdown", sig_name)
        self._daemon_shutdown = True

    def run(self):
        log.info(
            "Pool manager started (queue=%s, scheduler=%s)", self._wq.name(), self._sched.name()
        )
        log.info(
            "Scaling policy: min=%d max=%d batch=%d cooldown_up=%.0fs cooldown_down=%.0fs",
            self._policy.min_workers,
            self._policy.max_workers,
            self._policy.batch_size,
            self._policy.scale_up_cooldown,
            self._policy.scale_down_cooldown,
        )
        if self._has_node_configs:
            log.info(
                "Node-aware placement: %d node config(s), resources from condor_q per task",
                len(self._config.scheduler.node_configs),
            )

        while self._running:
            try:
                self._tick()
            except Exception:
                log.exception("Unhandled error in main loop")

            if self._daemon_shutdown:
                self._drain_all()
                self._running = False
                break

            time.sleep(self._config.poll_interval)

        log.info("Pool manager stopped")

    def _tick(self):
        tasks = self._wq.list_idle()
        plan = self._planner.plan_for_tasks(tasks)
        target = sum(p.count for p in plan)
        target = max(self._policy.min_workers, min(self._policy.max_workers, target))
        log.debug(
            "Tick: idle=%d target=%d active=%d draining=%d",
            len(tasks),
            target,
            self._active_count(),
            self._draining_count(),
        )

        self._reconcile()
        self._scale(tasks, plan, target)

    def _reconcile(self):
        active = self._sched.list_active()
        active_ids = {j.job_id for j in active}

        for aj in active:
            existing = self._tracked.get(aj.job_id)
            if existing is None:
                log.debug("Tracking new job %s (state=%s)", aj.job_id, aj.state.value)
                self._tracked[aj.job_id] = aj
            elif existing.state != aj.state:
                log.debug(
                    "Job %s state change: %s -> %s", aj.job_id, existing.state.value, aj.state.value
                )
                self._tracked[aj.job_id] = aj

        lost = [
            jid
            for jid in self._tracked
            if jid not in active_ids and self._tracked[jid].state not in (JobState.EXITED,)
        ]
        for jid in lost:
            tracked = self._tracked[jid]
            if tracked.state == JobState.DRAINING:
                log.info("Drained job %s exited gracefully", jid)
            else:
                log.info("Job %s no longer active (was %s)", jid, tracked.state.value)
            self._tracked[jid] = JobInfo(job_id=jid, state=JobState.EXITED)
            self._node_assignments.pop(jid, None)

    def _scale(self, tasks: list[TaskResources], plan: list[Placement], target: int):
        active = self._active_count()
        now = time.monotonic()

        if target > active:
            if now - self._last_scale_up < self._policy.scale_up_cooldown:
                log.debug("Scale-up cooldown active, skipping")
                return
            to_add = target - active
            log.debug("Scaling UP: adding %d workers (target=%d active=%d)", to_add, target, active)
            self._start_workers(plan, to_add)
            self._last_scale_up = now
            self._drain_start = None

        elif target < active:
            if not self._daemon_shutdown:
                if self._drain_start is None:
                    log.debug(
                        "Idle count %d below target %d; starting scale-down cooldown",
                        len(tasks),
                        target,
                    )
                    self._drain_start = now + self._policy.scale_down_cooldown
                    return
                if now < self._drain_start:
                    return

            excess = active - target
            log.debug(
                "Scaling DOWN: removing %d workers (target=%d active=%d)", excess, target, active
            )
            self._signal_workers(excess, plan=plan)
            self._last_scale_down = now

            if self._draining_count() > 0 and self._policy.drain_timeout > 0:
                deadline = self._drain_start + self._policy.drain_timeout
                if now > deadline and self._config.scale_down_cooldown > 0:
                    self._force_cancel_draining()

        elif self._daemon_shutdown:
            self._drain_all()

    def _start_workers(self, plan: list[Placement], count: int):
        if self._has_node_configs:
            self._start_workers_from_plan(plan, count)
        else:
            self._start_workers_simple(count)

    def _signal_workers(self, count: int, plan: list[Placement] | None = None):
        active = sorted(
            jid
            for jid, ji in self._tracked.items()
            if ji.state in (JobState.RUNNING, JobState.PENDING)
        )
        if count <= 0 or not active:
            return

        if self._has_node_configs and plan is not None:
            desired: dict[str, int] = {}
            for p in plan:
                desired[p.node_config.name] = desired.get(p.node_config.name, 0) + p.count

            by_type: dict[str, list[str]] = {}
            for jid in active:
                nt = self._node_assignments.get(jid, "unknown")
                by_type.setdefault(nt, []).append(jid)

            node_costs: dict[str, int] = {}
            for nc in self._config.scheduler.node_configs:
                node_costs[nc.name] = nc.cpus * max(nc.memory_mb, 1) * max(nc.gpus, 1)

            candidates: list[str] = []
            for nt, jids in by_type.items():
                jids.sort()
                max_keep = desired.get(nt, 0)
                if len(jids) > max_keep:
                    candidates.extend(jids[: len(jids) - max_keep])

            candidates.sort(
                key=lambda jid: node_costs.get(self._node_assignments.get(jid, ""), 0),
                reverse=True,
            )

            to_drain = candidates[:count]
        else:
            to_drain = active[:count]

        for jid in to_drain:
            log.info("Signalling worker %s to drain (SIGTERM)", jid)
            try:
                self._sched.signal(jid, "SIGTERM")
                self._tracked[jid] = JobInfo(job_id=jid, state=JobState.DRAINING)
            except Exception:
                log.exception("Failed to signal worker %s", jid)

    def _start_workers_simple(self, count: int):
        script = self._config.scheduler.worker_script
        if not script:
            log.error("Cannot start workers: no worker_script configured")
            return
        log.debug("Starting %d worker(s) via %s", count, script)
        for i in range(count):
            try:
                job_id = self._sched.submit(script, self._config.scheduler.submit_args)
                self._tracked[job_id] = JobInfo(job_id=job_id, state=JobState.PENDING)
                self._node_assignments[job_id] = "default"
                log.info("Started worker %s (%s)", job_id, self._sched.name())
            except Exception:
                log.exception("Failed to start worker %d/%d", i + 1, count)

    def _start_workers_from_plan(self, placements: list[Placement], count: int):
        script = self._config.scheduler.worker_script
        if not script:
            log.error("Cannot start workers: no worker_script configured")
            return
        log.debug("Starting %d worker(s) from placement plan", count)
        remaining = count
        for p in placements:
            batch = min(p.count, remaining)
            if batch <= 0:
                continue
            args = dict(self._config.scheduler.submit_args)
            nc = p.node_config
            args["cpus-per-task"] = str(nc.cpus)
            args["mem"] = f"{nc.memory_mb}M"
            if nc.gpus > 0:
                args["gpus"] = str(nc.gpus)
            for _ in range(batch):
                try:
                    job_id = self._sched.submit(script, args)
                    self._tracked[job_id] = JobInfo(job_id=job_id, state=JobState.PENDING)
                    self._node_assignments[job_id] = nc.name
                    log.info("Started worker %s (%s) on %s", job_id, self._sched.name(), nc.name)
                except Exception:
                    log.exception("Failed to start worker on %s", nc.name)
            remaining -= batch
            if remaining <= 0:
                break

    def _force_cancel_draining(self):
        draining = [j for j in self._tracked.values() if j.state == JobState.DRAINING]
        for ji in draining:
            log.warning("Force-cancelling draining worker %s (timed out)", ji.job_id)
            try:
                self._sched.cancel(ji.job_id)
                self._tracked[ji.job_id] = JobInfo(job_id=ji.job_id, state=JobState.EXITED)
            except Exception:
                log.exception("Failed to force-cancel worker %s", ji.job_id)

    def _drain_all(self):
        log.info("Draining all workers")
        active = self._active_count()
        if active == 0:
            log.info("No active workers to drain")
            return
        plan = self._planner.plan_for_tasks([])
        self._signal_workers(active, plan=plan)
        deadline = time.monotonic() + self._policy.drain_timeout
        while time.monotonic() < deadline:
            self._reconcile()
            if self._active_count() == 0:
                log.info("All workers drained")
                return
            time.sleep(2)
        self._force_cancel_draining()

    def _active_count(self) -> int:
        return sum(
            1
            for j in self._tracked.values()
            if j.state in (JobState.PENDING, JobState.RUNNING, JobState.DRAINING)
        )

    def _draining_count(self) -> int:
        return sum(1 for j in self._tracked.values() if j.state == JobState.DRAINING)

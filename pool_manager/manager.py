import os
import signal
import time
from pathlib import Path

try:
    import htcondor
except ImportError:
    htcondor = None

from loguru import logger

from pool_manager.config import Config
from pool_manager.metrics import (
    SCALE_DOWN_EVENTS,
    SCALE_UP_EVENTS,
    TICK_DURATION,
    WORKERS_STARTED,
    WORKERS_STOPPED,
    start_metrics_server,
    update_metrics,
)
from pool_manager.placement import Placement, PlacementPlanner, TaskResources
from pool_manager.scheduler import (
    HTCondorRESTAPIBackend,
    LocalSubprocessBackend,
    PBSSubprocessBackend,
    SchedulerWrapper,
    SlurmRESTAPIBackend,
    SlurmSFAPIBackend,
    SlurmSubprocessBackend,
)
from pool_manager.scheduler.base import JobInfo, JobState, SchedulerBackend, parse_config_name
from pool_manager.work_queue import (
    CondorPythonBackend,
    CondorRESTAPIBackend,
    CondorSubprocessBackend,
    CondorWorkQueue,
)
from pool_manager.work_queue.base import WorkQueue


def _make_work_queue(cfg) -> WorkQueue:
    wk = cfg.work_queue
    match wk.backend:
        case "condor_python":
            if htcondor is None:
                logger.warning("htcondor package not available, falling back to condor_subprocess")
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


def _make_scheduler(cfg) -> SchedulerBackend:
    sch = cfg.scheduler
    user = sch.user or os.environ.get("USER", "")
    job_name_prefix = sch.job_name_prefix

    match sch.backend:
        case "slurm_subprocess":
            backend = SlurmSubprocessBackend(
                job_name_prefix=job_name_prefix, test_mode=sch.test_mode, user=user
            )
        case "slurm_rest":
            backend = SlurmRESTAPIBackend(
                url=sch.rest_url,
                token=sch.rest_token,
                user=user,
                job_name_prefix=job_name_prefix,
                test_mode=sch.test_mode,
            )
        case "slurm_sfapi":
            backend = SlurmSFAPIBackend(
                machine=sch.machine,
                client_id=sch.sfapi_client_id,
                client_secret=sch.sfapi_client_secret,
                key_path=sch.sfapi_key_path,
                user=sch.sfapi_user or user,
                job_name_prefix=job_name_prefix,
                test_mode=sch.test_mode,
            )
        case "pbs_subprocess":
            backend = PBSSubprocessBackend(
                job_name_prefix=job_name_prefix, test_mode=sch.test_mode, user=user
            )
        case "local_subprocess":
            backend = LocalSubprocessBackend(test_mode=sch.test_mode)
        case "htcondor_rest":
            backend = HTCondorRESTAPIBackend(
                url=sch.rest_url,
                token=sch.rest_token,
                owner=user,
                job_name_prefix=job_name_prefix,
            )
        case _:
            raise ValueError(f"Unknown scheduler backend: {sch.backend}")

    return SchedulerWrapper(backend=backend)


class PoolManager:
    def __init__(self, config: Config, work_queue: WorkQueue, scheduler: SchedulerBackend):
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
        self._last_plan: list[Placement] = []

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self._daemon_shutdown = True
        self._running = False
        raise KeyboardInterrupt()

    def run(self):
        logger.info(
            "Pool manager started (queue={}, scheduler={})", self._wq.name(), self._sched.name()
        )
        logger.info(
            "Scaling policy: min={} max={} batch={} cooldown_up={} cooldown_down={}",
            self._policy.min_workers,
            self._policy.max_workers,
            self._policy.batch_size,
            self._policy.scale_up_cooldown,
            self._policy.scale_down_cooldown,
        )
        if self._has_node_configs:
            logger.info(
                "Node-aware placement: {} node config(s), resources from condor_q per task",
                len(self._config.scheduler.node_configs),
            )

        if self._config.metrics_port > 0:
            try:
                start_metrics_server(self._config.metrics_port)
                logger.info("Metrics server started on port {}", self._config.metrics_port)
            except Exception:
                logger.exception(
                    "Failed to start metrics server on port {}", self._config.metrics_port
                )

        self._recover_state()

        try:
            while self._running:
                try:
                    self._tick()
                except Exception:
                    logger.exception("Unhandled error in main loop")

                if not self._running:
                    break

                time.sleep(self._config.poll_interval)
        except KeyboardInterrupt:
            logger.info("Shutting down")

        if self._daemon_shutdown and self._policy.drain_on_stop:
            self._drain_all()
        logger.info("Pool manager stopped")

    def _tick(self):
        import time as time_mod

        start_time = time_mod.monotonic()
        try:
            tasks = self._wq.list_idle()
            plan = self._planner.plan_for_tasks(tasks)
            self._last_plan = plan
            target = sum(p.count for p in plan)
            target = max(self._policy.min_workers, min(self._policy.max_workers, target))
            logger.debug(
                "Tick: idle={} target={} active={} draining={}",
                len(tasks),
                target,
                self._active_count(),
                self._draining_count(),
            )

            self._reconcile()
            self._scale(tasks, plan, target)

            states = {jid: ji.state for jid, ji in self._tracked.items()}
            update_metrics(
                idle_count=len(tasks),
                target=target,
                tracked=states,
                node_assignments=self._node_assignments,
                placements=plan,
            )
        finally:
            duration = time_mod.monotonic() - start_time
            TICK_DURATION.observe(duration)

    def _recover_state(self):
        active = self._sched.list_active()
        prefix = self._config.scheduler.job_name_prefix
        for aj in active:
            self._tracked[aj.job_id] = aj
            config_name = parse_config_name(aj.job_name, prefix)
            self._node_assignments.setdefault(aj.job_id, config_name)
        if active:
            logger.info("Recovered {} active worker(s) from scheduler", len(active))

    def _reconcile(self):
        active = self._sched.list_active()
        active_ids = {j.job_id for j in active}
        prefix = self._config.scheduler.job_name_prefix

        for aj in active:
            existing = self._tracked.get(aj.job_id)
            if existing is None:
                logger.debug("Tracking new job {} (state={})", aj.job_id, aj.state.value)
                self._tracked[aj.job_id] = aj
                config_name = parse_config_name(aj.job_name, prefix)
                self._node_assignments.setdefault(aj.job_id, config_name)
            elif existing.state != aj.state:
                logger.debug(
                    "Job {} state change: {} -> {}", aj.job_id, existing.state.value, aj.state.value
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
                logger.info("Drained job {} exited gracefully", jid)
            else:
                logger.info("Job {} no longer active (was {})", jid, tracked.state.value)
            self._tracked[jid] = JobInfo(job_id=jid, state=JobState.EXITED)
            self._node_assignments.pop(jid, None)
            WORKERS_STOPPED.inc()

    def _scale(self, tasks: list[TaskResources], plan: list[Placement], target: int):
        active = self._active_count()
        now = time.monotonic()

        if target > active:
            if now - self._last_scale_up < self._policy.scale_up_cooldown:
                logger.debug("Scale-up cooldown active, skipping")
                return
            to_add = target - active
            logger.debug(
                "Scaling UP: adding {} workers (target={} active={})", to_add, target, active
            )
            self._start_workers(plan, to_add)
            self._last_scale_up = now
            self._drain_start = None
            SCALE_UP_EVENTS.inc()

        elif target < active:
            if not self._daemon_shutdown:
                if self._drain_start is None:
                    logger.debug(
                        "Idle count {} below target {}; starting scale-down cooldown",
                        len(tasks),
                        target,
                    )
                    self._drain_start = now + self._policy.scale_down_cooldown
                    return
                if now < self._drain_start:
                    return

            excess = active - target
            logger.debug(
                "Scaling DOWN: removing {} workers (target={} active={})", excess, target, active
            )
            self._signal_workers(excess, plan=plan)
            self._last_scale_down = now
            SCALE_DOWN_EVENTS.inc()

            can_force = (
                self._draining_count() > 0
                and self._policy.drain_timeout > 0
                and self._drain_start is not None
            )
            if can_force:
                deadline = self._drain_start + self._policy.drain_timeout
                if now > deadline and self._policy.scale_down_cooldown > 0:
                    self._force_cancel_draining()

        elif self._daemon_shutdown:
            self._drain_all()

    def _start_workers(self, plan: list[Placement], count: int):
        if self._has_node_configs:
            existing_per_type: dict[str, int] = {}
            for jid, ji in self._tracked.items():
                if ji.state in (JobState.PENDING, JobState.RUNNING, JobState.DRAINING):
                    nt = self._node_assignments.get(jid, "unknown")
                    existing_per_type[nt] = existing_per_type.get(nt, 0) + 1

            adjusted_plan = []
            total_needed = 0
            for p in plan:
                existing = existing_per_type.get(p.node_config.name, 0)
                needed = max(0, p.count - existing)
                if needed > 0:
                    adjusted_plan.append(Placement(node_config=p.node_config, count=needed))
                    total_needed += needed

            if total_needed > 0:
                self._start_workers_from_plan(adjusted_plan, min(count, total_needed))
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
            logger.info("Signalling worker {} to drain (SIGTERM)", jid)
            try:
                self._sched.signal(jid, "SIGTERM")
                self._tracked[jid] = JobInfo(job_id=jid, state=JobState.DRAINING)
            except Exception:
                logger.exception("Failed to signal worker {}", jid)

    def _start_workers_simple(self, count: int):
        script = self._config.scheduler.worker_script
        if not script:
            logger.error("Cannot start workers: no worker_script configured")
            return
        script_path = Path(script)
        if not script_path.exists():
            logger.error("Worker script not found: {}", script)
            return
        logger.debug("Starting {} worker(s) via {}", count, script)
        prefix = self._config.scheduler.job_name_prefix
        for i in range(count):
            try:
                args = dict(self._config.scheduler.submit_args)
                args.setdefault("job-name", f"{prefix}default")
                job_id = self._sched.submit(script, args)
                self._tracked[job_id] = JobInfo(job_id=job_id, state=JobState.PENDING)
                self._node_assignments[job_id] = "default"
                logger.info("Started worker {} ({})", job_id, self._sched.name())
                WORKERS_STARTED.inc()
            except Exception:
                logger.exception("Failed to start worker {}/{}", i + 1, count)

    def _start_workers_from_plan(self, placements: list[Placement], count: int):
        script = self._config.scheduler.worker_script
        if not script:
            logger.error("Cannot start workers: no worker_script configured")
            return
        script_path = Path(script)
        if not script_path.exists():
            logger.error("Worker script not found: {}", script)
            return
        logger.debug("Starting {} worker(s) from placement plan", count)
        prefix = self._config.scheduler.job_name_prefix
        remaining = count
        for p in placements:
            batch = min(p.count, remaining)
            if batch <= 0:
                continue
            args = dict(self._config.scheduler.submit_args)
            nc = p.node_config
            if nc.submit_args:
                args.update(nc.submit_args)
            args["job-name"] = f"{prefix}{nc.name}"
            args["cpus-per-task"] = str(nc.cpus)
            args["mem"] = f"{nc.memory_mb}M"
            if nc.gpus > 0:
                args["gpus"] = str(nc.gpus)
            for _ in range(batch):
                try:
                    job_id = self._sched.submit(script, args)
                    self._tracked[job_id] = JobInfo(job_id=job_id, state=JobState.PENDING)
                    self._node_assignments[job_id] = nc.name
                    logger.info("Started worker {} ({}) on {}", job_id, self._sched.name(), nc.name)
                    WORKERS_STARTED.inc()
                except Exception:
                    logger.exception("Failed to start worker on {}", nc.name)
            remaining -= batch
            if remaining <= 0:
                break

    def _force_cancel_draining(self):
        draining = [j for j in self._tracked.values() if j.state == JobState.DRAINING]
        for ji in draining:
            logger.warning("Force-cancelling draining worker {} (timed out)", ji.job_id)
            try:
                self._sched.cancel(ji.job_id)
                self._tracked[ji.job_id] = JobInfo(job_id=ji.job_id, state=JobState.EXITED)
            except Exception:
                logger.exception("Failed to force-cancel worker {}", ji.job_id)

    def _drain_all(self):
        logger.info("Draining all workers")
        active = self._active_count()
        if active == 0:
            logger.info("No active workers to drain")
            return
        plan = self._planner.plan_for_tasks([])
        self._signal_workers(active, plan=plan)
        deadline = time.monotonic() + self._policy.drain_timeout
        while time.monotonic() < deadline:
            self._reconcile()
            if self._active_count() == 0:
                logger.info("All workers drained")
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

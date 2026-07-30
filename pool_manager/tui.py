from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Static

from pool_manager.config import Config
from pool_manager.manager import _make_scheduler, _make_work_queue
from pool_manager.placement import Placement, PlacementPlanner, TaskResources
from pool_manager.scheduler.base import JobInfo, JobState, SchedulerBackend, parse_config_name
from pool_manager.work_queue.base import WorkQueue

REFRESH_INTERVAL = 1.0


@dataclass
class WorkerInfo:
    job_id: str
    state: str
    node_type: str


class PoolStateWidget(Static):
    idle_jobs = reactive(0)
    running_jobs = reactive(0)
    total_cpus = reactive(0.0)
    total_memory_mb = reactive(0)
    total_gpus = reactive(0)
    active_count = reactive(0)
    draining_count = reactive(0)
    pending_count = reactive(0)
    running_count = reactive(0)
    target_workers = reactive(0)
    uptime = reactive(0.0)
    started_at: float = 0.0
    scheduler_status = reactive("")
    work_queue_status = reactive("")
    scale_action = reactive("")

    def on_mount(self) -> None:
        self.started_at = time.monotonic()
        self.set_interval(REFRESH_INTERVAL, self._tick_uptime)

    def _tick_uptime(self) -> None:
        self.uptime = time.monotonic() - self.started_at

    def render(self) -> Text:
        lines = [
            Text.from_markup("[bold cyan]Pool Manager Status[/bold cyan]"),
            Text.from_markup(f"  Uptime: [green]{self._format_uptime(self.uptime)}[/green]"),
            Text(),
            Text.from_markup("[bold]Connections[/bold]"),
            Text.from_markup(f"  Scheduler:  {self.scheduler_status}"),
            Text.from_markup(f"  Work Queue: {self.work_queue_status}"),
            Text(),
            Text.from_markup("[bold]HTCondor Queue[/bold]"),
            Text.from_markup(f"  Idle:    [yellow]{self.idle_jobs}[/yellow]"),
            Text.from_markup(f"  Running: [green]{self.running_jobs}[/green]"),
            Text.from_markup(
                f"  Resources: [cyan]{self.total_cpus:.0f}[/cyan] CPUs  "
                f"[cyan]{self.total_memory_mb / 1024:.0f}[/cyan] GB  "
                f"[cyan]{self.total_gpus}[/cyan] GPUs"
            ),
            Text(),
            Text.from_markup("[bold]Workers[/bold]"),
            Text.from_markup(f"  Target:   [blue]{self.target_workers}[/blue]"),
            Text.from_markup(f"  Active:   [green]{self.active_count}[/green]"),
            Text.from_markup(f"  Running:  [green]{self.running_count}[/green]"),
            Text.from_markup(f"  Pending:  [yellow]{self.pending_count}[/yellow]"),
            Text.from_markup(f"  Draining: [red]{self.draining_count}[/red]"),
        ]
        if self.scale_action:
            lines.append(Text())
            lines.append(Text.from_markup(f"[bold]Action:[/bold] {self.scale_action}"))
        return Text("\n").join(lines)

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


class WorkersTable(DataTable):
    def on_mount(self) -> None:
        self.add_columns("Job ID", "State", "Node Type", "Queue Time")
        self.set_interval(REFRESH_INTERVAL, self._update)

    def _update(self) -> None:
        pass


class CompletedJobsTable(DataTable):
    def on_mount(self) -> None:
        self.add_columns("Job ID", "Node Type", "Queue Time", "Runtime")
        self.set_interval(REFRESH_INTERVAL, self._update)

    def _update(self) -> None:
        pass


class PlacementWidget(Static):
    placements: list[tuple[str, int, int, int, int, int]] = []

    def update_placements(self, placements: list[tuple[str, int, int, int, int, int]]) -> None:
        self.placements = placements
        self.refresh()

    def render(self) -> Text:
        lines = [Text.from_markup("[bold]Placement Plan[/bold]")]
        if not self.placements:
            lines.append(Text.from_markup("  [yellow]No change needed[/yellow]"))
        else:
            for name, delta, cpus, mem, gpus, existing in self.placements:
                if delta > 0:
                    tag = f"[green]+{delta}[/green]"
                elif delta < 0:
                    tag = f"[red]{delta}[/red]"
                else:
                    tag = "[blue]0[/blue]"
                lines.append(
                    Text.from_markup(
                        f"  {name}: {tag} nodes"
                        f" ({existing} running, cpus={cpus} mem={mem}MB gpus={gpus})"
                    )
                )
        return Text("\n").join(lines)


class PoolManagerTUI(App):
    CSS = """
    #main {
        layout: horizontal;
        height: 1fr;
    }

    .left-panel {
        width: 1fr;
        height: 1fr;
        layout: vertical;
    }

    .right-panel {
        width: 1fr;
        height: 1fr;
        layout: vertical;
    }

    PoolStateWidget {
        height: auto;
        padding: 1;
        border: solid green;
    }

    PlacementWidget {
        height: auto;
        padding: 1;
        border: solid blue;
        min-height: 3;
    }

    WorkersTable {
        height: 1fr;
        border: solid yellow;
    }

    CompletedJobsTable {
        height: 1fr;
        border: solid magenta;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    workers_data: list[WorkerInfo] = []
    placements_data: list[tuple[str, int, int, int, int]] = []

    def __init__(self, config_path: str = "pool-manager.yaml", **kwargs):
        super().__init__(**kwargs)
        self.config_path = Path(config_path)
        self.config = Config.from_file(self.config_path)
        self._sched: SchedulerBackend | None = None
        self._wq: WorkQueue | None = None
        self._planner: PlacementPlanner | None = None
        self._init_backends()

    def _init_backends(self) -> None:
        try:
            self._sched = _make_scheduler(self.config)
        except Exception as e:
            logger.warning("Failed to create scheduler backend: {}", e)
        try:
            self._wq = _make_work_queue(self.config)
        except Exception as e:
            logger.warning("Failed to create work queue backend: {}", e)

        self._planner = PlacementPlanner(
            node_configs=self.config.scheduler.node_configs or None,
            task_resources=self.config.scaling.task_resources,
            batch_size=self.config.scaling.batch_size,
            max_workers=self.config.scaling.max_workers,
            min_workers=self.config.scaling.min_workers,
        )

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with Container(classes="left-panel"):
                yield PoolStateWidget()
                yield PlacementWidget()
            with Container(classes="right-panel"):
                yield WorkersTable()
                yield CompletedJobsTable()
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Pool Manager"
        self.set_interval(REFRESH_INTERVAL, self._refresh)

    def _refresh(self) -> None:
        state_widget = self.query_one(PoolStateWidget)

        all_tasks: list[TaskResources] = []
        if self._wq is not None:
            try:
                all_tasks = self._wq.list_idle()
                state_widget.work_queue_status = "[green]Connected[/green]"
                idle_count = sum(1 for t in all_tasks if t.job_status == 1)
                running_count = sum(1 for t in all_tasks if t.job_status == 2)
                state_widget.idle_jobs = idle_count
                state_widget.running_jobs = running_count
                state_widget.total_cpus = sum(t.cpus for t in all_tasks)
                state_widget.total_memory_mb = sum(t.memory_mb for t in all_tasks)
                state_widget.total_gpus = sum(t.gpus for t in all_tasks)
            except Exception as e:
                msg = _format_error(e)
                state_widget.work_queue_status = f"[red]{msg}[/red]"
                state_widget.idle_jobs = 0
                state_widget.running_jobs = 0
                state_widget.total_cpus = 0.0
                state_widget.total_memory_mb = 0
                state_widget.total_gpus = 0
        else:
            state_widget.work_queue_status = "[dim]Not configured[/dim]"
            state_widget.idle_jobs = 0
            state_widget.running_jobs = 0
            state_widget.total_cpus = 0.0
            state_widget.total_memory_mb = 0
            state_widget.total_gpus = 0

        active_jobs: list[JobInfo] = []
        all_jobs: list[JobInfo] = []
        if self._sched is not None:
            try:
                all_jobs = self._sched.list_active()
                active_jobs = [
                    j
                    for j in all_jobs
                    if j.state in (JobState.PENDING, JobState.RUNNING, JobState.DRAINING)
                ]
                state_widget.scheduler_status = (
                    f"[green]Connected[/green] ({len(active_jobs)} active)"
                )
            except Exception as e:
                msg = _format_error(e)
                state_widget.scheduler_status = f"[red]{msg}[/red]"
        else:
            state_widget.scheduler_status = "[dim]Not configured[/dim]"

        idle_tasks = [t for t in all_tasks if t.job_status == 1]
        plan: list[Placement] = []
        target = 0
        if self._planner is not None:
            plan = self._planner.plan_for_tasks(idle_tasks)
            target = sum(p.count for p in plan)
            target = max(
                self.config.scaling.min_workers,
                min(self.config.scaling.max_workers, target),
            )

        pending_count = sum(1 for j in active_jobs if j.state == JobState.PENDING)
        running_count = sum(1 for j in active_jobs if j.state == JobState.RUNNING)
        draining_count = sum(1 for j in active_jobs if j.state == JobState.DRAINING)
        active_count = pending_count + running_count + draining_count

        state_widget.active_count = active_count
        state_widget.pending_count = pending_count
        state_widget.running_count = running_count
        state_widget.draining_count = draining_count
        state_widget.target_workers = target

        delta = target - active_count
        if delta > 0:
            state_widget.scale_action = f"[green]+{delta} workers needed[/green]"
        elif delta < 0:
            state_widget.scale_action = f"[red]{delta} workers to remove[/red]"
        else:
            state_widget.scale_action = "[blue]No change needed[/blue]"

        prefix = self.config.scheduler.job_name_prefix
        table = self.query_one(WorkersTable)
        table.clear()
        for j in active_jobs:
            node_type = parse_config_name(j.job_name, prefix)
            qt = _format_duration(j.queue_time) if j.queue_time > 0 else ""
            table.add_row(j.job_id, j.state.value, node_type, qt)

        completed = [j for j in all_jobs if j.state == JobState.COMPLETED]
        completed_table = self.query_one(CompletedJobsTable)
        completed_table.clear()
        for j in completed:
            node_type = parse_config_name(j.job_name, prefix)
            qt = _format_duration(j.queue_time) if j.queue_time > 0 else ""
            rt = _format_duration(j.runtime) if j.runtime > 0 else "N/A"
            completed_table.add_row(j.job_id, node_type, qt, rt)

        existing_per_type: dict[str, int] = {}
        for j in active_jobs:
            nt = parse_config_name(j.job_name, prefix)
            existing_per_type[nt] = existing_per_type.get(nt, 0) + 1

        all_types = set(existing_per_type.keys()) | {p.node_config.name for p in plan}
        placements_out: list[tuple[str, int, int, int, int, int]] = []
        for nt in sorted(all_types):
            nc = next((p.node_config for p in plan if p.node_config.name == nt), None)
            desired = next((p.count for p in plan if p.node_config.name == nt), 0)
            existing = existing_per_type.get(nt, 0)
            delta = desired - existing
            if nc:
                placements_out.append((nt, delta, nc.cpus, nc.memory_mb, nc.gpus, existing))
            else:
                placements_out.append((nt, delta, 0, 0, 0, existing))
        placement_widget = self.query_one(PlacementWidget)
        placement_widget.update_placements(placements_out)

    def action_refresh(self) -> None:
        self._refresh()

    async def action_quit(self) -> None:
        self.exit()


def _format_error(exc: Exception) -> str:
    msg = str(exc)
    if not msg:
        return type(exc).__name__
    return msg[:120]


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def run_tui(config_path: str = "pool-manager.yaml") -> None:
    app = PoolManagerTUI(config_path=config_path)
    app.run()

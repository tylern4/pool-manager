from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Static

from pool_manager.config import Config
from pool_manager.metrics import get_snapshot

if TYPE_CHECKING:
    pass

REFRESH_INTERVAL = 1.0


@dataclass
class WorkerInfo:
    job_id: str
    state: str
    node_type: str


class PoolStateWidget(Static):
    idle_jobs = reactive(0)
    active_workers = reactive(0)
    draining_workers = reactive(0)
    pending_workers = reactive(0)
    running_workers = reactive(0)
    target_workers = reactive(0)
    uptime = reactive(0.0)
    started_at: float = 0.0

    def on_mount(self) -> None:
        self.started_at = time.monotonic()
        self.set_interval(REFRESH_INTERVAL, self._update)

    def _update(self) -> None:
        snapshot = get_snapshot()
        self.idle_jobs = snapshot.idle_jobs
        self.active_workers = snapshot.active_workers
        self.draining_workers = snapshot.draining_workers
        self.pending_workers = snapshot.pending_workers
        self.running_workers = snapshot.running_workers
        self.target_workers = snapshot.target_workers
        self.uptime = time.monotonic() - self.started_at

    def render(self) -> Text:
        lines = [
            Text.from_markup("[bold cyan]Pool Manager Status[/bold cyan]"),
            Text.from_markup(f"  Uptime: [green]{self._format_uptime(self.uptime)}[/green]"),
            Text(),
            Text.from_markup("[bold]HTCondor Queue[/bold]"),
            Text.from_markup(f"  Idle jobs: [yellow]{self.idle_jobs}[/yellow]"),
            Text(),
            Text.from_markup("[bold]Workers[/bold]"),
            Text.from_markup(f"  Target:   [blue]{self.target_workers}[/blue]"),
            Text.from_markup(f"  Active:   [green]{self.active_workers}[/green]"),
            Text.from_markup(f"  Running:  [green]{self.running_workers}[/green]"),
            Text.from_markup(f"  Pending:  [yellow]{self.pending_workers}[/yellow]"),
            Text.from_markup(f"  Draining: [red]{self.draining_workers}[/red]"),
        ]
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
        self.add_columns("Job ID", "State", "Node Type")
        self.set_interval(REFRESH_INTERVAL, self._update)

    def _update(self) -> None:
        pass


class PlacementWidget(Static):
    placements: list[tuple[str, int, int, int, int]] = []

    def update_placements(self, placements: list[tuple[str, int, int, int, int]]) -> None:
        self.placements = placements
        self.refresh()

    def render(self) -> Text:
        if not self.placements:
            return Text.from_markup("[dim]No placements[/dim]")

        lines = [Text.from_markup("[bold]Placement Plan[/bold]")]
        for name, count, cpus, mem, gpus in self.placements:
            lines.append(
                Text.from_markup(
                    f"  {name}: [green]{count}[/green] nodes (cpus={cpus} mem={mem}MB gpus={gpus})"
                )
            )
        return Text("\n").join(lines)


class PoolManagerTUI(App):
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
    }

    .left-panel {
        column-span: 1;
    }

    .right-panel {
        column-span: 1;
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
    }

    WorkersTable {
        height: 1fr;
        border: solid yellow;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    idle_jobs = reactive(0)
    target_workers = reactive(0)
    workers_data: list[WorkerInfo] = []
    placements_data: list[tuple[str, int, int, int, int]] = []

    mock_workers: dict[str, WorkerInfo] = {}
    mock_placements: list[tuple[str, int, int, int, int]] = []

    def __init__(self, config_path: str = "pool-manager.yaml", **kwargs):
        super().__init__(**kwargs)
        self.config_path = Path(config_path)
        self.config = Config.from_file(self.config_path)

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(classes="left-panel"):
            yield PoolStateWidget()
            yield PlacementWidget()
        with Container(classes="right-panel"):
            yield WorkersTable()
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Pool Manager"
        self.set_interval(REFRESH_INTERVAL, self._refresh)

    def _refresh(self) -> None:
        snapshot = get_snapshot()
        self.idle_jobs = snapshot.idle_jobs
        self.target_workers = snapshot.target_workers

        state_widget = self.query_one(PoolStateWidget)
        state_widget._update()

        table = self.query_one(WorkersTable)
        table.clear()
        for worker in self.workers_data:
            table.add_row(worker.job_id, worker.state, worker.node_type)

        placement_widget = self.query_one(PlacementWidget)
        placement_widget.update_placements(self.placements_data)

    def set_workers(self, workers: list[WorkerInfo]) -> None:
        self.workers_data = workers

    def set_placements(self, placements: list[tuple[str, int, int, int, int]]) -> None:
        self.placements_data = placements

    def action_refresh(self) -> None:
        self._refresh()

    def action_quit(self) -> None:
        self.exit()


def run_tui(config_path: str = "pool-manager.yaml") -> None:
    app = PoolManagerTUI(config_path=config_path)
    app.run()

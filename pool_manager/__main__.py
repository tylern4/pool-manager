import json
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from pool_manager.config import Config
from pool_manager.log import setup_logging
from pool_manager.manager import PoolManager, _make_scheduler, _make_work_queue
from pool_manager.placement import TaskResources, make_placement_strategy
from pool_manager.tui import run_tui

app = typer.Typer(
    name="pool-manager",
    help="HTCondor → HPC scheduler pool manager.",
    no_args_is_help=True,
)

CONFIG_OPT = Annotated[
    str,
    typer.Option("-c", "--config", help="Path to config file.", show_default=True),
]
LOG_LEVEL_OPT = Annotated[
    str | None,
    typer.Option("--log-level", help="Log level override (TRACE, DEBUG, INFO, WARNING)."),
]


@app.command()
def run(config: CONFIG_OPT = "pool-manager.yaml", log_level: LOG_LEVEL_OPT = None):
    """Run the pool manager daemon."""
    cfg = Config.from_file(Path(config))

    level = log_level or cfg.log_level
    setup_logging(level, log_mode=cfg.log_mode, log_file=cfg.log_file)
    logger.info("Loading config from {}", config)
    logger.debug(
        "Config: poll_interval={} min={} max={} batch={} backend={} scheduler={}",
        cfg.poll_interval,
        cfg.scaling.min_workers,
        cfg.scaling.max_workers,
        cfg.scaling.batch_size,
        cfg.work_queue.backend,
        cfg.scheduler.backend,
    )

    try:
        wq = _make_work_queue(cfg)
        sched = _make_scheduler(cfg)
    except ValueError as e:
        logger.error("Configuration error: {}", e)
        raise typer.Exit(1)

    pm = PoolManager(config=cfg, work_queue=wq, scheduler=sched)
    try:
        pm.run()
    except KeyboardInterrupt:
        logger.info("Interrupted")


@app.command()
def tui(config: CONFIG_OPT = "pool-manager.yaml", log_level: LOG_LEVEL_OPT = None):
    """Run the TUI dashboard."""
    run_tui(config)


@app.command()
def test_strategy(
    json_file: Annotated[Path, typer.Argument(help="Path to condor_q -json output file.")],
    config: CONFIG_OPT = "pool-manager.yaml",
    log_level: LOG_LEVEL_OPT = None,
    running: Annotated[
        int | None, typer.Option("-r", "--running", help="Current number of running workers.")
    ] = None,
    running_type: Annotated[
        list[str] | None,
        typer.Option(
            "-rt",
            "--running-type",
            help="Running count per node type (repeatable, e.g. -rt small=3 -rt large=2).",
        ),
    ] = None,
    strategy: Annotated[
        str | None,
        typer.Option(
            "-s",
            "--strategy",
            help="Override strategy (high-throughput or runtime-packing).",
        ),
    ] = None,
    max_walltime: Annotated[
        float | None, typer.Option("--max-walltime", help="Override max_walltime_minutes.")
    ] = None,
    runtime_buffer: Annotated[
        float | None, typer.Option("--runtime-buffer", help="Override runtime_buffer.")
    ] = None,
):
    """Test placement strategy with condor_q -json output."""
    setup_logging(log_level or "WARNING")

    cfg = Config.from_file(Path(config))
    ncs = cfg.scheduler.node_configs
    policy = cfg.scaling

    strategy_name = strategy or policy.strategy
    max_walltime_val = max_walltime or policy.max_walltime_minutes
    runtime_buffer_val = runtime_buffer or policy.runtime_buffer

    raw = json.loads(Path(json_file).read_text())

    if not isinstance(raw, list):
        typer.echo("Error: JSON file must contain a list of job classads", err=True)
        raise typer.Exit(1)

    tasks = []
    for job in raw:
        job = {k.lower(): v for k, v in job.items()}
        tasks.append(
            TaskResources(
                cpus=float(job.get("requestcpus", 1)),
                memory_mb=int(job.get("requestmemory", 1024)),
                gpus=int(job.get("requestgpus", 0)),
                runtime_minutes=float(job.get("runtime_minutes", 0)),
            )
        )

    planner = make_placement_strategy(
        strategy=strategy_name,
        node_configs=ncs if ncs else None,
        task_resources=policy.task_resources,
        batch_size=policy.batch_size,
        max_workers=policy.max_workers,
        min_workers=policy.min_workers,
        max_walltime_minutes=max_walltime_val,
        runtime_buffer=runtime_buffer_val,
    )

    placements = planner.plan_for_tasks(tasks)
    target = max(
        planner._min_workers,
        min(planner._max_workers, sum(p.count for p in placements)),
    )

    running_total = running
    running_per_type: dict[str, int] = {}
    if running_type:
        for rt in running_type:
            if "=" not in rt:
                typer.echo(f"Error: --running-type must be TYPE=COUNT, got '{rt}'", err=True)
                raise typer.Exit(1)
            name, count_str = rt.split("=", 1)
            try:
                running_per_type[name] = int(count_str)
            except ValueError:
                typer.echo(f"Error: invalid count for --running-type '{rt}'", err=True)
                raise typer.Exit(1)

    if running_total is None and running_per_type:
        running_total = sum(running_per_type.values())

    typer.echo(f"Strategy: {strategy_name}")
    typer.echo(f"Tasks: {len(tasks)}")
    nc_list = ", ".join(n.name for n in ncs) if ncs else "none"
    typer.echo(f"Node configs: {len(ncs)} ({nc_list})")
    target_info = ""
    if ncs:
        target_info = f" (max={policy.max_workers}, min={policy.min_workers})"
    typer.echo(f"Target workers: {target}{target_info}")

    if strategy_name == "runtime-packing":
        runtimes = [t.runtime_minutes for t in tasks if t.runtime_minutes > 0]
        if runtimes:
            typer.echo(
                f"Task runtimes: min={min(runtimes):.0f}m max={max(runtimes):.0f}m "
                f"avg={sum(runtimes) / len(runtimes):.0f}m"
            )
        else:
            typer.echo(
                f"Task runtimes: none reported (using max_walltime={max_walltime_val}m as default)"
            )
        typer.echo(f"Max walltime: {max_walltime_val}m  Buffer: {runtime_buffer_val * 100:.0f}%")

    if running_total is not None:
        delta = target - running_total
        if delta > 0:
            typer.echo(f"Current workers: {running_total}  (add {delta})")
        elif delta < 0:
            typer.echo(f"Current workers: {running_total}  (remove {-delta})")
        else:
            typer.echo(f"Current workers: {running_total}  (no change)")

    if running_per_type and placements:
        typer.echo("")
        typer.echo("Per-type scaling:")
        desired_counts: dict[str, int] = {}
        for p in placements:
            desired_counts[p.node_config.name] = desired_counts.get(p.node_config.name, 0) + p.count
        all_types = sorted(set(list(running_per_type.keys()) + list(desired_counts.keys())))
        for t in all_types:
            cur = running_per_type.get(t, 0)
            des = desired_counts.get(t, 0)
            if cur < des:
                typer.echo(f"  {t}: {cur} -> {des}  (+{des - cur})")
            elif cur > des:
                typer.echo(f"  {t}: {cur} -> {des}  (-{cur - des})")
            else:
                typer.echo(f"  {t}: {cur} -> {des}  (no change)")

    typer.echo("")

    if not placements:
        typer.echo("No placement needed")
        return

    typer.echo("Placement plan:")
    total = 0
    for p in placements:
        nc = p.node_config
        detail = ""
        if ncs:
            detail = f" (cpus={nc.cpus} mem={nc.memory_mb}MB gpus={nc.gpus})"
        walltime_info = f" walltime={p.walltime}" if p.walltime else ""
        typer.echo(f"  {nc.name} x {p.count}{detail}{walltime_info}")
        total += p.count

    typer.echo(f"\nTotal nodes: {total}")
    if tasks:
        msg = f"Total tasks placed: {len(tasks)}"
        if total > 0:
            msg += f" ({len(tasks) // total} avg tasks/node)"
        typer.echo(msg)

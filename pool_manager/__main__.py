import json
import sys
from pathlib import Path

import typer
from loguru import logger

from pool_manager.config import Config
from pool_manager.log import setup_logging
from pool_manager.manager import PoolManager, _make_scheduler, _make_work_queue
from pool_manager.placement import PlacementPlanner, TaskResources, resolve_placement_strategy
from pool_manager.tui import run_tui

app = typer.Typer(
    name="pool-manager",
    help="HTCondor → HPC scheduler pool manager",
    no_args_is_help=True,
)


def _common_config(config: str, log_level: str | None) -> Config:
    cfg = Config.from_file(Path(config))
    level = log_level or cfg.log_level
    setup_logging(level, log_mode=cfg.log_mode, log_file=cfg.log_file)
    logger.info("Loading config from {}", config)
    return cfg


@app.command()
def run(
    config: str = typer.Option("pool-manager.yaml", "-c", "--config", help="Path to config file"),
    log_level: str | None = typer.Option(
        None, "--log-level", help="Log level override (TRACE, DEBUG, INFO, WARNING)"
    ),
):
    cfg = _common_config(config, log_level)
    try:
        wq = _make_work_queue(cfg)
        sched = _make_scheduler(cfg)
    except ValueError as e:
        logger.error("Configuration error: {}", e)
        sys.exit(1)

    pm = PoolManager(config=cfg, work_queue=wq, scheduler=sched)
    try:
        pm.run()
    except KeyboardInterrupt:
        logger.info("Interrupted")


@app.command()
def tui(
    config: str = typer.Option("pool-manager.yaml", "-c", "--config", help="Path to config file"),
):
    Config.from_file(Path(config))
    logger.remove()
    run_tui(config)


def _parse_running_types(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for rt in values:
        if "=" not in rt:
            typer.echo(f"Error: --running-type must be TYPE=COUNT, got '{rt}'", err=True)
            raise typer.Exit(1)
        name, count_str = rt.split("=", 1)
        try:
            result[name] = int(count_str)
        except ValueError:
            typer.echo(f"Error: invalid count for --running-type '{rt}'", err=True)
            raise typer.Exit(1)
    return result


@app.command()
def test_strategy(
    json_file: Path = typer.Argument(
        ..., help="Path to JSON file containing condor_q -json output"
    ),
    config: str = typer.Option("pool-manager.yaml", "-c", "--config", help="Path to config file"),
    log_level: str | None = typer.Option(
        None, "--log-level", help="Log level override (TRACE, DEBUG, INFO, WARNING)"
    ),
    running: int | None = typer.Option(
        None, "-r", "--running", help="Current number of running workers"
    ),
    running_type: list[str] = typer.Option(
        [],
        "-rt",
        "--running-type",
        help="Current running count per node type (repeatable, e.g. -rt small=3 -rt large=2)",
    ),
):
    _common_config(config, log_level)
    cfg = Config.from_file(Path(config))
    ncs = cfg.scheduler.node_configs
    policy = cfg.scaling

    if not json_file.exists():
        typer.echo(f"Error: JSON file not found: {json_file}", err=True)
        raise typer.Exit(1)
    raw = json.loads(json_file.read_text())

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
                runtime_minutes=int(job.get("runtime_minutes", 0)),
            )
        )

    ncs, runtime_aware = resolve_placement_strategy(cfg.scheduler.placement_strategy, ncs)
    planner = PlacementPlanner(
        node_configs=ncs,
        task_resources=policy.task_resources,
        batch_size=policy.batch_size,
        max_workers=policy.max_workers,
        min_workers=policy.min_workers,
        runtime_aware=runtime_aware,
    )

    placements = planner.plan_for_tasks(tasks)
    target = planner.target_size(len(tasks))

    running_per_type = _parse_running_types(running_type)
    running_total = running
    if running_total is None and running_per_type:
        running_total = sum(running_per_type.values())

    typer.echo(f"Tasks: {len(tasks)}")
    nc_list = ", ".join(n.name for n in ncs) if ncs else "none"
    typer.echo(f"Node configs: {len(ncs or [])} ({nc_list})")
    typer.echo(f"Placement strategy: {cfg.scheduler.placement_strategy}")
    target_info = ""
    if ncs:
        target_info = f" (max={policy.max_workers}, min={policy.min_workers})"
    typer.echo(f"Target workers: {target}{target_info}")

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
        typer.echo(f"  {nc.name} x {p.count}{detail}")
        total += p.count

    typer.echo(f"\nTotal nodes: {total}")
    if tasks:
        msg = f"Total tasks placed: {len(tasks)}"
        if total > 0:
            msg += f" ({len(tasks) // total} avg tasks/node)"
        typer.echo(msg)


def main():
    app()


if __name__ == "__main__":
    main()

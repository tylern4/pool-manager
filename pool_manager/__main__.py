import argparse
import json
import sys
from pathlib import Path

from pool_manager.config import Config
from pool_manager.log import setup_logging
from pool_manager.manager import PoolManager, _make_scheduler, _make_work_queue
from pool_manager.placement import PlacementPlanner, TaskResources


def main():
    parser = argparse.ArgumentParser(description="HTCondor → HPC scheduler pool manager")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand (default: run daemon)")

    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "-c",
        "--config",
        default="pool-manager.yaml",
        help="Path to config file (default: pool-manager.yaml)",
    )
    base_parser.add_argument(
        "--log-level", default=None, help="Log level override (TRACE, DEBUG, INFO, WARNING)"
    )

    run_parser = subparsers.add_parser(
        "run", parents=[base_parser], help="Run the pool manager daemon"
    )
    run_parser.set_defaults(command="run")

    strategy_parser = subparsers.add_parser(
        "test-strategy",
        parents=[base_parser],
        help="Test placement strategy with condor_q -json output",
    )
    strategy_parser.add_argument(
        "json_file",
        help="Path to JSON file containing condor_q -json output",
    )
    strategy_parser.add_argument(
        "--running",
        "-r",
        type=int,
        default=None,
        help="Current number of running workers",
    )
    strategy_parser.add_argument(
        "--running-type",
        "-rt",
        action="append",
        default=[],
        metavar="TYPE=COUNT",
        help=("Current running count per node type (repeatable, e.g. -rt small=3 -rt large=2)"),
    )

    args = parser.parse_args()

    if args.command is None:
        args.command = "run"

    if args.command == "run":
        _run_daemon(args)
    elif args.command == "test-strategy":
        _run_test_strategy(args)


def _run_daemon(args):
    config_path = getattr(args, "config", "pool-manager.yaml")
    config = Config.from_file(Path(config_path))

    level = getattr(args, "log_level", None) or config.log_level
    logger = setup_logging(level, log_mode=config.log_mode, log_file=config.log_file)
    logger.info("Loading config from %s", config_path)
    logger.debug(
        "Config: poll_interval=%s min=%d max=%d batch=%d backend=%s scheduler=%s",
        config.poll_interval,
        config.scaling.min_workers,
        config.scaling.max_workers,
        config.scaling.batch_size,
        config.work_queue.backend,
        config.scheduler.backend,
    )

    try:
        wq = _make_work_queue(config)
        sched = _make_scheduler(config)
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    pm = PoolManager(config=config, work_queue=wq, scheduler=sched)
    try:
        pm.run()
    except KeyboardInterrupt:
        logger.info("Interrupted")


def _run_test_strategy(args):
    level = getattr(args, "log_level", None) or "WARNING"
    setup_logging(level)

    config_path = getattr(args, "config", "pool-manager.yaml")
    config = Config.from_file(Path(config_path))
    ncs = config.scheduler.node_configs
    policy = config.scaling

    with open(args.json_file) as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        print("Error: JSON file must contain a list of job classads", file=sys.stderr)
        sys.exit(1)

    tasks = []
    for job in raw:
        tasks.append(
            TaskResources(
                cpus=float(job.get("RequestCpus", 1) or 1),
                memory_mb=int(job.get("RequestMemory", 1024) or 1024),
                gpus=int(job.get("RequestGpus", 0) or 0),
            )
        )

    planner = PlacementPlanner(
        node_configs=ncs if ncs else None,
        task_resources=policy.task_resources,
        batch_size=policy.batch_size,
        max_workers=policy.max_workers,
        min_workers=policy.min_workers,
    )

    placements = planner.plan_for_tasks(tasks)
    target = planner.target_size(len(tasks))

    running_total = args.running
    running_per_type: dict[str, int] = {}
    for rt in args.running_type:
        if "=" not in rt:
            print(f"Error: --running-type must be TYPE=COUNT, got '{rt}'", file=sys.stderr)
            sys.exit(1)
        name, count_str = rt.split("=", 1)
        try:
            running_per_type[name] = int(count_str)
        except ValueError:
            print(f"Error: invalid count for --running-type '{rt}'", file=sys.stderr)
            sys.exit(1)

    if running_total is None and running_per_type:
        running_total = sum(running_per_type.values())

    print(f"Tasks: {len(tasks)}")
    nc_list = ", ".join(n.name for n in ncs) if ncs else "none"
    print(f"Node configs: {len(ncs)} ({nc_list})")
    target_info = ""
    if ncs:
        target_info = f" (max={policy.max_workers}, min={policy.min_workers})"
    print(f"Target workers: {target}{target_info}")

    if running_total is not None:
        delta = target - running_total
        if delta > 0:
            print(f"Current workers: {running_total}  (add {delta})")
        elif delta < 0:
            print(f"Current workers: {running_total}  (remove {-delta})")
        else:
            print(f"Current workers: {running_total}  (no change)")

    if running_per_type and placements:
        print()
        print("Per-type scaling:")
        desired_counts: dict[str, int] = {}
        for p in placements:
            desired_counts[p.node_config.name] = desired_counts.get(p.node_config.name, 0) + p.count
        all_types = sorted(set(list(running_per_type.keys()) + list(desired_counts.keys())))
        for t in all_types:
            cur = running_per_type.get(t, 0)
            des = desired_counts.get(t, 0)
            if cur < des:
                print(f"  {t}: {cur} -> {des}  (+{des - cur})")
            elif cur > des:
                print(f"  {t}: {cur} -> {des}  (-{cur - des})")
            else:
                print(f"  {t}: {cur} -> {des}  (no change)")

    print()

    if not placements:
        print("No placement needed")
        return

    print("Placement plan:")
    total = 0
    for p in placements:
        nc = p.node_config
        detail = ""
        if ncs:
            detail = f" (cpus={nc.cpus} mem={nc.memory_mb}MB gpus={nc.gpus})"
        print(f"  {nc.name} x {p.count}{detail}")
        total += p.count

    print(f"\nTotal nodes: {total}")
    if tasks:
        print(f"Total tasks placed: {len(tasks)}", end="")
        if total > 0:
            print(f" ({len(tasks) // total} avg tasks/node)", end="")
        print()


if __name__ == "__main__":
    main()

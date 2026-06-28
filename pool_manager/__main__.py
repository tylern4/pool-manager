import argparse
import sys

from pool_manager.config import Config
from pool_manager.log import setup_logging
from pool_manager.manager import PoolManager, _make_scheduler, _make_work_queue


def main():
    parser = argparse.ArgumentParser(description="HTCondor → HPC scheduler pool manager")
    parser.add_argument(
        "-c",
        "--config",
        default="pool-manager.yaml",
        help="Path to config file (default: pool-manager.yaml)",
    )
    parser.add_argument(
        "--log-level", default=None, help="Log level override (TRACE, DEBUG, INFO, WARNING)"
    )
    args = parser.parse_args()

    config = Config.from_file(args.config)

    level = args.log_level or config.log_level
    logger = setup_logging(level)
    logger.info("Loading config from %s", args.config)
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


if __name__ == "__main__":
    main()

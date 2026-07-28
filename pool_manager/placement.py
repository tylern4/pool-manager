from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from loguru import logger

from pool_manager.scheduler.base import NodeConfig


@dataclass
class TaskResources:
    cpus: float = 1.0
    memory_mb: int = 1024
    gpus: int = 0
    runtime_minutes: float = 0.0


@dataclass
class Placement:
    node_config: NodeConfig
    count: int
    submit_args: dict[str, str] | None = None
    walltime: str | None = None


class PlacementStrategy(ABC):
    def __init__(
        self,
        node_configs: list[NodeConfig] | None = None,
        task_resources: TaskResources | None = None,
        batch_size: int = 1,
        max_workers: int = 16,
        min_workers: int = 0,
    ):
        self._node_configs = node_configs or []
        self._task_resources = task_resources or TaskResources()
        self._batch_size = batch_size
        self._max_workers = max_workers
        self._min_workers = min_workers

    @abstractmethod
    def plan(self, idle_count: int) -> list[Placement]: ...

    @abstractmethod
    def plan_for_tasks(
        self,
        tasks: list[TaskResources],
        existing: list[tuple[str, float]] | None = None,
    ) -> list[Placement]: ...

    def target_size(self, idle_count: int) -> int:
        plan = self.plan(idle_count)
        total = sum(p.count for p in plan)
        return max(self._min_workers, min(self._max_workers, total))

    def _plan_no_configs(self, idle_count: int) -> list[Placement]:
        if idle_count > 0:
            desired = (idle_count + self._batch_size - 1) // self._batch_size
            return [Placement(node_config=NodeConfig(name="default"), count=desired)]
        if self._min_workers > 0:
            return [
                Placement(
                    node_config=NodeConfig(name="default"),
                    count=self._min_workers,
                )
            ]
        return []

    def _plan_for_tasks_no_configs(self, tasks: list[TaskResources]) -> list[Placement]:
        if not tasks:
            if self._min_workers > 0:
                return [
                    Placement(
                        node_config=NodeConfig(name="default"),
                        count=self._min_workers,
                    )
                ]
            return []
        desired = (len(tasks) + self._batch_size - 1) // self._batch_size
        desired = max(self._min_workers, min(self._max_workers, desired))
        return [Placement(node_config=NodeConfig(name="default"), count=desired)]

    def _calculate_fit_per_node(self, node_config: NodeConfig) -> int:
        return self._fit_for_task(node_config, self._task_resources)

    @staticmethod
    def _fit_for_task(node_config: NodeConfig, task: TaskResources) -> int:
        fit = node_config.cpus // int(task.cpus) if task.cpus > 0 else 0
        if task.memory_mb > 0:
            fit = min(fit, node_config.memory_mb // task.memory_mb)
        if task.gpus > 0:
            if node_config.gpus <= 0:
                fit = 0
            else:
                fit = min(fit, node_config.gpus // task.gpus)
        return max(fit, 0)

    def _bin_pack_with_configs(
        self, tasks: list[TaskResources]
    ) -> list[tuple[NodeConfig, list[TaskResources]]]:
        sorted_configs = sorted(
            self._node_configs,
            key=lambda n: n.cpus * max(n.memory_mb, 1) * max(n.gpus, 1),
            reverse=True,
        )

        sorted_tasks = sorted(
            tasks,
            key=lambda t: t.cpus * max(t.memory_mb, 1) * max(t.gpus, 1),
            reverse=True,
        )

        nodes: list[tuple[NodeConfig, list[TaskResources]]] = []

        for task in sorted_tasks:
            placed = False

            for nc, task_list in nodes:
                if self._tasks_fit_on_node(nc, task_list + [task]):
                    task_list.append(task)
                    placed = True
                    break

            if placed:
                continue

            if len(nodes) >= self._max_workers:
                logger.trace(
                    "Cannot place all tasks within {} max workers; "
                    "task requiring cpus={} mem={}MB gpus={} unplaced",
                    self._max_workers,
                    task.cpus,
                    task.memory_mb,
                    task.gpus,
                )
                continue

            for nc in sorted_configs:
                if task.gpus == 0 and nc.gpus > 0:
                    continue
                if self._tasks_fit_on_node(nc, [task]):
                    nodes.append((nc, [task]))
                    placed = True
                    break

            if not placed:
                logger.warning(
                    "No node type can fit task requiring cpus={} mem={}MB gpus={}",
                    task.cpus,
                    task.memory_mb,
                    task.gpus,
                )

        return nodes

    @staticmethod
    def _aggregate_placements(
        nodes: list[tuple[NodeConfig, list[TaskResources]]],
    ) -> list[Placement]:
        config_counts: dict[str, int] = {}
        config_map: dict[str, NodeConfig] = {}
        for nc, _ in nodes:
            config_counts[nc.name] = config_counts.get(nc.name, 0) + 1
            config_map[nc.name] = nc

        return [
            Placement(node_config=config_map[name], count=count)
            for name, count in sorted(config_counts.items())
        ]

    @staticmethod
    def _tasks_fit_on_node(node_config: NodeConfig, tasks: list[TaskResources]) -> bool:
        total_cpus = sum(t.cpus for t in tasks)
        total_mem = sum(t.memory_mb for t in tasks)
        total_gpus = sum(t.gpus for t in tasks)

        if total_cpus > node_config.cpus:
            return False
        if total_mem > node_config.memory_mb:
            return False
        if total_gpus > 0 and node_config.gpus <= 0:
            return False
        if total_gpus > node_config.gpus:
            return False
        return True

    def _min_plan(self) -> list[Placement]:
        if not self._node_configs or self._min_workers <= 0:
            return []
        configs = [nc for nc in self._node_configs if nc.gpus == 0] or self._node_configs
        nc = configs[0]
        count = min(self._min_workers, self._max_workers)
        return [Placement(node_config=nc, count=count)]


class HighThroughputPlanner(PlacementStrategy):
    """Scale out to match idle task count (current/default behavior)."""

    def plan(self, idle_count: int) -> list[Placement]:
        if not self._node_configs:
            return self._plan_no_configs(idle_count)

        if idle_count <= 0:
            return self._min_plan()

        configs = [
            nc for nc in self._node_configs if (nc.gpus > 0) == (self._task_resources.gpus > 0)
        ] or self._node_configs

        sorted_configs = sorted(
            configs,
            key=lambda n: n.cpus * max(n.memory_mb, 1) * max(n.gpus, 1),
            reverse=True,
        )

        remaining = idle_count
        placements: list[Placement] = []
        total_nodes = 0

        for nc in sorted_configs:
            if remaining <= 0:
                break
            if total_nodes >= self._max_workers:
                break
            if nc.max_nodes is not None and total_nodes >= nc.max_nodes:
                break

            fit_per_node = self._calculate_fit_per_node(nc)
            if fit_per_node == 0:
                logger.debug(
                    "Node {} cannot fit any task "
                    "(cpus={} mem={}MB gpus={} vs task cpus={} mem={}MB gpus={})",
                    nc.name,
                    nc.cpus,
                    nc.memory_mb,
                    nc.gpus,
                    self._task_resources.cpus,
                    self._task_resources.memory_mb,
                    self._task_resources.gpus,
                )
                continue

            tasks_per_node = (
                fit_per_node if self._node_configs else min(fit_per_node, self._batch_size)
            )
            max_nodes_by_policy = self._max_workers - total_nodes
            if nc.max_nodes is not None:
                max_nodes_by_policy = min(max_nodes_by_policy, nc.max_nodes - total_nodes)
            nodes_needed = min(
                max_nodes_by_policy,
                (remaining + tasks_per_node - 1) // tasks_per_node,
            )

            if nodes_needed <= 0:
                continue

            placements.append(
                Placement(
                    node_config=nc,
                    count=nodes_needed,
                )
            )
            tasks_covered = nodes_needed * tasks_per_node
            remaining -= tasks_covered
            total_nodes += nodes_needed

            logger.debug(
                "Placed {} tasks on {} x {} ({} tasks/node, {} remaining)",
                tasks_covered,
                nodes_needed,
                nc.name,
                tasks_per_node,
                max(0, remaining),
            )

        if remaining > 0:
            logger.trace(
                "Could not place all {} tasks within {} max workers; {} tasks unplaced",
                idle_count,
                self._max_workers,
                remaining,
            )

        if not placements:
            return self._min_plan()

        return placements

    def plan_for_tasks(
        self,
        tasks: list[TaskResources],
        existing: list[tuple[str, float]] | None = None,
    ) -> list[Placement]:
        if not self._node_configs:
            return self._plan_for_tasks_no_configs(tasks)

        if not tasks:
            return self._min_plan()

        nodes = self._bin_pack_with_configs(tasks)
        return self._aggregate_placements(nodes)


class RuntimePackingPlanner(PlacementStrategy):
    """Pack tasks into few nodes with long walltimes to minimize queued jobs."""

    def __init__(
        self,
        node_configs: list[NodeConfig] | None = None,
        task_resources: TaskResources | None = None,
        batch_size: int = 1,
        max_workers: int = 16,
        min_workers: int = 0,
        max_walltime_minutes: float = 1440,
        runtime_buffer: float = 0.1,
    ):
        super().__init__(
            node_configs=node_configs,
            task_resources=task_resources,
            batch_size=batch_size,
            max_workers=max_workers,
            min_workers=min_workers,
        )
        self._max_walltime_minutes = max_walltime_minutes
        self._runtime_buffer = runtime_buffer

    def plan(self, idle_count: int) -> list[Placement]:
        if not self._node_configs:
            return self._plan_no_configs(idle_count)

        if idle_count <= 0:
            return self._min_plan()

        configs = [
            nc for nc in self._node_configs if (nc.gpus > 0) == (self._task_resources.gpus > 0)
        ] or self._node_configs

        sorted_configs = sorted(
            configs,
            key=lambda n: n.cpus * max(n.memory_mb, 1) * max(n.gpus, 1),
            reverse=True,
        )

        remaining = idle_count
        placements: list[Placement] = []
        total_nodes = 0

        for nc in sorted_configs:
            if remaining <= 0:
                break
            if total_nodes >= self._max_workers:
                break
            if nc.max_nodes is not None and total_nodes >= nc.max_nodes:
                break

            fit_per_node = self._calculate_fit_per_node(nc)
            if fit_per_node == 0:
                continue

            node_max_walltime = self._node_max_walltime(nc)
            tasks_per_node = fit_per_node
            total_batches = math.ceil(remaining / tasks_per_node)
            max_runtime = self._task_resources.runtime_minutes or node_max_walltime
            raw_walltime = total_batches * max_runtime * (1 + self._runtime_buffer)

            if raw_walltime <= node_max_walltime:
                nodes_needed = 1
                walltime_str = self._format_walltime(raw_walltime)
            else:
                nodes_needed = min(
                    self._max_workers - total_nodes,
                    math.ceil(raw_walltime / node_max_walltime),
                )
                if nc.max_nodes is not None:
                    nodes_needed = min(nodes_needed, nc.max_nodes - total_nodes)
                batches_per_node = math.ceil(total_batches / nodes_needed)
                walltime_str = self._format_walltime(
                    min(
                        batches_per_node * max_runtime * (1 + self._runtime_buffer),
                        node_max_walltime,
                    )
                )

            if nodes_needed <= 0:
                continue

            placements.append(Placement(node_config=nc, count=nodes_needed, walltime=walltime_str))
            tasks_covered = nodes_needed * tasks_per_node
            remaining -= tasks_covered
            total_nodes += nodes_needed

            logger.debug(
                "Placed {} tasks on {} x {} ({} tasks/node, walltime={}, {} remaining)",
                tasks_covered,
                nodes_needed,
                nc.name,
                tasks_per_node,
                walltime_str,
                max(0, remaining),
            )

        if remaining > 0:
            logger.trace(
                "Could not place all {} tasks within {} max workers; {} tasks unplaced",
                idle_count,
                self._max_workers,
                remaining,
            )

        if not placements:
            return self._min_plan()

        return placements

    def plan_for_tasks(
        self,
        tasks: list[TaskResources],
        existing: list[tuple[str, float]] | None = None,
    ) -> list[Placement]:
        if not self._node_configs:
            return self._plan_for_tasks_no_configs(tasks)

        if not tasks:
            return self._min_plan()

        representative = self._representative_task(tasks)

        existing_tasks = 0
        if existing:
            max_runtime = self._max_runtime_from_tasks(tasks)
            for node_name, remaining_min in existing:
                nc = self._find_node_config(node_name)
                if nc is None:
                    continue
                fit = self._fit_for_task(nc, representative)
                if fit == 0 or remaining_min <= 0:
                    continue
                batches = int(remaining_min / (max_runtime * (1 + self._runtime_buffer)))
                existing_tasks += batches * fit

        remaining_tasks = max(0, len(tasks) - existing_tasks)
        if remaining_tasks == 0:
            return [
                Placement(
                    node_config=self._node_configs[0],
                    count=0,
                    walltime=None,
                )
            ]

        sorted_configs = sorted(
            self._node_configs,
            key=lambda n: n.cpus * max(n.memory_mb, 1) * max(n.gpus, 1),
            reverse=True,
        )

        sorted_tasks = sorted(tasks[:remaining_tasks], key=lambda t: t.runtime_minutes)
        task_idx = 0
        node_walltimes: list[float] = []
        oversized: list[float] = []

        for nc in sorted_configs:
            fit_per_node = self._fit_for_task(nc, representative)
            if fit_per_node == 0:
                continue

            node_max_walltime = self._node_max_walltime(nc)

            while task_idx < len(sorted_tasks):
                node_walltime = 0.0
                node_task_count = 0

                while task_idx < len(sorted_tasks) and node_task_count < fit_per_node:
                    task = sorted_tasks[task_idx]
                    task_walltime = task.runtime_minutes * (1 + self._runtime_buffer)

                    if task_walltime > node_max_walltime:
                        oversized.append(task_walltime)
                        task_idx += 1
                        continue

                    node_walltime = max(node_walltime, task_walltime)
                    node_task_count += 1
                    task_idx += 1

                if node_task_count > 0:
                    node_walltimes.append(node_walltime)

            break

        node_walltimes.extend(oversized)

        if not node_walltimes:
            return self._min_plan()

        nc = sorted_configs[0]
        node_max_walltime = self._node_max_walltime(nc)

        groups: list[list[float]] = []
        for w in node_walltimes:
            if groups and w <= groups[-1][0] * 2:
                groups[-1].append(w)
            else:
                groups.append([w])

        placements: list[Placement] = []
        total_nodes = 0

        for group in groups:
            count = min(len(group), self._max_workers - total_nodes)
            if nc.max_nodes is not None:
                count = min(count, nc.max_nodes - total_nodes)
            if count <= 0:
                break
            avg_walltime = sum(group) / len(group)
            placements.append(
                Placement(
                    node_config=nc,
                    count=count,
                    walltime=self._format_walltime(min(avg_walltime, node_max_walltime)),
                )
            )
            total_nodes += count

        return placements if placements else self._min_plan()

    def _find_node_config(self, name: str) -> NodeConfig | None:
        for nc in self._node_configs:
            if nc.name == name:
                return nc
        return None

    def _node_max_walltime(self, nc: NodeConfig) -> float:
        return nc.max_walltime_minutes or self._max_walltime_minutes

    @staticmethod
    def _representative_task(tasks: list[TaskResources]) -> TaskResources:
        return TaskResources(
            cpus=max(t.cpus for t in tasks),
            memory_mb=max(t.memory_mb for t in tasks),
            gpus=0,
            runtime_minutes=max((t.runtime_minutes for t in tasks), default=0),
        )

    def _max_runtime_from_tasks(self, tasks: list[TaskResources]) -> float:
        runtimes = [t.runtime_minutes for t in tasks if t.runtime_minutes > 0]
        if not runtimes:
            return self._max_walltime_minutes
        return max(runtimes)

    @staticmethod
    def _format_walltime(minutes: float) -> str:
        total_seconds = int(minutes * 60)
        rounded_seconds = math.ceil(total_seconds / 1800) * 1800
        hours = rounded_seconds // 3600
        mins = (rounded_seconds % 3600) // 60
        secs = rounded_seconds % 60
        return f"{hours:02d}:{mins:02d}:{secs:02d}"


def make_placement_strategy(
    strategy: str,
    node_configs: list[NodeConfig] | None = None,
    task_resources: TaskResources | None = None,
    batch_size: int = 1,
    max_workers: int = 16,
    min_workers: int = 0,
    max_walltime_minutes: float = 1440,
    runtime_buffer: float = 0.1,
) -> PlacementStrategy:
    if strategy == "runtime-packing":
        return RuntimePackingPlanner(
            node_configs=node_configs,
            task_resources=task_resources,
            batch_size=batch_size,
            max_workers=max_workers,
            min_workers=min_workers,
            max_walltime_minutes=max_walltime_minutes,
            runtime_buffer=runtime_buffer,
        )
    return HighThroughputPlanner(
        node_configs=node_configs,
        task_resources=task_resources,
        batch_size=batch_size,
        max_workers=max_workers,
        min_workers=min_workers,
    )

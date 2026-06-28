from __future__ import annotations

import logging
from dataclasses import dataclass

from pool_manager.scheduler.base import NodeConfig

log = logging.getLogger("pool_manager.placement")


@dataclass
class TaskResources:
    cpus: float = 1.0
    memory_mb: int = 1024
    gpus: int = 0


@dataclass
class Placement:
    node_config: NodeConfig
    count: int
    submit_args: dict[str, str] | None = None


class PlacementPlanner:
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

    def target_size(self, idle_count: int) -> int:
        if not self._node_configs:
            if idle_count == 0:
                return self._min_workers
            desired = (idle_count + self._batch_size - 1) // self._batch_size
            return max(self._min_workers, min(self._max_workers, desired))

        plan = self.plan(idle_count)
        total = sum(p.count for p in plan)
        return max(self._min_workers, min(self._max_workers, total))

    def plan(self, idle_count: int) -> list[Placement]:
        if not self._node_configs:
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

        if idle_count <= 0:
            return self._min_plan()

        sorted_configs = sorted(
            self._node_configs,
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

            t = self._task_resources
            fit_per_node = nc.cpus // int(t.cpus) if t.cpus > 0 else 0
            if t.memory_mb > 0:
                fit_per_node = min(fit_per_node, nc.memory_mb // t.memory_mb)
            if t.gpus > 0:
                if nc.gpus <= 0:
                    fit_per_node = 0
                else:
                    fit_per_node = min(fit_per_node, nc.gpus // t.gpus)

            fit_per_node = max(fit_per_node, 0)
            if fit_per_node == 0:
                log.debug(
                    "Node %s cannot fit any task "
                    "(cpus=%d mem=%dMB gpus=%d vs task cpus=%.1f mem=%dMB gpus=%d)",
                    nc.name,
                    nc.cpus,
                    nc.memory_mb,
                    nc.gpus,
                    t.cpus,
                    t.memory_mb,
                    t.gpus,
                )
                continue

            tasks_per_node = (
                fit_per_node if self._node_configs else min(fit_per_node, self._batch_size)
            )
            max_nodes_by_policy = self._max_workers - total_nodes
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

            log.debug(
                "Placed %d tasks on %d x %s (%d tasks/node, %d remaining)",
                tasks_covered,
                nodes_needed,
                nc.name,
                tasks_per_node,
                max(0, remaining),
            )

        if remaining > 0:
            log.warning(
                "Could not place all %d tasks within %d max workers; %d tasks unplaced",
                idle_count,
                self._max_workers,
                remaining,
            )

        if not placements:
            return self._min_plan()

        return placements

    def plan_for_tasks(self, tasks: list[TaskResources]) -> list[Placement]:
        if not self._node_configs:
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

        if not tasks:
            return self._min_plan()

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
                log.warning(
                    "Cannot place all tasks within %d max workers; "
                    "task requiring cpus=%.1f mem=%dMB gpus=%d unplaced",
                    self._max_workers,
                    task.cpus,
                    task.memory_mb,
                    task.gpus,
                )
                continue

            for nc in sorted_configs:
                if self._tasks_fit_on_node(nc, [task]):
                    nodes.append((nc, [task]))
                    placed = True
                    break

            if not placed:
                log.warning(
                    "No node type can fit task requiring cpus=%.1f mem=%dMB gpus=%d",
                    task.cpus,
                    task.memory_mb,
                    task.gpus,
                )

        config_counts: dict[str, int] = {}
        config_map: dict[str, NodeConfig] = {}
        for nc, _ in nodes:
            config_counts[nc.name] = config_counts.get(nc.name, 0) + 1
            config_map[nc.name] = nc

        placements = [
            Placement(node_config=config_map[name], count=count)
            for name, count in sorted(config_counts.items())
        ]

        return placements

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
        nc = self._node_configs[0]
        count = min(self._min_workers, self._max_workers)
        return [Placement(node_config=nc, count=count)]

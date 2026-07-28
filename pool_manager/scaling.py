from __future__ import annotations

from dataclasses import dataclass, field

from pool_manager.placement import PlacementStrategy, TaskResources, make_placement_strategy


@dataclass
class ScalingPolicy:
    min_workers: int = 0
    max_workers: int = 16
    batch_size: int = 1
    scale_up_cooldown: float = 30.0
    scale_down_cooldown: float = 60.0
    drain_timeout: float = 120.0
    drain_on_stop: bool = False
    task_resources: TaskResources = field(default_factory=TaskResources)
    strategy: str = "high-throughput"
    max_walltime_minutes: float = 1440
    runtime_buffer: float = 0.1

    def target_size(self, idle_count: int) -> int:
        if idle_count == 0:
            return self.min_workers
        desired = (idle_count + self.batch_size - 1) // self.batch_size
        return max(self.min_workers, min(self.max_workers, desired))

    @property
    def placement_planner(self) -> PlacementStrategy:
        return make_placement_strategy(
            strategy=self.strategy,
            node_configs=None,
            task_resources=self.task_resources,
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            min_workers=self.min_workers,
            max_walltime_minutes=self.max_walltime_minutes,
            runtime_buffer=self.runtime_buffer,
        )

from dataclasses import dataclass


@dataclass
class ScalingPolicy:
    min_workers: int = 0
    max_workers: int = 16
    batch_size: int = 1
    scale_up_cooldown: float = 30.0
    scale_down_cooldown: float = 60.0
    drain_timeout: float = 120.0

    def target_size(self, idle_count: int) -> int:
        if idle_count == 0:
            return self.min_workers
        desired = (idle_count + self.batch_size - 1) // self.batch_size
        return max(self.min_workers, min(self.max_workers, desired))

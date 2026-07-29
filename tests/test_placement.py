from loguru import logger

from pool_manager.placement import NodeConfig, PlacementPlanner, TaskResources


class TestTargetSizeWithoutConfigs:
    def test_returns_min_when_idle_zero(self):
        p = PlacementPlanner(min_workers=0, max_workers=16, batch_size=1)
        assert p.target_size(0) == 0

    def test_returns_min_when_min_is_set(self):
        p = PlacementPlanner(min_workers=2, max_workers=16, batch_size=1)
        assert p.target_size(0) == 2

    def test_one_job_one_worker(self):
        p = PlacementPlanner(batch_size=1)
        assert p.target_size(1) == 1

    def test_batch_size_groups_jobs(self):
        p = PlacementPlanner(batch_size=5)
        assert p.target_size(6) == 2
        assert p.target_size(10) == 2
        assert p.target_size(11) == 3

    def test_capped_by_max(self):
        p = PlacementPlanner(max_workers=3, batch_size=1)
        assert p.target_size(100) == 3


class TestTargetSizeWithConfigs:
    def test_one_small_node_fits_one_task(self):
        nc = [NodeConfig(name="small", cpus=1, memory_mb=1024)]
        p = PlacementPlanner(
            node_configs=nc, task_resources=TaskResources(cpus=1, memory_mb=1024), batch_size=1
        )
        assert p.target_size(1) == 1

    def test_one_large_node_fits_many_tasks(self):
        nc = [NodeConfig(name="big", cpus=16, memory_mb=65536)]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024),
            batch_size=1,
        )
        assert p.target_size(10) == 1  # one big node fits all 10 tasks

    def test_multiple_nodes_needed(self):
        nc = [NodeConfig(name="small", cpus=2, memory_mb=2048)]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024),
            batch_size=1,
        )
        assert p.target_size(5) == 3  # 2 tasks/node, ceil(5/2) = 3 nodes

    def test_capped_by_max_workers(self):
        nc = [NodeConfig(name="small", cpus=2, memory_mb=2048)]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024),
            batch_size=1,
            max_workers=2,
        )
        assert p.target_size(10) == 2

    def test_min_workers_when_idle_zero(self):
        nc = [NodeConfig(name="small", cpus=2, memory_mb=2048)]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024),
            min_workers=1,
            max_workers=16,
        )
        assert p.target_size(0) == 1

    def test_min_workers_when_no_jobs_with_configs(self):
        nc = [NodeConfig(name="small", cpus=2, memory_mb=2048)]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024),
            min_workers=3,
            max_workers=16,
        )
        assert p.target_size(0) == 3


class TestPlan:
    def test_no_configs_returns_simple(self):
        p = PlacementPlanner(min_workers=0, batch_size=2)
        placements = p.plan(3)
        assert len(placements) == 1
        assert placements[0].count == 2

    def test_no_configs_with_zero_idle(self):
        p = PlacementPlanner(min_workers=0)
        assert p.plan(0) == []

    def test_no_configs_with_min_workers(self):
        p = PlacementPlanner(min_workers=3)
        placements = p.plan(0)
        assert len(placements) == 1
        assert placements[0].count == 3

    def test_single_node_type_fits_all(self):
        nc = [NodeConfig(name="big", cpus=16, memory_mb=65536, gpus=0)]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024),
        )
        placements = p.plan(8)
        assert len(placements) == 1
        assert placements[0].node_config.name == "big"
        assert placements[0].count == 1  # 8 tasks fit in 1 big node

    def test_prefers_larger_nodes_to_minimize_count(self):
        nc = [
            NodeConfig(name="small", cpus=2, memory_mb=2048),
            NodeConfig(name="big", cpus=16, memory_mb=65536),
        ]
        p = PlacementPlanner(node_configs=nc, task_resources=TaskResources(cpus=1, memory_mb=1024))
        placements = p.plan(8)
        assert len(placements) >= 1
        assert placements[0].node_config.name == "big"

    def test_uses_smaller_nodes_when_big_cannot_fit_remaining(self):
        nc = [
            NodeConfig(name="big", cpus=16, memory_mb=65536),
            NodeConfig(name="small", cpus=4, memory_mb=4096),
        ]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024),
            max_workers=16,
        )
        # 4 big nodes × 16 = 64 capacity > 50, so all in big (no small)
        placements = p.plan(50)
        assert len(placements) == 1
        assert placements[0].node_config.name == "big"
        assert placements[0].count == 4

    def test_gpu_task_skips_cpu_nodes(self):
        nc = [
            NodeConfig(name="cpu", cpus=16, memory_mb=65536, gpus=0),
            NodeConfig(name="gpu", cpus=4, memory_mb=8192, gpus=4),
        ]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024, gpus=1),
            max_workers=16,
        )
        # CPU is ranked higher but can't fit GPU tasks; falls through to GPU
        placements = p.plan(8)
        assert len(placements) == 1
        assert placements[0].node_config.name == "gpu"

    def test_batch_size_is_auto_when_node_configs_defined(self):
        nc = [NodeConfig(name="huge", cpus=64, memory_mb=262144)]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024),
        )
        placements = p.plan(20)
        # batch_size is ignored when node configs exist; 1 huge node fits all
        assert placements[0].count == 1

    def test_skips_gpu_node_when_no_gpu_tasks(self):
        nc = [
            NodeConfig(name="gpu", cpus=4, memory_mb=8192, gpus=1),
            NodeConfig(name="cpu", cpus=4, memory_mb=8192, gpus=0),
        ]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024, gpus=0),
        )
        placements = p.plan(4)
        assert len(placements) == 1
        assert placements[0].node_config.name == "cpu"  # GPU node skipped for non-GPU tasks

    def test_gpu_task_requires_gpu_node(self):
        nc = [
            NodeConfig(name="cpu", cpus=16, memory_mb=65536, gpus=0),
            NodeConfig(name="gpu", cpus=4, memory_mb=8192, gpus=4),
        ]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024, gpus=1),
        )
        placements = p.plan(8)
        assert len(placements) == 1
        assert placements[0].node_config.name == "gpu"

    def test_plan_with_zero_idle_no_configs(self):
        p = PlacementPlanner(min_workers=0)
        assert p.plan(0) == []

    def test_plan_with_zero_idle_with_configs_no_min(self):
        nc = [NodeConfig(name="small", cpus=2, memory_mb=2048)]
        p = PlacementPlanner(node_configs=nc, min_workers=0)
        assert p.plan(0) == []

    def test_plan_with_zero_idle_with_min(self):
        nc = [NodeConfig(name="small", cpus=2, memory_mb=2048)]
        p = PlacementPlanner(node_configs=nc, min_workers=3, max_workers=16)
        placements = p.plan(0)
        assert len(placements) == 1
        assert placements[0].count == 3


class TestPlanForTasks:
    def test_no_configs_returns_simple(self):
        p = PlacementPlanner(min_workers=0, batch_size=2)
        tasks = [TaskResources(cpus=1, memory_mb=1024)] * 3
        placements = p.plan_for_tasks(tasks)
        assert len(placements) == 1
        assert placements[0].count == 2

    def test_no_configs_with_empty_tasks(self):
        p = PlacementPlanner(min_workers=0)
        assert p.plan_for_tasks([]) == []

    def test_no_configs_with_min_workers_empty_tasks(self):
        p = PlacementPlanner(min_workers=3)
        placements = p.plan_for_tasks([])
        assert len(placements) == 1
        assert placements[0].count == 3

    def test_single_node_type_fits_all(self):
        nc = [NodeConfig(name="big", cpus=16, memory_mb=65536)]
        p = PlacementPlanner(node_configs=nc)
        tasks = [TaskResources(cpus=1, memory_mb=1024)] * 8
        placements = p.plan_for_tasks(tasks)
        assert len(placements) == 1
        assert placements[0].node_config.name == "big"
        assert placements[0].count == 1

    def test_heterogeneous_tasks_packed_into_min_nodes(self):
        nc = [NodeConfig(name="big", cpus=8, memory_mb=16384)]
        p = PlacementPlanner(node_configs=nc)
        tasks = [
            TaskResources(cpus=4, memory_mb=4096),
            TaskResources(cpus=4, memory_mb=4096),
            TaskResources(cpus=4, memory_mb=4096),
        ]
        placements = p.plan_for_tasks(tasks)
        assert len(placements) == 1
        assert placements[0].count == 2

    def test_gpu_task_assigned_to_gpu_node(self):
        nc = [
            NodeConfig(name="cpu", cpus=16, memory_mb=65536, gpus=0),
            NodeConfig(name="gpu", cpus=4, memory_mb=8192, gpus=4),
        ]
        p = PlacementPlanner(node_configs=nc)
        tasks = [
            TaskResources(cpus=1, memory_mb=1024, gpus=1),
            TaskResources(cpus=1, memory_mb=1024, gpus=0),
        ]
        placements = p.plan_for_tasks(tasks)
        assert len(placements) == 1
        assert placements[0].node_config.name == "gpu"

    def test_many_gpu_tasks_need_multiple_gpu_nodes(self):
        nc = [
            NodeConfig(name="cpu", cpus=16, memory_mb=65536, gpus=0),
            NodeConfig(name="gpu", cpus=4, memory_mb=8192, gpus=1),
        ]
        p = PlacementPlanner(node_configs=nc)
        tasks = [
            TaskResources(cpus=1, memory_mb=1024, gpus=1),
            TaskResources(cpus=1, memory_mb=1024, gpus=1),
            TaskResources(cpus=1, memory_mb=1024, gpus=0),
        ]
        placements = p.plan_for_tasks(tasks)
        assert len(placements) == 1
        assert placements[0].node_config.name == "gpu"
        assert placements[0].count == 2  # 2 GPU nodes + CPU task on first GPU node

    def test_large_tasks_fill_small_node(self):
        nc = [NodeConfig(name="small", cpus=4, memory_mb=4096)]
        p = PlacementPlanner(node_configs=nc)
        tasks = [
            TaskResources(cpus=4, memory_mb=4096),
            TaskResources(cpus=4, memory_mb=4096),
        ]
        placements = p.plan_for_tasks(tasks)
        assert len(placements) == 1
        assert placements[0].count == 2

    def test_empty_tasks_with_configs_no_min(self):
        nc = [NodeConfig(name="small", cpus=2, memory_mb=2048)]
        p = PlacementPlanner(node_configs=nc, min_workers=0)
        assert p.plan_for_tasks([]) == []

    def test_empty_tasks_with_configs_and_min(self):
        nc = [NodeConfig(name="small", cpus=2, memory_mb=2048)]
        p = PlacementPlanner(node_configs=nc, min_workers=2, max_workers=16)
        placements = p.plan_for_tasks([])
        assert len(placements) == 1
        assert placements[0].count == 2

    def test_tasks_fit_exactly_in_one_node(self):
        nc = [NodeConfig(name="med", cpus=8, memory_mb=16384)]
        p = PlacementPlanner(node_configs=nc)
        tasks = [
            TaskResources(cpus=4, memory_mb=8192),
            TaskResources(cpus=4, memory_mb=8192),
        ]
        placements = p.plan_for_tasks(tasks)
        assert len(placements) == 1
        assert placements[0].count == 1

    def test_limited_by_max_workers(self):
        nc = [NodeConfig(name="small", cpus=4, memory_mb=4096)]
        p = PlacementPlanner(node_configs=nc, max_workers=2)
        tasks = [
            TaskResources(cpus=1, memory_mb=1024),
            TaskResources(cpus=1, memory_mb=1024),
            TaskResources(cpus=1, memory_mb=1024),
        ]
        placements = p.plan_for_tasks(tasks)
        total = sum(pl.count for pl in placements)
        assert total <= 2


class TestPlanEdgeCases:
    def test_plan_max_workers_limit_hit(self):
        nc = [
            NodeConfig(name="small", cpus=1, memory_mb=1024),
            NodeConfig(name="big", cpus=16, memory_mb=65536),
        ]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024),
            max_workers=2,
        )
        placements = p.plan(10)
        total = sum(pl.count for pl in placements)
        assert total <= 2

    def test_plan_nodes_needed_zero_skipped(self):
        nc = [NodeConfig(name="small", cpus=1, memory_mb=1024)]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024),
            max_workers=0,
        )
        placements = p.plan(5)
        assert placements == []

    def test_plan_node_cannot_fit_any_task(self):
        nc = [NodeConfig(name="tiny", cpus=1, memory_mb=512)]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=2, memory_mb=4096),
        )
        placements = p.plan(1)
        assert placements == []  # falls to _min_plan which is empty

    def test_plan_gpu_task_no_gpu_node_skips(self):
        nc = [NodeConfig(name="gpu", cpus=4, memory_mb=8192, gpus=0)]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024, gpus=1),
        )
        placements = p.plan(1)
        assert placements == []

    def test_plan_no_placements_falls_to_min_plan(self):
        nc = [NodeConfig(name="small", cpus=2, memory_mb=2048)]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=8, memory_mb=16384),
            min_workers=2,
        )
        placements = p.plan(1)
        # Tasks can't fit, so falls to _min_plan with 2 min workers
        assert len(placements) == 1
        assert placements[0].count == 2

    def test_plan_task_too_large_for_gpu_node_skips(self):
        nc = [
            NodeConfig(name="gpu", cpus=1, memory_mb=1024, gpus=1),
        ]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=2, memory_mb=2048, gpus=1),
        )
        placements = p.plan(1)
        assert placements == []

    def test_plan_remaining_tasks_trace(self):
        nc = [NodeConfig(name="small", cpus=1, memory_mb=1024)]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024),
            max_workers=1,
        )
        records = []
        handler_id = logger.add(lambda m: records.append(m), level="TRACE")
        try:
            p.plan(10)
        finally:
            logger.remove(handler_id)
        assert any("Could not place all" in m for m in records)

    def test_plan_for_tasks_skips_gpu_node_for_cpu_task(self):
        nc = [
            NodeConfig(name="gpu", cpus=4, memory_mb=8192, gpus=1),
            NodeConfig(name="cpu", cpus=4, memory_mb=8192, gpus=0),
        ]
        p = PlacementPlanner(node_configs=nc)
        tasks = [TaskResources(cpus=1, memory_mb=1024, gpus=0)]
        placements = p.plan_for_tasks(tasks)
        assert len(placements) == 1
        assert placements[0].node_config.name == "cpu"

    def test_plan_for_tasks_cannot_place_any(self):
        nc = [NodeConfig(name="small", cpus=1, memory_mb=512)]
        p = PlacementPlanner(node_configs=nc)
        tasks = [TaskResources(cpus=8, memory_mb=16384)]
        logger.remove()
        placements = p.plan_for_tasks(tasks)
        assert placements == []

    def test_plan_for_tasks_unplaceable_due_to_gpu(self):
        nc = [NodeConfig(name="cpu", cpus=16, memory_mb=65536, gpus=0)]
        p = PlacementPlanner(node_configs=nc)
        tasks = [TaskResources(cpus=1, memory_mb=1024, gpus=1)]
        logger.remove()
        placements = p.plan_for_tasks(tasks)
        assert placements == []

    def test_plan_for_tasks_max_workers_exceeded(self):
        nc = [NodeConfig(name="big", cpus=16, memory_mb=65536)]
        p = PlacementPlanner(node_configs=nc, max_workers=1)
        tasks = [TaskResources(cpus=1, memory_mb=1024)] * 20
        logger.remove()
        placements = p.plan_for_tasks(tasks)
        total = sum(pl.count for pl in placements)
        assert total <= 1

    def test_tasks_fit_on_node_mem_overflow(self):
        nc = NodeConfig(name="small", cpus=4, memory_mb=2048)
        tasks = [TaskResources(cpus=1, memory_mb=4096)]
        assert not PlacementPlanner._tasks_fit_on_node(nc, tasks)

    def test_tasks_fit_on_node_gpu_overflow(self):
        nc = NodeConfig(name="small", cpus=4, memory_mb=8192, gpus=1)
        tasks = [TaskResources(cpus=1, memory_mb=1024, gpus=2)]
        assert not PlacementPlanner._tasks_fit_on_node(nc, tasks)

    def test_tasks_fit_on_node_no_gpu_on_node(self):
        nc = NodeConfig(name="small", cpus=4, memory_mb=8192, gpus=0)
        tasks = [TaskResources(cpus=1, memory_mb=1024, gpus=1)]
        assert not PlacementPlanner._tasks_fit_on_node(nc, tasks)

    def test_min_plan_with_all_gpu_configs_uses_first(self):
        nc = [NodeConfig(name="gpu1", cpus=4, memory_mb=8192, gpus=1)]
        p = PlacementPlanner(node_configs=nc, min_workers=2, max_workers=16)
        placements = p._min_plan()
        assert len(placements) == 1
        assert placements[0].node_config.name == "gpu1"

    def test_plan_for_tasks_empty_with_min_workers_and_gpu_configs(self):
        nc = [NodeConfig(name="gpu1", cpus=4, memory_mb=8192, gpus=1)]
        p = PlacementPlanner(node_configs=nc, min_workers=2, max_workers=16)
        placements = p.plan_for_tasks([])
        assert len(placements) == 1
        assert placements[0].node_config.name == "gpu1"

    def test_plan_for_tasks_cpu_only_on_gpu_config(self):
        nc = [NodeConfig(name="gpu1", cpus=4, memory_mb=8192, gpus=1)]
        p = PlacementPlanner(node_configs=nc)
        tasks = [TaskResources(cpus=1, memory_mb=1024, gpus=0)]
        # GPU-only node is skipped for CPU tasks in plan_for_tasks (line 214)
        placements = p.plan_for_tasks(tasks)
        assert placements == []


class TestPlanRuntimeAware:
    def test_runtime_fits_within_node_limit(self):
        nc = [NodeConfig(name="short", cpus=4, memory_mb=8192, runtime_minutes=60)]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024, runtime_minutes=30),
        )
        placements = p.plan(4)
        assert len(placements) == 1
        assert placements[0].node_config.name == "short"

    def test_runtime_exceeds_node_limit_skips_config(self):
        nc = [
            NodeConfig(name="short", cpus=128, memory_mb=262144, runtime_minutes=30),
            NodeConfig(name="long", cpus=128, memory_mb=262144, runtime_minutes=120),
        ]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024, runtime_minutes=60),
        )
        placements = p.plan(4)
        assert len(placements) == 1
        assert placements[0].node_config.name == "long"

    def test_runtime_no_limit_on_node_allows_any_task(self):
        nc = [NodeConfig(name="unlimited", cpus=4, memory_mb=8192, runtime_minutes=0)]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024, runtime_minutes=999),
        )
        placements = p.plan(1)
        assert len(placements) == 1
        assert placements[0].node_config.name == "unlimited"

    def test_plan_prefers_shorter_runtime_nodes_for_sorting(self):
        nc = [
            NodeConfig(name="short", cpus=4, memory_mb=8192, runtime_minutes=30),
            NodeConfig(name="long", cpus=4, memory_mb=8192, runtime_minutes=120),
        ]
        p = PlacementPlanner(
            node_configs=nc,
            task_resources=TaskResources(cpus=1, memory_mb=1024, runtime_minutes=10),
        )
        placements = p.plan(2)
        assert len(placements) == 1
        assert placements[0].node_config.name == "short"

    def test_tasks_fit_on_node_runtime_overflow(self):
        nc = NodeConfig(name="short", cpus=4, memory_mb=8192, runtime_minutes=30)
        tasks = [TaskResources(cpus=1, memory_mb=1024, runtime_minutes=60)]
        assert not PlacementPlanner._tasks_fit_on_node(nc, tasks)

    def test_tasks_fit_on_node_runtime_exact_fit(self):
        nc = NodeConfig(name="short", cpus=4, memory_mb=8192, runtime_minutes=60)
        tasks = [TaskResources(cpus=1, memory_mb=1024, runtime_minutes=60)]
        assert PlacementPlanner._tasks_fit_on_node(nc, tasks)

    def test_tasks_fit_on_node_runtime_mixed_tasks_max_wins(self):
        nc = NodeConfig(name="short", cpus=8, memory_mb=16384, runtime_minutes=60)
        tasks = [
            TaskResources(cpus=1, memory_mb=1024, runtime_minutes=30),
            TaskResources(cpus=1, memory_mb=1024, runtime_minutes=90),
        ]
        assert not PlacementPlanner._tasks_fit_on_node(nc, tasks)

    def test_plan_for_tasks_runtime_exceeds_skips_node(self):
        nc = [
            NodeConfig(name="short", cpus=4, memory_mb=8192, runtime_minutes=30),
            NodeConfig(name="long", cpus=4, memory_mb=8192, runtime_minutes=120),
        ]
        p = PlacementPlanner(node_configs=nc)
        tasks = [TaskResources(cpus=1, memory_mb=1024, runtime_minutes=60)]
        placements = p.plan_for_tasks(tasks)
        assert len(placements) == 1
        assert placements[0].node_config.name == "long"

    def test_plan_for_tasks_runtime_fits_on_first_node(self):
        nc = [
            NodeConfig(name="short", cpus=4, memory_mb=8192, runtime_minutes=60),
            NodeConfig(name="long", cpus=4, memory_mb=8192, runtime_minutes=120),
        ]
        p = PlacementPlanner(node_configs=nc)
        tasks = [TaskResources(cpus=1, memory_mb=1024, runtime_minutes=30)]
        placements = p.plan_for_tasks(tasks)
        assert len(placements) == 1
        assert placements[0].node_config.name == "short"

    def test_plan_for_tasks_runtime_mixed_packing(self):
        nc = [
            NodeConfig(name="short", cpus=4, memory_mb=8192, runtime_minutes=60),
            NodeConfig(name="long", cpus=4, memory_mb=8192, runtime_minutes=120),
        ]
        p = PlacementPlanner(node_configs=nc)
        tasks = [
            TaskResources(cpus=1, memory_mb=1024, runtime_minutes=30),
            TaskResources(cpus=1, memory_mb=1024, runtime_minutes=90),
        ]
        placements = p.plan_for_tasks(tasks)
        assert len(placements) == 2
        names = {pl.node_config.name for pl in placements}
        assert names == {"short", "long"}

    def test_plan_for_tasks_runtime_unplaceable(self):
        nc = [NodeConfig(name="short", cpus=4, memory_mb=8192, runtime_minutes=30)]
        p = PlacementPlanner(node_configs=nc)
        tasks = [TaskResources(cpus=1, memory_mb=1024, runtime_minutes=60)]
        logger.remove()
        placements = p.plan_for_tasks(tasks)
        assert placements == []

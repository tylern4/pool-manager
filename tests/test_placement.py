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

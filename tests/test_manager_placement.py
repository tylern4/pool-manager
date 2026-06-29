from unittest.mock import MagicMock

import pytest

from pool_manager.config import Config, SchedulerConfig, WorkQueueConfig
from pool_manager.manager import PoolManager
from pool_manager.placement import NodeConfig, Placement, TaskResources
from pool_manager.scaling import ScalingPolicy
from pool_manager.scheduler.base import JobInfo, JobState


@pytest.fixture
def mock_scheduler():
    sched = MagicMock()
    sched.list_active.return_value = []
    sched.submit.return_value = "42"
    sched.name.return_value = "test_sched"
    return sched


@pytest.fixture
def mock_work_queue():
    wq = MagicMock()
    wq.list_idle.return_value = [
        TaskResources(cpus=1, memory_mb=1024, gpus=0),
        TaskResources(cpus=1, memory_mb=1024, gpus=0),
        TaskResources(cpus=1, memory_mb=1024, gpus=0),
        TaskResources(cpus=1, memory_mb=1024, gpus=0),
        TaskResources(cpus=1, memory_mb=1024, gpus=0),
    ]
    wq.name.return_value = "test_queue"
    return wq


def make_config(node_configs=None, **overrides):
    sc = overrides.get("scaling", {})
    policy = ScalingPolicy(
        min_workers=sc.get("min_workers", 0),
        max_workers=sc.get("max_workers", 16),
        batch_size=sc.get("batch_size", 1),
    )
    return Config(
        poll_interval=0.1,
        log_level="DEBUG",
        work_queue=WorkQueueConfig(backend="condor_subprocess"),
        scheduler=SchedulerConfig(
            backend="local_subprocess",
            worker_script="/fake/worker.sh",
            node_configs=node_configs or [],
            submit_args={"account": "myproject"},
        ),
        scaling=policy,
    )


class TestManagerPlacement:
    def test_no_node_configs_uses_simple_scaling(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        assert mgr._has_node_configs is False
        assert mgr._planner._node_configs == []

    def test_with_node_configs_creates_planner(self, mock_scheduler, mock_work_queue):
        ncs = [NodeConfig(name="small", cpus=4, memory_mb=8192)]
        cfg = make_config(node_configs=ncs)
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        assert mgr._has_node_configs is True
        assert len(mgr._planner._node_configs) == 1

    def test_start_workers_simple_no_configs(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        plan = [Placement(node_config=NodeConfig(name="default"), count=3)]
        mgr._start_workers(plan, 3)
        assert mock_scheduler.submit.call_count == 3
        for call in mock_scheduler.submit.call_args_list:
            args, kwargs = call
            assert args[0] == "/fake/worker.sh"
            assert args[1] == {"account": "myproject"}

    def test_start_workers_from_plan_adds_resource_args(self, mock_scheduler, mock_work_queue):
        ncs = [NodeConfig(name="big", cpus=16, memory_mb=65536, gpus=0)]
        cfg = make_config(node_configs=ncs)
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        plan = mgr._planner.plan_for_tasks([TaskResources(cpus=1, memory_mb=1024)] * 8)
        mgr._start_workers(plan, 1)
        assert mock_scheduler.submit.call_count == 1
        call = mock_scheduler.submit.call_args
        _script, submit_args = call[0]
        assert submit_args["cpus-per-task"] == "16"
        assert submit_args["mem"] == "65536M"
        assert "gpus" not in submit_args

    def test_start_workers_from_plan_with_gpus(self, mock_scheduler, mock_work_queue):
        ncs = [NodeConfig(name="gpu", cpus=4, memory_mb=8192, gpus=4)]
        cfg = make_config(node_configs=ncs)
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        plan = mgr._planner.plan_for_tasks([TaskResources(cpus=1, memory_mb=1024)] * 4)
        mgr._start_workers(plan, 1)
        call = mock_scheduler.submit.call_args
        _script, submit_args = call[0]
        assert submit_args["gpus"] == "4"

    def test_tick_uses_target_from_planner(self, mock_scheduler, mock_work_queue):
        ncs = [NodeConfig(name="small", cpus=4, memory_mb=8192)]
        cfg = make_config(node_configs=ncs)
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tick()
        assert mock_scheduler.submit.called

    def test_scale_up_distributes_across_node_types(self, mock_scheduler, mock_work_queue):
        ncs = [
            NodeConfig(name="big", cpus=4, memory_mb=4096),
            NodeConfig(name="small", cpus=2, memory_mb=2048),
        ]
        cfg = make_config(
            node_configs=ncs,
            scaling={"batch_size": 1, "max_workers": 16, "min_workers": 0},
        )
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        tasks = [TaskResources(cpus=1, memory_mb=1024)] * 9
        # 9 idle tasks: big fits 4 per node → 3 ceil(9/4) big nodes
        plan = mgr._planner.plan_for_tasks(tasks)
        assert len(plan) == 1
        assert plan[0].node_config.name == "big"
        assert plan[0].count == 3


class TestSignalWorkers:
    def test_tracks_node_assignments_on_start(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mock_scheduler.submit.side_effect = ["101", "102"]
        mgr._start_workers_simple(2)
        assert len(mgr._node_assignments) == 2
        for v in mgr._node_assignments.values():
            assert v == "default"

    def test_signal_workers_simple_drains_oldest(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._tracked["2"] = JobInfo(job_id="2", state=JobState.RUNNING)
        mgr._tracked["3"] = JobInfo(job_id="3", state=JobState.RUNNING)
        mgr._signal_workers(2)
        assert mock_scheduler.signal.call_count == 2
        called_ids = [call[0][0] for call in mock_scheduler.signal.call_args_list]
        assert "1" in called_ids
        assert "2" in called_ids
        assert "3" not in called_ids

    def test_signal_workers_drains_excess_node_type(self, mock_scheduler, mock_work_queue):
        ncs = [NodeConfig(name="small", cpus=4, memory_mb=8192)]
        cfg = make_config(node_configs=ncs)
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._tracked["2"] = JobInfo(job_id="2", state=JobState.RUNNING)
        mgr._tracked["3"] = JobInfo(job_id="3", state=JobState.RUNNING)
        mgr._node_assignments["1"] = "small"
        mgr._node_assignments["2"] = "small"
        mgr._node_assignments["3"] = "small"
        # plan_for_tasks([]) with min_workers=0 returns [] → no nodes desired
        # so all 3 should be drained (excess = 3)
        plan = mgr._planner.plan_for_tasks([])
        mgr._signal_workers(3, plan=plan)
        assert mock_scheduler.signal.call_count == 3

    def test_signal_workers_keeps_required_nodes(self, mock_scheduler, mock_work_queue):
        ncs = [NodeConfig(name="small", cpus=4, memory_mb=8192)]
        cfg = make_config(node_configs=ncs, scaling={"min_workers": 1, "max_workers": 16})
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._tracked["2"] = JobInfo(job_id="2", state=JobState.RUNNING)
        mgr._tracked["3"] = JobInfo(job_id="3", state=JobState.RUNNING)
        mgr._node_assignments["1"] = "small"
        mgr._node_assignments["2"] = "small"
        mgr._node_assignments["3"] = "small"
        # plan_for_tasks([]) with min_workers=1 returns [small x 1]
        plan = mgr._planner.plan_for_tasks([])
        mgr._signal_workers(3, plan=plan)
        # 3 active - 1 desired = 2 excess
        assert mock_scheduler.signal.call_count == 2

    def test_signal_workers_drains_unused_node_type_first(self, mock_scheduler, mock_work_queue):
        ncs = [
            NodeConfig(name="small", cpus=4, memory_mb=8192),
            NodeConfig(name="large", cpus=16, memory_mb=65536),
        ]
        cfg = make_config(node_configs=ncs, scaling={"min_workers": 0})
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._tracked["2"] = JobInfo(job_id="2", state=JobState.RUNNING)
        mgr._tracked["3"] = JobInfo(job_id="3", state=JobState.RUNNING)
        mgr._node_assignments["1"] = "small"
        mgr._node_assignments["2"] = "large"
        mgr._node_assignments["3"] = "large"
        # plan_for_tasks([]) with min=0 returns []
        plan = mgr._planner.plan_for_tasks([])
        mgr._signal_workers(2, plan=plan)
        # both large nodes should be drained first (higher cost, not in plan)
        called_ids = [call[0][0] for call in mock_scheduler.signal.call_args_list]
        assert "2" in called_ids
        assert "3" in called_ids
        assert "1" not in called_ids

    def test_signal_workers_drains_expensive_nodes_first(self, mock_scheduler, mock_work_queue):
        ncs = [
            NodeConfig(name="small", cpus=4, memory_mb=8192),
            NodeConfig(name="large", cpus=16, memory_mb=65536),
        ]
        cfg = make_config(node_configs=ncs, scaling={"min_workers": 0})
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._tracked["2"] = JobInfo(job_id="2", state=JobState.RUNNING)
        mgr._node_assignments["1"] = "large"
        mgr._node_assignments["2"] = "small"
        # plan_for_tasks([]) with min=0 returns []
        plan = mgr._planner.plan_for_tasks([])
        mgr._signal_workers(1, plan=plan)
        assert mock_scheduler.signal.call_count == 1
        called_id = mock_scheduler.signal.call_args[0][0]
        assert called_id == "1"  # large drained first

    def test_signal_workers_cleans_assignments_on_reconcile(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._node_assignments["1"] = "default"
        mgr._tracked["2"] = JobInfo(job_id="2", state=JobState.RUNNING)
        mgr._node_assignments["2"] = "default"
        mock_scheduler.list_active.return_value = [
            JobInfo(job_id="1", state=JobState.RUNNING),
        ]
        mgr._reconcile()
        assert "1" in mgr._node_assignments  # still active
        assert "2" not in mgr._node_assignments  # lost job, cleaned up

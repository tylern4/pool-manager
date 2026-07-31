from unittest.mock import MagicMock, patch

import pytest

from pool_manager.config import Config, SchedulerConfig, WorkQueueConfig
from pool_manager.manager import PoolManager, _make_scheduler, _make_work_queue
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
            backend="slurm_subprocess",
            worker_script=overrides.get("worker_script", "/fake/worker.sh"),
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

    def test_start_workers_simple_no_configs(self, mock_scheduler, mock_work_queue, tmp_path):
        script = tmp_path / "worker.sh"
        script.write_text("#!/bin/bash\n")
        cfg = make_config(worker_script=str(script))
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        plan = [Placement(node_config=NodeConfig(name="default"), count=3)]
        mgr._start_workers(plan, 3)
        assert mock_scheduler.submit.call_count == 3
        for call in mock_scheduler.submit.call_args_list:
            args, kwargs = call
            assert args[0] == str(script)
            assert args[1]["account"] == "myproject"
            assert args[1]["job-name"] == "htcondor_worker_default"

    def test_start_workers_from_plan_adds_resource_args(
        self, mock_scheduler, mock_work_queue, tmp_path
    ):
        script = tmp_path / "worker.sh"
        script.write_text("#!/bin/bash\n")
        ncs = [NodeConfig(name="big", cpus=16, memory_mb=65536, gpus=0)]
        cfg = make_config(node_configs=ncs, worker_script=str(script))
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        plan = mgr._planner.plan_for_tasks([TaskResources(cpus=1, memory_mb=1024)] * 8)
        mgr._start_workers(plan, 1)
        assert mock_scheduler.submit.call_count == 1
        call = mock_scheduler.submit.call_args
        _script, submit_args = call[0]
        assert submit_args["cpus-per-task"] == "16"
        assert submit_args["mem"] == "65536M"
        assert "gpus" not in submit_args

    def test_start_workers_from_plan_with_gpus(self, mock_scheduler, mock_work_queue, tmp_path):
        script = tmp_path / "worker.sh"
        script.write_text("#!/bin/bash\n")
        ncs = [NodeConfig(name="gpu", cpus=4, memory_mb=8192, gpus=4)]
        cfg = make_config(node_configs=ncs, worker_script=str(script))
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        plan = mgr._planner.plan_for_tasks([TaskResources(cpus=1, memory_mb=1024, gpus=1)] * 4)
        mgr._start_workers(plan, 1)
        call = mock_scheduler.submit.call_args
        _script, submit_args = call[0]
        assert submit_args["gpus"] == "4"

    def test_start_workers_from_plan_injects_time_arg(
        self, mock_scheduler, mock_work_queue, tmp_path
    ):
        script = tmp_path / "worker.sh"
        script.write_text("#!/bin/bash\n")
        ncs = [NodeConfig(name="short", cpus=4, memory_mb=8192, runtime_minutes=90)]
        cfg = make_config(node_configs=ncs, worker_script=str(script))
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        plan = mgr._planner.plan_for_tasks([TaskResources(cpus=1, memory_mb=1024)] * 2)
        mgr._start_workers(plan, 1)
        call = mock_scheduler.submit.call_args
        _script, submit_args = call[0]
        assert submit_args["time"] == "01:30:00"

    def test_tick_uses_target_from_planner(self, mock_scheduler, mock_work_queue, tmp_path):
        script = tmp_path / "worker.sh"
        script.write_text("#!/bin/bash\n")
        ncs = [NodeConfig(name="small", cpus=4, memory_mb=8192)]
        cfg = make_config(node_configs=ncs, worker_script=str(script))
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tick()
        assert mock_scheduler.submit.called

    def test_start_workers_from_plan_with_node_submit_args(
        self, mock_scheduler, mock_work_queue, tmp_path
    ):
        script = tmp_path / "worker.sh"
        script.write_text("#!/bin/bash\n")
        ncs = [
            NodeConfig(
                name="big",
                cpus=16,
                memory_mb=65536,
                gpus=0,
                submit_args={"partition": "highmem"},
            ),
        ]
        cfg = make_config(node_configs=ncs, worker_script=str(script))
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        plan = mgr._planner.plan_for_tasks([TaskResources(cpus=1, memory_mb=1024)] * 4)
        mgr._start_workers(plan, 1)
        call = mock_scheduler.submit.call_args
        _script, submit_args = call[0]
        assert submit_args["partition"] == "highmem"
        assert submit_args["account"] == "myproject"
        assert submit_args["cpus-per-task"] == "16"

    def test_start_workers_from_plan_node_submit_args_per_type(
        self, mock_scheduler, mock_work_queue, tmp_path
    ):
        script = tmp_path / "worker.sh"
        script.write_text("#!/bin/bash\n")
        ncs = [
            NodeConfig(
                name="gpu",
                cpus=4,
                memory_mb=8192,
                gpus=4,
                submit_args={"partition": "gpu", "qos": "high"},
            ),
            NodeConfig(name="cpu", cpus=16, memory_mb=65536, gpus=0),
        ]
        cfg = make_config(node_configs=ncs, worker_script=str(script))
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._planner._task_resources = TaskResources(cpus=1, memory_mb=1024, gpus=1)
        plan = mgr._planner.plan_for_tasks([TaskResources(cpus=1, memory_mb=1024, gpus=1)] * 4)
        mgr._start_workers(plan, 2)
        assert mock_scheduler.submit.call_count == 1
        call = mock_scheduler.submit.call_args
        _script, submit_args = call[0]
        assert submit_args["partition"] == "gpu"
        assert submit_args["qos"] == "high"
        assert submit_args["account"] == "myproject"
        assert submit_args["gpus"] == "4"

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
    def test_tracks_node_assignments_on_start(self, mock_scheduler, mock_work_queue, tmp_path):
        script = tmp_path / "worker.sh"
        script.write_text("#!/bin/bash\n")
        cfg = make_config(worker_script=str(script))
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


class TestManagerExtended:
    def test_recover_state(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mock_scheduler.list_active.return_value = [
            JobInfo(job_id="101", state=JobState.RUNNING, job_name="htcondor_worker_default"),
            JobInfo(job_id="102", state=JobState.RUNNING, job_name="htcondor_worker_small"),
        ]
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._recover_state()
        assert "101" in mgr._tracked
        assert "102" in mgr._tracked
        assert mgr._node_assignments["101"] == "default"
        assert mgr._node_assignments["102"] == "small"

    def test_reconcile_new_job(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mock_scheduler.list_active.return_value = [
            JobInfo(job_id="42", state=JobState.RUNNING, job_name="htcondor_worker_default"),
        ]
        mgr._reconcile()
        assert "42" in mgr._tracked
        assert mgr._tracked["42"].state == JobState.RUNNING

    def test_reconcile_state_change(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.PENDING)
        mock_scheduler.list_active.return_value = [
            JobInfo(job_id="1", state=JobState.RUNNING),
        ]
        mgr._reconcile()
        assert mgr._tracked["1"].state == JobState.RUNNING

    def test_reconcile_lost_job(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mock_scheduler.list_active.return_value = []
        mgr._reconcile()
        assert mgr._tracked["1"].state == JobState.EXITED
        assert "1" not in mgr._node_assignments

    def test_reconcile_lost_draining_job(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.DRAINING)
        mock_scheduler.list_active.return_value = []
        mgr._reconcile()
        assert mgr._tracked["1"].state == JobState.EXITED

    def test_scale_up_cooldown_active(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        import time

        mgr._last_scale_up = time.monotonic() - 1.0  # 1 second ago, cooldown=30
        plan = [Placement(node_config=NodeConfig(name="default"), count=3)]
        mgr._scale([TaskResources(cpus=1, memory_mb=1024)] * 5, plan, 3)
        assert not mock_scheduler.submit.called

    def test_scale_down_starts_cooldown(self, mock_scheduler, mock_work_queue):
        cfg = make_config(scaling={"min_workers": 0, "max_workers": 16, "scale_down_cooldown": 60})
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._tracked["2"] = JobInfo(job_id="2", state=JobState.RUNNING)
        mgr._node_assignments["1"] = "default"
        mgr._node_assignments["2"] = "default"
        plan = [Placement(node_config=NodeConfig(name="default"), count=0)]
        mgr._scale([], plan, 0)
        assert mgr._drain_start is not None
        assert not mock_scheduler.signal.called

    def test_scale_down_within_cooldown(self, mock_scheduler, mock_work_queue):
        cfg = make_config(scaling={"min_workers": 0, "scale_down_cooldown": 60})
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._node_assignments["1"] = "default"
        import time

        mgr._drain_start = time.monotonic() + 99999.0
        plan = [Placement(node_config=NodeConfig(name="default"), count=0)]
        mgr._scale([], plan, 0)
        assert not mock_scheduler.signal.called

    def test_scale_down_drains(self, mock_scheduler, mock_work_queue):
        cfg = make_config(scaling={"min_workers": 0, "scale_down_cooldown": 0})
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._node_assignments["1"] = "default"
        mgr._drain_start = -1.0
        plan = [Placement(node_config=NodeConfig(name="default"), count=0)]
        mgr._scale([], plan, 0)
        assert mock_scheduler.signal.called

    def test_force_cancel_draining(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.DRAINING)
        mgr._force_cancel_draining()
        assert mock_scheduler.cancel.called
        assert mgr._tracked["1"].state == JobState.EXITED

    def test_drain_all_no_active(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._drain_all()

    def test_active_count(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._tracked["2"] = JobInfo(job_id="2", state=JobState.PENDING)
        mgr._tracked["3"] = JobInfo(job_id="3", state=JobState.DRAINING)
        mgr._tracked["4"] = JobInfo(job_id="4", state=JobState.EXITED)
        assert mgr._active_count() == 3

    def test_draining_count(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.DRAINING)
        mgr._tracked["2"] = JobInfo(job_id="2", state=JobState.RUNNING)
        assert mgr._draining_count() == 1

    def test_start_workers_adjusts_plan(self, mock_scheduler, mock_work_queue, tmp_path):
        script = tmp_path / "worker.sh"
        script.write_text("#!/bin/bash\n")
        ncs = [NodeConfig(name="big", cpus=16, memory_mb=65536)]
        cfg = make_config(node_configs=ncs, worker_script=str(script))
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._node_assignments["1"] = "big"
        plan = [Placement(node_config=ncs[0], count=2)]
        mgr._start_workers(plan, 2)
        assert mock_scheduler.submit.call_count == 1

    def test_start_workers_simple_no_script(self, mock_scheduler, mock_work_queue):
        cfg = make_config(worker_script="")
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._start_workers_simple(3)
        assert not mock_scheduler.submit.called

    def test_start_workers_simple_script_not_found(self, mock_scheduler, mock_work_queue):
        cfg = make_config(worker_script="/nonexistent/worker.sh")
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._start_workers_simple(3)
        assert not mock_scheduler.submit.called

    def test_start_workers_simple_submit_error(self, mock_scheduler, mock_work_queue, tmp_path):
        script = tmp_path / "worker.sh"
        script.write_text("#!/bin/bash\n")
        cfg = make_config(worker_script=str(script))
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mock_scheduler.submit.side_effect = RuntimeError("submit failed")
        mgr._start_workers_simple(1)
        assert mock_scheduler.submit.called

    def test_start_workers_from_plan_no_script(self, mock_scheduler, mock_work_queue):
        ncs = [NodeConfig(name="big", cpus=16, memory_mb=65536)]
        cfg = make_config(node_configs=ncs, worker_script="")
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        pl = [Placement(node_config=ncs[0], count=2)]
        mgr._start_workers_from_plan(pl, 2)
        assert not mock_scheduler.submit.called

    def test_start_workers_from_plan_script_not_found(self, mock_scheduler, mock_work_queue):
        ncs = [NodeConfig(name="big", cpus=16, memory_mb=65536)]
        cfg = make_config(node_configs=ncs, worker_script="/nonexistent/worker.sh")
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        pl = [Placement(node_config=ncs[0], count=2)]
        mgr._start_workers_from_plan(pl, 2)
        assert not mock_scheduler.submit.called

    def test_start_workers_from_plan_submit_error(self, mock_scheduler, mock_work_queue, tmp_path):
        script = tmp_path / "worker.sh"
        script.write_text("#!/bin/bash\n")
        ncs = [NodeConfig(name="big", cpus=16, memory_mb=65536)]
        cfg = make_config(node_configs=ncs, worker_script=str(script))
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mock_scheduler.submit.side_effect = RuntimeError("submit failed")
        pl = [Placement(node_config=ncs[0], count=1)]
        mgr._start_workers_from_plan(pl, 1)
        assert mock_scheduler.submit.called

    def test_signal_workers_empty_active(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._signal_workers(5)  # no tracked jobs
        assert not mock_scheduler.signal.called

    def test_signal_workers_with_node_configs_no_plan(self, mock_scheduler, mock_work_queue):
        ncs = [NodeConfig(name="small", cpus=4, memory_mb=8192)]
        cfg = make_config(node_configs=ncs)
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._node_assignments["1"] = "small"
        mgr._signal_workers(1)  # no plan → simple path (plan=None)
        assert mock_scheduler.signal.called

    def test_signal_workers_failure_logged(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mock_scheduler.signal.side_effect = RuntimeError("signal failed")
        mgr._signal_workers(1)
        # state unchanged because tracked is not updated on failure
        assert mgr._tracked["1"].state == JobState.RUNNING

    def test_daemon_shutdown_drains_all(self, mock_scheduler, mock_work_queue):
        cfg = make_config(scaling={"drain_on_stop": True})
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._node_assignments["1"] = "default"
        mgr._daemon_shutdown = True
        mgr._scale([], [Placement(node_config=NodeConfig(name="default"), count=0)], 0)
        assert mock_scheduler.signal.called

    def test_scale_down_drain_timeout_force_cancel(self, mock_scheduler, mock_work_queue):
        cfg = make_config(scaling={"min_workers": 0, "scale_down_cooldown": 1, "drain_timeout": 1})
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.DRAINING)
        mgr._drain_start = -9999.0
        mgr._last_scale_down = -9999.0
        plan = [Placement(node_config=NodeConfig(name="default"), count=0)]
        mgr._scale([], plan, 0)
        assert mock_scheduler.cancel.called
        assert mgr._tracked["1"].state == JobState.EXITED

    def test_signal_workers_count_zero(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._signal_workers(0)
        assert not mock_scheduler.signal.called

    def test_start_workers_from_plan_batch_zero_skipped(
        self, mock_scheduler, mock_work_queue, tmp_path
    ):
        script = tmp_path / "worker.sh"
        script.write_text("#!/bin/bash\n")
        ncs = [NodeConfig(name="big", cpus=16, memory_mb=65536)]
        cfg = make_config(node_configs=ncs, worker_script=str(script))
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        pl = [Placement(node_config=ncs[0], count=0)]
        mgr._start_workers_from_plan(pl, 0)
        assert not mock_scheduler.submit.called

    def test_start_workers_from_plan_with_gpu_submit_args(
        self, mock_scheduler, mock_work_queue, tmp_path
    ):
        script = tmp_path / "worker.sh"
        script.write_text("#!/bin/bash\n")
        ncs = [NodeConfig(name="gpu", cpus=4, memory_mb=8192, gpus=4)]
        cfg = make_config(node_configs=ncs, worker_script=str(script))
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        pl = [Placement(node_config=ncs[0], count=1)]
        mgr._start_workers_from_plan(pl, 1)
        call = mock_scheduler.submit.call_args
        _script, args = call[0]
        assert args["gpus"] == "4"


class TestMakeFunctions:
    def test_make_work_queue_condor_subprocess(self):
        cfg = Config(
            work_queue=WorkQueueConfig(
                backend="condor_subprocess", schedd_name="schedd.example.com"
            ),
            scheduler=SchedulerConfig(backend="slurm_subprocess"),
        )
        wq = _make_work_queue(cfg)
        assert wq.name() == "condor_subprocess(schedd=schedd.example.com)"

    def test_make_work_queue_condor_rest(self):
        cfg = Config(
            work_queue=WorkQueueConfig(backend="condor_rest", rest_url="http://example.com"),
            scheduler=SchedulerConfig(backend="slurm_subprocess"),
        )
        wq = _make_work_queue(cfg)
        assert "condor_rest" in wq.name()

    def test_make_work_queue_unknown_backend(self):
        cfg = Config(
            work_queue=WorkQueueConfig(backend="unknown"),
            scheduler=SchedulerConfig(backend="slurm_subprocess"),
        )
        with pytest.raises(ValueError, match="Unknown work_queue backend"):
            _make_work_queue(cfg)

    @patch("pool_manager.manager.htcondor", None)
    def test_make_work_queue_condor_python_fallback(self):
        cfg = Config(
            work_queue=WorkQueueConfig(backend="condor_python"),
            scheduler=SchedulerConfig(backend="slurm_subprocess"),
        )
        wq = _make_work_queue(cfg)
        assert "condor_subprocess" in wq.name()

    def test_make_scheduler_slurm_subprocess(self):
        cfg = Config(scheduler=SchedulerConfig(backend="slurm_subprocess"))
        sched = _make_scheduler(cfg)
        assert sched is not None

    def test_make_scheduler_slurm_rest(self):
        cfg = Config(
            scheduler=SchedulerConfig(
                backend="slurm_rest",
                rest_url="http://slurm:6820",
                rest_token="token123",
            )
        )
        sched = _make_scheduler(cfg)
        assert sched is not None

    @patch("pool_manager.scheduler.slurm_sfapi.Client")
    def test_make_scheduler_slurm_sfapi(self, mock_client):
        cfg = Config(
            scheduler=SchedulerConfig(
                backend="slurm_sfapi",
                machine="perlmutter",
                sfapi_client_id="client",
                sfapi_client_secret='"secret"',
                sfapi_key_path="/fake/key",
            )
        )
        sched = _make_scheduler(cfg)
        assert sched is not None

    def test_make_scheduler_pbs_subprocess(self):
        cfg = Config(scheduler=SchedulerConfig(backend="pbs_subprocess"))
        sched = _make_scheduler(cfg)
        assert sched is not None

    def test_make_scheduler_htcondor_rest(self):
        cfg = Config(
            scheduler=SchedulerConfig(
                backend="htcondor_rest",
                rest_url="http://condor:8080",
                rest_token="token123",
            )
        )
        sched = _make_scheduler(cfg)
        assert sched is not None

    def test_make_scheduler_unknown_backend(self):
        cfg = Config(scheduler=SchedulerConfig(backend="unknown"))
        with pytest.raises(ValueError, match="Unknown scheduler backend"):
            _make_scheduler(cfg)

    def test_handle_signal(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        assert mgr._running is True
        with pytest.raises(KeyboardInterrupt):
            mgr._handle_signal(None, None)
        assert mgr._daemon_shutdown is True
        assert mgr._running is False

    @patch("pool_manager.manager.htcondor", MagicMock())
    def test_make_work_queue_condor_python_with_htcondor(self):
        cfg = Config(
            work_queue=WorkQueueConfig(backend="condor_python"),
            scheduler=SchedulerConfig(backend="slurm_subprocess"),
        )
        wq = _make_work_queue(cfg)
        assert "condor_python" in wq.name()

    def test_make_scheduler_slurm_subprocess_with_user(self):
        cfg = Config(
            scheduler=SchedulerConfig(
                backend="slurm_subprocess",
                user="testuser",
            )
        )
        sched = _make_scheduler(cfg)
        assert sched is not None

    def test_start_workers_simple_with_submit_args(self, mock_scheduler, mock_work_queue, tmp_path):
        script = tmp_path / "worker.sh"
        script.write_text("#!/bin/bash\n")
        cfg = make_config(worker_script=str(script))
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._start_workers_simple(2)
        assert mock_scheduler.submit.call_count == 2
        for call in mock_scheduler.submit.call_args_list:
            _script, args = call[0]
            assert args["job-name"] == "htcondor_worker_default"

    def test_scale_up_cooldown_logged(self, mock_scheduler, mock_work_queue):
        cfg = make_config(scaling={"scale_up_cooldown": 99999})
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        plan = [Placement(node_config=NodeConfig(name="default"), count=3)]
        mgr._scale([TaskResources(cpus=1, memory_mb=1024)] * 5, plan, 3)
        assert not mock_scheduler.submit.called

    def test_daemon_shutdown_equal_target_drains(self, mock_scheduler, mock_work_queue):
        cfg = make_config(scaling={"drain_on_stop": True, "scale_down_cooldown": 0})
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._node_assignments["1"] = "default"
        mgr._daemon_shutdown = True
        mgr._scale([], [Placement(node_config=NodeConfig(name="default"), count=0)], 1)
        assert mgr._daemon_shutdown is True

    def test_force_cancel_draining_exception(self, mock_scheduler, mock_work_queue):
        cfg = make_config()
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.DRAINING)
        mock_scheduler.cancel.side_effect = RuntimeError("cancel failed")
        mgr._force_cancel_draining()
        assert mgr._tracked["1"].state == JobState.DRAINING  # unchanged on failure

    def test_run_keyboard_interrupt(self, mock_scheduler, mock_work_queue):
        cfg = make_config(scaling={"drain_on_stop": False, "min_workers": 0})
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        call_count = [0]

        def tick_once():
            call_count[0] += 1
            raise KeyboardInterrupt()

        mgr._tick = tick_once
        mgr.run()
        assert call_count[0] == 1

    def test_run_with_node_configs_keyboard_interrupt(self, mock_scheduler, mock_work_queue):
        ncs = [NodeConfig(name="small", cpus=4, memory_mb=8192)]
        cfg = make_config(node_configs=ncs, scaling={"drain_on_stop": False})
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)

        def tick_once():
            raise KeyboardInterrupt()

        mgr._tick = tick_once
        mgr.run()
        assert mgr._has_node_configs is True

    def test_run_keyboard_interrupt_with_drain(self, mock_scheduler, mock_work_queue):
        cfg = make_config(scaling={"drain_on_stop": True, "min_workers": 0})
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)

        def tick_once():
            raise KeyboardInterrupt()

        mgr._tick = tick_once
        mgr._daemon_shutdown = True
        mgr.run()
        assert mgr._daemon_shutdown is True

    def test_run_unhandled_error_then_keyboard_interrupt(self, mock_scheduler, mock_work_queue):
        cfg = make_config(scaling={"drain_on_stop": False}, poll_interval=0.01)
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        tick_calls = [0]

        def tick_with_error():
            tick_calls[0] += 1
            if tick_calls[0] == 1:
                raise RuntimeError("unhandled error")
            raise KeyboardInterrupt()

        mgr._tick = tick_with_error
        mgr.run()
        assert tick_calls[0] == 2

    def test_drain_all_while_loop_sleep_and_force(self, mock_scheduler, mock_work_queue):
        cfg = make_config(scaling={"drain_timeout": 0, "min_workers": 0})
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._node_assignments["1"] = "default"
        mgr._drain_all()
        # _drain_all calls _signal_workers → state becomes DRAINING
        # then _force_cancel_draining → state becomes EXITED
        assert mgr._tracked["1"].state == JobState.EXITED

    def test_drain_all_force_cancel_after_timeout(self, mock_scheduler, mock_work_queue):
        cfg = make_config(scaling={"drain_timeout": 0})
        mgr = PoolManager(config=cfg, work_queue=mock_work_queue, scheduler=mock_scheduler)
        mgr._tracked["1"] = JobInfo(job_id="1", state=JobState.RUNNING)
        mgr._node_assignments["1"] = "default"
        mgr._drain_all()
        assert mock_scheduler.signal.called

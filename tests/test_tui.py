from pool_manager.tui import PoolManagerTUI, PoolStateWidget, WorkerInfo, _format_error


class TestWorkerInfo:
    def test_dataclass_fields(self):
        info = WorkerInfo(job_id="job123", state="running", node_type="small")
        assert info.job_id == "job123"
        assert info.state == "running"
        assert info.node_type == "small"


class TestPoolStateWidget:
    def test_format_uptime_seconds(self):
        assert PoolStateWidget._format_uptime(5.0) == "5s"
        assert PoolStateWidget._format_uptime(59.0) == "59s"

    def test_format_uptime_minutes(self):
        assert PoolStateWidget._format_uptime(60.0) == "1m 0s"
        assert PoolStateWidget._format_uptime(125.0) == "2m 5s"

    def test_format_uptime_hours(self):
        assert PoolStateWidget._format_uptime(3600.0) == "1h 0m"
        assert PoolStateWidget._format_uptime(3665.0) == "1h 1m"

    def test_default_render_empty(self):
        widget = PoolStateWidget()
        rendered = widget.render().plain
        assert widget.idle_jobs == 0
        assert widget.target_workers == 0
        assert widget.scale_action == ""
        assert "Idle:    0" in rendered
        assert "Target:" in rendered

    def test_render_with_connection_errors(self):
        widget = PoolStateWidget()
        widget.scheduler_status = "[red]Connection refused[/red]"
        widget.work_queue_status = "[red]condor_q failed[/red]"
        rendered = widget.render().plain
        assert "Connection refused" in rendered
        assert "condor_q failed" in rendered

    def test_render_with_scale_action(self):
        widget = PoolStateWidget()
        widget.scale_action = "[green]+3 workers needed[/green]"
        widget.target_workers = 5
        widget.active_count = 2
        rendered = widget.render().plain
        assert "+3 workers needed" in rendered


class TestPoolManagerTUI:
    def test_init_with_config_path(self, tmp_path):
        config_file = tmp_path / "test-config.yaml"
        config_file.write_text("poll_interval: 10\n")

        app = PoolManagerTUI(config_path=str(config_file))
        assert app.config_path == config_file
        assert app.config.poll_interval == 10.0
        assert app._sched is not None
        assert app._wq is not None
        assert app._planner is not None


class TestFormatError:
    def test_with_message(self):
        assert _format_error(RuntimeError("sbatch failed")) == "sbatch failed"

    def test_without_message(self):
        assert _format_error(ValueError()) == "ValueError"

    def test_long_message_truncated(self):
        long = "x" * 200
        assert len(_format_error(RuntimeError(long))) == 120

from pool_manager.tui import PoolManagerTUI, PoolStateWidget, WorkerInfo


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


class TestPoolManagerTUI:
    def test_init_with_config_path(self, tmp_path):
        config_file = tmp_path / "test-config.yaml"
        config_file.write_text("poll_interval: 10\n")

        app = PoolManagerTUI(config_path=str(config_file))
        assert app.config_path == config_file
        assert app.config.poll_interval == 10.0

    def test_set_workers(self):
        app = PoolManagerTUI()
        workers = [
            WorkerInfo(job_id="job1", state="running", node_type="small"),
            WorkerInfo(job_id="job2", state="pending", node_type="large"),
        ]
        app.set_workers(workers)
        assert app.workers_data == workers

    def test_set_placements(self):
        app = PoolManagerTUI()
        placements = [("small", 2, 4, 8000, 0), ("large", 1, 16, 64000, 0)]
        app.set_placements(placements)
        assert app.placements_data == placements

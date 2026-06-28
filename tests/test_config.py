import tempfile
from pathlib import Path

from pool_manager.config import Config


class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.poll_interval == 15.0
        assert cfg.log_level == "INFO"
        assert cfg.work_queue.backend == "condor_python"
        assert cfg.scheduler.backend == "slurm_subprocess"
        assert cfg.scaling.min_workers == 0
        assert cfg.scaling.max_workers == 16

    def test_from_file_defaults_when_missing(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("")
            path = f.name
        try:
            cfg = Config.from_file(path)
            assert cfg.poll_interval == 15.0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_from_file_missing_file(self):
        cfg = Config.from_file("/nonexistent/path.yaml")
        assert isinstance(cfg, Config)

    def test_from_file_parses_values(self):
        yaml_content = """
poll_interval: 30
log_level: DEBUG
work_queue:
  backend: condor_subprocess
  schedd_name: my-schedd.example.com
scheduler:
  backend: pbs_subprocess
  worker_script: /path/to/worker.sh
  submit_args:
    walltime: "01:00:00"
    nodes: 2
scaling:
  min_workers: 1
  max_workers: 8
  batch_size: 2
  scale_up_cooldown: 15
  scale_down_cooldown: 30
  drain_timeout: 60
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            cfg = Config.from_file(path)
            assert cfg.poll_interval == 30
            assert cfg.log_level == "DEBUG"
            assert cfg.work_queue.backend == "condor_subprocess"
            assert cfg.work_queue.schedd_name == "my-schedd.example.com"
            assert cfg.scheduler.backend == "pbs_subprocess"
            assert cfg.scheduler.worker_script == "/path/to/worker.sh"
            assert cfg.scheduler.submit_args["walltime"] == "01:00:00"
            assert cfg.scheduler.submit_args["nodes"] == 2
            assert cfg.scaling.min_workers == 1
            assert cfg.scaling.max_workers == 8
            assert cfg.scaling.batch_size == 2
            assert cfg.scaling.scale_up_cooldown == 15
            assert cfg.scaling.scale_down_cooldown == 30
            assert cfg.scaling.drain_timeout == 60
        finally:
            Path(path).unlink(missing_ok=True)

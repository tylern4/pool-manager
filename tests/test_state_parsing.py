from pool_manager.scheduler.base import JobState
from pool_manager.scheduler.pbs_subprocess import _parse_pbs_state
from pool_manager.scheduler.slurm_rest import _parse_slurm_rest_state
from pool_manager.scheduler.slurm_subprocess import _parse_slurm_state


class TestSlurmStateParsing:
    def test_running(self):
        assert _parse_slurm_state("R") == JobState.RUNNING

    def test_pending(self):
        assert _parse_slurm_state("PD") == JobState.PENDING

    def test_configuring(self):
        assert _parse_slurm_state("CF") == JobState.PENDING

    def test_completing(self):
        assert _parse_slurm_state("CG") == JobState.RUNNING

    def test_unknown(self):
        assert _parse_slurm_state("FOO") == JobState.UNKNOWN

    def test_empty(self):
        assert _parse_slurm_state("") == JobState.UNKNOWN


class TestPBSStateParsing:
    def test_running(self):
        assert _parse_pbs_state("R") == JobState.RUNNING

    def test_queued(self):
        assert _parse_pbs_state("Q") == JobState.PENDING

    def test_held(self):
        assert _parse_pbs_state("H") == JobState.PENDING

    def test_waiting(self):
        assert _parse_pbs_state("W") == JobState.PENDING

    def test_suspended(self):
        assert _parse_pbs_state("S") == JobState.RUNNING

    def test_exiting(self):
        assert _parse_pbs_state("E") == JobState.RUNNING

    def test_unknown(self):
        assert _parse_pbs_state("Z") == JobState.UNKNOWN


class TestSlurmRESTStateParsing:
    def test_running(self):
        assert _parse_slurm_rest_state("RUNNING") == JobState.RUNNING

    def test_pending(self):
        assert _parse_slurm_rest_state("PENDING") == JobState.PENDING

    def test_configuring(self):
        assert _parse_slurm_rest_state("CONFIGURING") == JobState.PENDING

    def test_completing(self):
        assert _parse_slurm_rest_state("COMPLETING") == JobState.RUNNING

    def test_unknown(self):
        assert _parse_slurm_rest_state("CANCELLED") == JobState.UNKNOWN

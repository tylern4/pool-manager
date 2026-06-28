from pool_manager.scaling import ScalingPolicy


class TestScalingPolicy:
    def test_default_min_zero(self):
        sp = ScalingPolicy()
        assert sp.target_size(0) == 0

    def test_one_job_one_worker(self):
        sp = ScalingPolicy()
        assert sp.target_size(1) == 1

    def test_batch_size_groups_jobs(self):
        sp = ScalingPolicy(batch_size=4)
        assert sp.target_size(0) == 0
        assert sp.target_size(1) == 1
        assert sp.target_size(4) == 1
        assert sp.target_size(5) == 2
        assert sp.target_size(8) == 2
        assert sp.target_size(9) == 3

    def test_capped_by_max(self):
        sp = ScalingPolicy(max_workers=4)
        assert sp.target_size(100) == 4

    def test_min_workers_floor(self):
        sp = ScalingPolicy(min_workers=2, batch_size=10)
        assert sp.target_size(0) == 2
        assert sp.target_size(5) == 2
        assert sp.target_size(15) == 2

    def test_min_workers_respected_with_jobs(self):
        sp = ScalingPolicy(min_workers=2)
        assert sp.target_size(1) == 2
        assert sp.target_size(3) == 3

    def test_all_params(self):
        sp = ScalingPolicy(min_workers=1, max_workers=10, batch_size=3)
        assert sp.target_size(0) == 1
        assert sp.target_size(1) == 1
        assert sp.target_size(3) == 1
        assert sp.target_size(4) == 2
        assert sp.target_size(30) == 10

import pytest

from src.visualize import prepare_ttft_data, prepare_speedup_data


class TestPrepareTTFTData:
    def test_basic(self, sample_results):
        df = prepare_ttft_data(sample_results)
        assert len(df) == 10  # 2 methods x 5 QPS levels
        assert "method" in df.columns
        assert "qps" in df.columns
        assert "ttft_p50" in df.columns
        assert "throughput_p50" in df.columns

    def test_values(self, sample_results):
        df = prepare_ttft_data(sample_results)
        baseline_q1 = df[(df["method"] == "baseline") & (df["qps"] == 1)]
        assert len(baseline_q1) == 1
        assert baseline_q1.iloc[0]["ttft_p50"] == 42.0


class TestPrepareSpeedupData:
    def test_basic(self, sample_results):
        df = prepare_speedup_data(sample_results, ["eagle3"], [1, 5, 10, 25, 50])
        assert len(df) == 1
        assert df.iloc[0]["method"] == "eagle3"

    def test_speedup_values(self, sample_results):
        df = prepare_speedup_data(sample_results, ["eagle3"], [1, 5, 10, 25, 50])
        # eagle3 throughput at qps=1: 170 / baseline 85 = 2.0x
        assert df.iloc[0]["qps_1"] == pytest.approx(2.0, abs=0.01)


class TestPlotFunctions:
    def test_plot_ttft_by_qps(self, sample_results, tmp_path):
        from src.visualize import plot_ttft_by_qps
        plot_ttft_by_qps(sample_results, str(tmp_path))
        assert (tmp_path / "ttft_by_qps.png").exists()

    def test_plot_throughput_by_qps(self, sample_results, tmp_path):
        from src.visualize import plot_throughput_by_qps
        plot_throughput_by_qps(sample_results, str(tmp_path))
        assert (tmp_path / "throughput_by_qps.png").exists()

    def test_plot_speedup_heatmap(self, sample_results, tmp_path):
        from src.visualize import plot_speedup_heatmap
        plot_speedup_heatmap(sample_results, str(tmp_path))
        assert (tmp_path / "speedup_heatmap.png").exists()

import pytest

from src.visualize import prepare_tps_data, prepare_validity_data


class TestPrepareTpsData:
    def test_basic(self, sample_results):
        df = prepare_tps_data(sample_results)
        assert len(df) == 6
        assert "backend" in df.columns
        assert "schema" in df.columns
        assert "throughput_p50" in df.columns

    def test_values(self, sample_results):
        df = prepare_tps_data(sample_results)
        xgrammar_simple = df[
            (df["backend"] == "xgrammar") & (df["schema"] == "simple_json")
        ]
        assert xgrammar_simple["throughput_p50"].values[0] == 95.0


class TestPrepareValidityData:
    def test_basic(self, sample_results):
        df = prepare_validity_data(sample_results)
        assert len(df) == 6
        assert "validity_rate" in df.columns

    def test_percentage_conversion(self, sample_results):
        df = prepare_validity_data(sample_results)
        xgrammar_simple = df[
            (df["backend"] == "xgrammar") & (df["schema"] == "simple_json")
        ]
        assert xgrammar_simple["validity_rate"].values[0] == 100.0


class TestPlotFunctions:
    def test_plot_tps_overhead(self, sample_results, tmp_path):
        from src.visualize import plot_tps_overhead
        plot_tps_overhead(sample_results, str(tmp_path))
        assert (tmp_path / "tps_overhead.png").exists()

    def test_plot_validity_rate(self, sample_results, tmp_path):
        from src.visualize import plot_validity_rate
        plot_validity_rate(sample_results, str(tmp_path))
        assert (tmp_path / "validity_rate.png").exists()

    def test_plot_latency_comparison(self, sample_results, tmp_path):
        from src.visualize import plot_latency_comparison
        plot_latency_comparison(sample_results, str(tmp_path))
        assert (tmp_path / "latency_comparison.png").exists()

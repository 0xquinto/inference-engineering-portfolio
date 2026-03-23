import os
import tempfile

import matplotlib
matplotlib.use("Agg")

from src.visualize import plot_goodput_vs_qps, plot_latency_cdf, plot_fairness_heatmap


class TestPlotGoodputVsQps:
    def test_creates_file(self):
        data = {
            "fcfs": {1: {"goodput": 0.95}, 10: {"goodput": 0.60}},
            "slo_aware": {1: {"goodput": 0.95}, 10: {"goodput": 0.90}},
        }
        with tempfile.TemporaryDirectory() as d:
            plot_goodput_vs_qps(data, d)
            assert os.path.exists(os.path.join(d, "goodput_vs_qps.png"))


class TestPlotLatencyCdf:
    def test_creates_file(self):
        data = {
            "fcfs": {10: {"latencies": [1.0, 2.0, 3.0, 5.0, 8.0]}},
            "slo_aware": {10: {"latencies": [1.0, 1.5, 2.0, 2.5, 3.0]}},
        }
        slos = {"short": 2.0, "medium": 8.0, "long": 20.0}
        with tempfile.TemporaryDirectory() as d:
            plot_latency_cdf(data, slos, d)
            assert os.path.exists(os.path.join(d, "latency_cdf.png"))


class TestPlotFairnessHeatmap:
    def test_creates_file(self):
        data = {
            "fcfs": {1: {"fairness": 0.9}, 10: {"fairness": 0.4}},
            "slo_aware": {1: {"fairness": 0.9}, 10: {"fairness": 0.85}},
        }
        with tempfile.TemporaryDirectory() as d:
            plot_fairness_heatmap(data, d)
            assert os.path.exists(os.path.join(d, "fairness_heatmap.png"))

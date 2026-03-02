from unittest.mock import patch, MagicMock
from pathlib import Path

from src.config import QuantFormat
from src.quantize import QuantizationRunner, QuantizeResult


class TestQuantizeResult:
    def test_dataclass_fields(self):
        r = QuantizeResult(
            format_name="gptq_int4",
            output_path="quantized_models/Qwen-gptq_int4",
            time_seconds=120.5,
            original_size_mb=14000,
            quantized_size_mb=4200,
        )
        assert r.compression_ratio == 14000 / 4200

    def test_compression_ratio_baseline(self):
        r = QuantizeResult(
            format_name="bf16",
            output_path="Qwen/Qwen2.5-7B-Instruct",
            time_seconds=0.0,
            original_size_mb=14000,
            quantized_size_mb=14000,
        )
        assert r.compression_ratio == 1.0


class TestQuantizationRunner:
    def test_skip_baseline(self):
        fmt = QuantFormat.from_dict("bf16", {"description": "baseline", "tool": None})
        runner = QuantizationRunner(model_name="Qwen/Qwen2.5-7B-Instruct")
        result = runner.quantize(fmt)
        assert result.format_name == "bf16"
        assert result.time_seconds == 0.0

    @patch("src.quantize.QuantizationRunner._run_gptq")
    def test_dispatches_gptq(self, mock_gptq):
        mock_gptq.return_value = QuantizeResult(
            format_name="gptq_int4",
            output_path="quantized_models/test-gptq_int4",
            time_seconds=60.0,
            original_size_mb=14000,
            quantized_size_mb=4200,
        )
        fmt = QuantFormat.from_dict("gptq_int4", {
            "description": "GPTQ", "tool": "auto_gptq", "bits": 4, "group_size": 128,
        })
        runner = QuantizationRunner(model_name="test/model")
        result = runner.quantize(fmt)
        mock_gptq.assert_called_once_with(fmt)
        assert result.format_name == "gptq_int4"

    @patch("src.quantize.QuantizationRunner._run_awq")
    def test_dispatches_awq(self, mock_awq):
        mock_awq.return_value = QuantizeResult(
            format_name="awq_int4",
            output_path="quantized_models/test-awq_int4",
            time_seconds=45.0,
            original_size_mb=14000,
            quantized_size_mb=4100,
        )
        fmt = QuantFormat.from_dict("awq_int4", {
            "description": "AWQ", "tool": "autoawq", "bits": 4, "group_size": 128,
        })
        runner = QuantizationRunner(model_name="test/model")
        result = runner.quantize(fmt)
        mock_awq.assert_called_once_with(fmt)

    def test_unknown_tool_raises(self):
        fmt = QuantFormat.from_dict("unknown", {
            "description": "Unknown", "tool": "nonexistent_tool",
        })
        runner = QuantizationRunner(model_name="test/model")
        import pytest
        with pytest.raises(ValueError, match="Unsupported quantization tool"):
            runner.quantize(fmt)

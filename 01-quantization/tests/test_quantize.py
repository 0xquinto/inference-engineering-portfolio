from unittest.mock import patch, MagicMock
from pathlib import Path

from src.config import QuantFormat
from src.quantize import QuantizationRunner, QuantizeResult


class TestQuantizeResult:
    def test_dataclass_fields(self):
        r = QuantizeResult(
            format_name="w4a16",
            output_path="quantized_models/Qwen-w4a16",
            time_seconds=120.5,
            original_size_mb=14000,
            quantized_size_mb=4200,
        )
        assert r.compression_ratio == 14000 / 4200

    def test_compression_ratio_baseline(self):
        r = QuantizeResult(
            format_name="bf16",
            output_path="Qwen/Qwen3.5-9B",
            time_seconds=0.0,
            original_size_mb=14000,
            quantized_size_mb=14000,
        )
        assert r.compression_ratio == 1.0


class TestQuantizationRunner:
    def test_skip_baseline(self):
        fmt = QuantFormat.from_dict("bf16", {"description": "baseline", "tool": None})
        runner = QuantizationRunner(model_name="Qwen/Qwen3.5-9B")
        result = runner.quantize(fmt)
        assert result.format_name == "bf16"
        assert result.time_seconds == 0.0

    @patch("src.quantize.QuantizationRunner._run_llmcompressor")
    def test_dispatches_llmcompressor(self, mock_run):
        mock_run.return_value = QuantizeResult(
            format_name="w4a16",
            output_path="quantized_models/test-w4a16",
            time_seconds=60.0,
            original_size_mb=14000,
            quantized_size_mb=4200,
        )
        fmt = QuantFormat.from_dict("w4a16", {
            "description": "W4A16", "tool": "llm_compressor",
            "bits": 4, "group_size": 128, "target_dtype": "w4a16",
        })
        runner = QuantizationRunner(model_name="test/model")
        result = runner.quantize(fmt)
        mock_run.assert_called_once_with(fmt)
        assert result.format_name == "w4a16"

    @patch("src.quantize.QuantizationRunner._run_llmcompressor")
    def test_dispatches_fp8(self, mock_run):
        mock_run.return_value = QuantizeResult(
            format_name="fp8",
            output_path="quantized_models/test-fp8",
            time_seconds=30.0,
            original_size_mb=0,
            quantized_size_mb=7500,
        )
        fmt = QuantFormat.from_dict("fp8", {
            "description": "FP8", "tool": "llm_compressor", "target_dtype": "fp8",
        })
        runner = QuantizationRunner(model_name="test/model")
        result = runner.quantize(fmt)
        mock_run.assert_called_once_with(fmt)
        assert result.format_name == "fp8"

    def test_unknown_tool_raises(self):
        fmt = QuantFormat.from_dict("unknown", {
            "description": "Unknown", "tool": "nonexistent_tool",
        })
        runner = QuantizationRunner(model_name="test/model")
        import pytest
        with pytest.raises(ValueError, match="Unsupported quantization tool"):
            runner.quantize(fmt)

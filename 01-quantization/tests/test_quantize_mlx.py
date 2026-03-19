from unittest.mock import patch
from src.quantize_mlx import MLXQuantizationRunner
from src.config import QuantFormat

class TestMLXQuantizationRunner:
    def test_init(self):
        runner = MLXQuantizationRunner(model_name="Qwen/Qwen3.5-4B")
        assert runner.model_name == "Qwen/Qwen3.5-4B"

    def test_skip_baseline(self):
        runner = MLXQuantizationRunner(model_name="Qwen/Qwen3.5-4B")
        fmt = QuantFormat.from_dict("bf16", {"description": "baseline", "tool": None})
        result = runner.quantize(fmt)
        assert result.format_name == "bf16"
        assert result.time_seconds == 0.0

    @patch("src.quantize_mlx.mlx_lm_convert")
    def test_quantize_int4(self, mock_convert):
        mock_convert.return_value = None
        runner = MLXQuantizationRunner(model_name="Qwen/Qwen3.5-4B")
        fmt = QuantFormat.from_dict("int4", {"description": "INT4 via MLX", "tool": "mlx", "bits": 4, "group_size": 64})
        result = runner.quantize(fmt)
        assert result.format_name == "int4"
        mock_convert.assert_called_once()

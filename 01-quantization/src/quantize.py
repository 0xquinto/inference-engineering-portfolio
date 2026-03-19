import time
from dataclasses import dataclass
from pathlib import Path

from .config import QuantFormat


@dataclass
class QuantizeResult:
    format_name: str
    output_path: str
    time_seconds: float
    original_size_mb: int
    quantized_size_mb: int

    @property
    def compression_ratio(self) -> float:
        if self.quantized_size_mb == 0:
            return 0.0
        return self.original_size_mb / self.quantized_size_mb


class QuantizationRunner:
    def __init__(self, model_name: str, output_dir: str = "quantized_models"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)

    def quantize(self, fmt: QuantFormat) -> QuantizeResult:
        if fmt.is_baseline:
            return QuantizeResult(
                format_name=fmt.name,
                output_path=self.model_name,
                time_seconds=0.0,
                original_size_mb=0,
                quantized_size_mb=0,
            )

        dispatch = {
            "llm_compressor": self._run_llmcompressor,
            "mlx": self._run_mlx,
        }

        if fmt.tool not in dispatch:
            raise ValueError(f"Unsupported quantization tool: {fmt.tool}")

        return dispatch[fmt.tool](fmt)

    def _output_path(self, fmt: QuantFormat) -> Path:
        short_name = self.model_name.split("/")[-1]
        return self.output_dir / f"{short_name}-{fmt.name}"

    def _run_llmcompressor(self, fmt: QuantFormat) -> QuantizeResult:
        from llmcompressor import oneshot

        output_path = self._output_path(fmt)
        dtype = fmt.target_dtype or "fp8"

        if dtype == "w4a16":
            recipe = self._w4a16_recipe(fmt)
            oneshot_kwargs = dict(
                dataset="HuggingFaceH4/ultrachat_200k",
                num_calibration_samples=fmt.calibration_samples or 128,
                max_seq_length=2048,
            )
        else:
            recipe = self._fp8_recipe()
            oneshot_kwargs = dict(pipeline="datafree")

        start = time.time()
        oneshot(
            model=self.model_name,
            recipe=recipe,
            output_dir=str(output_path),
            **oneshot_kwargs,
        )
        elapsed = time.time() - start

        return QuantizeResult(
            format_name=fmt.name,
            output_path=str(output_path),
            time_seconds=elapsed,
            original_size_mb=0,
            quantized_size_mb=_dir_size_mb(output_path),
        )

    def _w4a16_recipe(self, fmt: QuantFormat):
        from llmcompressor.modifiers.quantization import GPTQModifier

        return GPTQModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=["lm_head"],
        )

    def _run_mlx(self, fmt: QuantFormat) -> QuantizeResult:
        from .quantize_mlx import MLXQuantizationRunner
        mlx_runner = MLXQuantizationRunner(self.model_name, str(self.output_dir))
        return mlx_runner.quantize(fmt)

    def _fp8_recipe(self):
        from llmcompressor.modifiers.quantization import QuantizationModifier

        return QuantizationModifier(
            targets="Linear", scheme="FP8", ignore=["lm_head"]
        )


def _dir_size_mb(path: Path) -> int:
    if not path.exists():
        return 0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return int(total / (1024 * 1024))

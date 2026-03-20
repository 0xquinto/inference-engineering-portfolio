import time
from pathlib import Path
from .config import QuantFormat
from .quantize import QuantizeResult, _dir_size_mb

def mlx_lm_convert(model: str, output_dir: str, bits: int, group_size: int) -> None:
    from mlx_lm import convert
    convert(hf_path=model, mlx_path=output_dir, quantize=True,
            q_bits=bits, q_group_size=group_size)

class MLXQuantizationRunner:
    def __init__(self, model_name: str, output_dir: str = "quantized_models"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)

    def quantize(self, fmt: QuantFormat) -> QuantizeResult:
        if fmt.is_baseline:
            return QuantizeResult(format_name=fmt.name, output_path=self.model_name,
                                 time_seconds=0.0, original_size_mb=0, quantized_size_mb=0)
        output_path = self._output_path(fmt)
        bits = fmt.bits or 4
        group_size = fmt.group_size or 64
        start = time.time()
        mlx_lm_convert(model=self.model_name, output_dir=str(output_path),
                        bits=bits, group_size=group_size)
        elapsed = time.time() - start
        return QuantizeResult(format_name=fmt.name, output_path=str(output_path),
                              time_seconds=elapsed, original_size_mb=0,
                              quantized_size_mb=_dir_size_mb(output_path))

    def _output_path(self, fmt: QuantFormat) -> Path:
        short_name = self.model_name.split("/")[-1]
        return self.output_dir / f"{short_name}-{fmt.name}-mlx"

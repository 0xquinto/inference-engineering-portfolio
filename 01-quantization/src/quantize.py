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
            "auto_gptq": self._run_gptq,
            "llm_compressor": self._run_fp8,
        }

        if fmt.tool not in dispatch:
            raise ValueError(f"Unsupported quantization tool: {fmt.tool}")

        return dispatch[fmt.tool](fmt)

    def _output_path(self, fmt: QuantFormat) -> Path:
        short_name = self.model_name.split("/")[-1]
        return self.output_dir / f"{short_name}-{fmt.name}"

    def _run_gptq(self, fmt: QuantFormat) -> QuantizeResult:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
        from datasets import load_dataset

        output_path = self._output_path(fmt)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        quant_config = BaseQuantizeConfig(
            bits=fmt.bits or 4,
            group_size=fmt.group_size or 128,
        )

        model = AutoGPTQForCausalLM.from_pretrained(
            self.model_name, quant_config
        )

        n_samples = fmt.calibration_samples or 128
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        examples = []
        for text in dataset["text"]:
            if len(text.strip()) > 100:
                tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                examples.append({"input_ids": tokens.input_ids[0], "attention_mask": tokens.attention_mask[0]})
            if len(examples) >= n_samples:
                break

        start = time.time()
        model.quantize(examples)
        elapsed = time.time() - start

        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))

        orig_size = _dir_size_mb(Path(model.config._name_or_path))
        quant_size = _dir_size_mb(output_path)

        return QuantizeResult(
            format_name=fmt.name,
            output_path=str(output_path),
            time_seconds=elapsed,
            original_size_mb=orig_size,
            quantized_size_mb=quant_size,
        )

    def _run_fp8(self, fmt: QuantFormat) -> QuantizeResult:
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor import oneshot
        from transformers import AutoModelForCausalLM, AutoTokenizer

        output_path = self._output_path(fmt)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        recipe = QuantizationModifier(
            targets="Linear", scheme="FP8", ignore=["lm_head"]
        )

        start = time.time()
        oneshot(model=model, recipe=recipe, output_dir=str(output_path))
        elapsed = time.time() - start

        tokenizer.save_pretrained(str(output_path))

        return QuantizeResult(
            format_name=fmt.name,
            output_path=str(output_path),
            time_seconds=elapsed,
            original_size_mb=0,
            quantized_size_mb=_dir_size_mb(output_path),
        )


def _dir_size_mb(path: Path) -> int:
    if not path.exists():
        return 0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return int(total / (1024 * 1024))

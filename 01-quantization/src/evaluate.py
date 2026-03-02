from dataclasses import dataclass


@dataclass
class EvalResult:
    format_name: str
    perplexity: float
    mmlu_accuracy: float

    def delta_from(self, baseline: "EvalResult") -> dict:
        ppl_delta = self.perplexity - baseline.perplexity
        mmlu_delta = self.mmlu_accuracy - baseline.mmlu_accuracy
        return {
            "perplexity_delta": ppl_delta,
            "perplexity_pct": (ppl_delta / baseline.perplexity) * 100,
            "mmlu_delta": mmlu_delta,
            "mmlu_pct": (mmlu_delta / baseline.mmlu_accuracy) * 100,
        }

    def to_dict(self) -> dict:
        return {
            "format_name": self.format_name,
            "perplexity": self.perplexity,
            "mmlu_accuracy": self.mmlu_accuracy,
        }


class QualityEvaluator:
    """Evaluates model quality via perplexity and MMLU accuracy.

    GPU-only: methods call transformers + datasets at runtime.
    """

    def __init__(self, model_path: str, max_samples: int = 500):
        self.model_path = model_path
        self.max_samples = max_samples

    def compute_perplexity(self) -> float:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto"
        )
        model.eval()

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in dataset["text"] if len(t.strip()) > 0][:self.max_samples]

        encodings = tokenizer("\n\n".join(texts), return_tensors="pt", truncation=True, max_length=2048)
        input_ids = encodings.input_ids.to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        return torch.exp(outputs.loss).item()

    def compute_mmlu_accuracy(self, num_tasks: int = 14, few_shot: int = 5) -> float:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto"
        )
        model.eval()

        tasks = [
            "abstract_algebra", "anatomy", "astronomy", "college_biology",
            "college_chemistry", "college_physics", "computer_science",
            "econometrics", "high_school_biology", "high_school_chemistry",
            "high_school_mathematics", "high_school_physics",
            "machine_learning", "professional_medicine",
        ][:num_tasks]

        correct, total = 0, 0
        choices = ["A", "B", "C", "D"]
        choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]

        for task in tasks:
            ds = load_dataset("cais/mmlu", task, split="test")
            for row in list(ds)[:100]:
                prompt = f"Question: {row['question']}\n"
                for i, opt in enumerate(row["choices"]):
                    prompt += f"{choices[i]}. {opt}\n"
                prompt += "Answer:"

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    logits = model(**inputs).logits[0, -1]

                probs = torch.softmax(logits[choice_ids], dim=0)
                pred = probs.argmax().item()

                if pred == row["answer"]:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    def evaluate(self, format_name: str) -> EvalResult:
        ppl = self.compute_perplexity()
        mmlu = self.compute_mmlu_accuracy()
        return EvalResult(format_name=format_name, perplexity=ppl, mmlu_accuracy=mmlu)

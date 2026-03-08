from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class Generator:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = self._resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        ).to(self.device)

    def generate(self, prompt: str, max_tokens: int = 512):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _resolve_device(self, requested_device: str) -> str:
        if requested_device.startswith("cuda") and torch.cuda.is_available():
            return requested_device
        return "cpu"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Generator:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate(self, prompt: str, max_tokens: int = 512):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
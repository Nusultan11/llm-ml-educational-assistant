from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate text response from model.

    Args:
        model: Loaded LLM model.
        tokenizer: Corresponding tokenizer.
        prompt: Input text prompt.
        max_new_tokens: Maximum generated tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.

    Returns:
        Generated text string.
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
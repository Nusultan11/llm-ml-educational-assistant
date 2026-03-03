from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:

    # Apply Mistral chat template
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from output
    return decoded.replace(formatted_prompt, "").strip()
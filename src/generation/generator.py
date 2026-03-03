import torch


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:

    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise ML tutor. "
                "Start immediately with the explanation. "
                "Do not repeat or restate the question."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # Получаем текст чата
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Токенизация
    enc = tokenizer(chat_text, return_tensors="pt")

    # ВАЖНО: переносим input_ids вручную (4-bit безопасно)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Берём только новую генерацию
    generated_tokens = outputs[0][input_ids.shape[-1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()
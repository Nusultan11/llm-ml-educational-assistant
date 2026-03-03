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
        {"role": "user", "content": prompt},
    ]

    # 1) Получаем готовый чат-промпт как строку
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 2) Токенизируем в тензоры
    enc = tokenizer(chat_text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}

    # 3) Генерация
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 4) Берём только сгенерированное (без входа)
    gen_tokens = out[0][enc["input_ids"].shape[-1] :]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
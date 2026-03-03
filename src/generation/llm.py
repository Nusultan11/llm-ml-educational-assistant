from typing import Tuple
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def setup_logger(name: str = "llm_project") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger(__name__)


def load_mistral_4bit(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:

    logger.info("Initializing 4-bit quantization config.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    logger.info("Loading tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading model in 4-bit mode.")

    # ❗ Убрали device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    logger.info("Model loaded successfully.")

    return tokenizer, model
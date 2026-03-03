from typing import Tuple
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def load_mistral(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:

    logger.info("Loading tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading model in FP16 mode.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Model loaded successfully.")
    return tokenizer, model
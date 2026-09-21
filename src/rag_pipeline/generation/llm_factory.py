from __future__ import annotations
from typing import Any, Dict, List
from rag_pipeline.utils import logger

def _load_single_llm(spec):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
    )
    from langchain_huggingface import HuggingFacePipeline

    hf_path = spec["hf_path"]
    logger.info("Loading LLM %s (%s)", spec.get("name", hf_path), hf_path)

    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Quantization config with optional CPU offload ---
    quantization_config = None
    if spec.get("load_in_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            # This is the critical fix:
            llm_int8_enable_fp32_cpu_offload=spec.get("cpu_offload", False),
        )

    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(hf_path, **model_kwargs)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=spec.get("max_new_tokens", 200),
        temperature=spec.get("temperature", 0.35),
        top_p=spec.get("top_p", 0.92),
        do_sample=spec.get("do_sample", True),
        repetition_penalty=spec.get("repetition_penalty", 1.05),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    return HuggingFacePipeline(pipeline=pipe)

def build_llms(cfg):
    llms = {}
    for spec in cfg.get("models", []):
        name = spec.get("name") or spec["hf_path"]
        llms[name] = _load_single_llm(spec)
    logger.info("Loaded %d LLM(s)", len(llms))
    return llms

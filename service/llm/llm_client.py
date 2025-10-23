# from typing import Optional, List, Dict
# import re
import os
from typing import Optional, List, Dict
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# (LoRA 어댑터 자동 로딩)
from peft import AutoPeftModelForCausalLM

load_dotenv()

# ======= 환경 변수 =======
# 예) FinGPT LoRA:
#   HF_REPO_ID="FinGPT/fingpt-mt_llama2-7b_lora"
#   HF_BASE_MODEL="meta-llama/Llama-2-7b-hf"
HF_REPO_ID   = os.getenv("HF_REPO_ID", "FinGPT/fingpt-mt_llama2-7b_lora")
HF_BASE_MODEL= os.getenv("HF_BASE_MODEL", "meta-llama/Llama-2-7b-hf")  # LoRA일 때 필수
HF_DEVICE_MAP= os.getenv("HF_DEVICE_MAP", "auto")    # "auto" | "cuda" | "cpu"
HF_DTYPE     = os.getenv("HF_DTYPE", "bfloat16")     # "float16" | "bfloat16" | "float32"
HF_4BIT      = os.getenv("HF_4BIT", "false").lower() == "true"
HF_TRUST     = os.getenv("HF_TRUST_REMOTE_CODE", "false").lower() == "true"
HF_TOKEN     = os.getenv("HUGGINGFACE_API_KEY")

_tokenizer = None
_model = None

def _torch_dtype():
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(HF_DTYPE, torch.bfloat16)

def _hf_auth_kwargs() -> Dict[str, str]:
    if not HF_TOKEN:
        return {}
    return {"token": HF_TOKEN}

def _load_tokenizer(name: str):
    auth_kwargs = _hf_auth_kwargs()
    return AutoTokenizer.from_pretrained(
        name,
        use_fast=True,
        trust_remote_code=HF_TRUST,
        **auth_kwargs,
    )

def _load_model(repo_id: str, is_lora: bool):
    kwargs = {}
    if HF_4BIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = _torch_dtype()
        kwargs["device_map"] = HF_DEVICE_MAP

    auth_kwargs = _hf_auth_kwargs()
    if is_lora:
        # LoRA 어댑터: base model 로드 후 어댑터 머지/로딩
        base = HF_BASE_MODEL
        if not base:
            raise ValueError("LoRA 어댑터를 쓰려면 HF_BASE_MODEL 환경변수를 지정하세요.")
        # 토크나이저는 베이스에서
        tok = _load_tokenizer(base)
        # 어댑터 자동 로드
        mdl = AutoPeftModelForCausalLM.from_pretrained(
            repo_id,
            trust_remote_code=HF_TRUST,
            **auth_kwargs,
            **kwargs,
        )
        mdl.eval()
        return tok, mdl
    else:
        tok = _load_tokenizer(repo_id)
        mdl = AutoModelForCausalLM.from_pretrained(
            repo_id,
            trust_remote_code=HF_TRUST,
            **auth_kwargs,
            **kwargs,
        )
        mdl.eval()
        return tok, mdl

def _is_lora_repo(repo_id: str) -> bool:
    # 매우 단순한 휴리스틱: 'lora' / 'adapter' 키워드 포함 시 LoRA 가정
    # (정확히 하려면 hf_hub로 파일 리스트 조회 후 adapter_config.json 존재 확인)
    key = repo_id.lower()
    return ("lora" in key) or ("adapter" in key)

def _load():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model
    is_lora = _is_lora_repo(HF_REPO_ID)
    _tokenizer, _model = _load_model(HF_REPO_ID, is_lora=is_lora)
    return _tokenizer, _model

def _apply_chat_template_safe(tokenizer, messages) -> torch.Tensor:
    # 지원 모델은 chat_template 사용
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    except Exception:
        # 템플릿이 없으면 Llama2 스타일로 포맷팅
        system = next((m["content"] for m in messages if m["role"]=="system"), "")
        user   = next((m["content"] for m in messages if m["role"]=="user"), "")
        prompt = f"""[INST] <<SYS>>
        {system.strip()}
        <</SYS>>

        {user.strip()} [/INST]
        """
        return tokenizer(prompt, return_tensors="pt").input_ids

def chat(system: str, user: str, max_tokens: int = 512) -> str:
    """
    LangGraph Generate 노드에서 호출되는 함수.
    system: 시스템 규칙/역할 (레벨별 템플릿 포함)
    user:   사용자 질문 + 컨텍스트
    max_tokens: 생성할 텍스트(답변)의 최대 토큰 수.
        - 이 값은 "최대 생성 토큰"을 의미하며, 모델이 한번에 얼마나 긴 출력을 생성할지 결정
        - 기본값 512는 실험적으로 적당히 "중간 길이 답변"을 커버할 정도로 설정
        - 길이에 엄격한 제한이 필요한 경우 컨텍스트별로 조정 가능
        - (참고: 입력 prompt와 생성 답변 토큰 합계는 모델 한계 context window를 초과하면 안 됨)
    """
    tokenizer, model = _load()

    messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]
    input_ids = _apply_chat_template_safe(tokenizer, messages).to(model.device)

    gen_out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    output_ids = gen_out[0][input_ids.shape[-1]:]
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return text

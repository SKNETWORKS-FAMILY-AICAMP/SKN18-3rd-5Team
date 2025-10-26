# service/llm_client.py
import os, sys
from typing import Optional, List, Dict
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # ← 핵심: AutoPeft 대신 명시 적용

load_dotenv()

# 오프라인 강제(사전 다운로드 했으므로 안전)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

HF_REPO_ID    = os.getenv("MODEL_DIR_ADAPTER", "./models/adapters/llama3.2-3b-ko-report-lora")  # LoRA 경로(로컬)
HF_BASE_MODEL = os.getenv("MODEL_DIR_BASE", "./models/base/Llama-3.2-3B")              # 베이스 경로(로컬)
HF_TRUST      = os.getenv("HF_TRUST_REMOTE_CODE", "false").lower() == "true"
HF_TOKEN      = os.getenv("HUGGINGFACE_HUB_TOKEN")     # 오프라인이면 없어도 됨

IS_RUNPOD = os.getenv("IS_RUNPOD", "false").lower() == "true"

if IS_RUNPOD:
    HF_DEVICE_MAP = os.getenv("HF_DEVICE_MAP", "cuda")
    HF_DTYPE = os.getenv("HF_DTYPE", "bfloat16")
    HF_4BIT = os.getenv("HF_4BIT", "false").lower() == "true"
else:
    HF_DEVICE_MAP = os.getenv("HF_DEVICE_MAP", "cpu")
    HF_DTYPE = os.getenv("HF_DTYPE", "float32")
    HF_4BIT = os.getenv("HF_4BIT", "false").lower() == "true"

print("[LLM] HF_REPO_ID =", HF_REPO_ID)
print("[LLM] HF_BASE_MODEL =", HF_BASE_MODEL)
print("[LLM] CWD =", os.getcwd(), file=sys.stderr)

_tokenizer = None
_model = None

def _torch_dtype():
    """환경 설정에 맞는 torch dtype을 반환 (CPU면 float32 강제)"""
    # CPU면 float32로 강제 (bf16/float16 미지원 환경 방지)
    if HF_DEVICE_MAP == "cpu":
        return torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(HF_DTYPE, torch.bfloat16)

def _hf_auth_kwargs() -> Dict[str, str]:
    """허깅페이스 인증 토큰이 있을 경우 kwargs 형태로 반환"""
    return {"token": HF_TOKEN} if HF_TOKEN else {}

def _load_tokenizer(path_or_id: str):
    """지정된 경로에서 토크나이저를 로드"""
    return AutoTokenizer.from_pretrained(
        path_or_id,
        use_fast=True,
        trust_remote_code=HF_TRUST,
        **_hf_auth_kwargs(),
    )

def _load_model_lora(base_path: str, lora_path: str):
    """베이스 모델을 로드하고 LoRA 어댑터를 적용해 파인튜닝 모델 생성"""
    kwargs = {}
    if HF_4BIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["device_map"] = "auto"
    else:
        # kwargs["torch_dtype"] = _torch_dtype()
        kwargs["dtype"] = _torch_dtype()
        kwargs["device_map"] = HF_DEVICE_MAP

    # 1) 베이스 모델(로컬 경로) 로드
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        trust_remote_code=HF_TRUST,
        low_cpu_mem_usage=True,             # ← 추가
        **_hf_auth_kwargs(),
        **kwargs,
    )
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        trust_remote_code=HF_TRUST,
        device_map=HF_DEVICE_MAP,
        torch_dtype=_torch_dtype(),  # 또는 dtype=_torch_dtype() (버전 호환성)
        **_hf_auth_kwargs(),
    )
    # 2) 어댑터(로컬 경로) 적용
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        **_hf_auth_kwargs(),
    )
    model.eval()
    return model

def _load():
    """토크나이저와 LoRA 적용 모델을 지연 로딩 후 캐싱"""
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    # 로컬 경로 사용 전제(B 방식): tokenizer는 베이스에서 로드
    _tokenizer = _load_tokenizer(HF_BASE_MODEL)
    _model = _load_model_lora(HF_BASE_MODEL, HF_REPO_ID)

    # pad_token 없으면 eos로 보정
    if _tokenizer.pad_token_id is None and _tokenizer.eos_token_id is not None:
        _tokenizer.pad_token = _tokenizer.eos_token

    return _tokenizer, _model

def _apply_chat_template_safe(tokenizer, messages) -> torch.Tensor:
    """chat_template 적용 실패 시 수동 포맷으로 프롬프트를 구성"""
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    except Exception:
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
    user  : 사용자 질문 + 컨텍스트
    """
    tokenizer, model = _load()

    messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]
    input_ids = _apply_chat_template_safe(tokenizer, messages).to(model.device)

    gen_out = model.generate(
        input_ids=input_ids,
        max_new_tokens=min(max_tokens, 128),  # ← 128로 캡
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # use_cache=False,                      # ← KV-cache OFF (메모리 급감)
        use_cache=True,                      # GPU에서는 KV-cache ON
    )
    output_ids = gen_out[0][input_ids.shape[-1]:]
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return text

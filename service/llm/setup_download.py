# setup_download.py
import os
from huggingface_hub import snapshot_download, login, HfApi
from dotenv import load_dotenv

# 1) .env 로드 (반드시 호출)
load_dotenv()

# 2) 토큰 읽기 (환경변수 이름: HUGGINGFACE_HUB_TOKEN)
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
if not HF_TOKEN:
    raise SystemExit("❌ HUGGINGFACE_HUB_TOKEN 이 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")

# 3) 로그인 (세션에 토큰 주입)
login(token=HF_TOKEN)

# 4) 다운로드 가속 옵션
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

IS_RUNPOD = os.getenv("IS_RUNPOD", "false").strip().lower() == "true"
BASE_DIR = "/workspace/SKN18-3rd-5Team/" if IS_RUNPOD else "./"
BASE_DIR = os.path.abspath(BASE_DIR)

def _resolve_model_dir(env_value: str, default_relative: str) -> str:
    """Resolve a model directory relative to BASE_DIR unless already absolute."""
    candidate = env_value.strip() if env_value else default_relative
    candidate = os.path.expanduser(candidate)
    if os.path.isabs(candidate):
        return os.path.normpath(candidate)
    return os.path.normpath(os.path.join(BASE_DIR, candidate))

HF_ADAPTER_REPO_ID = os.getenv("HF_ADAPTER_REPO_ID", "has0327/Llama-3.2-3B-ko-finetuned").strip()
MODEL_DIR_ADAPTER = _resolve_model_dir(
    os.getenv("MODEL_DIR_ADAPTER") or os.getenv("HF_REPO_ID", ""),
    "models/adapters/Llama-3.2-3B-ko-finetuned",
)

HF_BASE_REPO_ID = os.getenv("HF_BASE_REPO_ID", "meta-llama/Llama-3.2-3B").strip()
MODEL_DIR_BASE = _resolve_model_dir(
    os.getenv("MODEL_DIR_BASE") or os.getenv("HF_BASE_MODEL", ""),
    "models/base/Llama-3.2-3B",
)

os.makedirs(MODEL_DIR_ADAPTER, exist_ok=True)
os.makedirs(MODEL_DIR_BASE, exist_ok=True)

# LoRA 어댑터
print("⬇️  FinGPT LoRA 어댑터 다운로드 중…")
snapshot_download(
    repo_id=HF_ADAPTER_REPO_ID,
    local_dir=MODEL_DIR_ADAPTER,
    local_dir_use_symlinks=False
)

# 베이스 모델
print("⬇️  Llama-2 7B 베이스 다운로드 중… (게이트 승인 필요)")
snapshot_download(
    repo_id=HF_BASE_REPO_ID,
    local_dir=MODEL_DIR_BASE,
    local_dir_use_symlinks=False
)

print("✅ 모든 모델 파일이 models에 저장되었습니다.")

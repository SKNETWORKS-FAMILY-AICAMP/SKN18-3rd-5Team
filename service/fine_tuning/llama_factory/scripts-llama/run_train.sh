#!/bin/bash
set -euo pipefail

######################################
# Hugging Face 로그인 및 학습 자동화 스크립트
######################################

# 1️⃣ .env 파일 자동 로드
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
  echo "✅ .env 파일에서 환경변수 로드 완료"
else
  echo "⚠️ .env 파일이 없습니다. 수동으로 HUGGINGFACE_TOKEN을 설정해주세요."
fi

# 2️⃣ 토큰 확인
if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
  echo "❌ ERROR: HUGGINGFACE_TOKEN is not set"
  exit 1
fi

# 3️⃣ Hugging Face CLI 로그인
echo "🔐 Hugging Face 로그인 중..."
huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential || true
echo "✅ 로그인 완료"

# 4️⃣ 학습 시작
echo "🚀 LLaMA Factory 학습 시작..."
llamafactory-cli train service/fine_tuning/llama_factory/config/llama-factory.yaml
echo "✅ 학습 완료"

# 5️⃣ 학습 결과 업로드
MODEL_REPO="yourname/finance-lora"
OUTPUT_DIR="./service/fine_tuning/llama_factory/output_lora"

echo "📦 모델 업로드 준비 중..."
huggingface-cli repo create "$MODEL_REPO" --type model --private || true

echo "⬆️ 모델 업로드 중..."
huggingface-cli upload "$OUTPUT_DIR" --repo "$MODEL_REPO" --repo-type model

echo "🎉 모델 업로드 완료! 👉 https://huggingface.co/$MODEL_REPO"

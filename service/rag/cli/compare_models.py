#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
여러 LLM 모델로 같은 질문에 대한 답변 비교
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from ..rag_system import RAGSystem
from ..generation.generator import OllamaGenerator, GenerationConfig
from ..models.config import EmbeddingModelType

import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_config() -> dict:
    """데이터베이스 설정"""
    return {
        'host': os.getenv('PG_HOST', 'localhost'),
        'port': os.getenv('PG_PORT', '5432'),
        'database': os.getenv('PG_DB', 'rey'),
        'user': os.getenv('PG_USER', 'postgres'),
        'password': os.getenv('PG_PASSWORD', 'post1234')
    }


def compare_models(query: str, models: list, embedding_model: str = "E5", save_results: bool = True):
    """
    여러 모델로 같은 질문에 답변 생성하고 비교

    Args:
        query: 질문
        models: 비교할 LLM 모델 리스트 (예: ["gemma3:4b", "llama3.2:latest", "qwen3:4b"])
        embedding_model: 임베딩 모델
        save_results: 결과 저장 여부
    """
    db_config = get_db_config()
    results = {}

    print(f"\n{'='*80}")
    print(f"질문: {query}")
    print(f"임베딩 모델: {embedding_model}")
    print(f"비교 모델: {', '.join(models)}")
    print(f"{'='*80}\n")

    # 각 모델별로 답변 생성
    for model in models:
        print(f"\n🤖 {model} 모델 테스트 중...")
        print("-" * 80)

        try:
            # Generator 초기화
            generator = OllamaGenerator(
                base_url="http://localhost:11434",
                default_model=model
            )

            # 모델 상태 확인
            if not generator.check_health():
                print(f"❌ Ollama 서버에 연결할 수 없습니다.")
                results[model] = {"error": "Ollama server not available"}
                continue

            available_models = generator.list_models()
            if model not in available_models:
                print(f"⚠️  모델 '{model}'을 찾을 수 없습니다.")
                print(f"다음 명령어로 설치: ollama pull {model}")
                results[model] = {"error": "Model not found"}
                continue

            # RAG 시스템 초기화
            rag_system = RAGSystem(
                model_type=EmbeddingModelType.MULTILINGUAL_E5_SMALL,
                db_config=db_config,
                llm_generator=generator,
                enable_generation=True
            )

            # 답변 생성
            response = rag_system.generate_answer(
                query=query,
                top_k=5,
                use_reranker=False
            )

            # 결과 저장
            results[model] = {
                "answer": response.generated_answer.answer,
                "generation_time_ms": response.generated_answer.generation_time_ms,
                "tokens_used": response.generated_answer.tokens_used,
                "num_docs": len(response.retrieved_documents),
                "avg_similarity": sum(d.get('similarity', 0) for d in response.retrieved_documents) / len(response.retrieved_documents) if response.retrieved_documents else 0
            }

            # 답변 출력
            print(f"\n답변:")
            print(response.generated_answer.answer)
            print(f"\n⏱️  생성 시간: {response.generated_answer.generation_time_ms:.2f}ms")
            print(f"📊 토큰 수: {response.generated_answer.tokens_used}")
            print(f"📚 참고 문서: {len(response.retrieved_documents)}개")

        except Exception as e:
            logger.error(f"모델 {model} 테스트 중 오류: {e}")
            results[model] = {"error": str(e)}

    # 비교 결과 출력
    print(f"\n\n{'='*80}")
    print("📊 모델 비교 결과")
    print(f"{'='*80}\n")

    print(f"{'모델':<20} {'생성시간(ms)':>15} {'토큰수':>10} {'답변길이':>10}")
    print("-" * 80)

    for model, result in results.items():
        if "error" not in result:
            gen_time = result['generation_time_ms']
            tokens = result['tokens_used']
            answer_len = len(result['answer'])
            print(f"{model:<20} {gen_time:>15.2f} {tokens:>10} {answer_len:>10}")
        else:
            print(f"{model:<20} {'ERROR':>15} {'-':>10} {'-':>10}")

    # 결과 저장
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_query = safe_query.replace(' ', '_')[:50]
        filename = f"model_comparison_{safe_query}_{timestamp}.json"

        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)

        comparison_data = {
            "query": query,
            "embedding_model": embedding_model,
            "timestamp": timestamp,
            "models": results
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 비교 결과 저장: {filepath}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="여러 LLM 모델 답변 비교")
    parser.add_argument("query", type=str, help="질문")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gemma3:4b", "llama3.2:latest", "qwen3:4b"],
        help="비교할 모델들"
    )
    parser.add_argument("--embedding", type=str, default="E5", help="임베딩 모델")
    parser.add_argument("--no-save", action="store_true", help="결과 저장 안 함")

    args = parser.parse_args()

    compare_models(
        query=args.query,
        models=args.models,
        embedding_model=args.embedding,
        save_results=not args.no_save
    )

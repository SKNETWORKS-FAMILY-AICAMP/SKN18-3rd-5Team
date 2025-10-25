#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
임베딩 모델별 답변 비교 (간단 버전)
메모리 효율적으로 한 번에 1개 모델씩 실행하여 답변만 비교
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import gc

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from ..rag_system import RAGSystem
from ..models.config import EmbeddingModelType
from ..generation.generator import OllamaGenerator, GenerationConfig
from ..augmentation.formatters import EnhancedPromptFormatter
import logging

logging.basicConfig(level=logging.WARNING)  # 로그 최소화
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


def test_single_model(
    model_name: str,
    model_type: EmbeddingModelType,
    query: str,
    llm_model: str,
    db_config: dict
) -> dict:
    """단일 모델 테스트 (메모리 효율적)"""

    print(f"\n{'='*60}")
    print(f"🔍 {model_name} 모델 테스트 중...")
    print(f"{'='*60}")

    try:
        # LLM Generator 초기화
        llm_generator = OllamaGenerator(
            base_url="http://localhost:11434",
            default_model=llm_model
        )

        # RAG 시스템 초기화
        rag_system = RAGSystem(
            model_type=model_type,
            db_config=db_config,
            formatter=EnhancedPromptFormatter(),
            llm_generator=llm_generator,
            enable_generation=True
        )

        # 검색
        print(f"1️⃣ 검색 중...")
        search_results = rag_system.search_only(query=query, top_k=3)
        avg_similarity = sum(d.get('similarity', 0) for d in search_results) / len(search_results) if search_results else 0
        print(f"   ✅ {len(search_results)}개 문서 (평균 유사도: {avg_similarity:.3f})")

        # 답변 생성
        print(f"2️⃣ 답변 생성 중...")
        full_response = rag_system.generate_answer(
            query=query,
            top_k=3,  # 메모리 절약을 위해 3개만
            generation_config=GenerationConfig(
                model=llm_model,
                temperature=0.7,
                max_tokens=1000
            )
        )

        answer = full_response.generated_answer.answer
        gen_time = full_response.generated_answer.generation_time_ms

        print(f"   ✅ 완료 ({gen_time:.0f}ms)")

        # 검색된 문서 정보
        doc_sources = [
            doc.get('metadata', {}).get('source', 'N/A')
            for doc in search_results
        ]

        result = {
            "model_name": model_name,
            "answer": answer,
            "generation_time_ms": gen_time,
            "avg_similarity": avg_similarity,
            "doc_sources": doc_sources,
            "success": True
        }

        # 메모리 정리
        del rag_system
        del llm_generator
        gc.collect()

        return result

    except Exception as e:
        print(f"   ❌ 오류: {e}")
        return {
            "model_name": model_name,
            "error": str(e),
            "success": False
        }


def generate_comparison_report(
    query: str,
    llm_model: str,
    results: list,
    output_dir: str
) -> str:
    """자연어 답변 중심 비교 보고서 생성"""

    os.makedirs(output_dir, exist_ok=True)

    # 파일명
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"answer_comparison_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)

    # 마크다운 생성
    md = f"""# 임베딩 모델별 답변 비교

## 📋 실험 정보

- **질문**: {query}
- **LLM 모델**: {llm_model}
- **비교 모델**: {len([r for r in results if r['success']])}개
- **생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 간단 요약

| 모델 | 생성 시간 | 유사도 | 상태 |
|------|-----------|--------|------|
"""

    for result in results:
        if result['success']:
            md += f"| {result['model_name']} | {result['generation_time_ms']:.0f}ms | {result['avg_similarity']:.3f} | ✅ |\n"
        else:
            md += f"| {result['model_name']} | - | - | ❌ |\n"

    md += "\n---\n\n"

    # 각 모델의 답변 비교
    md += "## 💬 답변 내용 비교\n\n"

    for i, result in enumerate(results, 1):
        if not result['success']:
            md += f"### {i}. ❌ {result['model_name']} 모델 - 오류\n\n"
            md += f"```\n{result.get('error', 'Unknown error')}\n```\n\n"
            md += "---\n\n"
            continue

        md += f"### {i}. {result['model_name']} 모델\n\n"

        # 검색 정보
        md += f"**검색 정보**:\n"
        md += f"- 평균 유사도: {result['avg_similarity']:.3f}\n"
        md += f"- 참고 문서: {len(result['doc_sources'])}개\n"
        for j, source in enumerate(result['doc_sources'], 1):
            md += f"  {j}. {source}\n"
        md += f"\n**생성 시간**: {result['generation_time_ms']:.0f}ms\n\n"

        # 답변
        md += f"**답변**:\n\n{result['answer']}\n\n"
        md += "---\n\n"

    # 분석
    successful = [r for r in results if r['success']]
    if len(successful) > 1:
        md += "## 🔍 답변 분석\n\n"

        # 가장 빠른 모델
        fastest = min(successful, key=lambda x: x['generation_time_ms'])
        md += f"### ⚡ 가장 빠른 답변\n**{fastest['model_name']}** ({fastest['generation_time_ms']:.0f}ms)\n\n"

        # 가장 높은 유사도
        best_sim = max(successful, key=lambda x: x['avg_similarity'])
        md += f"### 🎯 가장 높은 검색 품질\n**{best_sim['model_name']}** (유사도: {best_sim['avg_similarity']:.3f})\n\n"

        # 가장 긴 답변
        longest = max(successful, key=lambda x: len(x['answer']))
        md += f"### 📝 가장 상세한 답변\n**{longest['model_name']}** ({len(longest['answer'])} 글자)\n\n"

        md += "### 💡 권장 사항\n\n"
        md += f"- **검색 품질 우선**: {best_sim['model_name']}\n"
        md += f"- **속도 우선**: {fastest['model_name']}\n"
        md += f"- **상세함 우선**: {longest['model_name']}\n\n"

    md += "---\n\n"
    md += f"*보고서 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

    # 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(md)

    return filepath


def main():
    import argparse

    parser = argparse.ArgumentParser(description="임베딩 모델별 답변 비교 (간단 버전)")
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default="청년 전세대출 조건과 금리",
        help="테스트 질문"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemma3:4b",
        help="LLM 모델 (기본: gemma3:4b)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["E5", "KAKAO", "QWEN", "GEMMA"],
        help="비교할 임베딩 모델들"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="결과 저장 디렉토리"
    )

    args = parser.parse_args()

    # 모델 매핑
    model_mapping = {
        "E5": EmbeddingModelType.MULTILINGUAL_E5_SMALL,
        "KAKAO": EmbeddingModelType.KAKAOBANK_DEBERTA,
        "QWEN": EmbeddingModelType.QWEN_EMBEDDING,
        "GEMMA": EmbeddingModelType.EMBEDDING_GEMMA
    }

    db_config = get_db_config()

    print(f"\n{'='*60}")
    print(f"임베딩 모델별 답변 비교")
    print(f"{'='*60}")
    print(f"질문: {args.query}")
    print(f"LLM: {args.llm_model}")
    print(f"모델: {', '.join(args.models)}")
    print(f"{'='*60}\n")

    # 각 모델 테스트 (순차 실행)
    results = []
    for model_name in args.models:
        if model_name not in model_mapping:
            print(f"⚠️  '{model_name}' 모델을 찾을 수 없습니다. 건너뜀.")
            continue

        result = test_single_model(
            model_name=model_name,
            model_type=model_mapping[model_name],
            query=args.query,
            llm_model=args.llm_model,
            db_config=db_config
        )
        results.append(result)

    # 보고서 생성
    print(f"\n{'='*60}")
    print(f"📝 보고서 생성 중...")
    report_path = generate_comparison_report(
        query=args.query,
        llm_model=args.llm_model,
        results=results,
        output_dir=args.output_dir
    )

    print(f"\n{'='*60}")
    print(f"✅ 완료!")
    print(f"📄 보고서: {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

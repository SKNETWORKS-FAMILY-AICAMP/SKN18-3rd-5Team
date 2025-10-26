#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
임베딩 모델별 RAG 비교 분석
4가지 임베딩 모델로 검색, 증강, 생성 결과를 비교하여 마크다운 보고서 생성
"""

import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from ..rag_system import RAGSystem
from ..models.config import EmbeddingModelType
from ..generation.generator import OllamaGenerator, GenerationConfig
from ..augmentation.formatters import EnhancedPromptFormatter, PolicyFormatter
import logging

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


def compare_all_models(
    query: str,
    llm_model: str = "gemma3:4b",
    formatter_name: str = "enhanced",
    output_dir: str = "results"
):
    """
    4가지 임베딩 모델로 전체 RAG 파이프라인 비교

    Args:
        query: 테스트 질문
        llm_model: 사용할 LLM 모델
        formatter_name: 사용할 포맷터 (enhanced 또는 policy)
        output_dir: 결과 저장 디렉토리
    """
    db_config = get_db_config()

    # LLM Generator 초기화
    llm_generator = OllamaGenerator(
        base_url="http://localhost:11434",
        default_model=llm_model
    )

    # Ollama 서버 확인
    if not llm_generator.check_health():
        logger.error("Ollama 서버가 실행 중이 아닙니다.")
        return None

    # 포맷터 선택
    if formatter_name == "policy":
        formatter = PolicyFormatter()
    else:
        formatter = EnhancedPromptFormatter()

    # 임베딩 모델 목록
    models = [
        ("E5", EmbeddingModelType.MULTILINGUAL_E5_SMALL),
        ("KAKAO", EmbeddingModelType.KAKAOBANK_DEBERTA),
        ("QWEN", EmbeddingModelType.QWEN_EMBEDDING),
        ("GEMMA", EmbeddingModelType.EMBEDDING_GEMMA)
    ]

    results = {}

    print(f"\n{'='*80}")
    print(f"임베딩 모델별 RAG 비교 분석")
    print(f"{'='*80}")
    print(f"질문: {query}")
    print(f"LLM 모델: {llm_model}")
    print(f"포맷터: {formatter_name}")
    print(f"{'='*80}\n")

    # 각 임베딩 모델로 테스트
    for name, model_type in models:
        print(f"\n{'='*80}")
        print(f"🔍 {name} 모델 테스트 중...")
        print(f"{'='*80}\n")

        try:
            # RAG 시스템 초기화
            rag_system = RAGSystem(
                model_type=model_type,
                db_config=db_config,
                formatter=formatter,
                llm_generator=llm_generator,
                enable_generation=True
            )

            # 1. 검색만 수행
            print(f"1️⃣ 검색 중...")
            search_results = rag_system.search_only(query=query, top_k=5)
            print(f"   ✅ {len(search_results)}개 문서 검색 완료")
            avg_similarity = sum(d.get('similarity', 0) for d in search_results) / len(search_results) if search_results else 0
            print(f"   평균 유사도: {avg_similarity:.3f}\n")

            # 2. 검색 + 증강
            print(f"2️⃣ 증강 중...")
            augment_response = rag_system.retrieve_and_augment(query=query, top_k=5)
            augmented_context = augment_response.augmented_context.context_text
            print(f"   ✅ 증강 완료")
            print(f"   컨텍스트 길이: {len(augmented_context)} 글자\n")

            # 3. 전체 파이프라인 (검색 + 증강 + 생성)
            print(f"3️⃣ LLM 답변 생성 중...")
            full_response = rag_system.generate_answer(
                query=query,
                top_k=5,
                generation_config=GenerationConfig(
                    model=llm_model,
                    temperature=0.7,
                    max_tokens=1500
                )
            )

            generated_answer = full_response.generated_answer.answer
            generation_time = full_response.generated_answer.generation_time_ms
            tokens_used = full_response.generated_answer.tokens_used

            print(f"   ✅ 답변 생성 완료")
            print(f"   생성 시간: {generation_time:.2f}ms")
            print(f"   토큰 수: {tokens_used}\n")

            # 결과 저장
            results[name] = {
                "model_type": model_type.name,
                "search_results": search_results,
                "avg_similarity": avg_similarity,
                "augmented_context": augmented_context,
                "context_length": len(augmented_context),
                "generated_answer": generated_answer,
                "generation_time_ms": generation_time,
                "tokens_used": tokens_used
            }

        except Exception as e:
            logger.error(f"{name} 모델 테스트 중 오류: {e}")
            results[name] = {
                "error": str(e)
            }

    # 마크다운 보고서 생성
    report_path = generate_markdown_report(
        query=query,
        llm_model=llm_model,
        formatter_name=formatter_name,
        results=results,
        output_dir=output_dir
    )

    print(f"\n{'='*80}")
    print(f"✅ 분석 완료!")
    print(f"📄 보고서 저장: {report_path}")
    print(f"{'='*80}\n")

    return report_path


def generate_markdown_report(
    query: str,
    llm_model: str,
    formatter_name: str,
    results: dict,
    output_dir: str
) -> str:
    """마크다운 보고서 생성"""

    os.makedirs(output_dir, exist_ok=True)

    # 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_query = safe_query.replace(' ', '_')[:50]
    filename = f"embedding_model_comparison_{safe_query}_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)

    # 마크다운 생성
    md_content = f"""# 임베딩 모델별 RAG 성능 비교 보고서

## 실험 정보

- **질문**: {query}
- **LLM 모델**: {llm_model}
- **포맷터**: {formatter_name}
- **생성 시간**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **비교 모델 수**: {len(results)}개

---

## 📊 종합 비교표

| 임베딩 모델 | 평균 유사도 | 컨텍스트 길이 | 생성 시간(ms) | 토큰 수 | 답변 길이 |
|------------|-------------|---------------|---------------|---------|-----------|
"""

    # 비교표 작성
    for model_name, result in results.items():
        if "error" in result:
            md_content += f"| {model_name} | ERROR | - | - | - | - |\n"
        else:
            avg_sim = result['avg_similarity']
            ctx_len = result['context_length']
            gen_time = result['generation_time_ms']
            tokens = result['tokens_used']
            ans_len = len(result['generated_answer'])
            md_content += f"| {model_name} | {avg_sim:.3f} | {ctx_len} | {gen_time:.2f} | {tokens} | {ans_len} |\n"

    md_content += "\n---\n\n"

    # 각 모델별 상세 결과
    for model_name, result in results.items():
        if "error" in result:
            md_content += f"## ❌ {model_name} 모�� - 오류 발생\n\n"
            md_content += f"```\n{result['error']}\n```\n\n---\n\n"
            continue

        md_content += f"## 🔍 {model_name} 모델\n\n"
        md_content += f"**모델 타입**: `{result['model_type']}`\n\n"

        # 1. 검색 결과
        md_content += f"### 1️⃣ 검색 결과\n\n"
        md_content += f"- **검색된 문서 수**: {len(result['search_results'])}개\n"
        md_content += f"- **평균 유사도**: {result['avg_similarity']:.3f}\n\n"

        md_content += "#### 검색된 문서 목록\n\n"
        for i, doc in enumerate(result['search_results'], 1):
            similarity = doc.get('similarity', 0)
            source = doc.get('metadata', {}).get('source', 'N/A')
            content = doc.get('content', '')[:200].replace('\n', ' ')

            md_content += f"{i}. **[유사도: {similarity:.3f}]** {source}\n"
            md_content += f"   > {content}...\n\n"

        # 2. 증강 결과
        md_content += f"### 2️⃣ 증강 결과\n\n"
        md_content += f"- **컨텍스트 길이**: {result['context_length']} 글자\n"
        md_content += f"- **포맷터**: {formatter_name}\n\n"

        md_content += "#### 증강된 컨텍스트\n\n"
        md_content += "```\n"
        # 컨텍스트가 너무 길면 앞부분만
        context = result['augmented_context']
        if len(context) > 2000:
            md_content += context[:2000] + "\n... (이하 생략)\n"
        else:
            md_content += context + "\n"
        md_content += "```\n\n"

        # 3. LLM 생성 결과
        md_content += f"### 3️⃣ LLM 생성 결과\n\n"
        md_content += f"- **생성 시간**: {result['generation_time_ms']:.2f}ms\n"
        md_content += f"- **사용 토큰**: {result['tokens_used']}개\n"
        md_content += f"- **답변 길이**: {len(result['generated_answer'])} 글자\n\n"

        md_content += "#### 생성된 답변\n\n"
        md_content += f"{result['generated_answer']}\n\n"

        md_content += "---\n\n"

    # 분석 및 결론
    md_content += "## 📈 분석 및 결론\n\n"

    # 유사도 비교
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        best_similarity = max(valid_results.items(), key=lambda x: x[1]['avg_similarity'])
        worst_similarity = min(valid_results.items(), key=lambda x: x[1]['avg_similarity'])

        md_content += "### 검색 품질 (평균 유사도)\n\n"
        md_content += f"- **최고**: {best_similarity[0]} ({best_similarity[1]['avg_similarity']:.3f})\n"
        md_content += f"- **최저**: {worst_similarity[0]} ({worst_similarity[1]['avg_similarity']:.3f})\n\n"

        # 생성 속도 비교
        fastest = min(valid_results.items(), key=lambda x: x[1]['generation_time_ms'])
        slowest = max(valid_results.items(), key=lambda x: x[1]['generation_time_ms'])

        md_content += "### 생성 속도\n\n"
        md_content += f"- **가장 빠름**: {fastest[0]} ({fastest[1]['generation_time_ms']:.2f}ms)\n"
        md_content += f"- **가장 느림**: {slowest[0]} ({slowest[1]['generation_time_ms']:.2f}ms)\n\n"

        # 답변 길이 비교
        longest = max(valid_results.items(), key=lambda x: len(x[1]['generated_answer']))
        shortest = min(valid_results.items(), key=lambda x: len(x[1]['generated_answer']))

        md_content += "### 답변 길이\n\n"
        md_content += f"- **가장 긴 답변**: {longest[0]} ({len(longest[1]['generated_answer'])} 글자)\n"
        md_content += f"- **가장 짧은 답변**: {shortest[0]} ({len(shortest[1]['generated_answer'])} 글자)\n\n"

    md_content += "### 권장 사항\n\n"
    md_content += "- **검색 품질 우선**: 평균 유사도가 높은 모델 선택\n"
    md_content += "- **속도 우선**: 생성 시간이 짧은 모델 선택\n"
    md_content += "- **균형**: 검색 품질과 속도를 모두 고려하여 선택\n\n"

    md_content += "---\n\n"
    md_content += f"*보고서 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

    # 파일 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(md_content)

    return filepath


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="임베딩 모델별 RAG 성능 비교")
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
        help="사용할 LLM 모델 (기본: gemma3:4b)"
    )
    parser.add_argument(
        "--formatter",
        type=str,
        default="enhanced",
        choices=["enhanced", "policy"],
        help="사용할 포맷터 (기본: enhanced)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="결과 저장 디렉토리 (기본: results)"
    )

    args = parser.parse_args()

    compare_all_models(
        query=args.query,
        llm_model=args.llm_model,
        formatter_name=args.formatter,
        output_dir=args.output_dir
    )

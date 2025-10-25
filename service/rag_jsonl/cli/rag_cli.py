#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 시스템 통합 CLI

Usage:
  rag-eval all              # 5개 모델 전체 평가
  rag-eval model --model E5  # 특정 모델 평가
  rag-eval reranking        # 리랭킹 효과 비교
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from ..models.config import EmbeddingModelType, get_model_config
from ..core.evaluator import RAGEvaluator
from ..core.embedder import MultiModelEmbedder
from ..rag_system import RAGSystem
from ..retrieval.reranker import KeywordReranker, SemanticReranker, CombinedReranker
from ..augmentation.formatters import (
    PromptFormatter,
    MarkdownFormatter,
    JSONFormatter,
    PolicyFormatter,
    EnhancedPromptFormatter
)
from ..generation.generator import OllamaGenerator, GenerationConfig
from backend.services.db.common.db_utils import test_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_db_config() -> dict:
    """데이터베이스 설정 가져오기"""
    return {
        'host': os.getenv('PG_HOST', 'localhost'),
        'port': os.getenv('PG_PORT', '5432'),
        'database': os.getenv('PG_DB', 'rey'),
        'user': os.getenv('PG_USER', 'postgres'),
        'password': os.getenv('PG_PASSWORD', 'post1234')
    }


def list_models():
    """사용 가능한 모델 목록 출력"""
    print("\n" + "="*80)
    print("사용 가능한 임베딩 모델")
    print("="*80)
    print(f"{'모델 타입':<30} {'이름':<40} {'차원':>6}")
    print("-"*80)

    for model_type in EmbeddingModelType:
        config = get_model_config(model_type)
        print(f"{model_type.name:<30} {config.display_name:<40} {config.dimension:>6}")

    print("="*80)
    print("\n사용 방법:")
    print("  rag-eval model --model E5")
    print("  rag-eval model --model KAKAOBANK_DEBERTA")
    print()


def evaluate_single_model(
    model_type: EmbeddingModelType,
    db_config: dict,
    top_k: int = 5,
    use_reranking: bool = False,
    save_results: bool = True,
    save_search_results: bool = False
) -> bool:
    """단일 모델 평가"""
    try:
        logger.info(f"=== {model_type.value} 모델 평가 시작 ===")

        evaluator = RAGEvaluator(db_config)
        result = evaluator.evaluate_model(
            model_type=model_type,
            top_k=top_k,
            use_reranking=use_reranking,
            save_search_results=save_search_results
        )

        if result.get('status') == 'failed':
            logger.error(f"평가 실패: {result.get('error')}")
            return False

        # 결과 저장
        if save_results:
            output_path = evaluator.save_results({'evaluation': result})
            print(f"\n✅ 평가 결과 저장: {output_path}")

        return True

    except Exception as e:
        logger.exception(f"모델 평가 중 오류 발생: {e}")
        return False


def evaluate_all_models(
    db_config: dict,
    top_k: int = 5,
    compare_reranking: bool = False,
    save_results: bool = True,
    save_search_results: bool = False
) -> bool:
    """전체 모델 평가"""
    try:
        logger.info("=== 5개 모델 전체 평가 시작 ===")

        # 모든 모델
        models = list(EmbeddingModelType)

        evaluator = RAGEvaluator(db_config)
        results = evaluator.evaluate_all_models(
            models=models,
            top_k=top_k,
            compare_reranking=compare_reranking,
            save_search_results=save_search_results
        )

        # 요약 출력
        print(results['summary'])

        # 상세 비교 출력
        print_comparison_table(results)

        # 리랭킹 비교 (옵션)
        if compare_reranking and results.get('reranking_comparison'):
            print_reranking_comparison(results['reranking_comparison'])

        # 결과 저장
        if save_results:
            output_path = evaluator.save_results(results)
            print(f"\n✅ 전체 평가 결과 저장: {output_path}")

        return True

    except Exception as e:
        logger.exception(f"전체 모델 평가 중 오류 발생: {e}")
        return False


def print_comparison_table(results: dict):
    """비교 결과를 표로 출력"""
    print("\n" + "="*120)
    print("모델 성능 비교")
    print("="*120)

    comparisons = results.get('comparison', {}).get('comparisons', {})

    if not comparisons:
        print("비교 결과가 없습니다.")
        return

    # 헤더
    print(f"{'모델명':<40} {'평균 Latency':>15} {'P95 Latency':>15} {'유사도':>10} {'Recall@5':>12} {'MRR':>10}")
    print("-"*120)

    # 각 모델 정보
    for model_name in comparisons.get('accuracy', {}).keys():
        latency = comparisons.get('latency', {}).get(model_name, {})
        accuracy = comparisons.get('accuracy', {}).get(model_name, 0)
        recall = comparisons.get('recall', {}).get(model_name, 0)
        mrr = comparisons.get('mrr', {}).get(model_name, 0)

        avg_lat = latency.get('avg', 0)
        p95_lat = latency.get('p95', 0)

        print(
            f"{model_name:<40} "
            f"{avg_lat:>13.2f}ms "
            f"{p95_lat:>13.2f}ms "
            f"{accuracy:>10.4f} "
            f"{recall:>12.4f} "
            f"{mrr:>10.4f}"
        )

    print("="*120)


def print_reranking_comparison(reranking_effects: dict):
    """리랭킹 효과 비교 출력"""
    print("\n" + "="*100)
    print("리랭킹 효과 비교")
    print("="*100)
    print(f"{'모델명':<30} {'유사도 개선':>15} {'Latency 오버헤드':>20} {'Recall@5 개선':>18} {'NDCG@5 개선':>18}")
    print("-"*100)

    for model_name, improvements in reranking_effects.items():
        sim_imp = improvements.get('similarity', 0)
        lat_over = improvements.get('latency_overhead', 0)
        recall_imp = improvements.get('recall_at_5', 0)
        ndcg_imp = improvements.get('ndcg_at_5', 0)

        print(
            f"{model_name:<30} "
            f"{sim_imp:>13.2f}% "
            f"{lat_over:>18.2f}ms "
            f"{recall_imp:>18.4f} "
            f"{ndcg_imp:>18.4f}"
        )

    print("="*100)


def embed_all_models(
    data_file: str,
    db_config: dict
) -> bool:
    """5개 모델로 데이터 임베딩"""
    try:
        logger.info("=== 5개 모델 임베딩 생성 시작 ===")

        if not Path(data_file).exists():
            logger.error(f"데이터 파일을 찾을 수 없습니다: {data_file}")
            return False

        embedder = MultiModelEmbedder(data_file, db_config, skip_chunking=True)
        results = embedder.embed_all_models()

        # 결과 요약 출력
        print(embedder.get_summary())

        # 성공한 모델 확인
        successful = [k for k, v in results.items() if v.get("status") == "success"]

        if successful:
            logger.info(f"✅ {len(successful)}개 모델 임베딩 완료")
            return True
        else:
            logger.error("❌ 모든 모델 임베딩 실패")
            return False

    except Exception as e:
        logger.exception(f"임베딩 생성 중 오류 발생: {e}")
        return False


def get_model_type_from_name(model_name: str) -> EmbeddingModelType:
    """모델 이름을 EmbeddingModelType으로 변환"""
    mapping = {
        "E5": EmbeddingModelType.MULTILINGUAL_E5_SMALL,
        "KAKAO": EmbeddingModelType.KAKAOBANK_DEBERTA,
        "QWEN": EmbeddingModelType.QWEN_EMBEDDING,
        "GEMMA": EmbeddingModelType.EMBEDDING_GEMMA
    }
    return mapping.get(model_name.upper(), EmbeddingModelType.MULTILINGUAL_E5_SMALL)


def get_formatter_from_name(format_name: str):
    """포맷 이름을 포맷터 객체로 변환"""
    formatters = {
        "prompt": PromptFormatter(),
        "markdown": MarkdownFormatter(),
        "json": JSONFormatter(),
        "policy": PolicyFormatter(),
        "enhanced": EnhancedPromptFormatter()
    }
    return formatters.get(format_name.lower(), PromptFormatter())


def save_augment_result_to_markdown(
    query: str,
    response: Any,
    model: str,
    output_dir: str = "results"
) -> str:
    """증강 결과를 마크다운 파일로 저장"""
    import os
    from datetime import datetime
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 생성 (쿼리 + 타임스탬프)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_query = safe_query.replace(' ', '_')[:50]  # 최대 50자
    filename = f"augment_{safe_query}_{model}_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    # 마크다운 내용 생성
    markdown_content = f"""# RAG 증강 결과

## 쿼리
{query}

## 모델 정보
- 모델: {model}
- 생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 검색된 문서 ({len(response.retrieved_documents)}개)

"""
    
    # 검색된 문서들 추가
    for i, doc in enumerate(response.retrieved_documents, 1):
        similarity = doc.get('similarity', 0)
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})
        
        markdown_content += f"""### 문서 {i} (유사도: {similarity:.3f})

**내용:**
{content}

**메타데이터:**
- 출처: {metadata.get('source', 'N/A')}
- 원본 ID: {metadata.get('original_id', 'N/A')}

---

"""
    
    # 증강된 컨텍스트 추가
    markdown_content += f"""## 증강된 컨텍스트

**토큰 수:** {response.augmented_context.token_count}
**처리 시간:** {response.augmented_context.processing_time_ms:.2f}ms

**컨텍스트:**
```
{response.augmented_context.context_text}
```

## 메타데이터

- 총 문서 수: {response.augmented_context.metadata.get('total_documents', 0)}
- 선택된 문서 수: {response.augmented_context.metadata.get('selected_documents', 0)}
- 컨텍스트 길이: {response.augmented_context.metadata.get('context_length', 0)}
- 평균 유사도: {response.augmented_context.metadata.get('avg_similarity', 0):.3f}
"""
    
    # 파일 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return filepath


def run_search_command(args, db_config: dict) -> bool:
    """검색 명령어 실행"""
    try:
        model_type = get_model_type_from_name(args.model)
        formatter = get_formatter_from_name(args.format)
        
        # Reranker 설정
        reranker = None
        if args.reranking:
            reranker = KeywordReranker()
        
        # RAG 시스템 초기화
        rag_system = RAGSystem(
            model_type=model_type,
            db_config=db_config,
            reranker=reranker,
            formatter=formatter
        )
        
        # 검색 실행
        print(f"\n🔍 검색 중... (모델: {args.model}, Top-K: {args.top_k})")
        search_results = rag_system.search_only(
            query=args.query,
            top_k=args.top_k,
            use_reranker=args.reranking
        )
        
        # 결과 출력
        print(f"\n📊 검색 결과 ({len(search_results)}개 문서)")
        print("=" * 80)
        
        for i, doc in enumerate(search_results, 1):
            print(f"\n[문서 {i}] (유사도: {doc.get('similarity', 0):.3f})")
            print(f"내용: {doc.get('content', '')[:200]}...")
            if doc.get('metadata'):
                print(f"메타데이터: {doc.get('metadata')}")
        
        return True
        
    except Exception as e:
        logger.exception(f"검색 중 오류 발생: {e}")
        return False


def run_augment_command(args, db_config: dict) -> bool:
    """증강 명령어 실행"""
    try:
        model_type = get_model_type_from_name(args.model)
        formatter = get_formatter_from_name(args.format)
        
        # Reranker 설정
        reranker = None
        if args.reranking:
            reranker = KeywordReranker()
        
        # RAG 시스템 초기화
        rag_system = RAGSystem(
            model_type=model_type,
            db_config=db_config,
            reranker=reranker,
            formatter=formatter
        )
        
        # 검색 및 증강 실행
        print(f"\n🔍 검색 및 증강 중... (모델: {args.model}, 포맷: {args.format})")
        response = rag_system.retrieve_and_augment(
            query=args.query,
            top_k=args.top_k,
            use_reranker=args.reranking,
            context_type=args.context_type
        )
        
        # 결과 출력
        print(f"\n📊 검색 결과 ({len(response.retrieved_documents)}개 문서)")
        print("=" * 80)
        
        for i, doc in enumerate(response.retrieved_documents, 1):
            print(f"\n[문서 {i}] (유사도: {doc.get('similarity', 0):.3f})")
            print(f"내용: {doc.get('content', '')[:200]}...")
        
        print(f"\n🤖 증강된 컨텍스트 (토큰 수: {response.augmented_context.token_count})")
        print("=" * 80)
        print(response.augmented_context.context_text)
        
        # 마크다운 파일 저장
        if args.save:
            filepath = save_augment_result_to_markdown(
                query=args.query,
                response=response,
                model=args.model,
                output_dir=args.output_dir
            )
            print(f"\n💾 결과가 저장되었습니다: {filepath}")
        
        return True
        
    except Exception as e:
        logger.exception(f"증강 중 오류 발생: {e}")
        return False


def run_rag_eval_command(args, db_config: dict) -> bool:
    """RAG 시스템 평가 명령어 실행"""
    try:
        model_type = get_model_type_from_name(args.model)

        # Reranker 설정
        reranker = None
        if args.reranking:
            reranker = KeywordReranker()

        # RAG 시스템 초기화
        rag_system = RAGSystem(
            model_type=model_type,
            db_config=db_config,
            reranker=reranker
        )

        # 테스트 쿼리 로드
        test_queries = [
            "주거복지사업이란 무엇인가요?",
            "청년 주거 지원 정책은 어떤 것들이 있나요?",
            "임대주택 신청 방법을 알려주세요",
            "주거급여 신청 자격은 무엇인가요?",
            "공공임대주택과 민간임대주택의 차이점은 무엇인가요?"
        ]

        print(f"\n🧪 RAG 시스템 평가 중... (모델: {args.model}, 쿼리: {len(test_queries)}개)")

        # 평가 실행
        eval_results = rag_system.evaluate_retrieval(
            queries=test_queries,
            top_k=args.top_k,
            use_reranker=args.reranking
        )

        # 결과 출력
        print(f"\n📊 RAG 시스템 평가 결과")
        print("=" * 80)
        print(f"총 쿼리 수: {eval_results['total_queries']}")
        print(f"성공한 쿼리: {eval_results['successful_queries']}")
        print(f"실패한 쿼리: {eval_results['failed_queries']}")
        print(f"평균 검색 시간: {eval_results['avg_time_ms']:.2f}ms")
        print(f"쿼리당 평균 문서 수: {eval_results['avg_documents_per_query']:.1f}")
        print(f"평균 유사도: {eval_results['avg_similarity']:.3f}")

        # 개별 쿼리 결과
        print(f"\n📋 개별 쿼리 결과")
        print("-" * 80)
        for i, query_result in enumerate(eval_results['queries'], 1):
            if query_result['success']:
                print(f"{i}. {query_result['query'][:50]}...")
                print(f"   문서 수: {query_result['documents_found']}, 유사도: {query_result['avg_similarity']:.3f}, 시간: {query_result['time_ms']:.1f}ms")
            else:
                print(f"{i}. {query_result['query'][:50]}... (실패: {query_result['error']})")

        return True

    except Exception as e:
        logger.exception(f"RAG 평가 중 오류 발생: {e}")
        return False


def run_generate_command(args, db_config: dict) -> bool:
    """전체 RAG 파이프라인 실행 (검색 + 증강 + 생성)"""
    try:
        model_type = get_model_type_from_name(args.model)
        formatter = get_formatter_from_name(args.format)

        # Reranker 설정
        reranker = None
        if args.reranking:
            reranker = KeywordReranker()

        # LLM Generator 초기화
        llm_generator = OllamaGenerator(
            base_url=args.ollama_url,
            default_model=args.llm_model
        )

        # Ollama 서버 상태 확인
        if not llm_generator.check_health():
            logger.error(f"Ollama 서버에 연결할 수 없습니다: {args.ollama_url}")
            print(f"\n❌ Ollama 서버가 실행 중이 아닙니다.")
            print(f"다음 명령어로 Ollama를 시작하세요: ollama serve")
            return False

        # 사용 가능한 모델 확인
        available_models = llm_generator.list_models()
        if args.llm_model not in available_models:
            logger.warning(f"모델 '{args.llm_model}'이 설치되어 있지 않습니다.")
            print(f"\n⚠️  모델 '{args.llm_model}'을 찾을 수 없습니다.")
            print(f"다음 명령어로 모델을 설치하세요: ollama pull {args.llm_model}")
            print(f"\n사용 가능한 모델: {', '.join(available_models)}")
            return False

        # RAG 시스템 초기화
        rag_system = RAGSystem(
            model_type=model_type,
            db_config=db_config,
            reranker=reranker,
            formatter=formatter,
            llm_generator=llm_generator,
            enable_generation=True
        )

        # 생성 설정
        gen_config = GenerationConfig(
            model=args.llm_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )

        # 전체 RAG 파이프라인 실행
        print(f"\n🔍 RAG 파이프라인 실행 중...")
        print(f"  - 임베딩 모델: {args.model}")
        print(f"  - LLM 모델: {args.llm_model}")
        print(f"  - 검색 문서 수: {args.top_k}")
        print(f"  - 리랭킹: {'사용' if args.reranking else '미사용'}")

        response = rag_system.generate_answer(
            query=args.query,
            top_k=args.top_k,
            use_reranker=args.reranking,
            context_type=args.context_type,
            generation_config=gen_config
        )

        # 결과 출력
        print(f"\n" + "="*80)
        print(f"📝 질문: {args.query}")
        print("="*80)

        print(f"\n🤖 AI 답변:")
        print("-"*80)
        print(response.generated_answer.answer)
        print("-"*80)

        print(f"\n📊 생성 정보:")
        print(f"  - 모델: {response.generated_answer.model}")
        print(f"  - 생성 시간: {response.generated_answer.generation_time_ms:.2f}ms")
        print(f"  - 토큰 수: {response.generated_answer.tokens_used}")

        print(f"\n📚 참고 문서 ({len(response.retrieved_documents)}개):")
        for i, doc in enumerate(response.retrieved_documents, 1):
            similarity = doc.get('similarity', 0)
            source = doc.get('metadata', {}).get('source', 'N/A')
            print(f"  {i}. [{similarity:.3f}] {source}")

        # 결과 저장
        if args.save:
            from datetime import datetime
            import json

            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query = "".join(c for c in args.query if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_query = safe_query.replace(' ', '_')[:50]
            filename = f"rag_result_{safe_query}_{args.llm_model.replace(':', '_')}_{timestamp}.json"
            filepath = os.path.join(args.output_dir, filename)

            result_data = {
                "query": args.query,
                "answer": response.generated_answer.answer,
                "model_info": {
                    "embedding_model": args.model,
                    "llm_model": response.generated_answer.model,
                    "generation_time_ms": response.generated_answer.generation_time_ms,
                    "tokens_used": response.generated_answer.tokens_used
                },
                "retrieved_documents": [
                    {
                        "similarity": doc.get('similarity'),
                        "content": doc.get('content'),
                        "metadata": doc.get('metadata')
                    }
                    for doc in response.retrieved_documents
                ],
                "context": response.augmented_context.context_text,
                "timestamp": timestamp
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            print(f"\n💾 결과가 저장되었습니다: {filepath}")

        return True

    except Exception as e:
        logger.exception(f"RAG 생성 중 오류 발생: {e}")
        return False


def main():
    # 환경 변수 설정
    os.environ.setdefault("PG_USER", "postgres")
    os.environ.setdefault("PG_PASSWORD", "post1234")
    os.environ.setdefault("PG_HOST", "localhost")
    os.environ.setdefault("PG_PORT", "5432")
    os.environ.setdefault("PG_DB", "rey")

    parser = argparse.ArgumentParser(
        description="rag: RAG 시스템 도구 (검색, 증강, 평가)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")

    subparsers = parser.add_subparsers(dest="command", help="사용 가능한 명령어")

    # 모델 목록
    p_list = subparsers.add_parser("list", help="사용 가능한 모델 목록")

    # 단일 모델 평가
    p_model = subparsers.add_parser("model", help="단일 모델 평가")
    p_model.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[m.name for m in EmbeddingModelType],
        help="평가할 모델"
    )
    p_model.add_argument("--top-k", type=int, default=5, help="검색할 결과 수 (기본: 5)")
    p_model.add_argument("--reranking", action="store_true", help="리랭킹 사용")
    p_model.add_argument("--no-save", action="store_true", help="결과 저장 안 함")
    p_model.add_argument("--save-search-results", action="store_true", help="검색 결과도 함께 저장 (파일 크기 증가)")

    # 전체 모델 평가
    p_all = subparsers.add_parser("all", help="전체 모델 평가")
    p_all.add_argument("--top-k", type=int, default=5, help="검색할 결과 수 (기본: 5)")
    p_all.add_argument("--no-save", action="store_true", help="결과 저장 안 함")
    p_all.add_argument("--save-search-results", action="store_true", help="검색 결과도 함께 저장 (파일 크기 증가)")

    # 리랭킹 비교
    p_rerank = subparsers.add_parser("reranking", help="리랭킹 전후 성능 비교")
    p_rerank.add_argument("--top-k", type=int, default=5, help="검색할 결과 수 (기본: 5)")
    p_rerank.add_argument("--no-save", action="store_true", help="결과 저장 안 함")
    p_rerank.add_argument("--save-search-results", action="store_true", help="검색 결과도 함께 저장 (파일 크기 증가)")

    # RAG 시스템 검색
    p_search = subparsers.add_parser("search", help="RAG 시스템으로 검색")
    p_search.add_argument("query", type=str, help="검색 쿼리")
    p_search.add_argument("--model", type=str, default="E5", choices=["E5", "KAKAO", "QWEN", "GEMMA"], help="사용할 모델")
    p_search.add_argument("--top-k", type=int, default=5, help="검색할 결과 수")
    p_search.add_argument("--reranking", action="store_true", help="리랭킹 사용")
    p_search.add_argument("--format", type=str, default="prompt", choices=["prompt", "markdown", "json", "policy", "enhanced"], help="출력 포맷")
    
    # RAG 시스템 증강
    p_augment = subparsers.add_parser("augment", help="검색 결과 증강")
    p_augment.add_argument("query", type=str, help="검색 쿼리")
    p_augment.add_argument("--model", type=str, default="E5", choices=["E5", "KAKAO", "QWEN", "GEMMA"], help="사용할 모델")
    p_augment.add_argument("--top-k", type=int, default=5, help="검색할 결과 수")
    p_augment.add_argument("--reranking", action="store_true", help="리랭킹 사용")
    p_augment.add_argument("--format", type=str, default="prompt", choices=["prompt", "markdown", "json", "policy", "enhanced"], help="출력 포맷")
    p_augment.add_argument("--context-type", type=str, default="general", choices=["general", "qa", "summarization"], help="컨텍스트 타입")
    p_augment.add_argument("--save", action="store_true", help="결과를 마크다운 파일로 저장")
    p_augment.add_argument("--output-dir", type=str, default="results", help="저장할 디렉토리")
    
    # RAG 시스템 평가
    p_rag_eval = subparsers.add_parser("rag-eval", help="RAG 시스템 전체 평가")
    p_rag_eval.add_argument("--model", type=str, default="E5", choices=["E5", "KAKAO", "QWEN", "GEMMA"], help="사용할 모델")
    p_rag_eval.add_argument("--top-k", type=int, default=5, help="검색할 결과 수")
    p_rag_eval.add_argument("--reranking", action="store_true", help="리랭킹 사용")
    p_rag_eval.add_argument("--no-save", action="store_true", help="결과 저장 안 함")

    # RAG 생성 (전체 파이프라인)
    p_generate = subparsers.add_parser("generate", help="RAG 전체 파이프라인 (검색 + 증강 + 생성)")
    p_generate.add_argument("query", type=str, help="질문")
    p_generate.add_argument("--model", type=str, default="E5", choices=["E5", "KAKAO", "QWEN", "GEMMA"], help="임베딩 모델")
    p_generate.add_argument("--llm-model", type=str, default="gemma2:2b", help="LLM 모델 (예: gemma2:2b)")
    p_generate.add_argument("--top-k", type=int, default=5, help="검색할 결과 수")
    p_generate.add_argument("--reranking", action="store_true", help="리랭킹 사용")
    p_generate.add_argument("--format", type=str, default="enhanced", choices=["prompt", "markdown", "json", "policy", "enhanced"], help="컨텍스트 포맷")
    p_generate.add_argument("--context-type", type=str, default="general", choices=["general", "qa", "summarization"], help="컨텍스트 타입")
    p_generate.add_argument("--temperature", type=float, default=0.7, help="생성 온도 (0.0-1.0)")
    p_generate.add_argument("--max-tokens", type=int, default=1000, help="최대 토큰 수")
    p_generate.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama API URL")
    p_generate.add_argument("--save", action="store_true", help="결과를 JSON 파일로 저장")
    p_generate.add_argument("--output-dir", type=str, default="results", help="저장할 디렉토리")

    # 임베딩 생성
    p_embed = subparsers.add_parser("embed", help="4개 모델로 데이터 임베딩")
    p_embed.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="임베딩할 JSON 데이터 파일"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # DB 연결 테스트
    if args.command not in ['list']:
        logger.info("DB 연결 테스트 중...")
        if not test_connection():
            logger.error("DB 연결 실패")
            sys.exit(1)
        logger.info("DB 연결 성공")

    db_config = get_db_config()

    success = False

    # 명령어 실행
    if args.command == "list":
        list_models()
        success = True

    elif args.command == "model":
        model_type = EmbeddingModelType[args.model]
        success = evaluate_single_model(
            model_type=model_type,
            db_config=db_config,
            top_k=args.top_k,
            use_reranking=args.reranking,
            save_results=not args.no_save,
            save_search_results=args.save_search_results
        )

    elif args.command == "all":
        success = evaluate_all_models(
            db_config=db_config,
            top_k=args.top_k,
            compare_reranking=False,
            save_results=not args.no_save,
            save_search_results=args.save_search_results
        )

    elif args.command == "reranking":
        success = evaluate_all_models(
            db_config=db_config,
            top_k=args.top_k,
            compare_reranking=True,
            save_results=not args.no_save,
            save_search_results=args.save_search_results
        )
    
    elif args.command == "search":
        success = run_search_command(args, db_config)
    
    elif args.command == "augment":
        success = run_augment_command(args, db_config)
    
    elif args.command == "rag-eval":
        success = run_rag_eval_command(args, db_config)

    elif args.command == "generate":
        success = run_generate_command(args, db_config)

    elif args.command == "embed":
        success = embed_all_models(
            data_file=args.data_file,
            db_config=db_config
        )

    if success:
        logger.info(f"{args.command} 명령이 성공적으로 완료되었습니다.")
        sys.exit(0)
    else:
        logger.error(f"{args.command} 명령 실행 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()

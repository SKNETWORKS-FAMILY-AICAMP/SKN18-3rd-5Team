#!/usr/bin/env python3
"""
RAG 시스템 평가 CLI
evaluation_queries.json을 사용한 자동 평가 실행
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from service.rag.evaluation import RAGEvaluator, EvaluationConfig
from service.rag.models.config import EmbeddingModelType

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="RAG 시스템 평가 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 평가 (E5 모델, 검색만)
  python evaluate_rag.py

  # 답변 생성 포함 평가
  python evaluate_rag.py --enable-generation

  # 특정 모델로 평가
  python evaluate_rag.py --model kakaobank

  # 여러 모델 비교
  python evaluate_rag.py --compare-models e5 kakaobank fine5

  # 특정 쿼리만 평가
  python evaluate_rag.py --query-ids Q001 Q002

  # 난이도별 필터링
  python evaluate_rag.py --difficulty easy medium
        """
    )
    
    # 기본 옵션
    parser.add_argument(
        "--model", 
        choices=["e5", "kakaobank", "fine5"],
        default="e5",
        help="사용할 임베딩 모델 (기본값: e5)"
    )
    
    parser.add_argument(
        "--enable-generation",
        action="store_true",
        help="답변 생성 활성화 (LLM 필요)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="검색할 문서 수 (기본값: 5)"
    )
    
    # 필터링 옵션
    parser.add_argument(
        "--query-ids",
        nargs="+",
        help="평가할 쿼리 ID 목록 (예: Q001 Q002)"
    )
    
    parser.add_argument(
        "--difficulty",
        nargs="+",
        choices=["easy", "medium", "hard"],
        help="평가할 난이도 필터"
    )
    
    parser.add_argument(
        "--query-type",
        nargs="+",
        help="평가할 쿼리 타입 필터"
    )
    
    # 비교 평가 옵션
    parser.add_argument(
        "--compare-models",
        nargs="+",
        choices=["e5", "kakaobank", "fine5"],
        help="여러 모델 비교 평가"
    )
    
    # 출력 옵션
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="결과 저장 디렉토리 (기본값: evaluation_results)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="상세 결과 저장하지 않음"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력"
    )
    
    args = parser.parse_args()
    
    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 모델 타입 매핑
    model_mapping = {
        "e5": EmbeddingModelType.MULTILINGUAL_E5_SMALL,
        "kakaobank": EmbeddingModelType.KAKAOBANK_DEBERTA,
        "fine5": EmbeddingModelType.FINE5_FINANCE
    }
    
    try:
        if args.compare_models:
            # 모델 비교 평가
            run_model_comparison(args, model_mapping)
        else:
            # 단일 모델 평가
            run_single_evaluation(args, model_mapping)
            
    except KeyboardInterrupt:
        logger.info("평가가 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"평가 실행 중 오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_single_evaluation(args, model_mapping):
    """단일 모델 평가 실행"""
    model_type = model_mapping[args.model]
    
    logger.info(f"RAG 평가 시작 - 모델: {args.model}")
    logger.info(f"답변 생성: {'활성화' if args.enable_generation else '비활성화'}")
    logger.info(f"검색 문서 수: {args.top_k}")
    
    # 평가 설정
    config = EvaluationConfig(
        top_k=args.top_k,
        enable_generation=args.enable_generation,
        model_type=model_type,
        output_dir=args.output_dir,
        save_detailed_results=not args.no_save
    )
    
    # 평가기 생성
    evaluator = RAGEvaluator(config=config)
    
    # 쿼리 필터링
    filtered_queries = filter_queries(evaluator.evaluation_queries, args)
    
    if not filtered_queries:
        logger.warning("필터 조건에 맞는 쿼리가 없습니다.")
        return
    
    logger.info(f"평가할 쿼리 수: {len(filtered_queries)}")
    
    # 평가 실행
    results = []
    for i, query_data in enumerate(filtered_queries, 1):
        logger.info(f"[{i}/{len(filtered_queries)}] {query_data['query_id']}: {query_data['query']}")
        
        result = evaluator.evaluate_single_query(query_data)
        results.append(result)
        
        # 즉시 결과 출력
        print(f"  점수: {result.overall_score:.1f}/100")
        print(f"  검색: Recall={result.retrieval.recall_at_k:.3f}, Precision={result.retrieval.precision_at_k:.3f}")
        print(f"  응답시간: {result.response_time_ms:.0f}ms")
        if result.error_message:
            print(f"  에러: {result.error_message}")
        print()
    
    # 요약 통계
    print_summary(results)


def run_model_comparison(args, model_mapping):
    """모델 비교 평가 실행"""
    model_types = [model_mapping[model] for model in args.compare_models]
    
    logger.info(f"모델 비교 평가 시작: {args.compare_models}")
    
    # 평가 설정
    config = EvaluationConfig(
        top_k=args.top_k,
        enable_generation=args.enable_generation,
        output_dir=args.output_dir,
        save_detailed_results=not args.no_save
    )
    
    # 첫 번째 모델로 평가기 생성 (쿼리 로드용)
    evaluator = RAGEvaluator(config=config)
    
    # 쿼리 필터링
    filtered_queries = filter_queries(evaluator.evaluation_queries, args)
    
    if not filtered_queries:
        logger.warning("필터 조건에 맞는 쿼리가 없습니다.")
        return
    
    logger.info(f"비교할 쿼리 수: {len(filtered_queries)}")
    
    # 모델별 평가
    comparison_results = {}
    
    for model_name in args.compare_models:
        logger.info(f"모델 평가 중: {model_name}")
        
        model_type = model_mapping[model_name]
        config.model_type = model_type
        
        # 새로운 평가기 생성
        model_evaluator = RAGEvaluator(config=config)
        
        # 평가 실행
        model_results = []
        for query_data in filtered_queries:
            result = model_evaluator.evaluate_single_query(query_data)
            model_results.append(result)
        
        comparison_results[model_name] = model_results
    
    # 비교 결과 출력
    print_comparison_summary(comparison_results)


def filter_queries(queries, args):
    """쿼리 필터링"""
    filtered = queries.copy()
    
    # 쿼리 ID 필터
    if args.query_ids:
        filtered = [q for q in filtered if q['query_id'] in args.query_ids]
    
    # 난이도 필터
    if args.difficulty:
        filtered = [q for q in filtered if q.get('difficulty') in args.difficulty]
    
    # 쿼리 타입 필터
    if args.query_type:
        filtered = [q for q in filtered if q.get('query_type') in args.query_type]
    
    return filtered


def print_summary(results):
    """평가 결과 요약 출력"""
    valid_results = [r for r in results if not r.error_message]
    error_count = len(results) - len(valid_results)
    
    print("\n" + "=" * 60)
    print("📊 평가 결과 요약")
    print("=" * 60)
    
    print(f"총 쿼리 수: {len(results)}")
    print(f"성공한 쿼리: {len(valid_results)}")
    print(f"실패한 쿼리: {error_count}")
    
    if valid_results:
        avg_score = sum(r.overall_score for r in valid_results) / len(valid_results)
        avg_time = sum(r.response_time_ms for r in valid_results) / len(valid_results)
        
        print(f"평균 점수: {avg_score:.2f}/100")
        print(f"평균 응답시간: {avg_time:.0f}ms")
        
        # 검색 메트릭 평균
        avg_recall = sum(r.retrieval.recall_at_k for r in valid_results) / len(valid_results)
        avg_precision = sum(r.retrieval.precision_at_k for r in valid_results) / len(valid_results)
        avg_mrr = sum(r.retrieval.mrr for r in valid_results) / len(valid_results)
        
        print(f"평균 Recall@{5}: {avg_recall:.3f}")
        print(f"평균 Precision@{5}: {avg_precision:.3f}")
        print(f"평균 MRR: {avg_mrr:.3f}")
        
        # 난이도별 성능
        difficulty_stats = {}
        for result in valid_results:
            diff = result.difficulty
            if diff not in difficulty_stats:
                difficulty_stats[diff] = []
            difficulty_stats[diff].append(result.overall_score)
        
        if difficulty_stats:
            print("\n📈 난이도별 성능:")
            for difficulty, scores in difficulty_stats.items():
                avg_score = sum(scores) / len(scores)
                print(f"  {difficulty}: {avg_score:.2f} ({len(scores)}개 쿼리)")


def print_comparison_summary(comparison_results):
    """모델 비교 결과 요약 출력"""
    print("\n" + "=" * 80)
    print("📊 모델 비교 결과")
    print("=" * 80)
    
    # 모델별 성능 요약
    print(f"{'모델':<15} {'평균점수':<10} {'평균응답시간':<15} {'Recall@5':<10} {'Precision@5':<12}")
    print("-" * 80)
    
    for model_name, results in comparison_results.items():
        valid_results = [r for r in results if not r.error_message]
        if valid_results:
            avg_score = sum(r.overall_score for r in valid_results) / len(valid_results)
            avg_time = sum(r.response_time_ms for r in valid_results) / len(valid_results)
            avg_recall = sum(r.retrieval.recall_at_k for r in valid_results) / len(valid_results)
            avg_precision = sum(r.retrieval.precision_at_k for r in valid_results) / len(valid_results)
            
            print(f"{model_name:<15} {avg_score:<10.2f} {avg_time:<15.0f} {avg_recall:<10.3f} {avg_precision:<12.3f}")
    
    # 쿼리별 상세 비교
    print("\n📋 쿼리별 상세 비교:")
    print("-" * 80)
    
    # 첫 번째 모델의 쿼리 순서를 기준으로
    first_model = list(comparison_results.keys())[0]
    first_results = comparison_results[first_model]
    
    for result in first_results:
        if result.error_message:
            continue
            
        print(f"[{result.query_id}] {result.query}")
        print("-" * 40)
        
        for model_name, results in comparison_results.items():
            model_result = next((r for r in results if r.query_id == result.query_id), None)
            if model_result and not model_result.error_message:
                print(f"{model_name}: {model_result.overall_score:.1f}점 "
                      f"(Recall={model_result.retrieval.recall_at_k:.3f}, "
                      f"응답시간={model_result.response_time_ms:.0f}ms)")
        
        print()


if __name__ == "__main__":
    main()

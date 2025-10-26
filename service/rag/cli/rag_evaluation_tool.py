#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 RAG 평가 도구
- RAG 검색 결과 생성 (rag_success 형식)
- 단일 모드, 단일 결과물
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 프로젝트 루트를 Python 경로에 추가 (직접 실행 시 경로 문제 해결)
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]  # service/rag/cli/ -> project_root (SKN18-3rd-5Team)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from service.rag.rag_system import RAGSystem
from service.rag.models.config import EmbeddingModelType
from service.rag.evaluation.evaluator import RAGEvaluator, EvaluationConfig
from service.rag.evaluation.metrics import MetricsCalculator
from service.rag.cli.rag_cli import RAGJSONLSystem, get_db_config
from config.vector_database import get_vector_db_config
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluationTool:
    """통합 RAG 평가 도구"""
    
    def __init__(self, embedding_model: str = "intfloat/multilingual-e5-small"):
        """초기화"""
        self.embedding_model = embedding_model
        self.rag_system = RAGJSONLSystem(
            db_config=get_db_config(),
            embedding_model="intfloat/multilingual-e5-small"
        )
        
        # 메트릭 계산기 초기화
        self.metrics_calculator = MetricsCalculator()
        
        logger.info(f"RAG Evaluation Tool 초기화 완료: {embedding_model}")
    
    def _load_evaluation_queries(self) -> List[Dict[str, Any]]:
        """평가 쿼리 로드"""
        queries_path = project_root / "service" / "rag" / "cli" / "evaluation_queries.json"
        try:
            with open(queries_path, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            logger.info(f"평가 쿼리 로드 완료: {len(queries)}개")
            return queries
        except Exception as e:
            logger.error(f"평가 쿼리 로드 실패: {e}")
            return []
    
    def run_evaluation(self, 
                      queries: List[Dict[str, Any]] = None,
                      top_k: int = 5,
                      min_similarity: float = 0.0,
                      corp_filter: str = None) -> List[Dict[str, Any]]:
        """
        RAG 평가 실행 (통합된 단일 메서드)
        
        Args:
            queries: 검색할 쿼리 리스트 (기본값: evaluation_queries.json)
            top_k: 상위 K개 결과 반환
            min_similarity: 최소 유사도 임계값
            corp_filter: 기업 필터 (예: "삼성전자")
        
        Returns:
            검색 결과 리스트
        """
        if queries is None:
            queries = self._load_evaluation_queries()
        
        logger.info("🚀 RAG 평가 시작")
        logger.info(f"📊 모델: {self.embedding_model}")
        logger.info(f"🔍 Top-K: {top_k}")
        if corp_filter:
            logger.info(f"🏢 기업 필터: {corp_filter}")
        logger.info("=" * 80)
        
        results = []
        
        for i, query_data in enumerate(queries, 1):
            query_id = query_data.get("query_id", f"Q{i:03d}")
            query_text = query_data.get("query", "")
            
            logger.info(f"[{i}/{len(queries)}] {query_id}: {query_text}")
            
            try:
                # 검색 실행
                start_time = time.time()
                search_results = self.rag_system.search(
                    query=query_text,
                    top_k=top_k,
                    min_similarity=min_similarity,
                    corp_filter=corp_filter
                )
                search_time = (time.time() - start_time) * 1000  # ms
                
                # 결과 포맷팅
                formatted_results = []
                for result in search_results:
                    formatted_result = {
                        "chunk_id": result["chunk_id"],
                        "content": result["natural_text"],  # natural_text를 content로 매핑
                        "similarity": result["similarity"],
                        "search_time_ms": search_time,
                        "metadata": result["metadata"]
                    }
                    formatted_results.append(formatted_result)
                
                # 메트릭 계산 (complete_evaluation.py의 기능)
                expected_keywords = query_data.get("expected_keywords", [])
                expected_docs = query_data.get("expected_docs", [])
                
                retrieval_metrics = self.metrics_calculator.calculate_retrieval_metrics(
                    query=query_text,
                    retrieved_docs=formatted_results,
                    expected_keywords=expected_keywords,
                    expected_docs=expected_docs,
                    k=top_k
                )
                
                # 쿼리 결과 저장 (메트릭 포함)
                query_result = {
                    "query_id": query_id,
                    "query": query_text,
                    "search_results": formatted_results,
                    "retrieval_metrics": {
                        "recall_at_k": retrieval_metrics.recall_at_k,
                        "precision_at_k": retrieval_metrics.precision_at_k,
                        "mrr": retrieval_metrics.mrr,
                        "ndcg_at_k": retrieval_metrics.ndcg_at_k,
                        "avg_similarity": retrieval_metrics.avg_similarity,
                        "keyword_coverage": retrieval_metrics.keyword_coverage
                    }
                }
                
                results.append(query_result)
                
                if formatted_results:
                    logger.info(f"✅ {len(formatted_results)}개 결과 반환 (유사도: {formatted_results[0]['similarity']:.4f} ~ {formatted_results[-1]['similarity']:.4f})")
                else:
                    logger.info(f"⚠️  결과 없음 (임베딩 데이터 부족 또는 필터 조건 불일치)")
                
            except Exception as e:
                logger.error(f"❌ 오류 발생: {str(e)}")
                # 오류가 발생해도 계속 진행
                error_result = {
                    "query_id": query_id,
                    "query": query_text,
                    "search_results": [],
                    "error": str(e)
                }
                results.append(error_result)
        
        logger.info(f"🎉 RAG 평가 완료!")
        logger.info(f"📈 총 {len(results)}개 쿼리 처리 완료")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]]) -> str:
        """
        결과를 파일로 저장 (JSON + 텍스트 파일)
        
        Args:
            results: 저장할 결과
            
        Returns:
            저장된 파일 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_int = int(time.time())
        
        # JSON 파일 저장
        json_filename = f"rag_evaluation_{timestamp}.json"
        json_path = project_root / "service" / "rag" / "results" / json_filename
        
        # 결과 디렉토리 생성
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # JSON 결과 저장
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"💾 JSON 결과 저장 완료: {json_path}")
        
        # 텍스트 파일들 저장 (complete_evaluation.py와 동일한 형식)
        self._save_detailed_results(results, timestamp_int)
        self._save_summary_report(results, timestamp_int)
        
        return str(json_path)
    
    def _save_detailed_results(self, results: List[Dict[str, Any]], timestamp: int):
        """상세 결과 텍스트 파일 저장"""
        detailed_filename = f"detailed_results_{timestamp}.json"
        detailed_path = project_root / "service" / "rag" / "results" / detailed_filename
        
        # 상세 결과 포맷팅 (complete_evaluation.py와 동일한 형식)
        detailed_results = []
        for result in results:
            detailed_result = {
                "query_id": result.get("query_id", "unknown"),
                "query": result.get("query", "unknown"),
                "query_type": "factual_numerical",  # 기본값
                "difficulty": "easy",  # 기본값
                "retrieval": result.get("retrieval_metrics", {}),
                "generation": None,
                "overall_score": 0.0,
                "response_time_ms": 0.0,
                "retrieved_docs_count": len(result.get("search_results", [])),
                "error_message": None
            }
            
            # 전체 점수 계산 (retrieval 메트릭 기반)
            if detailed_result["retrieval"]:
                recall = detailed_result["retrieval"].get("recall_at_k", 0.0)
                precision = detailed_result["retrieval"].get("precision_at_k", 0.0)
                mrr = detailed_result["retrieval"].get("mrr", 0.0)
                detailed_result["overall_score"] = (recall + precision + mrr) / 3 * 100
            
            # 응답 시간 계산
            if result.get("search_results"):
                detailed_result["response_time_ms"] = result["search_results"][0].get("search_time_ms", 0.0)
            
            detailed_results.append(detailed_result)
        
        # 상세 결과 저장
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📄 상세 결과 저장 완료: {detailed_path}")
    
    def _save_summary_report(self, results: List[Dict[str, Any]], timestamp: int):
        """요약 리포트 텍스트 파일 저장"""
        summary_filename = f"summary_report_{timestamp}.txt"
        summary_path = project_root / "service" / "rag" / "results" / summary_filename
        
        # 요약 리포트 생성
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAG 평가 요약 리포트\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"평가 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"모델: {self.embedding_model}\n")
            f.write(f"총 쿼리 수: {len(results)}\n\n")
            
            # 성공한 쿼리 수
            successful_queries = [r for r in results if r.get("search_results")]
            f.write(f"성공한 쿼리: {len(successful_queries)}/{len(results)}\n\n")
            
            # 평균 메트릭 계산
            valid_results = [r for r in results if "retrieval_metrics" in r and r["retrieval_metrics"]]
            if valid_results:
                avg_recall = sum(r["retrieval_metrics"]["recall_at_k"] for r in valid_results) / len(valid_results)
                avg_precision = sum(r["retrieval_metrics"]["precision_at_k"] for r in valid_results) / len(valid_results)
                avg_mrr = sum(r["retrieval_metrics"]["mrr"] for r in valid_results) / len(valid_results)
                avg_ndcg = sum(r["retrieval_metrics"]["ndcg_at_k"] for r in valid_results) / len(valid_results)
                avg_keyword_coverage = sum(r["retrieval_metrics"]["keyword_coverage"] for r in valid_results) / len(valid_results)
                
                f.write("평균 성능 지표:\n")
                f.write(f"  - Recall@K: {avg_recall:.4f}\n")
                f.write(f"  - Precision@K: {avg_precision:.4f}\n")
                f.write(f"  - MRR: {avg_mrr:.4f}\n")
                f.write(f"  - NDCG@K: {avg_ndcg:.4f}\n")
                f.write(f"  - Keyword Coverage: {avg_keyword_coverage:.4f}\n\n")
            
            # 평균 유사도
            if successful_queries:
                all_similarities = []
                for result in successful_queries:
                    for search_result in result["search_results"]:
                        all_similarities.append(search_result.get("similarity", 0.0))
                
                if all_similarities:
                    avg_similarity = sum(all_similarities) / len(all_similarities)
                    f.write(f"평균 유사도: {avg_similarity:.4f}\n\n")
            
            # 개별 쿼리 결과
            f.write("개별 쿼리 결과:\n")
            f.write("-" * 80 + "\n")
            
            for i, result in enumerate(results, 1):
                query_id = result.get("query_id", f"Q{i:03d}")
                query = result.get("query", "unknown")
                search_count = len(result.get("search_results", []))
                
                f.write(f"{i}. {query_id}: {query}\n")
                f.write(f"   검색 결과: {search_count}개\n")
                
                if "retrieval_metrics" in result and result["retrieval_metrics"]:
                    metrics = result["retrieval_metrics"]
                    f.write(f"   Recall@K: {metrics.get('recall_at_k', 0.0):.4f}\n")
                    f.write(f"   Precision@K: {metrics.get('precision_at_k', 0.0):.4f}\n")
                    f.write(f"   MRR: {metrics.get('mrr', 0.0):.4f}\n")
                    f.write(f"   Keyword Coverage: {metrics.get('keyword_coverage', 0.0):.4f}\n")
                
                f.write("\n")
        
        logger.info(f"📋 요약 리포트 저장 완료: {summary_path}")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """결과 요약 출력"""
        print(f"\n📊 평가 결과 요약:")
        print(f"   - 총 쿼리 수: {len(results)}")
        print(f"   - 모델: {self.embedding_model}")
        
        # 성공한 쿼리 수 계산
        successful_queries = [r for r in results if r.get("search_results")]
        print(f"   - 성공한 쿼리: {len(successful_queries)}/{len(results)}")
        
        # 메트릭이 있는 결과만 필터링
        valid_results = [r for r in results if "retrieval_metrics" in r and r["retrieval_metrics"]]
        
        if valid_results:
            # 평균 메트릭 계산 (complete_evaluation.py와 동일)
            avg_recall_at_k = sum(r["retrieval_metrics"]["recall_at_k"] for r in valid_results) / len(valid_results)
            avg_precision_at_k = sum(r["retrieval_metrics"]["precision_at_k"] for r in valid_results) / len(valid_results)
            avg_mrr = sum(r["retrieval_metrics"]["mrr"] for r in valid_results) / len(valid_results)
            avg_ndcg_at_k = sum(r["retrieval_metrics"]["ndcg_at_k"] for r in valid_results) / len(valid_results)
            avg_keyword_coverage = sum(r["retrieval_metrics"]["keyword_coverage"] for r in valid_results) / len(valid_results)
            
            print(f"\n📈 평균 성능 지표:")
            print(f"   - Recall@K: {avg_recall_at_k:.4f}")
            print(f"   - Precision@K: {avg_precision_at_k:.4f}")
            print(f"   - MRR: {avg_mrr:.4f}")
            print(f"   - NDCG@K: {avg_ndcg_at_k:.4f}")
            print(f"   - Keyword Coverage: {avg_keyword_coverage:.4f}")
        
        if successful_queries:
            # 평균 유사도 계산
            all_similarities = []
            for result in successful_queries:
                for search_result in result["search_results"]:
                    all_similarities.append(search_result.get("similarity", 0.0))
            
            if all_similarities:
                avg_similarity = sum(all_similarities) / len(all_similarities)
                print(f"   - 평균 유사도: {avg_similarity:.4f}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="통합 RAG 평가 도구")
    parser.add_argument("--top-k", type=int, default=5, help="검색할 상위 K개 문서")
    parser.add_argument("--min-similarity", type=float, default=0.0, help="최소 유사도 임계값")
    parser.add_argument("--corp-filter", type=str, help="기업 필터 (예: 삼성전자)")
    parser.add_argument("--model", choices=["multilingual-e5-small", "kakaobank", "fine5"], 
                       default="multilingual-e5-small", help="임베딩 모델")
    
    args = parser.parse_args()
    
    try:
        # 도구 초기화
        tool = RAGEvaluationTool(embedding_model=args.model)
        
        # 평가 실행
        results = tool.run_evaluation(
            top_k=args.top_k,
            min_similarity=args.min_similarity,
            corp_filter=args.corp_filter
        )
        
        # 결과 저장 및 요약 출력
        output_path = tool.save_results(results)
        tool.print_summary(results)
        
        print(f"\n💾 결과 저장 위치: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ 오류 발생: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
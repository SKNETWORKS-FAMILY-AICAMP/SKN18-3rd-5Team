#!/usr/bin/env python3
"""
RAG 시스템 평가기
evaluation_queries.json을 사용한 자동 평가
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from .metrics import MetricsCalculator, EvaluationMetrics, RetrievalMetrics, GenerationMetrics
from ..rag_system import RAGSystem
from ..models.config import EmbeddingModelType
import psycopg2
from psycopg2.extras import RealDictCursor
import os

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """평가 설정"""
    queries_path: str = "service/rag_jsonl/cli/evaluation_queries.json"
    top_k: int = 5
    enable_generation: bool = False
    model_type: EmbeddingModelType = EmbeddingModelType.MULTILINGUAL_E5_SMALL
    output_dir: str = "service/rag_jsonl/results"
    save_detailed_results: bool = True
    save_to_db: bool = True  # DB 저장 여부
    db_config: Optional[Dict[str, str]] = None  # DB 설정


class RAGEvaluator:
    """RAG 시스템 평가기"""
    
    def __init__(
        self, 
        rag_system: Optional[RAGSystem] = None,
        config: Optional[EvaluationConfig] = None
    ):
        """
        Args:
            rag_system: 평가할 RAG 시스템 (None이면 새로 생성)
            config: 평가 설정
        """
        self.config = config or EvaluationConfig()
        self.metrics_calculator = MetricsCalculator()
        
        # RAG 시스템 초기화
        if rag_system:
            self.rag_system = rag_system
        else:
            self.rag_system = RAGSystem(
                model_type=self.config.model_type,
                db_config=self.config.db_config.get_db_config() if hasattr(self.config.db_config, 'get_db_config') else self.config.db_config,
                enable_generation=self.config.enable_generation
            )
        
        # 평가 쿼리 로드
        self.evaluation_queries = self._load_evaluation_queries()
        
        # DB 연결 (평가 결과 저장용)
        self.db_conn = None
        if self.config.save_to_db:
            self._connect_db()
        
        logger.info(f"RAG Evaluator 초기화 완료: {len(self.evaluation_queries)}개 쿼리")
    
    def _load_evaluation_queries(self) -> List[Dict[str, Any]]:
        """평가 쿼리 로드"""
        # 프로젝트 루트를 기준으로 경로 계산
        project_root = Path(__file__).parents[3]  # service/rag_jsonl/evaluation/ -> project_root
        queries_path = project_root / self.config.queries_path
        
        try:
            with open(queries_path, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            logger.info(f"평가 쿼리 로드 완료: {len(queries)}개")
            return queries
        except FileNotFoundError:
            logger.error(f"평가 쿼리 파일을 찾을 수 없습니다: {queries_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"평가 쿼리 JSON 파싱 오류: {e}")
            return []
    
    def _connect_db(self):
        """DB 연결"""
        try:
            db_config = self.config.db_config or {
                'host': os.getenv('PG_HOST', 'localhost'),
                'port': os.getenv('PG_PORT', '5432'),
                'database': os.getenv('PG_DB', 'postgres'),
                'user': os.getenv('PG_USER', 'postgres'),
                'password': os.getenv('PG_PASSWORD', 'postgres')
            }
            
            self.db_conn = psycopg2.connect(**db_config)
            logger.info("평가 결과 DB 연결 성공")
            
        except Exception as e:
            logger.warning(f"평가 결과 DB 연결 실패: {e}. DB 저장 비활성화.")
            self.db_conn = None
            self.config.save_to_db = False
    
    def _save_to_db(self, result: EvaluationMetrics, model_name: str):
        """평가 결과를 DB에 저장"""
        if not self.db_conn:
            return
        
        try:
            cursor = self.db_conn.cursor()
            
            sql = """
                INSERT INTO vector_db.evaluation_results (
                    model_name, query_id, query_text, query_type, difficulty,
                    recall_at_k, precision_at_k, mrr, ndcg_at_k, 
                    avg_similarity, keyword_coverage,
                    exact_match, keyword_f1, keyword_precision, keyword_recall,
                    contains_ground_truth,
                    overall_score, response_time_ms, retrieved_docs_count,
                    error_message
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s, %s, %s,
                    %s,
                    %s, %s, %s,
                    %s
                )
            """
            
            values = (
                model_name,
                result.query_id,
                result.query,
                result.query_type,
                result.difficulty,
                result.retrieval.recall_at_k,
                result.retrieval.precision_at_k,
                result.retrieval.mrr,
                result.retrieval.ndcg_at_k,
                result.retrieval.avg_similarity,
                result.retrieval.keyword_coverage,
                result.generation.exact_match if result.generation else None,
                result.generation.keyword_f1 if result.generation else None,
                result.generation.keyword_precision if result.generation else None,
                result.generation.keyword_recall if result.generation else None,
                result.generation.contains_ground_truth if result.generation else None,
                result.overall_score,
                result.response_time_ms,
                result.retrieved_docs_count,
                result.error_message
            )
            
            cursor.execute(sql, values)
            self.db_conn.commit()
            cursor.close()
            
            logger.debug(f"평가 결과 DB 저장 완료: {result.query_id}")
            
        except Exception as e:
            logger.error(f"평가 결과 DB 저장 실패: {e}")
            self.db_conn.rollback()
    
    def evaluate_all(self) -> List[EvaluationMetrics]:
        """모든 쿼리 평가"""
        results = []
        
        logger.info(f"전체 {len(self.evaluation_queries)}개 쿼리 평가 시작")
        
        for i, query_data in enumerate(self.evaluation_queries, 1):
            logger.info(f"[{i}/{len(self.evaluation_queries)}] 평가 중: {query_data['query']}")
            
            try:
                result = self.evaluate_single_query(query_data)
                results.append(result)
                
                # DB에 저장
                if self.config.save_to_db:
                    self._save_to_db(result, self.config.model_type.value)
                
                # 진행률 표시
                if i % 3 == 0 or i == len(self.evaluation_queries):
                    logger.info(f"진행률: {i}/{len(self.evaluation_queries)} ({i/len(self.evaluation_queries)*100:.1f}%)")
                    
            except Exception as e:
                logger.error(f"쿼리 {query_data['query_id']} 평가 실패: {e}")
                # 에러 결과 생성
                error_result = EvaluationMetrics(
                    query_id=query_data['query_id'],
                    query=query_data['query'],
                    query_type=query_data.get('query_type', 'unknown'),
                    difficulty=query_data.get('difficulty', 'unknown'),
                    retrieval=RetrievalMetrics(),
                    error_message=str(e)
                )
                results.append(error_result)
        
        # 결과 저장
        if self.config.save_detailed_results:
            self._save_results(results)
        
        return results
    
    def evaluate_single_query(self, query_data: Dict[str, Any]) -> EvaluationMetrics:
        """단일 쿼리 평가"""
        start_time = time.time()
        
        query = query_data['query']
        expected_keywords = query_data.get('expected_keywords', [])
        expected_docs = query_data.get('expected_docs', [])
        ground_truth = query_data.get('ground_truth_answer', '')
        
        # RAG 시스템으로 쿼리 실행
        try:
            if self.config.enable_generation:
                rag_response = self.rag_system.query(query)
                retrieved_docs = rag_response.retrieved_documents
                generated_answer = rag_response.generated_answer.text if rag_response.generated_answer else ""
            else:
                # 검색만 수행
                retrieved_docs = self.rag_system.search_only(query, top_k=self.config.top_k)
                generated_answer = ""
        except Exception as e:
            logger.error(f"RAG 시스템 실행 오류: {str(e)}")
            raise
        
        response_time = (time.time() - start_time) * 1000  # ms
        
        # 검색 메트릭 계산
        retrieval_metrics = self.metrics_calculator.calculate_retrieval_metrics(
            query=query,
            retrieved_docs=retrieved_docs,
            expected_keywords=expected_keywords,
            expected_docs=expected_docs,
            k=self.config.top_k
        )
        
        # 생성 메트릭 계산 (답변 생성이 활성화된 경우)
        generation_metrics = None
        if self.config.enable_generation and generated_answer:
            generation_metrics = self.metrics_calculator.calculate_generation_metrics(
                generated_answer=generated_answer,
                ground_truth_answer=ground_truth,
                expected_keywords=expected_keywords
            )
        
        # 전체 점수 계산
        overall_score = self._calculate_overall_score(retrieval_metrics, generation_metrics)
        
        return EvaluationMetrics(
            query_id=query_data['query_id'],
            query=query,
            query_type=query_data.get('query_type', 'unknown'),
            difficulty=query_data.get('difficulty', 'unknown'),
            retrieval=retrieval_metrics,
            generation=generation_metrics,
            overall_score=overall_score,
            response_time_ms=response_time,
            retrieved_docs_count=len(retrieved_docs)
        )
    
    def _calculate_overall_score(
        self, 
        retrieval: RetrievalMetrics, 
        generation: Optional[GenerationMetrics]
    ) -> float:
        """전체 점수 계산 (0-100)"""
        
        # 검색 점수 (70% 가중치)
        retrieval_score = (
            retrieval.recall_at_k * 0.3 +
            retrieval.precision_at_k * 0.3 +
            retrieval.mrr * 0.2 +
            retrieval.keyword_coverage * 0.2
        ) * 70
        
        # 생성 점수 (30% 가중치, 생성이 활성화된 경우만)
        generation_score = 0.0
        if generation:
            generation_score = (
                generation.keyword_f1 * 0.5 +
                (1.0 if generation.contains_ground_truth else 0.0) * 0.3 +
                (1.0 if generation.exact_match else 0.0) * 0.2
            ) * 30
        
        return retrieval_score + generation_score
    
    def _save_results(self, results: List[EvaluationMetrics]):
        """평가 결과 저장"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 상세 결과 (JSON)
        detailed_results = [asdict(result) for result in results]
        detailed_path = output_dir / f"detailed_results_{int(time.time())}.json"
        
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"상세 결과 저장: {detailed_path}")
        
        # 요약 리포트 (텍스트)
        summary_path = output_dir / f"summary_report_{int(time.time())}.txt"
        self._generate_summary_report(results, summary_path)
        
        logger.info(f"요약 리포트 저장: {summary_path}")
    
    def _generate_summary_report(self, results: List[EvaluationMetrics], output_path: Path):
        """요약 리포트 생성"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAG 시스템 평가 리포트\n")
            f.write("=" * 80 + "\n\n")
            
            # 전체 통계
            f.write("📊 전체 통계\n")
            f.write("-" * 40 + "\n")
            
            valid_results = [r for r in results if not r.error_message]
            error_count = len(results) - len(valid_results)
            
            f.write(f"총 쿼리 수: {len(results)}\n")
            f.write(f"성공한 쿼리: {len(valid_results)}\n")
            f.write(f"실패한 쿼리: {error_count}\n")
            
            if valid_results:
                avg_score = sum(r.overall_score for r in valid_results) / len(valid_results)
                avg_response_time = sum(r.response_time_ms for r in valid_results) / len(valid_results)
                
                f.write(f"평균 점수: {avg_score:.2f}/100\n")
                f.write(f"평균 응답시간: {avg_response_time:.0f}ms\n\n")
                
                # 검색 메트릭 평균
                f.write("🔍 검색 품질 (평균)\n")
                f.write("-" * 40 + "\n")
                
                avg_recall = sum(r.retrieval.recall_at_k for r in valid_results) / len(valid_results)
                avg_precision = sum(r.retrieval.precision_at_k for r in valid_results) / len(valid_results)
                avg_mrr = sum(r.retrieval.mrr for r in valid_results) / len(valid_results)
                avg_keyword_coverage = sum(r.retrieval.keyword_coverage for r in valid_results) / len(valid_results)
                
                f.write(f"Recall@{self.config.top_k}: {avg_recall:.3f}\n")
                f.write(f"Precision@{self.config.top_k}: {avg_precision:.3f}\n")
                f.write(f"MRR: {avg_mrr:.3f}\n")
                f.write(f"키워드 커버리지: {avg_keyword_coverage:.3f}\n\n")
                
                # 난이도별 성능
                f.write("📈 난이도별 성능\n")
                f.write("-" * 40 + "\n")
                
                difficulty_stats = {}
                for result in valid_results:
                    diff = result.difficulty
                    if diff not in difficulty_stats:
                        difficulty_stats[diff] = []
                    difficulty_stats[diff].append(result.overall_score)
                
                for difficulty, scores in difficulty_stats.items():
                    avg_score = sum(scores) / len(scores)
                    f.write(f"{difficulty}: {avg_score:.2f} ({len(scores)}개 쿼리)\n")
                
                f.write("\n")
                
                # 개별 쿼리 결과
                f.write("📋 개별 쿼리 결과\n")
                f.write("-" * 40 + "\n")
                
                for result in valid_results:
                    f.write(f"[{result.query_id}] {result.query}\n")
                    f.write(f"  점수: {result.overall_score:.1f}/100\n")
                    f.write(f"  검색: Recall={result.retrieval.recall_at_k:.3f}, "
                           f"Precision={result.retrieval.precision_at_k:.3f}\n")
                    f.write(f"  응답시간: {result.response_time_ms:.0f}ms\n")
                    
                    if result.generation:
                        f.write(f"  생성: F1={result.generation.keyword_f1:.3f}, "
                               f"정확일치={result.generation.exact_match}\n")
                    f.write("\n")
            
            # 에러 쿼리
            if error_count > 0:
                f.write("❌ 실패한 쿼리\n")
                f.write("-" * 40 + "\n")
                for result in results:
                    if result.error_message:
                        f.write(f"[{result.query_id}] {result.query}\n")
                        f.write(f"  에러: {result.error_message}\n\n")
    
    def compare_models(
        self, 
        model_types: List[EmbeddingModelType],
        output_dir: str = "model_comparison"
    ) -> Dict[str, List[EvaluationMetrics]]:
        """여러 모델 비교 평가"""
        
        logger.info(f"모델 비교 평가 시작: {len(model_types)}개 모델")
        
        comparison_results = {}
        
        for model_type in model_types:
            logger.info(f"모델 평가 중: {model_type.value}")
            
            # 새로운 RAG 시스템 생성
            rag_system = RAGSystem(
                model_type=model_type,
                enable_generation=self.config.enable_generation
            )
            
            # 평가기 생성 및 실행
            evaluator = RAGEvaluator(rag_system, self.config)
            results = evaluator.evaluate_all()
            
            comparison_results[model_type.value] = results
        
        # 비교 리포트 생성
        self._generate_comparison_report(comparison_results, output_dir)
        
        return comparison_results
    
    def _generate_comparison_report(
        self, 
        comparison_results: Dict[str, List[EvaluationMetrics]],
        output_dir: str
    ):
        """모델 비교 리포트 생성"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report_path = output_path / f"model_comparison_{int(time.time())}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAG 모델 비교 리포트\n")
            f.write("=" * 80 + "\n\n")
            
            # 모델별 성능 요약
            f.write("📊 모델별 성능 요약\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'모델':<30} {'평균점수':<10} {'평균응답시간':<15} {'Recall@5':<10}\n")
            f.write("-" * 60 + "\n")
            
            for model_name, results in comparison_results.items():
                valid_results = [r for r in results if not r.error_message]
                if valid_results:
                    avg_score = sum(r.overall_score for r in valid_results) / len(valid_results)
                    avg_time = sum(r.response_time_ms for r in valid_results) / len(valid_results)
                    avg_recall = sum(r.retrieval.recall_at_k for r in valid_results) / len(valid_results)
                    
                    f.write(f"{model_name:<30} {avg_score:<10.2f} {avg_time:<15.0f} {avg_recall:<10.3f}\n")
            
            f.write("\n")
            
            # 쿼리별 상세 비교
            f.write("📋 쿼리별 상세 비교\n")
            f.write("-" * 60 + "\n")
            
            # 첫 번째 모델의 쿼리 순서를 기준으로
            first_model = list(comparison_results.keys())[0]
            first_results = comparison_results[first_model]
            
            for result in first_results:
                if result.error_message:
                    continue
                    
                f.write(f"[{result.query_id}] {result.query}\n")
                f.write("-" * 40 + "\n")
                
                for model_name, results in comparison_results.items():
                    model_result = next((r for r in results if r.query_id == result.query_id), None)
                    if model_result and not model_result.error_message:
                        f.write(f"{model_name}: {model_result.overall_score:.1f}점 "
                               f"(Recall={model_result.retrieval.recall_at_k:.3f}, "
                               f"응답시간={model_result.response_time_ms:.0f}ms)\n")
                
                f.write("\n")
        
        logger.info(f"모델 비교 리포트 저장: {report_path}")

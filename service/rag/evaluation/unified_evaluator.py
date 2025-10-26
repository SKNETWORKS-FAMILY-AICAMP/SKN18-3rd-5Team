#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 RAG 평가 도구
- RAG 검색 결과 생성 (rag_success 형식)
- RAGAs 기반 평가 (Faithfulness, Answer Relevancy 등)
- 단일 모드, 통합 결과물
"""

import sys
import json
import argparse
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# 프로젝트 루트를 Python 경로에 추가 (직접 실행 시 경로 문제 해결)
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]  # service/rag/evaluation/ -> project_root (SKN18-3rd-5Team)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from service.rag.rag_system import RAGSystem
from service.rag.models.config import EmbeddingModelType
from service.rag.evaluation.evaluator import RAGEvaluator, EvaluationConfig
from service.rag.evaluation.metrics import MetricsCalculator
from service.rag.cli.rag_cli import RAGJSONLSystem, get_db_config
from config.vector_database import get_vector_db_config
from service.rag.generation.generator import OllamaGenerator, OpenAIGenerator, GenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGEvaluation:
    """RAG 평가 결과"""
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

    # 메트릭 점수
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None

    # 평균 점수
    average_score: Optional[float] = None

    def calculate_average(self):
        """평균 점수 계산"""
        scores = [
            s for s in [
                self.faithfulness,
                self.answer_relevancy,
                self.context_precision,
                self.context_recall
            ] if s is not None
        ]
        if scores:
            self.average_score = sum(scores) / len(scores)


class RAGASEvaluator:
    """
    RAGAs 기반 평가기

    Note: 실제 RAGAs는 LLM(OpenAI GPT-4 등)을 사용하여 평가합니다.
    여기서는 간소화된 버전을 구현하고, 실제 RAGAs 라이브러리 사용법도 제공합니다.
    """

    def __init__(self, use_ragas_library: bool = False):
        """
        Args:
            use_ragas_library: ragas 라이브러리 사용 여부
        """
        self.use_ragas_library = use_ragas_library

        if use_ragas_library:
            try:
                from ragas import evaluate
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                )
                self.ragas_evaluate = evaluate
                self.metrics = {
                    'faithfulness': faithfulness,
                    'answer_relevancy': answer_relevancy,
                    'context_precision': context_precision,
                    'context_recall': context_recall
                }
                logger.info("RAGAs library loaded successfully")
            except ImportError:
                logger.warning("RAGAs library not found. Install with: pip install ragas")
                self.use_ragas_library = False

    def evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Faithfulness (충실성) 평가
        답변의 각 문장이 제공된 컨텍스트에서 추론 가능한지 평가

        간소화된 버전: 답변의 키워드가 컨텍스트에 포함되어 있는지 확인

        Args:
            answer: 생성된 답변
            contexts: 검색된 컨텍스트 리스트

        Returns:
            Faithfulness 점수 (0.0 ~ 1.0)
        """
        if not answer or not contexts:
            return 0.0

        # 답변을 문장 단위로 분리 (간단히 마침표 기준)
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        if not sentences:
            return 0.0

        # 전체 컨텍스트 합치기
        full_context = ' '.join(contexts)

        # 각 문장의 키워드가 컨텍스트에 있는지 확인
        supported_sentences = 0
        for sentence in sentences:
            # 문장에서 주요 키워드 추출 (3글자 이상)
            words = [w for w in sentence.split() if len(w) >= 3]
            if not words:
                continue

            # 키워드 중 50% 이상이 컨텍스트에 있으면 지원된 것으로 간주
            supported_words = sum(1 for w in words if w in full_context)
            if supported_words / len(words) >= 0.5:
                supported_sentences += 1

        return supported_sentences / len(sentences)

    def evaluate_answer_relevancy(
        self,
        query: str,
        answer: str
    ) -> float:
        """
        Answer Relevancy (답변 관련성) 평가
        답변이 질문과 얼마나 관련 있는지 평가

        간소화된 버전: 질문의 키워드가 답변에 포함되어 있는지 확인

        Args:
            query: 질문
            answer: 생성된 답변

        Returns:
            Answer Relevancy 점수 (0.0 ~ 1.0)
        """
        if not query or not answer:
            return 0.0

        # 질문에서 키워드 추출 (2글자 이상)
        query_words = set(w for w in query.split() if len(w) >= 2)
        if not query_words:
            return 0.0

        # 답변에 있는 질문 키워드 수
        matching_words = sum(1 for w in query_words if w in answer)

        return matching_words / len(query_words)

    def evaluate_context_precision(
        self,
        query: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> float:
        """
        Context Precision (컨텍스트 정밀도) 평가
        검색된 컨텍스트가 질문에 대한 정답 정보를 포함하는지 평가

        간소화된 버전: 컨텍스트가 질문 키워드를 포함하는지 확인

        Args:
            query: 질문
            contexts: 검색된 컨텍스트 리스트
            ground_truth: 정답 (있는 경우)

        Returns:
            Context Precision 점수 (0.0 ~ 1.0)
        """
        if not contexts:
            return 0.0

        # 질문 키워드
        query_words = set(w for w in query.split() if len(w) >= 2)
        if not query_words:
            return 0.0

        # 각 컨텍스트가 질문과 관련있는지 확인
        relevant_contexts = 0
        for context in contexts:
            matching_words = sum(1 for w in query_words if w in context)
            if matching_words / len(query_words) >= 0.3:  # 30% 이상 매칭
                relevant_contexts += 1

        return relevant_contexts / len(contexts)

    def evaluate_context_recall(
        self,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> float:
        """
        Context Recall (컨텍스트 재현율) 평가
        정답에 필요한 정보를 모두 검색했는지 평가

        Note: ground_truth가 없으면 평가 불가

        Args:
            contexts: 검색된 컨텍스트 리스트
            ground_truth: 정답

        Returns:
            Context Recall 점수 (0.0 ~ 1.0)
        """
        if not ground_truth or not contexts:
            return 0.0

        # 정답의 키워드
        truth_words = set(w for w in ground_truth.split() if len(w) >= 2)
        if not truth_words:
            return 0.0

        # 전체 컨텍스트
        full_context = ' '.join(contexts)

        # 정답 키워드 중 컨텍스트에 있는 비율
        covered_words = sum(1 for w in truth_words if w in full_context)

        return covered_words / len(truth_words)

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> RAGEvaluation:
        """
        전체 RAG 평가 수행

        Args:
            query: 질문
            answer: 생성된 답변
            contexts: 검색된 컨텍스트 리스트
            ground_truth: 정답 (선택)

        Returns:
            RAGEvaluation 결과
        """
        evaluation = RAGEvaluation(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )

        # 각 메트릭 평가
        evaluation.faithfulness = self.evaluate_faithfulness(answer, contexts)
        evaluation.answer_relevancy = self.evaluate_answer_relevancy(query, answer)
        evaluation.context_precision = self.evaluate_context_precision(query, contexts, ground_truth)

        if ground_truth:
            evaluation.context_recall = self.evaluate_context_recall(contexts, ground_truth)

        # 평균 점수 계산
        evaluation.calculate_average()

        return evaluation

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> List[RAGEvaluation]:
        """
        배치 평가

        Args:
            test_cases: 테스트 케이스 리스트
                각 케이스는 {'query', 'answer', 'contexts', 'ground_truth'} 포함

        Returns:
            평가 결과 리스트
        """
        results = []
        for case in test_cases:
            evaluation = self.evaluate(
                query=case['query'],
                answer=case['answer'],
                contexts=case['contexts'],
                ground_truth=case.get('ground_truth')
            )
            results.append(evaluation)

        return results

    def get_summary_statistics(
        self,
        evaluations: List[RAGEvaluation]
    ) -> Dict[str, float]:
        """
        평가 결과 통계 요약

        Args:
            evaluations: 평가 결과 리스트

        Returns:
            통계 딕셔너리
        """
        metrics = {
            'faithfulness': [],
            'answer_relevancy': [],
            'context_precision': [],
            'context_recall': [],
            'average': []
        }

        for eval_result in evaluations:
            if eval_result.faithfulness is not None:
                metrics['faithfulness'].append(eval_result.faithfulness)
            if eval_result.answer_relevancy is not None:
                metrics['answer_relevancy'].append(eval_result.answer_relevancy)
            if eval_result.context_precision is not None:
                metrics['context_precision'].append(eval_result.context_precision)
            if eval_result.context_recall is not None:
                metrics['context_recall'].append(eval_result.context_recall)
            if eval_result.average_score is not None:
                metrics['average'].append(eval_result.average_score)

        # 통계 계산
        summary = {}
        for metric_name, values in metrics.items():
            if values:
                summary[f'{metric_name}_mean'] = np.mean(values)
                summary[f'{metric_name}_std'] = np.std(values)
                summary[f'{metric_name}_min'] = np.min(values)
                summary[f'{metric_name}_max'] = np.max(values)

        return summary


class UnifiedRAGEvaluator:
    """통합 RAG 평가 도구"""

    def __init__(self,
                 model: str = "gemma"):  # "gemma" 또는 "openai"
        """
        초기화

        Args:
            model: 사용할 모델 ("gemma" 또는 "openai")
        """
        self.model = model

        # 심플 RAG 시스템 사용 (안정성 우선)
        self.rag_system = RAGJSONLSystem(
            db_config=get_db_config(),
            embedding_model="intfloat/multilingual-e5-small"
        )

        # LLM Generator 초기화
        if model == "openai":
            logger.info("Using OpenAI Generator (gpt-5-nano)")
            self.generator = OpenAIGenerator(default_model="gpt-5-nano")
            model_name = "gpt-5-nano"
            timeout = 60  # OpenAI는 더 빠름
        else:  # gemma
            logger.info("Using Ollama Generator (gemma3:4b)")
            self.generator = OllamaGenerator(
                base_url="http://localhost:11434",
                default_model="gemma3:4b"
            )
            model_name = "gemma3:4b"
            timeout = 120  # Ollama는 더 느림

        # Generation Config
        self.generation_config = GenerationConfig(
            model=model_name,
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            timeout=timeout
        )

        # 메트릭 계산기 초기화
        self.metrics_calculator = MetricsCalculator()

        # RAGAs 평가기 초기화
        self.ragas_evaluator = RAGASEvaluator(use_ragas_library=False)

        logger.info(f"Unified RAG Evaluation Tool 초기화 완료")
        logger.info(f"  - Model: {model_name}")
    
    def _load_evaluation_queries(self) -> List[Dict[str, Any]]:
        """평가 쿼리 로드"""
        queries_path = project_root / "service" / "rag" / "evaluation" / "evaluation_queries.json"
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
        logger.info(f"📊 모델: {self.model}")
        logger.info(f"🔍 Top-K: {top_k}")
        if corp_filter:
            logger.info(f"🏢 기업 필터: {corp_filter}")
        logger.info("=" * 80)
        
        results = []
        
        for i, query_data in enumerate(queries, 1):
            query_id = query_data.get("query_id", f"Q{i:03d}")
            query_text = query_data.get("query", "")
            
            # 쿼리별 회사명 필터 추출
            query_company_filter = None
            if "company_name" in query_data:
                company_name = query_data["company_name"]
                if isinstance(company_name, list):
                    # 여러 회사 비교 쿼리의 경우 필터 없이 검색
                    query_company_filter = None
                    logger.info(f"  다중 회사 비교 쿼리: {company_name}")
                else:
                    query_company_filter = company_name
                    logger.info(f"  회사 필터 적용: {company_name}")
            
            logger.info(f"[{i}/{len(queries)}] {query_id}: {query_text}")
            
            try:
                # 심플 검색 실행 (쿼리별 회사 필터 적용)
                search_results = self.rag_system.search(
                    query=query_text,
                    top_k=top_k,
                    min_similarity=min_similarity,
                    corp_filter=query_company_filter
                )
                
                # 결과 처리
                if search_results:
                    # 컨텍스트 추출 (natural_text 또는 chunk_text 필드 사용)
                    contexts = [
                        result.get("natural_text", result.get("chunk_text", "")) 
                        for result in search_results
                    ]
                    
                    # 빈 컨텍스트 필터링
                    contexts = [ctx for ctx in contexts if ctx.strip()]
                    
                    # 컨텍스트를 하나의 문자열로 결합
                    context_text = "\n\n".join(contexts)
                    
                    # Ollama Generator로 답변 생성 (테스트용)
                    try:
                        generated_result = self.generator.generate(
                            query=query_text,
                            context=context_text,
                            config=self.generation_config
                        )
                        generated_answer = generated_result.answer
                        logger.info(f"  답변 생성 완료: {len(generated_answer)} chars")
                        if generated_result.generation_time_ms:
                            logger.info(f"  생성 시간: {generated_result.generation_time_ms:.0f}ms")
                    except Exception as e:
                        logger.error(f"  답변 생성 실패: {e}")
                        generated_answer = "답변 생성에 실패했습니다."
                    
                    # RAGAs 평가 수행 (Reference-Free)
                    ragas_evaluation = self.ragas_evaluator.evaluate(
                        query=query_text,
                        answer=generated_answer,
                        contexts=contexts,
                        ground_truth=None  # Reference-Free 평가
                    )
                    
                    # 결과 구성
                    result = {
                        "query_id": query_id,
                        "query": query_text,
                        "query_type": query_data.get("query_type", "unknown"),
                        "difficulty": query_data.get("difficulty", "medium"),
                        "search_results": search_results,
                        "contexts": contexts,
                        "answer": generated_answer,
                        "ragas_scores": {
                            "faithfulness": ragas_evaluation.faithfulness,
                            "answer_relevancy": ragas_evaluation.answer_relevancy,
                            "context_precision": ragas_evaluation.context_precision,
                            "context_recall": ragas_evaluation.context_recall,
                            "average_score": ragas_evaluation.average_score
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    results.append(result)
                    logger.info(f"✅ {query_id} 완료 - 평균 점수: {ragas_evaluation.average_score:.3f}")
                else:
                    logger.warning(f"⚠️ {query_id}: 검색 결과 없음")
                    
            except Exception as e:
                logger.error(f"❌ {query_id} 오류: {e}")
                continue
        
        logger.info("🎉 RAG 평가 완료!")
        logger.info(f"📈 총 {len(results)}개 쿼리 처리 완료")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: str = "results") -> Dict[str, str]:
        """
        결과를 파일로 저장
        
        Args:
            results: 평가 결과 리스트
            output_dir: 출력 디렉토리
            
        Returns:
            저장된 파일 경로 딕셔너리
        """
        # 현재 파일의 위치를 기준으로 올바른 경로 계산
        current_file = Path(__file__).resolve()
        # service/rag/evaluation -> service/rag (rag 디렉토리)
        rag_dir = current_file.parents[1]
        
        # 절대 경로로 변환
        if Path(output_dir).is_absolute():
            output_path = Path(output_dir)
        else:
            # "results"만 입력된 경우 rag/results로 설정
            if output_dir == "results":
                output_path = rag_dir / "results"
            else:
                # rag 디렉토리 기준으로 상대 경로 계산
                output_path = rag_dir / output_dir
            
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 상세 결과 저장 (JSON)
        detailed_file = output_path / f"unified_evaluation_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 요약 통계 계산
        ragas_evaluations = []
        for result in results:
            ragas_eval = RAGEvaluation(
                query=result["query"],
                answer=result["answer"],
                contexts=result["contexts"],
                ground_truth=None  # Reference-Free 평가
            )
            ragas_eval.faithfulness = result["ragas_scores"]["faithfulness"]
            ragas_eval.answer_relevancy = result["ragas_scores"]["answer_relevancy"]
            ragas_eval.context_precision = result["ragas_scores"]["context_precision"]
            ragas_eval.context_recall = result["ragas_scores"]["context_recall"]
            ragas_eval.average_score = result["ragas_scores"]["average_score"]
            ragas_evaluations.append(ragas_eval)
        
        summary_stats = self.ragas_evaluator.get_summary_statistics(ragas_evaluations)

        # 요약 보고서 저장 (TXT)
        summary_file = output_path / f"unified_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Unified RAG Evaluation Summary Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples: {len(results)}\n")
            f.write(f"Model: {self.model}\n\n")
            
            f.write("=== RAGAs Metric Statistics ===\n")
            for metric_name, values in summary_stats.items():
                if 'mean' in metric_name:
                    base_name = metric_name.replace('_mean', '')
                    f.write(f"{base_name.upper()}:\n")
                    f.write(f"  Mean: {values:.3f}\n")
                    f.write(f"  Std:  {summary_stats.get(f'{base_name}_std', 0):.3f}\n")
                    f.write(f"  Min:  {summary_stats.get(f'{base_name}_min', 0):.3f}\n")
                    f.write(f"  Max:  {summary_stats.get(f'{base_name}_max', 0):.3f}\n\n")
            
            f.write("=== Individual Results ===\n")
            for i, result in enumerate(results):
                f.write(f"Sample {i+1} ({result['query_id']}):\n")
                f.write(f"  Query: {result['query'][:50]}...\n")
                f.write(f"  Type: {result['query_type']}\n")
                f.write(f"  Difficulty: {result['difficulty']}\n")
                f.write(f"  Faithfulness: {result['ragas_scores']['faithfulness']:.3f}\n")
                f.write(f"  Answer Relevancy: {result['ragas_scores']['answer_relevancy']:.3f}\n")
                f.write(f"  Context Precision: {result['ragas_scores']['context_precision']:.3f}\n")
                context_recall_str = f"{result['ragas_scores']['context_recall']:.3f}" if result['ragas_scores']['context_recall'] is not None else "N/A"
                f.write(f"  Context Recall: {context_recall_str}\n")
                f.write(f"  Average Score: {result['ragas_scores']['average_score']:.3f}\n\n")
        
        logger.info(f"💾 JSON 결과 저장 완료: {detailed_file}")
        logger.info(f"📄 상세 결과 저장 완료: {summary_file}")
        
        return {
            "detailed_results": str(detailed_file),
            "summary_report": str(summary_file)
        }


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="통합 RAG 평가 도구 - evaluation_queries.json을 사용한 자동 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # Gemma 모델로 평가 (기본값)
  python unified_evaluator.py
  python unified_evaluator.py --model gemma

  # OpenAI 모델로 평가
  python unified_evaluator.py --model openai
        """
    )
    parser.add_argument("--model", type=str, default="gemma", choices=["gemma", "openai"],
                       help="사용할 모델 (기본값: gemma)")

    args = parser.parse_args()

    # 평가 도구 초기화
    evaluator = UnifiedRAGEvaluator(model=args.model)
    
    # 평가 실행 (evaluation_queries.json 사용)
    results = evaluator.run_evaluation()

    # 결과 저장
    saved_files = evaluator.save_results(results, "results")
    
    # 결과 출력
    print(f"\n📊 평가 결과 요약:")
    print(f"   - 총 쿼리 수: {len(results)}")
    print(f"   - 모델: {evaluator.model}")
    print(f"   - 성공한 쿼리: {len(results)}")
    
    print(f"\n💾 결과 저장 위치:")
    print(f"   - 상세 결과: {saved_files['detailed_results']}")
    print(f"   - 요약 보고서: {saved_files['summary_report']}")


if __name__ == "__main__":
    main()

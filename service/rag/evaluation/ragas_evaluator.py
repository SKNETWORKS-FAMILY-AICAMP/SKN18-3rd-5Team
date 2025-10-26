#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAGAs 기반 RAG 평가 시스템
- Faithfulness: 답변이 컨텍스트에 근거하는가?
- Answer Relevancy: 답변이 질문과 관련 있는가?
- Context Precision: 검색된 컨텍스트가 정확한가?
- Context Recall: 필요한 정보를 모두 검색했는가?
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

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

    def save_results(
        self,
        evaluations: List[RAGEvaluation],
        output_dir: str = "results",
        prefix: str = "ragas_evaluation"
    ) -> Dict[str, str]:
        """
        평가 결과를 파일로 저장
        
        Args:
            evaluations: 평가 결과 리스트
            output_dir: 출력 디렉토리
            prefix: 파일명 접두사
            
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
        detailed_file = output_path / f"{prefix}_{timestamp}.json"
        detailed_data = []
        
        for eval_result in evaluations:
            eval_dict = {
                "query": eval_result.query,
                "answer": eval_result.answer,
                "contexts": eval_result.contexts,
                "ground_truth": eval_result.ground_truth,
                "scores": {
                    "faithfulness": eval_result.faithfulness,
                    "answer_relevancy": eval_result.answer_relevancy,
                    "context_precision": eval_result.context_precision,
                    "context_recall": eval_result.context_recall,
                    "average_score": eval_result.average_score
                }
            }
            detailed_data.append(eval_dict)
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)
        
        # 요약 통계 저장 (TXT)
        summary_file = output_path / f"summary_report_{timestamp}.txt"
        summary_stats = self.get_summary_statistics(evaluations)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"RAGAs Evaluation Summary Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples: {len(evaluations)}\n\n")
            
            f.write("=== Metric Statistics ===\n")
            for metric_name, values in summary_stats.items():
                if 'mean' in metric_name:
                    base_name = metric_name.replace('_mean', '')
                    f.write(f"{base_name.upper()}:\n")
                    f.write(f"  Mean: {values:.3f}\n")
                    f.write(f"  Std:  {summary_stats.get(f'{base_name}_std', 0):.3f}\n")
                    f.write(f"  Min:  {summary_stats.get(f'{base_name}_min', 0):.3f}\n")
                    f.write(f"  Max:  {summary_stats.get(f'{base_name}_max', 0):.3f}\n\n")
            
            f.write("=== Individual Results ===\n")
            for i, eval_result in enumerate(evaluations):
                f.write(f"Sample {i+1}:\n")
                f.write(f"  Query: {eval_result.query[:50]}...\n")
                f.write(f"  Faithfulness: {eval_result.faithfulness:.3f}\n")
                f.write(f"  Answer Relevancy: {eval_result.answer_relevancy:.3f}\n")
                f.write(f"  Context Precision: {eval_result.context_precision:.3f}\n")
                f.write(f"  Context Recall: {eval_result.context_recall:.3f}\n")
                f.write(f"  Average Score: {eval_result.average_score:.3f}\n\n")
        
        logger.info(f"Results saved to {detailed_file} and {summary_file}")
        
        return {
            "detailed_results": str(detailed_file),
            "summary_report": str(summary_file)
        }


def evaluate_with_ragas_library(
    test_cases: List[Dict[str, Any]],
    llm_provider: str = "openai"
) -> Dict[str, Any]:
    """
    실제 RAGAs 라이브러리 사용 예제

    Args:
        test_cases: 테스트 케이스 리스트
        llm_provider: LLM 제공자 ('openai', 'anthropic' 등)

    Returns:
        RAGAs 평가 결과
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        )
        from datasets import Dataset

        # 데이터셋 변환
        data = {
            'question': [case['query'] for case in test_cases],
            'answer': [case['answer'] for case in test_cases],
            'contexts': [case['contexts'] for case in test_cases],
        }

        # ground_truth가 있는 경우 추가
        if 'ground_truth' in test_cases[0]:
            data['ground_truth'] = [case['ground_truth'] for case in test_cases]

        dataset = Dataset.from_dict(data)

        # 평가 수행
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )

        return result

    except ImportError as e:
        logger.error(f"RAGAs library not installed: {e}")
        logger.info("Install with: pip install ragas datasets")
        return {}


def load_evaluation_queries(json_file: str = "cli/evaluation_queries.json") -> List[Dict[str, Any]]:
    """
    evaluation_queries.json 파일에서 평가 쿼리 로드
    
    Args:
        json_file: JSON 파일 경로
        
    Returns:
        평가 쿼리 리스트
    """
    try:
        # 현재 파일의 위치를 기준으로 올바른 경로 계산
        current_file = Path(__file__).resolve()
        # service/rag/evaluation -> service/rag (rag 디렉토리)
        rag_dir = current_file.parents[1]
        
        # 절대 경로로 변환
        if Path(json_file).is_absolute():
            json_path = Path(json_file)
        else:
            # rag 디렉토리 기준으로 상대 경로 계산
            json_path = rag_dir / json_file
            
        with open(json_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
            
        logger.info(f"Loaded {len(queries)} evaluation queries from {json_path}")
        return queries
        
    except FileNotFoundError:
        logger.error(f"Evaluation queries file not found: {json_file}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {json_file}: {e}")
        return []


def convert_queries_to_test_cases(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    evaluation_queries.json 형식을 RAGAs 테스트 케이스 형식으로 변환
    
    Args:
        queries: evaluation_queries.json에서 로드한 쿼리 리스트
        
    Returns:
        RAGAs 테스트 케이스 리스트
    """
    test_cases = []
    
    for query_data in queries:
        # 실제 답변과 컨텍스트는 모의 데이터로 생성 (실제 RAG 시스템에서는 검색 결과 사용)
        mock_answer = f"{query_data['query']}에 대한 답변입니다. {query_data.get('ground_truth_answer', '관련 정보를 찾을 수 없습니다.')}"
        
        # 모의 컨텍스트 생성 (실제로는 검색 시스템에서 가져와야 함)
        mock_contexts = []
        if 'expected_docs' in query_data:
            for doc in query_data['expected_docs']:
                mock_contexts.append(f"{doc} 관련 내용: {query_data['query']}에 대한 정보가 포함되어 있습니다.")
        
        # expected_keywords를 활용한 추가 컨텍스트
        if 'expected_keywords' in query_data:
            keyword_context = f"키워드 {', '.join(query_data['expected_keywords'])}와 관련된 정보입니다."
            mock_contexts.append(keyword_context)
        
        # 최소 3개의 컨텍스트 보장
        while len(mock_contexts) < 3:
            mock_contexts.append(f"추가 컨텍스트 {len(mock_contexts) + 1}: {query_data['query']} 관련 정보입니다.")
        
        test_case = {
            'query': query_data['query'],
            'answer': mock_answer,
            'contexts': mock_contexts,
            'ground_truth': query_data.get('ground_truth_answer', ''),
            'query_id': query_data.get('query_id', ''),
            'difficulty': query_data.get('difficulty', 'medium'),
            'query_type': query_data.get('query_type', 'factual')
        }
        
        test_cases.append(test_case)
    
    return test_cases


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)

    print("=== RAGAs Evaluator Test ===\n")

    evaluator = RAGASEvaluator(use_ragas_library=False)

    # evaluation_queries.json에서 쿼리 로드
    print("Loading evaluation queries...")
    queries = load_evaluation_queries()
    
    if not queries:
        print("No evaluation queries found. Using fallback test cases.")
        # 폴백 테스트 케이스
        test_cases = [
            {
                'query': '2차전지 산업 전망은?',
                'answer': '2025년 2차전지 산업은 전기차 수요 증가로 성장이 예상됩니다.',
                'contexts': [
                    '2025년 2차전지 산업은 전기차 수요 증가로 성장 전망이 밝습니다.',
                    'LFP 배터리 기술이 저렴한 가격과 안정성으로 주목받고 있습니다.',
                    '국내 배터리 3사는 해외 공장 증설을 계획 중입니다.'
                ],
                'ground_truth': '2차전지 산업은 전기차 수요로 성장하며, LFP 기술이 핵심입니다.'
            }
        ]
    else:
        # evaluation_queries.json을 테스트 케이스로 변환
        test_cases = convert_queries_to_test_cases(queries)

    # 배치 평가 수행
    print(f"Evaluating {len(test_cases)} test cases...\n")
    results = evaluator.evaluate_batch(test_cases)

    # 결과 출력
    for i, result in enumerate(results):
        test_case = test_cases[i]
        print(f"=== Sample {i+1} ===")
        print(f"Query ID: {test_case.get('query_id', 'N/A')}")
        print(f"Query Type: {test_case.get('query_type', 'N/A')}")
        print(f"Difficulty: {test_case.get('difficulty', 'N/A')}")
        print(f"Query: {result.query}")
        print(f"Answer: {result.answer[:50]}...")
        print(f"Contexts: {len(result.contexts)} documents")
        print("Scores:")
        print(f"  Faithfulness:       {result.faithfulness:.3f}")
        print(f"  Answer Relevancy:   {result.answer_relevancy:.3f}")
        print(f"  Context Precision:  {result.context_precision:.3f}")
        context_recall_str = f"{result.context_recall:.3f}" if result.context_recall is not None else "N/A"
        print(f"  Context Recall:     {context_recall_str}")
        print(f"  Average Score:      {result.average_score:.3f}")
        print()

    # 통계 요약
    summary = evaluator.get_summary_statistics(results)
    print("=== Summary Statistics ===")
    for metric_name, value in summary.items():
        if 'mean' in metric_name:
            print(f"{metric_name}: {value:.3f}")

    # 결과 저장
    print("\n=== Saving Results ===")
    saved_files = evaluator.save_results(results)
    print(f"Detailed results saved to: {saved_files['detailed_results']}")
    print(f"Summary report saved to: {saved_files['summary_report']}")

    print("\n=== RAGAs Library Usage ===")
    print("""
To use the official RAGAs library:

1. Install dependencies:
   pip install ragas datasets

2. Set up OpenAI API key:
   export OPENAI_API_KEY='your-key-here'

3. Use evaluate_with_ragas_library() function

Example:
    test_cases = [
        {
            'query': 'What is the outlook for battery industry?',
            'answer': 'The battery industry is expected to grow...',
            'contexts': ['Context 1...', 'Context 2...'],
            'ground_truth': 'Expected answer...'
        }
    ]

    results = evaluate_with_ragas_library(test_cases)
    print(results)
""")

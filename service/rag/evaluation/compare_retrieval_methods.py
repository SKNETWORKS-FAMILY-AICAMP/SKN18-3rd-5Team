#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
검색 방법 비교 실험 스크립트
Vector DB vs RDB vs Hybrid Search 성능 비교 + RAGAs 평가
"""

import asyncio
import logging
import json
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from datetime import datetime

# 프로젝트 모듈
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from service.rag.retrieval.retriever import Retriever
from service.rag.models.config import EmbeddingModelType
from service.rag.evaluation.ragas_evaluator import RAGASEvaluator
from service.pgv_temp.reranker_crossencoder import CrossEncoderReranker

logger = logging.getLogger(__name__)


class RetrievalComparisonExperiment:
    """검색 방법 비교 실험"""

    def __init__(
        self,
        db_config: Dict[str, str],
        output_dir: str = "evaluation_results"
    ):
        """
        Args:
            db_config: PostgreSQL 연결 설정
            output_dir: 결과 저장 디렉토리
        """
        self.db_config = db_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 컴포넌트 초기화
        self.retriever = Retriever(
            model_type=EmbeddingModelType.MULTILINGUAL_E5_SMALL,
            enable_temporal_filter=True
        )
        self.reranker = None  # Lazy loading
        self.evaluator = RAGASEvaluator(use_ragas_library=False)

    async def initialize(self):
        """컴포넌트 초기화"""
        await self.retriever.initialize()
        logger.info("Retriever initialized")

    async def close(self):
        """리소스 정리"""
        await self.retriever.close()
        logger.info("Resources closed")

    def get_reranker(self):
        """Reranker lazy loading"""
        if self.reranker is None:
            try:
                self.reranker = CrossEncoderReranker()
                logger.info("Reranker loaded")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")
        return self.reranker

    async def run_single_experiment(
        self,
        query: str,
        query_embedding: List[float],
        search_method: SearchMethod,
        use_reranker: bool = False,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        단일 검색 실험 수행

        Args:
            query: 검색 쿼리
            query_embedding: 쿼리 임베딩
            search_method: 검색 방법
            use_reranker: Reranker 사용 여부
            top_k: 반환할 문서 수

        Returns:
            실험 결과
        """
        config = SearchConfig(search_method=search_method)

        # 검색 수행
        import time
        start_time = time.time()

        results = await self.retriever.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            config=config,
            top_k=top_k * 2 if use_reranker else top_k
        )

        search_time = time.time() - start_time

        # Reranking (옵션)
        if use_reranker and results:
            reranker = self.get_reranker()
            if reranker:
                start_time = time.time()
                results = reranker.rerank(query, results, top_k=top_k)
                rerank_time = time.time() - start_time
            else:
                rerank_time = 0.0
                results = results[:top_k]
        else:
            rerank_time = 0.0

        return {
            'method': search_method.value,
            'use_reranker': use_reranker,
            'num_results': len(results),
            'search_time_ms': search_time * 1000,
            'rerank_time_ms': rerank_time * 1000,
            'total_time_ms': (search_time + rerank_time) * 1000,
            'results': results
        }

    async def compare_methods(
        self,
        test_queries: List[Dict[str, Any]],
        top_k: int = 5,
        use_reranker: bool = True
    ) -> pd.DataFrame:
        """
        여러 검색 방법 비교

        Args:
            test_queries: 테스트 쿼리 리스트
                [{'query': str, 'embedding': List[float], 'ground_truth': str}, ...]
            top_k: 반환할 문서 수
            use_reranker: Reranker 사용 여부

        Returns:
            비교 결과 DataFrame
        """
        results = []

        # 검색 방법들
        methods = [
            SearchMethod.VECTOR_ONLY,
            SearchMethod.KEYWORD_ONLY,
            SearchMethod.HYBRID
        ]

        for query_data in test_queries:
            query = query_data['query']
            embedding = query_data['embedding']
            ground_truth = query_data.get('ground_truth')

            logger.info(f"Testing query: {query[:50]}...")

            for method in methods:
                # 검색 실험
                exp_result = await self.run_single_experiment(
                    query=query,
                    query_embedding=embedding,
                    search_method=method,
                    use_reranker=use_reranker,
                    top_k=top_k
                )

                # 검색 결과에서 컨텍스트 추출
                contexts = [r.get('chunk_text', '') for r in exp_result['results']]

                # RAGAs 평가를 위한 더미 답변 (실제로는 LLM 생성 필요)
                answer = f"검색된 {len(contexts)}개 문서 기반 답변"

                # RAGAs 평가
                evaluation = self.evaluator.evaluate(
                    query=query,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth
                )

                # 결과 저장
                results.append({
                    'query': query,
                    'method': method.value,
                    'use_reranker': use_reranker,
                    'num_results': exp_result['num_results'],
                    'search_time_ms': exp_result['search_time_ms'],
                    'rerank_time_ms': exp_result['rerank_time_ms'],
                    'total_time_ms': exp_result['total_time_ms'],
                    'faithfulness': evaluation.faithfulness,
                    'answer_relevancy': evaluation.answer_relevancy,
                    'context_precision': evaluation.context_precision,
                    'context_recall': evaluation.context_recall,
                    'average_score': evaluation.average_score
                })

        return pd.DataFrame(results)

    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        실험 결과 분석

        Args:
            df: 실험 결과 DataFrame

        Returns:
            분석 결과
        """
        analysis = {}

        # 검색 방법별 평균 성능
        for method in df['method'].unique():
            method_df = df[df['method'] == method]

            analysis[method] = {
                'avg_search_time_ms': method_df['search_time_ms'].mean(),
                'avg_total_time_ms': method_df['total_time_ms'].mean(),
                'avg_faithfulness': method_df['faithfulness'].mean(),
                'avg_answer_relevancy': method_df['answer_relevancy'].mean(),
                'avg_context_precision': method_df['context_precision'].mean(),
                'avg_context_recall': method_df['context_recall'].mean() if 'context_recall' in method_df else None,
                'avg_score': method_df['average_score'].mean()
            }

        # 최고 성능 방법
        best_by_speed = df.loc[df['total_time_ms'].idxmin()]['method']
        best_by_quality = df.loc[df['average_score'].idxmax()]['method']

        analysis['summary'] = {
            'best_by_speed': best_by_speed,
            'best_by_quality': best_by_quality,
            'total_queries': len(df['query'].unique())
        }

        return analysis

    def save_results(
        self,
        df: pd.DataFrame,
        analysis: Dict[str, Any],
        filename_prefix: str = "retrieval_comparison"
    ):
        """
        결과 저장

        Args:
            df: 결과 DataFrame
            analysis: 분석 결과
            filename_prefix: 파일명 접두사
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV 저장
        csv_path = self.output_dir / f"{filename_prefix}_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"Results saved to {csv_path}")

        # JSON 분석 결과 저장
        json_path = self.output_dir / f"{filename_prefix}_analysis_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        logger.info(f"Analysis saved to {json_path}")

        # 요약 출력
        print("\n" + "="*60)
        print("RETRIEVAL COMPARISON RESULTS")
        print("="*60)
        print(f"\nTotal Queries: {analysis['summary']['total_queries']}")
        print(f"Best by Speed: {analysis['summary']['best_by_speed']}")
        print(f"Best by Quality: {analysis['summary']['best_by_quality']}")
        print("\n" + "-"*60)
        print("METHOD COMPARISON")
        print("-"*60)

        for method, metrics in analysis.items():
            if method == 'summary':
                continue
            print(f"\n{method.upper()}:")
            print(f"  Avg Search Time: {metrics['avg_search_time_ms']:.2f} ms")
            print(f"  Avg Total Time:  {metrics['avg_total_time_ms']:.2f} ms")
            print(f"  Avg Faithfulness: {metrics['avg_faithfulness']:.3f}")
            print(f"  Avg Answer Relevancy: {metrics['avg_answer_relevancy']:.3f}")
            print(f"  Avg Context Precision: {metrics['avg_context_precision']:.3f}")
            print(f"  Avg Overall Score: {metrics['avg_score']:.3f}")

        print("\n" + "="*60)


async def run_comparison_experiment():
    """비교 실험 실행"""
    logging.basicConfig(level=logging.INFO)

    # 데이터베이스 설정
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'skn_project',
        'user': 'postgres',
        'password': 'post1234'
    }

    # 테스트 쿼리 (실제로는 더 많은 쿼리 필요)
    test_queries = [
        {
            'query': '2차전지 산업 전망은?',
            'embedding': [0.1] * 384,  # 실제로는 임베딩 모델 사용
            'ground_truth': '2차전지 산업은 전기차 수요로 성장 전망'
        },
        {
            'query': '반도체 업황은 어떤가요?',
            'embedding': [0.2] * 384,
            'ground_truth': '반도체는 AI 수요로 호황'
        },
        # 더 많은 테스트 쿼리 추가...
    ]

    # 실험 실행
    experiment = RetrievalComparisonExperiment(db_config)

    try:
        await experiment.initialize()

        # 비교 실험 수행
        results_df = await experiment.compare_methods(
            test_queries=test_queries,
            top_k=5,
            use_reranker=True
        )

        # 결과 분석
        analysis = experiment.analyze_results(results_df)

        # 결과 저장
        experiment.save_results(results_df, analysis)

    finally:
        await experiment.close()


if __name__ == "__main__":
    print("""
=== Retrieval Methods Comparison Experiment ===

This script compares:
1. Vector Search (pgvector)
2. Keyword Search (PostgreSQL Full-Text Search)
3. Hybrid Search (Reciprocal Rank Fusion)

With optional Cross-Encoder reranking.

Evaluation metrics (RAGAs):
- Faithfulness: Answer grounded in context
- Answer Relevancy: Answer relevant to query
- Context Precision: Retrieved contexts are precise
- Context Recall: All necessary info retrieved

Usage:
    python compare_retrieval_methods.py

Or import and use programmatically:
    from compare_retrieval_methods import run_comparison_experiment
    asyncio.run(run_comparison_experiment())
""")

    # 실행
    # asyncio.run(run_comparison_experiment())

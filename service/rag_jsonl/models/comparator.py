#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
임베딩 모델 비교 도구
여러 모델의 검색 결과를 비교하고 성능을 측정
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict

from .config import EmbeddingModelType, ExperimentConfig
from .encoder import EmbeddingEncoder, MultiModelEncoder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    model_name: str
    model_type: str
    query: str
    documents: List[str]
    similarities: List[float]
    top_k: int
    search_time: float  # 검색 소요 시간 (초)
    embedding_dimension: int
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class KoreanEvaluationMetrics:
    """한국어 이해도 평가 메트릭"""
    # 기본 검색 성능
    avg_similarity: float
    max_similarity: float
    min_similarity: float
    
    # 한국어 특화 평가
    keyword_precision: float  # 키워드 정확도 (찾은 키워드 중 정확한 비율)
    keyword_recall: float     # 키워드 재현율 (전체 키워드 중 찾은 비율)
    keyword_f1: float         # 키워드 F1 점수
    
    # 의미적 이해도
    semantic_coherence: float  # 의미적 일관성
    context_relevance: float   # 맥락 관련성
    
    # 한국어 특성 평가
    korean_entity_recognition: float  # 한국어 개체명 인식
    domain_specificity: float         # 도메인 특화성 (금융/주거)
    
    # 종합 점수
    overall_korean_score: float


@dataclass
class ComparisonMetrics:
    """비교 메트릭"""
    model_name: str
    model_type: str
    avg_search_time: float
    total_queries: int
    avg_top1_similarity: float
    avg_top3_similarity: float
    embedding_dimension: int
    memory_usage_mb: Optional[float] = None
    korean_evaluation: Optional[KoreanEvaluationMetrics] = None


class ModelComparator:
    """모델 비교기"""

    def __init__(
        self,
        model_types: List[EmbeddingModelType],
        config: Optional[ExperimentConfig] = None
    ):
        """
        Args:
            model_types: 비교할 모델 타입 리스트
            config: 실험 설정
        """
        self.model_types = model_types
        self.config = config or ExperimentConfig()

        # 결과 저장 디렉토리
        self.results_dir = Path(self.config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 다중 모델 인코더
        self.multi_encoder = MultiModelEncoder(model_types)

        # 결과 저장용
        self.search_results: List[SearchResult] = []
        self.metrics: Dict[str, ComparisonMetrics] = {}

        logger.info(f"ModelComparator initialized with {len(model_types)} models")

    def evaluate_korean_understanding(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        expected_keywords: List[str],
        domain_context: str = "finance_housing"
    ) -> KoreanEvaluationMetrics:
        """
        한국어 이해도 평가
        
        Args:
            query: 검색 쿼리
            search_results: 검색 결과 리스트
            expected_keywords: 예상 키워드 리스트
            domain_context: 도메인 컨텍스트
            
        Returns:
            한국어 평가 메트릭
        """
        if not search_results:
            return KoreanEvaluationMetrics(
                avg_similarity=0.0, max_similarity=0.0, min_similarity=0.0,
                keyword_precision=0.0, keyword_recall=0.0, keyword_f1=0.0,
                semantic_coherence=0.0, context_relevance=0.0,
                korean_entity_recognition=0.0, domain_specificity=0.0,
                overall_korean_score=0.0
            )
        
        # 1. 기본 유사도 메트릭
        similarities = [r['similarity'] for r in search_results]
        avg_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)
        min_similarity = np.min(similarities)
        
        # 2. 키워드 정확도 평가
        keyword_metrics = self._evaluate_keyword_accuracy(
            search_results, expected_keywords
        )
        
        # 3. 의미적 일관성 평가
        semantic_coherence = self._evaluate_semantic_coherence(
            query, search_results
        )
        
        # 4. 맥락 관련성 평가
        context_relevance = self._evaluate_context_relevance(
            query, search_results, domain_context
        )
        
        # 5. 한국어 개체명 인식 평가
        entity_recognition = self._evaluate_korean_entities(
            search_results, query
        )
        
        # 6. 도메인 특화성 평가
        domain_specificity = self._evaluate_domain_specificity(
            search_results, domain_context
        )
        
        # 7. 종합 점수 계산
        overall_score = self._calculate_overall_korean_score(
            keyword_metrics, semantic_coherence, context_relevance,
            entity_recognition, domain_specificity, avg_similarity
        )
        
        return KoreanEvaluationMetrics(
            avg_similarity=avg_similarity,
            max_similarity=max_similarity,
            min_similarity=min_similarity,
            keyword_precision=keyword_metrics['precision'],
            keyword_recall=keyword_metrics['recall'],
            keyword_f1=keyword_metrics['f1'],
            semantic_coherence=semantic_coherence,
            context_relevance=context_relevance,
            korean_entity_recognition=entity_recognition,
            domain_specificity=domain_specificity,
            overall_korean_score=overall_score
        )

    def _evaluate_keyword_accuracy(
        self,
        search_results: List[Dict[str, Any]],
        expected_keywords: List[str]
    ) -> Dict[str, float]:
        """키워드 정확도 평가"""
        found_keywords = set()
        total_content = ""
        
        for result in search_results:
            content = result['content'].lower()
            total_content += content + " "
            
            for keyword in expected_keywords:
                if keyword.lower() in content:
                    found_keywords.add(keyword.lower())
        
        # Precision: 찾은 키워드 중 정확한 비율
        precision = len(found_keywords) / len(expected_keywords) if expected_keywords else 0.0
        
        # Recall: 전체 키워드 중 찾은 비율
        recall = len(found_keywords) / len(expected_keywords) if expected_keywords else 0.0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _evaluate_semantic_coherence(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> float:
        """의미적 일관성 평가"""
        if not search_results:
            return 0.0
        
        # 쿼리와 결과 간의 의미적 연관성 평가
        query_words = set(self._extract_korean_words(query))
        coherence_scores = []
        
        for result in search_results:
            content_words = set(self._extract_korean_words(result['content']))
            
            # 단어 겹침 비율
            overlap_ratio = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0.0
            
            # 유사도 가중치
            similarity_weight = result['similarity']
            
            # 종합 점수
            coherence_score = overlap_ratio * 0.6 + similarity_weight * 0.4
            coherence_scores.append(coherence_score)
        
        return np.mean(coherence_scores)

    def _evaluate_context_relevance(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        domain_context: str
    ) -> float:
        """맥락 관련성 평가"""
        if not search_results:
            return 0.0
        
        # 도메인별 키워드 정의
        domain_keywords = {
            'finance_housing': [
                '대출', '이자', '보증금', '임차', '주택', '금리', '신청', '자격',
                '소득', '한도', '서류', '절차', '지원', '서울시', '신혼부부'
            ],
            'general': ['정보', '안내', '문의', '확인', '신청', '지원']
        }
        
        relevant_keywords = domain_keywords.get(domain_context, domain_keywords['general'])
        relevance_scores = []
        
        for result in search_results:
            content = result['content'].lower()
            found_domain_keywords = sum(1 for kw in relevant_keywords if kw in content)
            relevance_score = found_domain_keywords / len(relevant_keywords)
            relevance_scores.append(relevance_score)
        
        return np.mean(relevance_scores)

    def _evaluate_korean_entities(
        self,
        search_results: List[Dict[str, Any]],
        query: str
    ) -> float:
        """한국어 개체명 인식 평가"""
        if not search_results:
            return 0.0
        
        # 한국어 개체명 패턴
        korean_entity_patterns = [
            r'[가-힣]+시',  # 도시명
            r'[가-힣]+구',  # 구명
            r'[가-힣]+동',  # 동명
            r'[가-힣]+은행',  # 은행명
            r'[가-힣]+공사',  # 공사명
            r'[가-힣]+센터',  # 센터명
            r'\d+억원',  # 금액
            r'\d+만원',  # 금액
            r'\d+%',  # 퍼센트
            r'\d+년',  # 연도
            r'\d+개월',  # 기간
        ]
        
        entity_scores = []
        for result in search_results:
            content = result['content']
            found_entities = 0
            
            for pattern in korean_entity_patterns:
                matches = re.findall(pattern, content)
                found_entities += len(matches)
            
            # 정규화된 점수 (문서 길이 대비)
            entity_score = found_entities / len(content) * 1000  # 1000자당 개체 수
            entity_scores.append(min(entity_score, 1.0))  # 최대 1.0으로 제한
        
        return np.mean(entity_scores)

    def _evaluate_domain_specificity(
        self,
        search_results: List[Dict[str, Any]],
        domain_context: str
    ) -> float:
        """도메인 특화성 평가"""
        if not search_results:
            return 0.0
        
        # 금융/주거 도메인 특화 키워드
        domain_specific_keywords = {
            'finance_housing': [
                '임차보증금', '이자지원', '대출한도', '소득기준', '신혼부부',
                '전세', '월세', '보증서', '금융공사', '주택금융', '연소득',
                '무주택자', '세대주', '혼인관계', '가족관계', '주민등록'
            ]
        }
        
        keywords = domain_specific_keywords.get(domain_context, [])
        specificity_scores = []
        
        for result in search_results:
            content = result['content'].lower()
            found_specific_keywords = sum(1 for kw in keywords if kw in content)
            specificity_score = found_specific_keywords / len(keywords) if keywords else 0.0
            specificity_scores.append(specificity_score)
        
        return np.mean(specificity_scores)

    def _extract_korean_words(self, text: str) -> List[str]:
        """한국어 단어 추출"""
        # 한글, 영문, 숫자만 추출
        words = re.findall(r'[가-힣a-zA-Z0-9]+', text.lower())
        # 2글자 이상만 유효한 단어로 간주
        return [word for word in words if len(word) >= 2]

    def _calculate_overall_korean_score(
        self,
        keyword_metrics: Dict[str, float],
        semantic_coherence: float,
        context_relevance: float,
        entity_recognition: float,
        domain_specificity: float,
        avg_similarity: float
    ) -> float:
        """종합 한국어 이해도 점수 계산"""
        # 가중치 설정
        weights = {
            'keyword_f1': 0.25,      # 키워드 정확도
            'semantic_coherence': 0.20,  # 의미적 일관성
            'context_relevance': 0.20,   # 맥락 관련성
            'entity_recognition': 0.15,  # 개체명 인식
            'domain_specificity': 0.15,  # 도메인 특화성
            'avg_similarity': 0.05       # 기본 유사도
        }
        
        overall_score = (
            keyword_metrics['f1'] * weights['keyword_f1'] +
            semantic_coherence * weights['semantic_coherence'] +
            context_relevance * weights['context_relevance'] +
            entity_recognition * weights['entity_recognition'] +
            domain_specificity * weights['domain_specificity'] +
            avg_similarity * weights['avg_similarity']
        )
        
        return min(overall_score, 1.0)  # 최대 1.0으로 제한

    def compare_query_embeddings(
        self,
        query: str,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        단일 쿼리에 대한 모든 모델의 임베딩 비교

        Args:
            query: 검색 쿼리
            save_results: 결과 저장 여부

        Returns:
            모델별 임베딩 정보
        """
        logger.info(f"Comparing query embeddings for: '{query}'")

        results = self.multi_encoder.encode_query_all(query)

        # 임베딩 간 유사도 계산
        embeddings = {
            model_name: np.array(data["embedding"])
            for model_name, data in results.items()
        }

        # 코사인 유사도 매트릭스
        similarity_matrix = {}
        model_names = list(embeddings.keys())

        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:  # 중복 제거
                    emb1 = embeddings[model1]
                    emb2 = embeddings[model2]

                    # 차원이 다르면 비교 불가
                    if len(emb1) != len(emb2):
                        similarity = None
                    else:
                        # 코사인 유사도
                        similarity = float(
                            np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        )

                    pair_key = f"{model1} <-> {model2}"
                    similarity_matrix[pair_key] = similarity

        comparison_result = {
            "query": query,
            "embeddings": results,
            "similarity_matrix": similarity_matrix,
            "timestamp": datetime.now().isoformat()
        }

        # 결과 저장
        if save_results:
            filename = f"query_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.results_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(comparison_result, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved comparison results to {filepath}")

        return comparison_result

    def compare_document_embeddings(
        self,
        documents: List[str],
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        문서 임베딩 비교

        Args:
            documents: 문서 리스트
            save_results: 결과 저장 여부

        Returns:
            모델별 임베딩 정보
        """
        logger.info(f"Comparing document embeddings for {len(documents)} documents")

        results = self.multi_encoder.encode_documents_all(documents, show_progress=True)

        # 통계 계산
        stats = {}
        for model_name, data in results.items():
            embeddings = np.array(data["embeddings"])

            stats[model_name] = {
                "model_name": data["model_name"],
                "dimension": data["dimension"],
                "count": data["count"],
                "embedding_stats": {
                    "mean": float(np.mean(embeddings)),
                    "std": float(np.std(embeddings)),
                    "min": float(np.min(embeddings)),
                    "max": float(np.max(embeddings))
                }
            }

        comparison_result = {
            "document_count": len(documents),
            "results": results,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }

        # 결과 저장
        if save_results and self.config.save_embeddings:
            filename = f"document_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.results_dir / filename

            # 임베딩은 크기가 크므로 통계만 저장
            save_data = {
                "document_count": len(documents),
                "statistics": stats,
                "timestamp": comparison_result["timestamp"]
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved comparison statistics to {filepath}")

        return comparison_result

    def add_search_result(self, result: SearchResult):
        """검색 결과 추가"""
        self.search_results.append(result)

    def calculate_metrics(self) -> Dict[str, ComparisonMetrics]:
        """
        누적된 검색 결과로부터 메트릭 계산

        Returns:
            모델별 메트릭
        """
        if not self.search_results:
            logger.warning("No search results to calculate metrics")
            return {}

        # 모델별로 그룹화
        results_by_model = {}
        for result in self.search_results:
            model_type = result.model_type
            if model_type not in results_by_model:
                results_by_model[model_type] = []
            results_by_model[model_type].append(result)

        # 메트릭 계산
        metrics = {}
        for model_type, results in results_by_model.items():
            search_times = [r.search_time for r in results]
            top1_sims = [r.similarities[0] if r.similarities else 0.0 for r in results]
            top3_sims = [
                np.mean(r.similarities[:3]) if len(r.similarities) >= 3 else np.mean(r.similarities)
                for r in results
            ]

            metric = ComparisonMetrics(
                model_name=results[0].model_name,
                model_type=model_type,
                avg_search_time=float(np.mean(search_times)),
                total_queries=len(results),
                avg_top1_similarity=float(np.mean(top1_sims)),
                avg_top3_similarity=float(np.mean(top3_sims)),
                embedding_dimension=results[0].embedding_dimension
            )

            metrics[model_type] = metric

        self.metrics = metrics
        return metrics

    def save_metrics(self, filename: Optional[str] = None):
        """메트릭 저장"""
        if not self.metrics:
            self.calculate_metrics()

        if not filename:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.results_dir / filename

        # dataclass를 dict로 변환
        metrics_dict = {
            model_type: asdict(metric)
            for model_type, metric in self.metrics.items()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved metrics to {filepath}")
        return filepath

    def print_comparison_table(self):
        """비교 결과를 표로 출력"""
        if not self.metrics:
            self.calculate_metrics()

        print("\n" + "=" * 100)
        print("모델 성능 비교표")
        print("=" * 100)

        # 헤더
        print(f"{'모델명':<30} {'차원':>6} {'쿼리수':>8} {'평균속도(s)':>12} {'Top-1 유사도':>14} {'Top-3 유사도':>14}")
        print("-" * 100)

        # 각 모델 정보
        for model_type, metric in self.metrics.items():
            print(
                f"{metric.model_name:<30} "
                f"{metric.embedding_dimension:>6} "
                f"{metric.total_queries:>8} "
                f"{metric.avg_search_time:>12.4f} "
                f"{metric.avg_top1_similarity:>14.4f} "
                f"{metric.avg_top3_similarity:>14.4f}"
            )

        print("=" * 100)

        # 추천 모델
        if self.metrics:
            # Top-1 유사도 기준 최고 모델
            best_similarity = max(self.metrics.values(), key=lambda m: m.avg_top1_similarity)
            # 속도 기준 최고 모델
            best_speed = min(self.metrics.values(), key=lambda m: m.avg_search_time)

            print(f"\n✅ 최고 정확도: {best_similarity.model_name} (Top-1 유사도: {best_similarity.avg_top1_similarity:.4f})")
            print(f"⚡ 최고 속도: {best_speed.model_name} (평균 {best_speed.avg_search_time:.4f}초)")
            print()

    def export_results(self, format: str = "json") -> Path:
        """
        전체 결과 내보내기

        Args:
            format: 'json' 또는 'csv'

        Returns:
            저장된 파일 경로
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if format == "json":
            filename = f"comparison_results_{timestamp}.json"
            filepath = self.results_dir / filename

            export_data = {
                "config": asdict(self.config),
                "models": [mt.value for mt in self.model_types],
                "metrics": {
                    model_type: asdict(metric)
                    for model_type, metric in self.metrics.items()
                },
                "search_results": [asdict(r) for r in self.search_results],
                "timestamp": datetime.now().isoformat()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

        elif format == "csv":
            import csv

            filename = f"comparison_results_{timestamp}.csv"
            filepath = self.results_dir / filename

            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # 헤더
                writer.writerow([
                    "모델명", "모델타입", "차원", "쿼리수", "평균속도(s)",
                    "Top-1 유사도", "Top-3 유사도"
                ])

                # 데이터
                for model_type, metric in self.metrics.items():
                    writer.writerow([
                        metric.model_name,
                        metric.model_type,
                        metric.embedding_dimension,
                        metric.total_queries,
                        f"{metric.avg_search_time:.4f}",
                        f"{metric.avg_top1_similarity:.4f}",
                        f"{metric.avg_top3_similarity:.4f}"
                    ])

        logger.info(f"Exported results to {filepath}")
        return filepath


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)

    print("=== Testing Model Comparator ===\n")

    # 2개 모델로 테스트 (시간 절약)
    models = [
        EmbeddingModelType.MULTILINGUAL_E5_SMALL,
        EmbeddingModelType.KAKAOBANK_DEBERTA
    ]

    comparator = ModelComparator(models)

    # 쿼리 임베딩 비교
    query = "강남구 청년주택 찾기"
    query_result = comparator.compare_query_embeddings(query)

    print(f"\nQuery: {query}")
    print(f"Models compared: {len(query_result['embeddings'])}")
    print("\nSimilarity between models:")
    for pair, similarity in query_result['similarity_matrix'].items():
        if similarity is not None:
            print(f"  {pair}: {similarity:.4f}")
        else:
            print(f"  {pair}: N/A (different dimensions)")

    # 문서 임베딩 비교
    documents = [
        "강남구에 위치한 청년주택입니다.",
        "서초구 원룸 월세 50만원",
        "송파구 투룸 전세 1억"
    ]

    doc_result = comparator.compare_document_embeddings(documents, save_results=False)

    print(f"\n\nDocument embeddings compared for {doc_result['document_count']} documents")
    print("\nStatistics:")
    for model_name, stats in doc_result['statistics'].items():
        print(f"  {stats['model_name']}:")
        print(f"    - Dimension: {stats['dimension']}")
        print(f"    - Mean: {stats['embedding_stats']['mean']:.6f}")
        print(f"    - Std: {stats['embedding_stats']['std']:.6f}")

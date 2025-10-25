#!/usr/bin/env python3
"""
RAG 평가 메트릭 계산 모듈
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np


@dataclass
class RetrievalMetrics:
    """검색 품질 메트릭"""
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_k: float = 0.0  # Normalized Discounted Cumulative Gain
    avg_similarity: float = 0.0
    keyword_coverage: float = 0.0  # 예상 키워드 커버리지


@dataclass
class GenerationMetrics:
    """생성 품질 메트릭"""
    exact_match: bool = False
    keyword_f1: float = 0.0
    keyword_precision: float = 0.0
    keyword_recall: float = 0.0
    answer_length: int = 0
    contains_ground_truth: bool = False


@dataclass
class EvaluationMetrics:
    """종합 평가 메트릭"""
    query_id: str
    query: str
    query_type: str
    difficulty: str
    
    # 검색 메트릭
    retrieval: RetrievalMetrics
    
    # 생성 메트릭 (답변 생성 시)
    generation: Optional[GenerationMetrics] = None
    
    # 전체 점수
    overall_score: float = 0.0
    
    # 메타데이터
    response_time_ms: float = 0.0
    retrieved_docs_count: int = 0
    error_message: Optional[str] = None


class MetricsCalculator:
    """메트릭 계산기"""
    
    def __init__(self):
        self.stop_words = {
            '은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '도', '로', '으로',
            '에서', '부터', '까지', '에게', '한테', '께', '보다', '처럼', '같이', '만큼',
            '조차', '마저', '까지', '뿐', '밖에', '외에', '대신', '위해', '통해', '따라',
            '관련', '대한', '대해', '대하', '위한', '위해', '통한', '통해', '따른', '따라'
        }
    
    def calculate_retrieval_metrics(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        expected_keywords: List[str],
        expected_docs: List[str],
        k: int = 5
    ) -> RetrievalMetrics:
        """검색 메트릭 계산"""
        
        if not retrieved_docs:
            return RetrievalMetrics()
        
        # 1. Recall@K: 예상 키워드가 검색된 문서에 포함된 비율
        recall_at_k = self._calculate_recall_at_k(
            retrieved_docs[:k], expected_keywords
        )
        
        # 2. Precision@K: 상위 K개 문서 중 관련 문서 비율
        precision_at_k = self._calculate_precision_at_k(
            retrieved_docs[:k], expected_keywords
        )
        
        # 3. MRR: 첫 번째 관련 문서의 순위 역수
        mrr = self._calculate_mrr(retrieved_docs, expected_keywords)
        
        # 4. NDCG@K: 정규화된 할인 누적 이득
        ndcg_at_k = self._calculate_ndcg_at_k(
            retrieved_docs[:k], expected_keywords
        )
        
        # 5. 평균 유사도
        similarities = [doc.get('similarity', 0.0) for doc in retrieved_docs[:k]]
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # 6. 키워드 커버리지
        keyword_coverage = self._calculate_keyword_coverage(
            retrieved_docs[:k], expected_keywords
        )
        
        return RetrievalMetrics(
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            avg_similarity=avg_similarity,
            keyword_coverage=keyword_coverage
        )
    
    def calculate_generation_metrics(
        self,
        generated_answer: str,
        ground_truth_answer: str,
        expected_keywords: List[str]
    ) -> GenerationMetrics:
        """생성 메트릭 계산"""
        
        if not generated_answer:
            return GenerationMetrics()
        
        # 1. Exact Match: 정확한 일치
        exact_match = self._check_exact_match(
            generated_answer, ground_truth_answer
        )
        
        # 2. 키워드 F1 점수
        keyword_f1, keyword_precision, keyword_recall = self._calculate_keyword_f1(
            generated_answer, expected_keywords
        )
        
        # 3. 답변 길이
        answer_length = len(generated_answer.strip())
        
        # 4. Ground Truth 포함 여부
        contains_ground_truth = self._check_contains_ground_truth(
            generated_answer, ground_truth_answer
        )
        
        return GenerationMetrics(
            exact_match=exact_match,
            keyword_f1=keyword_f1,
            keyword_precision=keyword_precision,
            keyword_recall=keyword_recall,
            answer_length=answer_length,
            contains_ground_truth=contains_ground_truth
        )
    
    def _calculate_recall_at_k(
        self, 
        docs: List[Dict[str, Any]], 
        expected_keywords: List[str]
    ) -> float:
        """Recall@K 계산"""
        if not expected_keywords:
            return 0.0
        
        found_keywords = set()
        for doc in docs:
            # natural_text 또는 content 필드에서 텍스트 가져오기
            text = doc.get('natural_text', doc.get('content', '')).lower()
            for keyword in expected_keywords:
                if keyword.lower() in text:
                    found_keywords.add(keyword.lower())
        
        return len(found_keywords) / len(expected_keywords)
    
    def _calculate_precision_at_k(
        self, 
        docs: List[Dict[str, Any]], 
        expected_keywords: List[str]
    ) -> float:
        """Precision@K 계산"""
        if not docs:
            return 0.0
        
        relevant_docs = 0
        for doc in docs:
            # natural_text 또는 content 필드에서 텍스트 가져오기
            text = doc.get('natural_text', doc.get('content', '')).lower()
            if any(keyword.lower() in text for keyword in expected_keywords):
                relevant_docs += 1
        
        return relevant_docs / len(docs)
    
    def _calculate_mrr(
        self, 
        docs: List[Dict[str, Any]], 
        expected_keywords: List[str]
    ) -> float:
        """MRR (Mean Reciprocal Rank) 계산"""
        for i, doc in enumerate(docs):
            # natural_text 또는 content 필드에서 텍스트 가져오기
            text = doc.get('natural_text', doc.get('content', '')).lower()
            if any(keyword.lower() in text for keyword in expected_keywords):
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_ndcg_at_k(
        self, 
        docs: List[Dict[str, Any]], 
        expected_keywords: List[str]
    ) -> float:
        """NDCG@K 계산"""
        if not docs:
            return 0.0
        
        # 각 문서의 관련성 점수 (0 또는 1)
        relevance_scores = []
        for doc in docs:
            # natural_text 또는 content 필드에서 텍스트 가져오기
            text = doc.get('natural_text', doc.get('content', '')).lower()
            score = 1.0 if any(keyword.lower() in text for keyword in expected_keywords) else 0.0
            relevance_scores.append(score)
        
        # DCG 계산
        dcg = 0.0
        for i, score in enumerate(relevance_scores):
            dcg += score / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # IDCG 계산 (완벽한 순서)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_keyword_coverage(
        self, 
        docs: List[Dict[str, Any]], 
        expected_keywords: List[str]
    ) -> float:
        """키워드 커버리지 계산"""
        if not expected_keywords:
            return 0.0
        
        # natural_text 또는 content 필드에서 텍스트 가져오기
        all_text = ' '.join([doc.get('natural_text', doc.get('content', '')) for doc in docs]).lower()
        found_keywords = sum(1 for keyword in expected_keywords 
                           if keyword.lower() in all_text)
        
        return found_keywords / len(expected_keywords)
    
    def _check_exact_match(self, answer: str, ground_truth: str) -> bool:
        """정확한 일치 확인"""
        answer_clean = self._normalize_text(answer)
        ground_truth_clean = self._normalize_text(ground_truth)
        return answer_clean == ground_truth_clean
    
    def _calculate_keyword_f1(
        self, 
        answer: str, 
        expected_keywords: List[str]
    ) -> Tuple[float, float, float]:
        """키워드 F1 점수 계산"""
        if not expected_keywords:
            return 0.0, 0.0, 0.0
        
        answer_words = set(self._extract_keywords(answer))
        expected_words = set(self._extract_keywords(' '.join(expected_keywords)))
        
        if not expected_words:
            return 0.0, 0.0, 0.0
        
        # Precision: 답변에 있는 키워드 중 예상 키워드 비율
        precision = len(answer_words & expected_words) / len(answer_words) if answer_words else 0.0
        
        # Recall: 예상 키워드 중 답변에 있는 비율
        recall = len(answer_words & expected_words) / len(expected_words)
        
        # F1: 조화평균
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1, precision, recall
    
    def _check_contains_ground_truth(self, answer: str, ground_truth: str) -> bool:
        """Ground Truth 포함 여부 확인"""
        if not ground_truth or ground_truth == "매출액 XX조원, 영업이익 XX조원":
            return False  # 플레이스홀더는 제외
        
        answer_clean = self._normalize_text(answer)
        ground_truth_clean = self._normalize_text(ground_truth)
        
        # 부분 일치 확인
        return ground_truth_clean in answer_clean
    
    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출"""
        # 한글, 영문, 숫자만 추출
        words = re.findall(r'[가-힣a-zA-Z0-9]+', text.lower())
        # 불용어 제거
        keywords = [word for word in words if word not in self.stop_words and len(word) > 1]
        return keywords
    
    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        # 공백 정리
        text = re.sub(r'\s+', ' ', text.strip())
        # 특수문자 제거
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()

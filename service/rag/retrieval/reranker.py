#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
리랭커 모듈
기본 리랭커 + 하이브리드 리랭커 (Rule-based + Cross-Encoder)
"""

import logging
import re
import math
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """리랭커 기본 클래스"""

    def __init__(self, name: str = "BaseReranker"):
        self.name = name
        logger.info(f"Initialized reranker: {self.name}")

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        후보 문서들을 리랭킹

        Args:
            query: 검색 쿼리
            candidates: 후보 문서 리스트
            top_k: 반환할 최대 개수

        Returns:
            리랭킹된 결과
        """
        raise NotImplementedError

    def _normalize_score(self, score: float, min_score: float, max_score: float) -> float:
        """점수를 0-1 범위로 정규화"""
        if max_score == min_score:
            return 0.5
        return (score - min_score) / (max_score - min_score)


class KeywordReranker(BaseReranker):
    """키워드 기반 리랭커"""

    def __init__(self, weight: float = 0.3):
        super().__init__("KeywordReranker")
        self.weight = weight

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """키워드 매칭을 기반으로 리랭킹"""
        if not candidates:
            return candidates

        query_words = set(self._extract_keywords(query))

        for candidate in candidates:
            content = candidate.get('content', candidate.get('chunk_text', ''))
            content_words = set(self._extract_keywords(content))

            # 키워드 겹침 점수 계산
            overlap = len(query_words.intersection(content_words))
            keyword_score = overlap / len(query_words) if query_words else 0.0

            # 기존 유사도와 결합
            original_similarity = candidate.get('similarity', 0.0)
            combined_score = (1 - self.weight) * original_similarity + self.weight * keyword_score

            candidate['keyword_score'] = keyword_score
            candidate['rerank_score'] = combined_score

        # 리랭킹 점수로 정렬
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

        return reranked[:top_k] if top_k else reranked

    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출"""
        # 한글, 영문, 숫자만 추출
        words = re.findall(r'[가-힣a-zA-Z0-9]+', text.lower())
        # 2글자 이상만 유효한 키워드로 간주
        return [word for word in words if len(word) >= 2]


class LengthReranker(BaseReranker):
    """길이 기반 리랭커"""

    def __init__(self, optimal_length: int = 200, weight: float = 0.1):
        super().__init__("LengthReranker")
        self.optimal_length = optimal_length
        self.weight = weight

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """콘텐츠 길이를 고려하여 리랭킹"""
        if not candidates:
            return candidates

        for candidate in candidates:
            content = candidate.get('content', candidate.get('chunk_text', ''))
            content_length = len(content)

            # 최적 길이에서의 거리 계산
            length_diff = abs(content_length - self.optimal_length)
            length_score = 1.0 / (1.0 + length_diff / self.optimal_length)

            # 기존 유사도와 결합
            original_similarity = candidate.get('similarity', 0.0)
            combined_score = (1 - self.weight) * original_similarity + self.weight * length_score

            candidate['length_score'] = length_score
            candidate['rerank_score'] = combined_score

        # 리랭킹 점수로 정렬
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

        return reranked[:top_k] if top_k else reranked


class PositionReranker(BaseReranker):
    """위치 기반 리랭커 (문서 내 위치 고려)"""

    def __init__(self, weight: float = 0.1):
        super().__init__("PositionReranker")
        self.weight = weight

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """문서 내 위치를 고려하여 리랭킹"""
        if not candidates:
            return candidates

        for candidate in candidates:
            # 메타데이터에서 위치 정보 추출
            metadata = candidate.get('metadata', {})
            chunk_index = metadata.get('chunk_index', 0)

            # 앞쪽 청크일수록 높은 점수 (문서 시작 부분이 중요)
            position_score = 1.0 / (1.0 + chunk_index * 0.1)

            # 기존 유사도와 결합
            original_similarity = candidate.get('similarity', 0.0)
            combined_score = (1 - self.weight) * original_similarity + self.weight * position_score

            candidate['position_score'] = position_score
            candidate['rerank_score'] = combined_score

        # 리랭킹 점수로 정렬
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

        return reranked[:top_k] if top_k else reranked


class SemanticReranker(BaseReranker):
    """의미적 유사도 기반 리랭커 (추가 임베딩 모델 사용)"""

    def __init__(
        self,
        encoder,
        weight: float = 0.4
    ):
        super().__init__("SemanticReranker")
        self.encoder = encoder
        self.weight = weight

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """의미적 유사도를 기반으로 리랭킹"""
        if not candidates:
            return candidates

        try:
            # 쿼리 임베딩
            query_embedding = self.encoder.encode_query(query)

            # 각 후보에 대한 의미적 유사도 계산
            for candidate in candidates:
                content = candidate.get('content', candidate.get('chunk_text', ''))
                content_embedding = self.encoder.encode_query(content)

                # 코사인 유사도 계산
                semantic_similarity = self._cosine_similarity(query_embedding, content_embedding)

                # 기존 유사도와 결합
                original_similarity = candidate.get('similarity', 0.0)
                combined_score = (1 - self.weight) * original_similarity + self.weight * semantic_similarity

                candidate['semantic_score'] = semantic_similarity
                candidate['rerank_score'] = combined_score

        except Exception as e:
            logger.warning(f"Semantic reranking failed: {e}")
            # 실패 시 원본 점수 유지
            for candidate in candidates:
                candidate['semantic_score'] = candidate.get('similarity', 0.0)
                candidate['rerank_score'] = candidate.get('similarity', 0.0)

        # 리랭킹 점수로 정렬
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

        return reranked[:top_k] if top_k else reranked

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(a * a for a in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class CombinedReranker(BaseReranker):
    """여러 리랭커를 조합한 통합 리랭커"""

    def __init__(
        self,
        rerankers: Optional[List[BaseReranker]] = None,
        weights: Optional[List[float]] = None
    ):
        super().__init__("CombinedReranker")
        
        # 기본 리랭커 조합
        if rerankers is None:
            rerankers = [
                KeywordReranker(weight=0.3),
                LengthReranker(optimal_length=200, weight=0.1),
                PositionReranker(weight=0.1)
            ]
        
        self.rerankers = rerankers
        self.weights = weights or [1.0] * len(rerankers)

        if len(self.weights) != len(self.rerankers):
            raise ValueError("Weights length must match rerankers length")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """여러 리랭커의 결과를 조합하여 최종 리랭킹"""
        if not candidates:
            return candidates

        # 각 리랭커로 점수 계산
        for reranker in self.rerankers:
            candidates = reranker.rerank(query, candidates, top_k=None)

        # 가중 평균으로 최종 점수 계산
        for candidate in candidates:
            scores = []
            for i, reranker in enumerate(self.rerankers):
                score_key = f"{reranker.name.lower().replace('reranker', '')}_score"
                if score_key in candidate:
                    scores.append(candidate[score_key] * self.weights[i])

            if scores:
                # 원본 유사도도 포함
                original_similarity = candidate.get('similarity', 0.0)
                final_score = (original_similarity * 0.5 + sum(scores) * 0.5) / (0.5 + sum(self.weights) * 0.5)
                candidate['final_rerank_score'] = final_score
            else:
                candidate['final_rerank_score'] = candidate.get('similarity', 0.0)

        # 최종 점수로 정렬
        reranked = sorted(candidates, key=lambda x: x['final_rerank_score'], reverse=True)

        return reranked[:top_k] if top_k else reranked


class HybridReranker:
    """
    2단계 하이브리드 리랭커
    
    Stage 1: 빠른 휴리스틱 필터링 (KeywordReranker, LengthReranker 등)
    Stage 2: 정밀 의미 평가 (Cross-Encoder)
    """

    def __init__(
        self,
        stage1_reranker: Optional[BaseReranker] = None,
        use_cross_encoder: bool = True,
        stage1_top_k_multiplier: float = 3.0,
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None
    ):
        """
        Args:
            stage1_reranker: 1단계 Reranker (None이면 기본 CombinedReranker)
            use_cross_encoder: Cross-Encoder 사용 여부
            stage1_top_k_multiplier: 1단계에서 가져올 배수 (final_k * multiplier)
            cross_encoder_model: Cross-Encoder 모델명
            device: 디바이스 ('cpu', 'cuda', 'mps')
        """
        self.stage1_reranker = stage1_reranker or create_default_reranker()
        self.use_cross_encoder = use_cross_encoder
        self.stage1_multiplier = stage1_top_k_multiplier

        # Cross-Encoder lazy loading
        self._cross_encoder = None
        self.cross_encoder_model = cross_encoder_model
        self.device = device

        logger.info(f"HybridReranker initialized (CE: {use_cross_encoder})")

    @property
    def cross_encoder(self):
        """Cross-Encoder lazy loading"""
        if self._cross_encoder is None and self.use_cross_encoder:
            try:
                from service.pgv_temp.reranker_crossencoder import CrossEncoderReranker
                self._cross_encoder = CrossEncoderReranker(
                    model_name=self.cross_encoder_model,
                    device=self.device
                )
                logger.info("Cross-Encoder loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Cross-Encoder: {e}")
                self.use_cross_encoder = False
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        2단계 하이브리드 리랭킹

        Args:
            query: 검색 쿼리
            candidates: 후보 문서 리스트
            top_k: 최종 반환할 문서 수

        Returns:
            리랭킹된 문서 리스트
        """
        if not candidates:
            return candidates

        num_candidates = len(candidates)
        logger.info(f"Starting hybrid reranking: {num_candidates} candidates → top_{top_k}")

        # Stage 1: 빠른 휴리스틱 필터링
        stage1_k = min(
            int(top_k * self.stage1_multiplier),
            num_candidates
        )

        stage1_results = self.stage1_reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=stage1_k
        )

        logger.info(f"Stage 1 complete: {len(stage1_results)} candidates")

        # Stage 2: Cross-Encoder 정밀 리랭킹
        if self.use_cross_encoder and self.cross_encoder:
            try:
                final_results = self.cross_encoder.rerank(
                    query=query,
                    candidates=stage1_results,
                    top_k=top_k
                )
                logger.info(f"Stage 2 complete: {len(final_results)} final results")
                return final_results
            except Exception as e:
                logger.error(f"Cross-Encoder reranking failed: {e}")
                # Fallback to stage 1 results
                return stage1_results[:top_k]
        else:
            # Cross-Encoder 미사용 시 Stage 1 결과만 반환
            return stage1_results[:top_k]

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return {
            'stage1_reranker': type(self.stage1_reranker).__name__,
            'cross_encoder_enabled': self.use_cross_encoder,
            'cross_encoder_loaded': self._cross_encoder is not None,
            'stage1_multiplier': self.stage1_multiplier
        }


def create_default_reranker() -> CombinedReranker:
    """기본 리랭커 조합 생성"""
    keyword_reranker = KeywordReranker(weight=0.3)
    length_reranker = LengthReranker(optimal_length=200, weight=0.1)
    position_reranker = PositionReranker(weight=0.1)

    return CombinedReranker(
        rerankers=[keyword_reranker, length_reranker, position_reranker],
        weights=[0.6, 0.2, 0.2]
    )


# LangGraph 노드용 비동기 함수
async def rerank(
    question: str,
    candidates: List[Dict],
    top_n: int = 4,
    use_hybrid: bool = True
) -> List[Dict]:
    """
    LangGraph 노드에서 사용할 비동기 rerank 함수

    Args:
        question: 사용자 질문
        candidates: 후보 문서 리스트
        top_n: 반환할 최대 개수
        use_hybrid: 하이브리드 모드 사용 여부

    Returns:
        리랭킹된 후보 리스트
    """
    if not use_hybrid:
        # 기본 Reranker만 사용
        reranker = create_default_reranker()
        return reranker.rerank(question, candidates, top_k=top_n)

    # 하이브리드 Reranker 사용
    hybrid_reranker = HybridReranker(
        use_cross_encoder=True,
        stage1_top_k_multiplier=2.0  # 최종 k의 2배를 1단계에서 선택
    )

    return hybrid_reranker.rerank(question, candidates, top_k=top_n)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== Reranker Test ===\n")

    # 테스트 데이터
    query = "2차전지 산업 전망은?"

    candidates = [
        {
            "chunk_text": "2025년 2차전지 산업은 전기차 수요 증가로 성장 전망",
            "content": "2025년 2차전지 산업은 전기차 수요 증가로 성장 전망",
            "similarity": 0.82,
            "metadata": {"chunk_index": 0}
        },
        {
            "chunk_text": "배터리 산업 전망: LFP 기술이 주목받고 있습니다",
            "content": "배터리 산업 전망: LFP 기술이 주목받고 있습니다",
            "similarity": 0.78,
            "metadata": {"chunk_index": 1}
        },
        {
            "chunk_text": "반도체 산업 전망은 AI 반도체 중심으로 회복세",
            "content": "반도체 산업 전망은 AI 반도체 중심으로 회복세",
            "similarity": 0.85,
            "metadata": {"chunk_index": 0}
        }
    ]

    # 1. 키워드 리랭커
    print("=== Keyword Reranker ===")
    keyword_reranker = KeywordReranker(weight=0.5)
    results = keyword_reranker.rerank(query, candidates.copy(), top_k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['rerank_score']:.3f}] {r['chunk_text'][:50]}...")

    # 2. Combined 리랭커
    print("\n=== Combined Reranker ===")
    combined_reranker = create_default_reranker()
    results = combined_reranker.rerank(query, candidates.copy(), top_k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['final_rerank_score']:.3f}] {r['chunk_text'][:50]}...")

    print("\nReranker module loaded successfully")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
검색 리트리버 모듈
기본 Retriever + 하이브리드 Retriever 통합
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor
import asyncpg

from ..models.encoder import EmbeddingEncoder
from ..models.config import EmbeddingModelType
from ..vectorstore.pgvector_store import PgVectorStore
from .reranker import BaseReranker, KeywordReranker, SemanticReranker, CombinedReranker
from ..query.temporal_parser import TemporalQueryParser

logger = logging.getLogger(__name__)


# ============================================================================
# 기본 Retriever 클래스 (임베딩 인코더 사용)
# ============================================================================

class Retriever:
    """
    통합 검색 리트리버 (벡터 + 키워드 + 하이브리드)
    
    지원 검색 방법:
    - vector: 벡터 유사도 검색 (의미적 유사성)
    - keyword: 키워드 Full-Text 검색 (정확한 용어 매칭)
    - hybrid: 벡터 + 키워드 결합 (RRF/Weighted Sum)
    """

    def __init__(
        self,
        model_type: EmbeddingModelType = EmbeddingModelType.MULTILINGUAL_E5_SMALL,
        db_config: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        reranker: Optional[BaseReranker] = None,
        enable_temporal_filter: bool = True,
        enable_hybrid: bool = True
    ):
        """
        Args:
            model_type: 사용할 임베딩 모델
            db_config: 데이터베이스 연결 설정
            device: 디바이스 ('cuda', 'cpu', None)
            reranker: 리랭킹 모듈 (선택사항)
            enable_temporal_filter: 시간 필터링 활성화 (기본값: True)
            enable_hybrid: 하이브리드 검색 활성화 (기본값: True)
        """
        self.model_type = model_type
        self.encoder = EmbeddingEncoder(model_type, device)
        self.vector_store = PgVectorStore(db_config)
        self.reranker = reranker
        self.enable_temporal_filter = enable_temporal_filter
        self.enable_hybrid = enable_hybrid
        self.db_config = db_config
        
        # 시간 쿼리 파서 초기화
        if self.enable_temporal_filter:
            self.temporal_parser = TemporalQueryParser()
        
        logger.info(f"Retriever initialized with {self.encoder.get_display_name()}")
        if self.reranker:
            logger.info(f"Reranker enabled: {self.reranker.name}")
        if self.enable_temporal_filter:
            logger.info(f"Temporal filtering enabled")
        if self.enable_hybrid:
            logger.info(f"Hybrid search enabled")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        include_metadata: bool = True,
        use_reranker: bool = True,
        include_context: bool = True,
        search_method: str = "vector"
    ) -> List[Dict[str, Any]]:
        """
        쿼리에 대한 검색 수행 (벡터/키워드/하이브리드)

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            min_similarity: 최소 유사도 임계값
            include_metadata: 메타데이터 포함 여부
            use_reranker: 리랭킹 사용 여부
            include_context: 앞뒤 문맥 포함 여부 (Context Window Management)
            search_method: 검색 방법 ("vector", "keyword", "hybrid")

        Returns:
            검색 결과 리스트
        """
        start_time = time.time()
        
        try:
            # 시간 정보 파싱 (Temporal Query Handler)
            temporal_info = None
            if self.enable_temporal_filter:
                temporal_info = self.temporal_parser.parse(query)
                if temporal_info.filters:
                    logger.info(f"Temporal filters applied: {temporal_info.filters}")
            
            # 쿼리 임베딩 생성
            query_embedding = self.encoder.encode_query(query)
            
            # 검색 방법에 따라 수행
            if search_method == "keyword":
                # 키워드 검색만
                results = self._keyword_search(query, top_k)
            elif search_method == "hybrid" and self.enable_hybrid:
                # 하이브리드 검색 (벡터 + 키워드)
                results = self._hybrid_search(query, query_embedding, top_k, min_similarity)
            else:
                # 벡터 검색 (기본값)
                results = self.vector_store.search_similar(
                    query_embedding=query_embedding,
                    model_type=self.model_type,
                    top_k=top_k,
                    min_similarity=min_similarity
                )
            
            # 검색 시간 계산
            search_time = (time.time() - start_time) * 1000  # ms
            
            # 결과 후처리
            processed_results = []
            for result in results:
                processed_result = {
                    'chunk_id': result.chunk_id,
                    'content': result.content,
                    'similarity': float(result.similarity),
                    'search_time_ms': search_time
                }
                
                # 메타데이터 추가
                if include_metadata:
                    processed_result.update({
                        'report_id': result.report_id,
                        'date': result.date,
                        'title': result.title,
                        'url': result.url,
                        'metadata': result.metadata
                    })
                
                processed_results.append(processed_result)
            
            # 리랭킹 적용
            if use_reranker and self.reranker and processed_results:
                logger.info(f"Applying reranker: {self.reranker.name}")
                processed_results = self.reranker.rerank(
                    query=query,
                    candidates=processed_results,
                    top_k=top_k
                )
            
            # 문맥 윈도우 추가
            if include_context:
                processed_results = self._enrich_with_context(processed_results)
            
            logger.info(f"Search completed: {len(processed_results)} results in {search_time:.2f}ms")
            return processed_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_only(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """검색만 수행 (리랭킹 없이)"""
        return self.search(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
            use_reranker=False,
            include_context=False
        )

    def get_model_stats(self) -> Dict[str, Any]:
        """모델 통계 정보 반환"""
        return {
            'model_name': self.encoder.get_display_name(),
            'model_type': self.model_type.value,
            'embedding_dimension': self.encoder.get_dimension(),
            'reranker_enabled': self.reranker is not None,
            'temporal_filter_enabled': self.enable_temporal_filter
        }

    def _keyword_search(self, query: str, top_k: int) -> List[Any]:
        """
        키워드 Full-Text 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        try:
            # PostgreSQL Full-Text Search 사용
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            sql_query = """
            SELECT
                d.report_id,
                d.date,
                d.title,
                d.url,
                e.chunk_id,
                e.chunk_text as content,
                ts_rank_cd(
                    to_tsvector('simple', e.chunk_text),
                    plainto_tsquery('simple', %s)
                ) AS similarity
            FROM embeddings_multilingual_e5_small e
            JOIN financial_documents d ON e.report_id = d.report_id
            WHERE to_tsvector('simple', e.chunk_text) @@ plainto_tsquery('simple', %s)
            ORDER BY similarity DESC
            LIMIT %s
            """
            
            cursor.execute(sql_query, (query, query, top_k))
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # PgVectorStore 결과 형식으로 변환
            from ..vectorstore.pgvector_store import SearchResult
            results = []
            for row in rows:
                results.append(SearchResult(
                    chunk_id=row['chunk_id'],
                    content=row['content'],
                    similarity=float(row['similarity']),
                    report_id=row['report_id'],
                    date=row['date'],
                    title=row['title'],
                    url=row['url'],
                    metadata={'source': 'keyword'}
                ))
            
            logger.info(f"Keyword search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int,
        min_similarity: float
    ) -> List[Any]:
        """
        하이브리드 검색 (벡터 + 키워드 RRF)
        
        Args:
            query: 검색 쿼리
            query_embedding: 쿼리 임베딩
            top_k: 반환할 결과 수
            min_similarity: 최소 유사도
            
        Returns:
            검색 결과 리스트
        """
        try:
            # 1. 벡터 검색
            vector_results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                model_type=self.model_type,
                top_k=top_k * 2,  # 더 많이 가져와서 융합
                min_similarity=min_similarity
            )
            
            # 2. 키워드 검색
            keyword_results = self._keyword_search(query, top_k * 2)
            
            # 3. Reciprocal Rank Fusion (RRF)
            fused_results = self._reciprocal_rank_fusion(
                vector_results, keyword_results, k=60
            )
            
            logger.info(f"Hybrid search found {len(fused_results)} results")
            return fused_results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to vector search
            return self.vector_store.search_similar(
                query_embedding=query_embedding,
                model_type=self.model_type,
                top_k=top_k,
                min_similarity=min_similarity
            )
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Any],
        keyword_results: List[Any],
        k: int = 60
    ) -> List[Any]:
        """
        Reciprocal Rank Fusion (RRF)
        
        Args:
            vector_results: 벡터 검색 결과
            keyword_results: 키워드 검색 결과
            k: RRF 파라미터
            
        Returns:
            융합된 결과 리스트
        """
        scores = {}
        chunk_map = {}
        
        # Vector 검색 결과 점수 계산
        for rank, result in enumerate(vector_results):
            chunk_id = result.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            chunk_map[chunk_id] = result
        
        # Keyword 검색 결과 점수 계산
        for rank, result in enumerate(keyword_results):
            chunk_id = result.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result
        
        # 점수로 정렬
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 결과 생성
        from ..vectorstore.pgvector_store import SearchResult
        results = []
        for chunk_id, fusion_score in sorted_chunks:
            original = chunk_map[chunk_id]
            # 융합 점수를 similarity로 설정
            result = SearchResult(
                chunk_id=original.chunk_id,
                content=original.content,
                similarity=fusion_score,
                report_id=original.report_id,
                date=original.date,
                title=original.title,
                url=original.url,
                metadata={'source': 'hybrid', 'original_similarity': original.similarity}
            )
            results.append(result)
        
        return results

    def _enrich_with_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        검색 결과에 앞뒤 문맥 추가 (Context Window Management)
        
        Args:
            results: 검색 결과 리스트
            
        Returns:
            문맥이 추가된 검색 결과
        """
        enriched_results = []
        
        for result in results:
            enriched_result = result.copy()
            
            # 메타데이터에서 문맥 정보 추출
            metadata = result.get('metadata', {})
            content = result.get('content', '')
            
            # 문맥 윈도우 구성
            context_parts = []
            
            # 이전 문맥 추가
            if 'prev_context' in metadata and metadata['prev_context']:
                context_parts.append(f"[이전 문맥]\n{metadata['prev_context']}\n")
            
            # 현재 내용
            context_parts.append(f"[검색된 내용]\n{content}\n")
            
            # 다음 문맥 추가
            if 'next_context' in metadata and metadata['next_context']:
                context_parts.append(f"[다음 문맥]\n{metadata['next_context']}")
            
            # enriched_text 필드에 저장
            enriched_result['enriched_text'] = '\n'.join(context_parts)
            
            # 기존 content는 유지 (원본 보존)
            enriched_result['original_content'] = content
            
            enriched_results.append(enriched_result)
        
        logger.debug(f"Enriched {len(enriched_results)} results with context windows")
        return enriched_results
    
    def close(self):
        """리트리버 정리"""
        if hasattr(self.vector_store, 'close'):
            self.vector_store.close()


# ============================================================================
# 하이브리드 Retriever 클래스 (벡터 + 키워드 검색)
# ============================================================================

class SearchMethod(Enum):
    """검색 방법"""
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"


class RankFusionMethod(Enum):
    """랭크 퓨전 방법"""
    RECIPROCAL_RANK_FUSION = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED_SUM = "weighted_sum"    # 가중 합산
    MAX_SCORE = "max_score"          # 최대 점수


@dataclass
class SearchConfig:
    """검색 설정"""
    search_method: SearchMethod = SearchMethod.HYBRID
    fusion_method: RankFusionMethod = RankFusionMethod.RECIPROCAL_RANK_FUSION
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    rrf_k: int = 60  # RRF 파라미터
    min_score: float = 0.0


class HybridRetriever:
    """하이브리드 검색기"""

    def __init__(
        self,
        db_config: Dict[str, str],
        table_name: str = "financial_documents",
        embedding_table: str = "embeddings_multilingual_e5_small"
    ):
        """
        Args:
            db_config: PostgreSQL 연결 설정
            table_name: 문서 테이블명
            embedding_table: 임베딩 테이블명
        """
        self.db_config = db_config
        self.table_name = table_name
        self.embedding_table = embedding_table
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """데이터베이스 연결 풀 초기화"""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(**self.db_config)
            logger.info("Database connection pool created")

    async def close(self):
        """연결 풀 종료"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    async def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        벡터 유사도 검색 (pgvector)

        Args:
            query_embedding: 쿼리 임베딩 벡터
            top_k: 반환할 최대 개수
            min_similarity: 최소 유사도

        Returns:
            검색 결과 리스트
        """
        if not self.pool:
            await self.initialize()

        query = f"""
        SELECT
            d.report_id,
            d.date,
            d.title,
            d.url,
            e.chunk_id,
            e.chunk_text,
            1 - (e.embedding <=> $1::vector) AS similarity
        FROM {self.embedding_table} e
        JOIN {self.table_name} d ON e.report_id = d.report_id
        WHERE 1 - (e.embedding <=> $1::vector) >= $2
        ORDER BY e.embedding <=> $1::vector
        LIMIT $3
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    query,
                    query_embedding,
                    min_similarity,
                    top_k
                )

                results = []
                for row in rows:
                    results.append({
                        'report_id': row['report_id'],
                        'date': row['date'],
                        'title': row['title'],
                        'url': row['url'],
                        'chunk_id': row['chunk_id'],
                        'chunk_text': row['chunk_text'],
                        'similarity': float(row['similarity']),
                        'source': 'vector'
                    })

                logger.info(f"Vector search found {len(results)} results")
                return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        language: str = 'korean'
    ) -> List[Dict[str, Any]]:
        """
        키워드 Full-Text Search (PostgreSQL tsvector)

        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 개수
            language: 검색 언어 ('korean', 'english')

        Returns:
            검색 결과 리스트
        """
        if not self.pool:
            await self.initialize()

        # PostgreSQL Full-Text Search 쿼리
        sql_query = f"""
        SELECT
            d.report_id,
            d.date,
            d.title,
            d.url,
            e.chunk_id,
            e.chunk_text,
            ts_rank_cd(
                to_tsvector($1, e.chunk_text),
                plainto_tsquery($1, $2)
            ) AS rank_score
        FROM {self.embedding_table} e
        JOIN {self.table_name} d ON e.report_id = d.report_id
        WHERE to_tsvector($1, e.chunk_text) @@ plainto_tsquery($1, $2)
        ORDER BY rank_score DESC
        LIMIT $3
        """

        # 언어 설정 매핑
        lang_config = {
            'korean': 'simple',  # PostgreSQL에 한국어 FTS가 없으면 simple 사용
            'english': 'english'
        }
        pg_language = lang_config.get(language, 'simple')

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    sql_query,
                    pg_language,
                    query,
                    top_k
                )

                results = []
                for row in rows:
                    results.append({
                        'report_id': row['report_id'],
                        'date': row['date'],
                        'title': row['title'],
                        'url': row['url'],
                        'chunk_id': row['chunk_id'],
                        'chunk_text': row['chunk_text'],
                        'similarity': float(row['rank_score']),
                        'source': 'keyword'
                    })

                logger.info(f"Keyword search found {len(results)} results")
                return results

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF)

        RRF Score = sum(1 / (k + rank))

        Args:
            vector_results: 벡터 검색 결과
            keyword_results: 키워드 검색 결과
            k: RRF 파라미터 (일반적으로 60)

        Returns:
            퓨전된 결과 리스트
        """
        # chunk_id를 키로 사용하여 점수 계산
        scores = {}
        chunk_map = {}

        # Vector 검색 결과 점수 계산
        for rank, result in enumerate(vector_results):
            chunk_id = result['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            chunk_map[chunk_id] = result

        # Keyword 검색 결과 점수 계산
        for rank, result in enumerate(keyword_results):
            chunk_id = result['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result

        # 점수로 정렬
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for chunk_id, fusion_score in sorted_chunks:
            result = chunk_map[chunk_id].copy()
            result['fusion_score'] = fusion_score
            result['source'] = 'hybrid_rrf'
            results.append(result)

        return results

    def weighted_sum_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        가중 합산 퓨전

        Args:
            vector_results: 벡터 검색 결과
            keyword_results: 키워드 검색 결과
            vector_weight: 벡터 검색 가중치
            keyword_weight: 키워드 검색 가중치

        Returns:
            퓨전된 결과 리스트
        """
        # chunk_id를 키로 사용하여 점수 계산
        scores = {}
        chunk_map = {}

        # Vector 검색 결과 점수 계산
        for result in vector_results:
            chunk_id = result['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0) + result['similarity'] * vector_weight
            chunk_map[chunk_id] = result

        # Keyword 검색 결과 점수 계산
        for result in keyword_results:
            chunk_id = result['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0) + result['similarity'] * keyword_weight
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result

        # 점수로 정렬
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for chunk_id, fusion_score in sorted_chunks:
            result = chunk_map[chunk_id].copy()
            result['fusion_score'] = fusion_score
            result['source'] = 'hybrid_weighted'
            results.append(result)

        return results

    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        config: Optional[SearchConfig] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 수행

        Args:
            query: 검색 쿼리 텍스트
            query_embedding: 쿼리 임베딩 벡터
            config: 검색 설정
            top_k: 반환할 최대 개수

        Returns:
            검색 결과 리스트
        """
        if config is None:
            config = SearchConfig()

        # 검색 방법에 따라 수행
        if config.search_method == SearchMethod.VECTOR_ONLY:
            return await self.vector_search(query_embedding, top_k=top_k)

        elif config.search_method == SearchMethod.KEYWORD_ONLY:
            return await self.keyword_search(query, top_k=top_k)

        else:  # HYBRID
            # 두 검색 동시 수행
            vector_results = await self.vector_search(query_embedding, top_k=top_k)
            keyword_results = await self.keyword_search(query, top_k=top_k)

            # 퓨전 방법에 따라 결합
            if config.fusion_method == RankFusionMethod.RECIPROCAL_RANK_FUSION:
                return self.reciprocal_rank_fusion(
                    vector_results, keyword_results, config.rrf_k
                )[:top_k]

            elif config.fusion_method == RankFusionMethod.WEIGHTED_SUM:
                return self.weighted_sum_fusion(
                    vector_results, keyword_results,
                    config.vector_weight, config.keyword_weight
                )[:top_k]

            else:  # MAX_SCORE
                return self._max_score_fusion(vector_results, keyword_results)[:top_k]

    def _max_score_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """최대 점수 퓨전"""
        # chunk_id를 키로 사용하여 최대 점수 계산
        scores = {}
        chunk_map = {}

        # Vector 검색 결과 점수 계산
        for result in vector_results:
            chunk_id = result['chunk_id']
            scores[chunk_id] = max(scores.get(chunk_id, 0), result['similarity'])
            chunk_map[chunk_id] = result

        # Keyword 검색 결과 점수 계산
        for result in keyword_results:
            chunk_id = result['chunk_id']
            scores[chunk_id] = max(scores.get(chunk_id, 0), result['similarity'])
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result

        # 점수로 정렬
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for chunk_id, fusion_score in sorted_chunks:
            result = chunk_map[chunk_id].copy()
            result['fusion_score'] = fusion_score
            result['source'] = 'hybrid_max'
            results.append(result)

        return results


# ============================================================================
# 다중 모델 Retriever (비교 분석용)
# ============================================================================

class MultiModelRetriever:
    """다중 모델 리트리버 (모델 성능 비교용)"""

    def __init__(self, db_config: Optional[Dict[str, str]] = None):
        """
        Args:
            db_config: 데이터베이스 연결 설정
        """
        self.db_config = db_config
        self.retrievers = {}
        
        # 지원하는 모델들
        self.supported_models = [
            EmbeddingModelType.MULTILINGUAL_E5_SMALL,
            EmbeddingModelType.MULTILINGUAL_E5_BASE,
            EmbeddingModelType.KOREAN_BERT_BASE
        ]
        
        logger.info(f"MultiModelRetriever initialized with {len(self.supported_models)} models")

    def _get_retriever(self, model_type: EmbeddingModelType) -> Retriever:
        """모델별 리트리버 가져오기 (lazy loading)"""
        if model_type not in self.retrievers:
            self.retrievers[model_type] = Retriever(
                model_type=model_type,
                db_config=self.db_config,
                enable_temporal_filter=True
            )
        return self.retrievers[model_type]

    def search_all_models(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        모든 모델로 검색 수행

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            min_similarity: 최소 유사도 임계값

        Returns:
            모델별 검색 결과 딕셔너리
        """
        results = {}
        
        for model_type in self.supported_models:
            try:
                retriever = self._get_retriever(model_type)
                model_results = retriever.search(
                    query=query,
                    top_k=top_k,
                    min_similarity=min_similarity,
                    use_reranker=False  # 비교를 위해 리랭킹 비활성화
                )
                results[model_type.value] = model_results
                logger.info(f"Search completed for {model_type.value}: {len(model_results)} results")
                
            except Exception as e:
                logger.error(f"Search failed for {model_type.value}: {e}")
                results[model_type.value] = []

        return results

    def compare_models(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> Dict[str, Any]:
        """
        모델 성능 비교

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            min_similarity: 최소 유사도 임계값

        Returns:
            비교 결과 딕셔너리
        """
        all_results = self.search_all_models(query, top_k, min_similarity)
        
        comparison = {
            'query': query,
            'top_k': top_k,
            'min_similarity': min_similarity,
            'models': {}
        }
        
        for model_name, results in all_results.items():
            if results:
                avg_similarity = sum(r['similarity'] for r in results) / len(results)
                search_time = results[0]['search_time_ms'] if results else 0.0
                
                comparison['models'][model_name] = {
                    'result_count': len(results),
                    'avg_similarity': avg_similarity,
                    'search_time_ms': search_time,
                    'top_result_similarity': results[0]['similarity'] if results else 0.0
                }
            else:
                comparison['models'][model_name] = {
                    'result_count': 0,
                    'avg_similarity': 0.0,
                    'search_time_ms': 0.0,
                    'top_result_similarity': 0.0
                }
        
        return comparison

    def close(self):
        """모든 리트리버 정리"""
        for retriever in self.retrievers.values():
            retriever.close()


if __name__ == "__main__":
    # 테스트
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Retriever Test ===\n")
    
    # 기본 Retriever 테스트
    retriever = Retriever(EmbeddingModelType.MULTILINGUAL_E5_SMALL)
    
    query = "신혼부부 임차보증금 이자지원"
    results = retriever.search(query, top_k=3)
    
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"{i}. Similarity: {result['similarity']:.4f}")
        print(f"   Content: {result['content'][:100]}...")
        print()
    
    # 모델 통계
    stats = retriever.get_model_stats()
    print("Model Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    retriever.close()
    
    print("\n=== Hybrid Retriever Test ===")
    print("HybridRetriever는 비동기 함수이므로 별도 테스트 필요")
    
    print("\nRetriever module loaded successfully")
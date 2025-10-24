#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
벡터 검색 리트리버
pgvector를 사용한 유사도 검색 및 결과 후처리
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor

from ..models.encoder import EmbeddingEncoder
from ..models.config import EmbeddingModelType
from ..vectorstore.pgvector_store import PgVectorStore
from .reranker import BaseReranker, KeywordReranker, SemanticReranker, CombinedReranker

logger = logging.getLogger(__name__)


class Retriever:
    """통합 검색 리트리버"""

    def __init__(
        self,
        model_type: EmbeddingModelType = EmbeddingModelType.MULTILINGUAL_E5_SMALL,
        db_config: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        reranker: Optional[BaseReranker] = None
    ):
        """
        Args:
            model_type: 사용할 임베딩 모델
            db_config: 데이터베이스 연결 설정
            device: 디바이스 ('cuda', 'cpu', None)
            reranker: 리랭킹 모듈 (선택사항)
        """
        self.model_type = model_type
        self.encoder = EmbeddingEncoder(model_type, device)
        self.vector_store = PgVectorStore(db_config)
        self.reranker = reranker
        
        logger.info(f"Retriever initialized with {self.encoder.get_display_name()}")
        if self.reranker:
            logger.info(f"Reranker enabled: {self.reranker.name}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        include_metadata: bool = True,
        use_reranker: bool = True
    ) -> List[Dict[str, Any]]:
        """
        쿼리에 대한 유사도 검색 수행

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            min_similarity: 최소 유사도 임계값
            include_metadata: 메타데이터 포함 여부
            use_reranker: 리랭킹 사용 여부

        Returns:
            검색 결과 리스트
        """
        start_time = time.time()
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.encoder.encode_query(query)
            
            # 벡터 검색 수행
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
                    'chunk_id': result['chunk_id'],
                    'content': result['content'],
                    'similarity': float(result['similarity']),
                    'search_time_ms': search_time
                }
                
                if include_metadata and result.get('metadata'):
                    processed_result['metadata'] = result['metadata']
                
                processed_results.append(processed_result)
            
            # 리랭킹 적용 (선택사항)
            if use_reranker and self.reranker:
                logger.info(f"Applying reranker: {self.reranker.name}")
                processed_results = self.reranker.rerank(query, processed_results, top_k)
            
            # 검색 로그 저장
            self._log_search(query, query_embedding, processed_results, search_time)
            
            logger.info(f"Search completed: {len(processed_results)} results in {search_time:.2f}ms")
            return processed_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def search_with_reranking(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        rerank_top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        리랭킹을 포함한 검색 (더 많은 후보에서 선별)

        Args:
            query: 검색 쿼리
            top_k: 최종 반환할 결과 수
            min_similarity: 최소 유사도 임계값
            rerank_top_k: 리랭킹할 후보 수

        Returns:
            리랭킹된 검색 결과
        """
        # 더 많은 후보 검색
        candidates = self.search(
            query=query,
            top_k=rerank_top_k,
            min_similarity=min_similarity,
            include_metadata=True
        )
        
        if len(candidates) <= top_k:
            return candidates
        
        # 간단한 리랭킹 (유사도 + 콘텐츠 길이 가중치)
        reranked = self._simple_rerank(query, candidates)
        
        return reranked[:top_k]

    def _simple_rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        간단한 리랭킹 로직

        Args:
            query: 원본 쿼리
            candidates: 후보 결과들

        Returns:
            리랭킹된 결과
        """
        query_words = set(query.lower().split())
        
        for candidate in candidates:
            content = candidate['content'].lower()
            content_words = set(content.split())
            
            # 단어 겹침 점수 계산
            word_overlap = len(query_words.intersection(content_words))
            word_overlap_score = word_overlap / len(query_words) if query_words else 0
            
            # 콘텐츠 길이 가중치 (너무 짧거나 긴 것에 불이익)
            content_length = len(candidate['content'])
            length_score = 1.0
            if content_length < 50:
                length_score = 0.8  # 너무 짧음
            elif content_length > 1000:
                length_score = 0.9  # 너무 김
            
            # 최종 점수 = 유사도 * 0.7 + 단어겹침 * 0.2 + 길이점수 * 0.1
            final_score = (
                candidate['similarity'] * 0.7 +
                word_overlap_score * 0.2 +
                length_score * 0.1
            )
            
            candidate['rerank_score'] = final_score
        
        # 리랭킹 점수로 정렬
        return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

    def _log_search(
        self,
        query: str,
        query_embedding: List[float],
        results: List[Dict[str, Any]],
        search_time: float
    ):
        """검색 로그 저장"""
        try:
            self.vector_store.connect()
            
            # 모델 ID 조회
            self.vector_store.cursor.execute(
                "SELECT id FROM vector_db.embedding_models WHERE model_name = %s",
                (self.encoder.get_model_name(),)
            )
            model_id = self.vector_store.cursor.fetchone()[0]
            
            # 평균 유사도 계산 (NaN 값 필터링)
            import math
            valid_similarities = [
                r['similarity'] for r in results 
                if not (math.isnan(r['similarity']) or math.isinf(r['similarity']))
            ]
            avg_similarity = sum(valid_similarities) / len(valid_similarities) if valid_similarities else 0.0
            
            # 검색 결과 JSON 생성 (NaN 값 필터링)
            results_json = {
                'chunk_ids': [r['chunk_id'] for r in results],
                'similarities': [
                    r['similarity'] if not (math.isnan(r['similarity']) or math.isinf(r['similarity'])) 
                    else 0.0 for r in results
                ]
            }
            
            # 로그 저장
            self.vector_store.cursor.execute("""
                INSERT INTO vector_db.search_logs 
                (model_id, query, query_embedding, top_k, search_time_ms, results, avg_similarity)
                VALUES (%s, %s, %s::vector, %s, %s, %s, %s)
            """, (
                model_id,
                query,
                '[' + ','.join(map(str, query_embedding)) + ']',
                len(results),
                search_time,
                psycopg2.extras.Json(results_json),
                avg_similarity
            ))
            
            self.vector_store.conn.commit()
            
        except Exception as e:
            logger.warning(f"Failed to log search: {e}")
            if self.vector_store.conn:
                self.vector_store.conn.rollback()

    def get_model_stats(self) -> Dict[str, Any]:
        """현재 모델의 통계 정보 반환"""
        try:
            self.vector_store.connect()
            
            # 모델 정보
            self.vector_store.cursor.execute("""
                SELECT display_name, dimension, COUNT(ce.id) as embedding_count
                FROM vector_db.embedding_models em
                LEFT JOIN vector_db.chunk_embeddings ce ON em.id = ce.model_id
                WHERE em.model_name = %s
                GROUP BY em.id, em.display_name, em.dimension
            """, (self.encoder.get_model_name(),))
            
            model_info = self.vector_store.cursor.fetchone()
            
            # 최근 검색 통계
            self.vector_store.cursor.execute("""
                SELECT 
                    AVG(search_time_ms) as avg_search_time,
                    AVG(avg_similarity) as avg_similarity,
                    COUNT(*) as total_searches
                FROM vector_db.search_logs sl
                JOIN vector_db.embedding_models em ON sl.model_id = em.id
                WHERE em.model_name = %s
                AND sl.created_at >= NOW() - INTERVAL '7 days'
            """, (self.encoder.get_model_name(),))
            
            search_stats = self.vector_store.cursor.fetchone()
            
            return {
                'model_name': model_info[0] if model_info else 'Unknown',
                'dimension': model_info[1] if model_info else 0,
                'embedding_count': model_info[2] if model_info else 0,
                'avg_search_time_ms': float(search_stats[0]) if search_stats[0] else 0.0,
                'avg_similarity': float(search_stats[1]) if search_stats[1] else 0.0,
                'total_searches_7d': search_stats[2] if search_stats[2] else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get model stats: {e}")
            return {}

    def close(self):
        """리소스 정리"""
        if hasattr(self, 'vector_store'):
            self.vector_store.disconnect()


class MultiModelRetriever:
    """다중 모델 리트리버 (모델 비교용)"""

    def __init__(
        self,
        model_types: List[EmbeddingModelType],
        db_config: Optional[Dict[str, str]] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            model_types: 사용할 모델 타입 리스트
            db_config: 데이터베이스 연결 설정
            device: 디바이스
        """
        self.retrievers = {}
        
        for model_type in model_types:
            try:
                self.retrievers[model_type] = VectorRetriever(
                    model_type=model_type,
                    db_config=db_config,
                    device=device
                )
                logger.info(f"Loaded retriever: {model_type.value}")
            except Exception as e:
                logger.error(f"Failed to load {model_type.value}: {e}")

    def search_all_models(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        모든 모델로 동시 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            min_similarity: 최소 유사도 임계값

        Returns:
            모델별 검색 결과
        """
        results = {}
        
        for model_type, retriever in self.retrievers.items():
            try:
                model_results = retriever.search(
                    query=query,
                    top_k=top_k,
                    min_similarity=min_similarity
                )
                results[model_type.value] = model_results
                
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
            비교 결과
        """
        all_results = self.search_all_models(query, top_k, min_similarity)
        
        comparison = {
            'query': query,
            'timestamp': time.time(),
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
    
    print("=== Vector Retriever Test ===\n")
    
    # 단일 모델 테스트
    retriever = VectorRetriever(EmbeddingModelType.MULTILINGUAL_E5_SMALL)
    
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

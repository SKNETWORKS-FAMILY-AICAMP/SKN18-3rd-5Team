#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PostgreSQL + pgvector 벡터 스토어 (개선된 버전)
중앙화된 설정과 표준화된 인터페이스 사용
"""

import logging
from typing import List, Dict, Any, Optional, Union
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np

from ..models.config import EmbeddingModelType
from ..interfaces.vector_store import VectorStoreInterface, SearchResult
from ..utils.error_handler import (
    DatabaseErrorHandler, 
    VectorStoreErrorHandler, 
    ErrorHandler,
    ErrorContext
)
from config.vector_database import get_vector_db_config

logger = logging.getLogger(__name__)


class PgVectorStore(VectorStoreInterface):
    """PostgreSQL + pgvector 벡터 스토어 (개선된 버전)"""

    def __init__(self, db_config: Optional[Dict[str, str]] = None):
        """
        Args:
            db_config: DB 연결 설정 (None이면 중앙화된 설정 사용)
        """
        # 중앙화된 설정 사용
        self.config = get_vector_db_config()
        self.db_config = db_config or self.config.get_db_config()
        
        # 에러 핸들러 초기화
        self.error_handler = ErrorHandler()
        
        # 연결 상태
        self.conn = None
        self._connect()

    def _connect(self):
        """PostgreSQL 연결 (에러 처리 개선)"""
        context = self.error_handler.create_context("database_connection")
        
        def _do_connect():
            self.conn = psycopg2.connect(**self.db_config)
            logger.info(f"Connected to {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
        
        try:
            self.error_handler.handle_database_operation(_do_connect, context)
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    def _get_cursor(self):
        """커서 반환 (재연결 지원)"""
        if self.conn is None or self.conn.closed:
            self._connect()
        return self.conn.cursor(cursor_factory=RealDictCursor)

    def search_similar(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        model_type: EmbeddingModelType,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        벡터 유사도 검색 (개선된 버전)

        Args:
            query_embedding: 쿼리 임베딩 벡터
            model_type: 모델 타입
            top_k: 반환할 결과 수
            min_similarity: 최소 유사도 (0~1)

        Returns:
            검색 결과 리스트
        """
        table_name = self.config.get_table_name(model_type)
        
        # 벡터를 리스트로 변환
        if isinstance(query_embedding, np.ndarray):
            emb_list = query_embedding.tolist()
        else:
            emb_list = query_embedding
        
        emb_str = '[' + ','.join(map(str, emb_list)) + ']'

        # 유사도 검색 쿼리 (pgvector <=> 연산자 사용)
        query = f"""
            SELECT
                c.id as chunk_db_id,
                c.chunk_id,
                c.natural_text as content,
                c.metadata->>'corp_name' as corp_name,
                c.metadata->>'document_name' as document_name,
                c.metadata->>'doc_type' as doc_type,
                c.metadata,
                (1 - (e.embedding <=> %s::vector)) as similarity
            FROM {table_name} e
            JOIN chunks c ON e.chunk_id = c.chunk_id
            WHERE (1 - (e.embedding <=> %s::vector)) >= %s
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
        """

        context = self.error_handler.create_context(
            "vector_search",
            model_type=model_type.value,
            additional_info={'top_k': top_k, 'min_similarity': min_similarity}
        )

        try:
            cursor = self._get_cursor()
            cursor.execute(query, (emb_str, emb_str, min_similarity, emb_str, top_k))
            results = cursor.fetchall()

            # SearchResult 객체로 변환
            search_results = []
            for row in results:
                result = SearchResult(
                    chunk_id=row['chunk_id'],
                    chunk_db_id=row['chunk_db_id'],
                    content=row['content'],
                    similarity=float(row['similarity']),
                    metadata=row['metadata'],
                    corp_name=row['corp_name'],
                    document_name=row['document_name'],
                    doc_type=row['doc_type']
                )
                search_results.append(result)

            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return VectorStoreErrorHandler.handle_search_error(e, context)

    def insert_embeddings(
        self,
        chunk_ids: List[Union[int, str]],
        embeddings: Union[np.ndarray, List[List[float]]],
        model_type: EmbeddingModelType
    ) -> int:
        """
        임베딩 삽입 (개선된 버전)

        Args:
            chunk_ids: 청크 ID 리스트
            embeddings: 임베딩 벡터 배열
            model_type: 모델 타입

        Returns:
            삽입된 개수
        """
        table_name = self.config.get_table_name(model_type)

        if len(chunk_ids) != len(embeddings):
            raise ValueError("chunk_ids and embeddings length mismatch")

        query = f"""
            INSERT INTO {table_name} (chunk_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                created_at = NOW()
        """

        context = self.error_handler.create_context(
            "insert_embeddings",
            model_type=model_type.value,
            additional_info={'table_name': table_name, 'batch_size': len(chunk_ids)}
        )

        try:
            cursor = self._get_cursor()

            # 배치 삽입
            for chunk_id, emb in zip(chunk_ids, embeddings):
                # 벡터를 리스트로 변환
                if isinstance(emb, np.ndarray):
                    emb_list = emb.tolist()
                elif hasattr(emb, 'tolist'):
                    emb_list = emb.tolist()
                else:
                    emb_list = emb
                
                emb_str = '[' + ','.join(map(str, emb_list)) + ']'
                cursor.execute(query, (chunk_id, emb_str))

            self.conn.commit()
            logger.info(f"Inserted {len(chunk_ids)} embeddings into {table_name}")
            return len(chunk_ids)

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Insert failed: {e}")
            
            # 에러 처리
            error_result = VectorStoreErrorHandler.handle_insert_error(e, context, embeddings)
            if error_result.get('retry', False):
                raise  # 재시도 가능한 에러
            else:
                return error_result.get('inserted', 0)

    def get_chunk_ids(
        self, 
        limit: Optional[int] = None,
        model_type: Optional[EmbeddingModelType] = None
    ) -> List[Union[int, str]]:
        """
        청크 ID 조회 (개선된 버전)

        Args:
            limit: 반환할 최대 개수
            model_type: 특정 모델의 임베딩이 없는 청크만 조회

        Returns:
            청크 ID 리스트
        """
        if model_type:
            # 특정 모델의 임베딩이 없는 청크만 조회
            table_name = self.config.get_table_name(model_type)
            query = f"""
                SELECT c.id FROM chunks c
                LEFT JOIN {table_name} e ON c.chunk_id = e.chunk_id
                WHERE e.chunk_id IS NULL
                  AND c.natural_text IS NOT NULL
                  AND c.natural_text != ''
            """
        else:
            # 모든 청크 조회
            query = "SELECT id FROM chunks"
        
        if limit:
            query += f" LIMIT {limit}"

        context = self.error_handler.create_context(
            "get_chunk_ids",
            model_type=model_type.value if model_type else None,
            additional_info={'limit': limit}
        )

        try:
            cursor = self._get_cursor()
            cursor.execute(query)
            return [row['id'] for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get chunk IDs: {e}")
            return []

    def get_embedding_count(self, model_type: EmbeddingModelType) -> int:
        """
        특정 모델의 임베딩 개수 조회
        
        Args:
            model_type: 모델 타입
            
        Returns:
            임베딩 개수
        """
        table_name = self.config.get_table_name(model_type)
        query = f"SELECT COUNT(*) FROM {table_name}"
        
        try:
            cursor = self._get_cursor()
            cursor.execute(query)
            return cursor.fetchone()['count']
        except Exception as e:
            logger.error(f"Failed to get embedding count: {e}")
            return 0

    def delete_embeddings(
        self,
        chunk_ids: List[Union[int, str]],
        model_type: EmbeddingModelType
    ) -> int:
        """
        임베딩 삭제
        
        Args:
            chunk_ids: 삭제할 청크 ID 리스트
            model_type: 모델 타입
            
        Returns:
            삭제된 개수
        """
        table_name = self.config.get_table_name(model_type)
        query = f"DELETE FROM {table_name} WHERE chunk_id = ANY(%s)"
        
        try:
            cursor = self._get_cursor()
            cursor.execute(query, (chunk_ids,))
            deleted_count = cursor.rowcount
            self.conn.commit()
            logger.info(f"Deleted {deleted_count} embeddings from {table_name}")
            return deleted_count
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to delete embeddings: {e}")
            return 0

    def is_connected(self) -> bool:
        """
        데이터베이스 연결 상태 확인
        
        Returns:
            연결 상태
        """
        try:
            if self.conn is None or self.conn.closed:
                return False
            
            # 간단한 쿼리로 연결 테스트
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except Exception:
            return False

    def close(self):
        """연결 종료"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Connection closed")

    def get_stats(self) -> Dict[str, Any]:
        """
        벡터 스토어 통계 조회
        
        Returns:
            통계 정보 딕셔너리
        """
        stats = {
            'connected': self.is_connected(),
            'models': {}
        }
        
        for model_type in self.config.table_config.get_all_models():
            try:
                count = self.get_embedding_count(model_type)
                stats['models'][model_type.value] = {
                    'table_name': self.config.get_table_name(model_type),
                    'dimension': self.config.get_dimension(model_type),
                    'embedding_count': count
                }
            except Exception as e:
                logger.warning(f"Failed to get stats for {model_type.value}: {e}")
                stats['models'][model_type.value] = {
                    'error': str(e)
                }
        
        return stats

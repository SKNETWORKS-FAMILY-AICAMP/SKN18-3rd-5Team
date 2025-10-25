#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PostgreSQL + pgvector 벡터 스토어
schema.sql의 3개 모델 테이블과 연동
"""

import os
import logging
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np

from ..models.config import EmbeddingModelType

logger = logging.getLogger(__name__)


class PgVectorStore:
    """PostgreSQL + pgvector 벡터 스토어"""

    # 모델별 임베딩 테이블 매핑 (schema.sql 기준)
    MODEL_TABLE_MAP = {
        EmbeddingModelType.MULTILINGUAL_E5_SMALL: "embeddings_e5_small",
        EmbeddingModelType.KAKAOBANK_DEBERTA: "embeddings_kakaobank",
        EmbeddingModelType.FINE5_FINANCE: "embeddings_fine5"
    }

    def __init__(self, db_config: Optional[Dict[str, str]] = None):
        """
        Args:
            db_config: DB 연결 설정 (None이면 환경변수 사용)
        """
        self.db_config = db_config or {
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': os.getenv('PG_PORT', '5432'),
            'database': os.getenv('PG_DB', 'postgres'),
            'user': os.getenv('PG_USER', 'postgres'),
            'password': os.getenv('PG_PASSWORD', 'postgres')
        }
        self.conn = None
        self._connect()

    def _connect(self):
        """PostgreSQL 연결"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info(f"Connected to {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
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
        query_embedding: np.ndarray,
        model_type: EmbeddingModelType,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        벡터 유사도 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            model_type: 모델 타입
            top_k: 반환할 결과 수
            min_similarity: 최소 유사도 (0~1)

        Returns:
            검색 결과 리스트
        """
        table_name = self.MODEL_TABLE_MAP.get(model_type)
        if not table_name:
            raise ValueError(f"Unknown model: {model_type}")

        # numpy array → PostgreSQL vector 문자열
        emb_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'

        # 유사도 검색 쿼리 (pgvector <=> 연산자 사용)
        query = f"""
            SELECT
                dc.id as chunk_db_id,
                dc.chunk_id,
                dc.natural_text as content,
                dc.corp_name,
                dc.document_name,
                dc.doc_type,
                dc.metadata,
                (1 - (e.embedding <=> %s::vector)) as similarity
            FROM vector_db.{table_name} e
            JOIN vector_db.document_chunks dc ON e.chunk_id = dc.id
            WHERE (1 - (e.embedding <=> %s::vector)) >= %s
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
        """

        try:
            cursor = self._get_cursor()
            cursor.execute(query, (emb_str, emb_str, min_similarity, emb_str, top_k))
            results = cursor.fetchall()

            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def insert_embeddings(
        self,
        chunk_ids: List[int],
        embeddings: np.ndarray,
        model_type: EmbeddingModelType
    ) -> int:
        """
        임베딩 삽입

        Args:
            chunk_ids: document_chunks.id 리스트
            embeddings: 임베딩 벡터 배열
            model_type: 모델 타입

        Returns:
            삽입된 개수
        """
        table_name = self.MODEL_TABLE_MAP.get(model_type)
        if not table_name:
            raise ValueError(f"Unknown model: {model_type}")

        if len(chunk_ids) != len(embeddings):
            raise ValueError("chunk_ids and embeddings length mismatch")

        query = f"""
            INSERT INTO vector_db.{table_name} (chunk_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                created_at = NOW()
        """

        try:
            cursor = self._get_cursor()

            # 배치 삽입
            for chunk_id, emb in zip(chunk_ids, embeddings):
                emb_str = '[' + ','.join(map(str, emb.tolist())) + ']'
                cursor.execute(query, (chunk_id, emb_str))

            self.conn.commit()
            logger.info(f"Inserted {len(chunk_ids)} embeddings into {table_name}")
            return len(chunk_ids)

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Insert failed: {e}")
            raise

    def get_chunk_ids(self, limit: Optional[int] = None) -> List[int]:
        """
        임베딩이 없는 청크 ID 조회

        Args:
            limit: 반환할 최대 개수

        Returns:
            청크 ID 리스트
        """
        query = "SELECT id FROM vector_db.document_chunks"
        if limit:
            query += f" LIMIT {limit}"

        try:
            cursor = self._get_cursor()
            cursor.execute(query)
            return [row['id'] for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get chunk IDs: {e}")
            raise

    def close(self):
        """연결 종료"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Connection closed")

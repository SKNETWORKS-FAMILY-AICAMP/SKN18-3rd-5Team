#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벡터 데이터베이스 설정 중앙화
PostgreSQL + pgvector 관련 모든 설정을 통합 관리
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from service.rag.models.config import EmbeddingModelType


@dataclass
class DatabaseConfig:
    """데이터베이스 연결 설정"""
    host: str = "localhost"
    port: int = 5432
    database: str = "skn_project"
    user: str = "postgres"
    password: str = "post1234"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user,
            'password': self.password
        }
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """환경변수에서 설정 로드"""
        return cls(
            host=os.getenv('PG_HOST', 'localhost'),
            port=int(os.getenv('PG_PORT', '5432')),
            database=os.getenv('PG_DB', 'skn_project'),
            user=os.getenv('PG_USER', 'postgres'),
            password=os.getenv('PG_PASSWORD', 'post1234')
        )


class VectorTableConfig:
    """벡터 테이블 설정"""
    
    # 모델별 임베딩 테이블 매핑
    MODEL_TABLE_MAP = {
        EmbeddingModelType.MULTILINGUAL_E5_SMALL: "embeddings_multilingual_e5_small",
        EmbeddingModelType.KAKAOBANK_DEBERTA: "embeddings_kakaobank",
        EmbeddingModelType.FINE5_FINANCE: "embeddings_fine5"
    }
    
    # 모델별 벡터 차원 매핑
    MODEL_DIMENSION_MAP = {
        EmbeddingModelType.MULTILINGUAL_E5_SMALL: 384,
        EmbeddingModelType.KAKAOBANK_DEBERTA: 768,
        EmbeddingModelType.FINE5_FINANCE: 4096
    }
    
    # HNSW 인덱스 설정
    HNSW_CONFIG = {
        'm': 16,                    # 연결 수
        'ef_construction': 64,      # 구성 시 탐색 깊이
        'ef_search': 40             # 검색 시 탐색 깊이
    }
    
    @classmethod
    def get_table_name(cls, model_type: EmbeddingModelType) -> str:
        """모델 타입으로 테이블명 조회"""
        table_name = cls.MODEL_TABLE_MAP.get(model_type)
        if not table_name:
            raise ValueError(f"Unknown model type: {model_type}")
        return table_name
    
    @classmethod
    def get_dimension(cls, model_type: EmbeddingModelType) -> int:
        """모델 타입으로 벡터 차원 조회"""
        dimension = cls.MODEL_DIMENSION_MAP.get(model_type)
        if not dimension:
            raise ValueError(f"Unknown model type: {model_type}")
        return dimension
    
    @classmethod
    def get_all_models(cls) -> list[EmbeddingModelType]:
        """모든 지원 모델 반환"""
        return list(cls.MODEL_TABLE_MAP.keys())


class VectorSearchConfig:
    """벡터 검색 설정"""
    
    # 기본 검색 파라미터
    DEFAULT_TOP_K = 5
    DEFAULT_MIN_SIMILARITY = 0.0
    DEFAULT_MAX_RESULTS = 100
    
    # 배치 처리 설정
    DEFAULT_BATCH_SIZE = 100
    MAX_BATCH_SIZE = 1000
    MIN_BATCH_SIZE = 1
    
    # 성능 설정
    SEARCH_TIMEOUT_SECONDS = 30
    INSERT_TIMEOUT_SECONDS = 60


class VectorDatabaseConfig:
    """벡터 데이터베이스 통합 설정"""
    
    def __init__(self, 
                 db_config: Optional[DatabaseConfig] = None,
                 use_env: bool = True):
        """
        Args:
            db_config: 데이터베이스 설정 (None이면 기본값 사용)
            use_env: 환경변수 사용 여부
        """
        if db_config is None:
            if use_env:
                self.db_config = DatabaseConfig.from_env()
            else:
                self.db_config = DatabaseConfig()
        else:
            self.db_config = db_config
        
        self.table_config = VectorTableConfig()
        self.search_config = VectorSearchConfig()
    
    def get_db_config(self) -> Dict[str, Any]:
        """데이터베이스 연결 설정 반환"""
        return self.db_config.to_dict()
    
    def get_table_name(self, model_type: EmbeddingModelType) -> str:
        """모델별 테이블명 반환"""
        return self.table_config.get_table_name(model_type)
    
    def get_dimension(self, model_type: EmbeddingModelType) -> int:
        """모델별 벡터 차원 반환"""
        return self.table_config.get_dimension(model_type)
    
    def get_hnsw_config(self) -> Dict[str, int]:
        """HNSW 인덱스 설정 반환"""
        return self.table_config.HNSW_CONFIG.copy()
    
    def get_search_config(self) -> Dict[str, Any]:
        """검색 설정 반환"""
        return {
            'default_top_k': self.search_config.DEFAULT_TOP_K,
            'default_min_similarity': self.search_config.DEFAULT_MIN_SIMILARITY,
            'default_batch_size': self.search_config.DEFAULT_BATCH_SIZE,
            'max_batch_size': self.search_config.MAX_BATCH_SIZE,
            'min_batch_size': self.search_config.MIN_BATCH_SIZE,
            'search_timeout': self.search_config.SEARCH_TIMEOUT_SECONDS,
            'insert_timeout': self.search_config.INSERT_TIMEOUT_SECONDS
        }


# 전역 설정 인스턴스
vector_db_config = VectorDatabaseConfig()


def get_vector_db_config() -> VectorDatabaseConfig:
    """전역 벡터 데이터베이스 설정 반환"""
    return vector_db_config


def update_db_config(db_config: DatabaseConfig):
    """데이터베이스 설정 업데이트"""
    global vector_db_config
    vector_db_config = VectorDatabaseConfig(db_config)


if __name__ == "__main__":
    # 설정 테스트
    config = get_vector_db_config()
    
    print("=== 벡터 데이터베이스 설정 ===")
    print(f"DB 설정: {config.get_db_config()}")
    print(f"HNSW 설정: {config.get_hnsw_config()}")
    print(f"검색 설정: {config.get_search_config()}")
    
    print("\n=== 모델별 설정 ===")
    for model_type in config.table_config.get_all_models():
        print(f"{model_type.value}:")
        print(f"  테이블: {config.get_table_name(model_type)}")
        print(f"  차원: {config.get_dimension(model_type)}")

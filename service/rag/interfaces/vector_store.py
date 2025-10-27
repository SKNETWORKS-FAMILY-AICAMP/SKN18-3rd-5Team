#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벡터 스토어 인터페이스 정의
모든 벡터 스토어 구현체가 준수해야 하는 표준 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterable
import numpy as np
from dataclasses import dataclass

from ..models.config import EmbeddingModelType


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    chunk_id: str
    content: str
    similarity: float
    metadata: Dict[str, Any]
    chunk_db_id: Optional[int] = None
    corp_name: Optional[str] = None
    document_name: Optional[str] = None
    doc_type: Optional[str] = None
    rcept_dt: Optional[str] = None
    report_id: Optional[str] = None
    date: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'chunk_id': self.chunk_id,
            'chunk_db_id': self.chunk_db_id,
            'content': self.content,
            'similarity': self.similarity,
            'metadata': self.metadata,
            'corp_name': self.corp_name,
            'document_name': self.document_name,
            'doc_type': self.doc_type,
            'rcept_dt': self.rcept_dt,
            'report_id': self.report_id,
            'date': self.date,
            'title': self.title,
            'url': self.url,
        }


class VectorStoreInterface(ABC):
    """벡터 스토어 표준 인터페이스"""
    
    @abstractmethod
    def search_similar(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        model_type: EmbeddingModelType,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
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
        pass
    
    @abstractmethod
    def insert_embeddings(
        self,
        chunk_ids: List[Union[int, str]],
        embeddings: Union[np.ndarray, List[List[float]]],
        model_type: EmbeddingModelType
    ) -> int:
        """
        임베딩 삽입
        
        Args:
            chunk_ids: 청크 ID 리스트
            embeddings: 임베딩 벡터 배열
            model_type: 모델 타입
            
        Returns:
            삽입된 개수
        """
        pass
    
    @abstractmethod
    def get_chunk_ids(
        self, 
        limit: Optional[int] = None,
        model_type: Optional[EmbeddingModelType] = None
    ) -> List[Union[int, str]]:
        """
        청크 ID 조회
        
        Args:
            limit: 반환할 최대 개수
            model_type: 특정 모델의 임베딩이 없는 청크만 조회
            
        Returns:
            청크 ID 리스트
        """
        pass

    @abstractmethod
    def count_chunks_to_process(
        self,
        model_type: Optional[EmbeddingModelType],
        skip_existing: bool = True
    ) -> int:
        """처리할 청크 수 조회"""
        pass

    @abstractmethod
    def iter_chunks_to_process(
        self,
        model_type: Optional[EmbeddingModelType],
        skip_existing: bool = True,
        limit: Optional[int] = None,
        fetch_size: int = 1000
    ) -> Iterable[Dict[str, Any]]:
        """처리할 청크를 스트리밍으로 반환"""
        pass
    
    @abstractmethod
    def get_embedding_count(
        self, 
        model_type: EmbeddingModelType
    ) -> int:
        """
        특정 모델의 임베딩 개수 조회
        
        Args:
            model_type: 모델 타입
            
        Returns:
            임베딩 개수
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        데이터베이스 연결 상태 확인
        
        Returns:
            연결 상태
        """
        pass
    
    @abstractmethod
    def close(self):
        """연결 종료"""
        pass
    
    # 선택적 메서드들 (구현체에서 필요에 따라 오버라이드)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        벡터 스토어 통계 조회
        
        Returns:
            통계 정보 딕셔너리
        """
        return {}
    
    def health_check(self) -> Dict[str, Any]:
        """
        헬스 체크
        
        Returns:
            헬스 체크 결과
        """
        return {
            'status': 'healthy' if self.is_connected() else 'unhealthy',
            'connected': self.is_connected()
        }
    
    def batch_insert_embeddings(
        self,
        batch_data: List[Dict[str, Any]],
        model_type: EmbeddingModelType,
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        배치 임베딩 삽입 (기본 구현)
        
        Args:
            batch_data: 배치 데이터 [{'chunk_id': ..., 'embedding': ...}, ...]
            model_type: 모델 타입
            batch_size: 배치 크기
            
        Returns:
            처리 결과 통계
        """
        total_inserted = 0
        total_errors = 0
        
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            chunk_ids = [item['chunk_id'] for item in batch]
            embeddings = [item['embedding'] for item in batch]
            
            try:
                inserted = self.insert_embeddings(chunk_ids, embeddings, model_type)
                total_inserted += inserted
            except Exception as e:
                total_errors += len(batch)
                # 로깅은 구현체에서 처리
        
        return {
            'total_inserted': total_inserted,
            'total_errors': total_errors,
            'total_processed': len(batch_data)
        }


class VectorStoreError(Exception):
    """벡터 스토어 관련 예외"""
    pass


class ConnectionError(VectorStoreError):
    """연결 관련 예외"""
    pass


class InsertError(VectorStoreError):
    """삽입 관련 예외"""
    pass


class SearchError(VectorStoreError):
    """검색 관련 예외"""
    pass


if __name__ == "__main__":
    # 인터페이스 테스트
    print("=== 벡터 스토어 인터페이스 ===")
    print("인터페이스 정의 완료")
    print("구현체: PgVectorStore")
    print("사용법: VectorStoreInterface를 상속받아 구현")

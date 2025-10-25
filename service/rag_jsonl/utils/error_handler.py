#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공통 에러 처리 유틸리티
데이터베이스, 벡터 스토어, 배치 처리 관련 에러를 통합 관리
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
from enum import Enum
import psycopg2
from psycopg2 import OperationalError, DatabaseError, IntegrityError

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """에러 심각도"""
    LOW = "low"           # 경고 수준, 처리 계속 가능
    MEDIUM = "medium"     # 중간 수준, 재시도 필요
    HIGH = "high"         # 높은 수준, 배치 실패
    CRITICAL = "critical" # 치명적, 전체 프로세스 중단


@dataclass
class ErrorContext:
    """에러 컨텍스트 정보"""
    operation: str                    # 수행 중인 작업
    batch_id: Optional[int] = None    # 배치 ID
    chunk_id: Optional[str] = None    # 청크 ID
    model_type: Optional[str] = None  # 모델 타입
    retry_count: int = 0              # 재시도 횟수
    additional_info: Dict[str, Any] = None  # 추가 정보
    
    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}


class DatabaseErrorHandler:
    """데이터베이스 에러 처리"""
    
    # PostgreSQL 에러 코드 매핑
    ERROR_CODE_MAPPING = {
        '08000': ErrorSeverity.MEDIUM,    # 연결 예외
        '08003': ErrorSeverity.HIGH,      # 연결이 존재하지 않음
        '08006': ErrorSeverity.HIGH,      # 연결 실패
        '23505': ErrorSeverity.LOW,       # 고유 제약 위반 (중복)
        '23503': ErrorSeverity.MEDIUM,    # 외래 키 제약 위반
        '42P01': ErrorSeverity.CRITICAL,  # 테이블이 존재하지 않음
        '42703': ErrorSeverity.CRITICAL,  # 컬럼이 존재하지 않음
    }
    
    @classmethod
    def handle_connection_error(
        cls, 
        error: Exception, 
        context: ErrorContext,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> bool:
        """
        연결 에러 처리 및 재시도
        
        Args:
            error: 발생한 에러
            context: 에러 컨텍스트
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간격 (초)
            
        Returns:
            재시도 가능 여부
        """
        severity = cls._get_error_severity(error)
        
        logger.error(f"데이터베이스 연결 에러 [{context.operation}]: {error}")
        logger.error(f"에러 심각도: {severity.value}")
        
        if severity == ErrorSeverity.CRITICAL:
            logger.critical("치명적 에러로 인해 재시도 불가")
            return False
        
        if context.retry_count >= max_retries:
            logger.error(f"최대 재시도 횟수({max_retries}) 초과")
            return False
        
        if severity in [ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]:
            logger.info(f"재시도 {context.retry_count + 1}/{max_retries} - {retry_delay}초 대기")
            time.sleep(retry_delay)
            return True
        
        return False
    
    @classmethod
    def handle_query_error(
        cls, 
        error: Exception, 
        context: ErrorContext
    ) -> ErrorSeverity:
        """
        쿼리 에러 처리
        
        Args:
            error: 발생한 에러
            context: 에러 컨텍스트
            
        Returns:
            에러 심각도
        """
        severity = cls._get_error_severity(error)
        
        logger.error(f"쿼리 에러 [{context.operation}]: {error}")
        
        if isinstance(error, IntegrityError):
            if "duplicate key" in str(error).lower():
                logger.warning(f"중복 키 에러 (무시): {context.chunk_id}")
                return ErrorSeverity.LOW
            else:
                logger.error(f"무결성 제약 위반: {error}")
                return ErrorSeverity.MEDIUM
        
        return severity
    
    @classmethod
    def _get_error_severity(cls, error: Exception) -> ErrorSeverity:
        """에러 심각도 판단"""
        if isinstance(error, OperationalError):
            return ErrorSeverity.HIGH
        elif isinstance(error, DatabaseError):
            # PostgreSQL 에러 코드 확인
            if hasattr(error, 'pgcode'):
                return cls.ERROR_CODE_MAPPING.get(error.pgcode, ErrorSeverity.MEDIUM)
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.MEDIUM


class VectorStoreErrorHandler:
    """벡터 스토어 에러 처리"""
    
    @classmethod
    def handle_insert_error(
        cls, 
        error: Exception, 
        context: ErrorContext,
        embeddings_data: List[Any]
    ) -> Dict[str, Any]:
        """
        임베딩 삽입 에러 처리
        
        Args:
            error: 발생한 에러
            context: 에러 컨텍스트
            embeddings_data: 삽입 시도한 임베딩 데이터
            
        Returns:
            처리 결과 통계
        """
        severity = DatabaseErrorHandler._get_error_severity(error)
        
        logger.error(f"임베딩 삽입 에러 [{context.operation}]: {error}")
        logger.error(f"배치 크기: {len(embeddings_data)}")
        
        if severity == ErrorSeverity.LOW:
            # 중복 키 등의 경미한 에러
            return {
                'success': True,
                'inserted': 0,
                'skipped': len(embeddings_data),
                'errors': 0,
                'message': '중복 데이터로 인해 건너뜀'
            }
        elif severity == ErrorSeverity.MEDIUM:
            # 재시도 가능한 에러
            return {
                'success': False,
                'inserted': 0,
                'skipped': 0,
                'errors': len(embeddings_data),
                'message': '재시도 필요',
                'retry': True
            }
        else:
            # 치명적 에러
            return {
                'success': False,
                'inserted': 0,
                'skipped': 0,
                'errors': len(embeddings_data),
                'message': '치명적 에러',
                'retry': False
            }
    
    @classmethod
    def handle_search_error(
        cls, 
        error: Exception, 
        context: ErrorContext
    ) -> List[Dict[str, Any]]:
        """
        벡터 검색 에러 처리
        
        Args:
            error: 발생한 에러
            context: 에러 컨텍스트
            
        Returns:
            빈 검색 결과
        """
        logger.error(f"벡터 검색 에러 [{context.operation}]: {error}")
        
        return []


class BatchProcessingErrorHandler:
    """배치 처리 에러 처리"""
    
    @classmethod
    def handle_batch_error(
        cls, 
        error: Exception, 
        context: ErrorContext,
        batch_data: List[Any],
        error_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        배치 처리 에러 처리
        
        Args:
            error: 발생한 에러
            context: 에러 컨텍스트
            batch_data: 배치 데이터
            error_callback: 에러 발생 시 호출할 콜백 함수
            
        Returns:
            배치 처리 결과
        """
        logger.error(f"배치 처리 에러 [{context.operation}]: {error}")
        logger.error(f"배치 ID: {context.batch_id}, 크기: {len(batch_data)}")
        
        # 스택 트레이스 로깅
        logger.debug(f"스택 트레이스:\n{traceback.format_exc()}")
        
        # 에러 콜백 호출
        if error_callback:
            try:
                error_callback(error, context, batch_data)
            except Exception as callback_error:
                logger.error(f"에러 콜백 실행 실패: {callback_error}")
        
        return {
            'success': False,
            'processed': 0,
            'errors': len(batch_data),
            'error_message': str(error),
            'batch_id': context.batch_id
        }
    
    @classmethod
    def handle_partial_batch_error(
        cls, 
        successful_items: List[Any],
        failed_items: List[Any],
        context: ErrorContext
    ) -> Dict[str, Any]:
        """
        부분적 배치 실패 처리
        
        Args:
            successful_items: 성공한 아이템들
            failed_items: 실패한 아이템들
            context: 에러 컨텍스트
            
        Returns:
            부분 처리 결과
        """
        total_items = len(successful_items) + len(failed_items)
        success_rate = len(successful_items) / total_items if total_items > 0 else 0
        
        logger.warning(f"부분적 배치 실패 [{context.operation}]")
        logger.warning(f"성공: {len(successful_items)}, 실패: {len(failed_items)}")
        logger.warning(f"성공률: {success_rate:.2%}")
        
        return {
            'success': success_rate > 0.5,  # 50% 이상 성공하면 부분 성공
            'processed': len(successful_items),
            'errors': len(failed_items),
            'success_rate': success_rate,
            'batch_id': context.batch_id
        }


class ErrorHandler:
    """통합 에러 핸들러"""
    
    def __init__(self, 
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 log_level: str = "ERROR"):
        """
        Args:
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간격 (초)
            log_level: 로그 레벨
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.log_level = getattr(logging, log_level.upper())
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)
    
    def handle_database_operation(
        self, 
        operation: Callable,
        context: ErrorContext,
        *args, 
        **kwargs
    ) -> Any:
        """
        데이터베이스 작업 에러 처리 래퍼
        
        Args:
            operation: 실행할 작업
            context: 에러 컨텍스트
            *args, **kwargs: 작업에 전달할 인자들
            
        Returns:
            작업 결과
        """
        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as error:
                context.retry_count = attempt
                
                if attempt < self.max_retries:
                    if DatabaseErrorHandler.handle_connection_error(
                        error, context, self.max_retries, self.retry_delay
                    ):
                        continue
                
                # 최종 실패
                self.logger.error(f"작업 실패 [{context.operation}]: {error}")
                raise
    
    def create_context(
        self, 
        operation: str,
        **kwargs
    ) -> ErrorContext:
        """에러 컨텍스트 생성"""
        return ErrorContext(operation=operation, **kwargs)


if __name__ == "__main__":
    # 에러 핸들러 테스트
    print("=== 에러 핸들러 테스트 ===")
    
    # 테스트용 에러 컨텍스트
    context = ErrorContext(
        operation="test_operation",
        batch_id=1,
        chunk_id="test_chunk_001"
    )
    
    # 에러 핸들러 인스턴스 생성
    handler = ErrorHandler()
    
    print(f"에러 컨텍스트: {context}")
    print(f"최대 재시도: {handler.max_retries}")
    print("에러 핸들러 초기화 완료")

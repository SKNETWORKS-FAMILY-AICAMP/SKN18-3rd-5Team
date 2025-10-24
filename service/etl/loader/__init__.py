"""
ETL (Load) 서비스 모듈

데이터 로드(Load) 단계를 담당하는 모듈들로 구성됩니다.
Transform 단계의 출력물을 최종 저장 포맷(Parquet, VectorDB 등)으로 변환합니다.

모듈 구성:
- jsonl_to_parquet: JSONL을 Parquet 파일로 변환 (컬럼 기반 압축 포맷)
- loader: Parquet 데이터를 pgvector(PostgreSQL)에 로드

사용법:
    from service.etl.loader import ParquetConverter
    
    # JSONL → Parquet 변환
    converter = ParquetConverter(compression='snappy')
    df = converter.jsonl_to_dataframe(jsonl_files)
    converter.save_parquet(df, output_path)
"""

# 주요 클래스 import
from .jsonl_to_parquet import (
    ParquetConverter,
    process_to_single_parquet,
    process_to_partitioned_parquet
)

__all__ = [
    'ParquetConverter',
    'process_to_single_parquet',
    'process_to_partitioned_parquet'
]

# 모듈 버전 정보
__version__ = "1.0.0"
__author__ = "SKN18-3rd-5Team"
__description__ = "ETL Load Service Module for Data Storage"


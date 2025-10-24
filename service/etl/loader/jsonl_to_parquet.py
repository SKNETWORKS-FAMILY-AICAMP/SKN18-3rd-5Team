#!/usr/bin/env python3
"""
JSONL to Parquet Converter - Optimized Version

JSONL 파일들을 효율적으로 Parquet 파일로 변환하는 도구
- 스트리밍 처리로 메모리 효율성 극대화
- 샘플 기반 스키마 생성으로 처리 속도 향상
- Row Group 기반 배치 처리로 안정성 보장
"""

import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime


class ParquetConverter:
    """JSONL을 Parquet으로 변환하는 최적화된 컨버터"""
    
    def __init__(self, compression: str = 'snappy'):
        """
        Args:
            compression: 압축 방식 ('snappy', 'gzip', 'brotli', 'zstd')
        """
        self.compression = compression
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame의 데이터 타입을 최적화하여 메모리 사용량 감소"""
        
        # 문자열 컬럼을 category로 변환 (중복값이 많은 경우)
        category_cols = ['chunk_type', 'doc_type', 'data_category']
        for col in category_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # 정수형 최적화
        if 'token_count' in df.columns:
            df['token_count'] = pd.to_numeric(df['token_count'], downcast='integer')
        
        if 'merged_count' in df.columns:
            df['merged_count'] = pd.to_numeric(df['merged_count'], downcast='integer')
        
        # fiscal_year는 nullable int로
        if 'fiscal_year' in df.columns:
            df['fiscal_year'] = df['fiscal_year'].astype('Int16')
        
        return df
    
    def save_parquet(
        self, 
        df: pd.DataFrame, 
        output_path: Path,
        partition_cols: Optional[List[str]] = None
    ) -> None:
        """DataFrame을 Parquet 파일로 저장
        
        Args:
            df: 저장할 DataFrame
            output_path: 출력 파일 경로
            partition_cols: 파티션 컬럼 (예: ['corp_name', 'doc_type'])
        """
        # 데이터 타입 최적화
        df = self._optimize_dtypes(df)
        
        if partition_cols:
            # 파티션별로 저장
            df.to_parquet(
                output_path,
                partition_cols=partition_cols,
                compression=self.compression,
                engine='pyarrow'
            )
        else:
            # 단일 파일로 저장
            df.to_parquet(
                output_path,
                compression=self.compression,
                engine='pyarrow'
            )
    
    def jsonl_to_dataframe(self, jsonl_files: List[Path]) -> pd.DataFrame:
        """JSONL 파일들을 DataFrame으로 변환 (메모리 집약적)
        
        Args:
            jsonl_files: JSONL 파일 경로 리스트
            
        Returns:
            변환된 DataFrame
        """
        all_data = []
        
        for file_path in jsonl_files:
            print(f"처리 중: {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  {file_path.name} Line {line_no} JSON 파싱 실패: {e}")
                        continue
        
        print(f"총 {len(all_data):,}개 청크 로드 완료")
        return pd.DataFrame(all_data)
    
    def jsonl_to_parquet_streaming(
        self, 
        jsonl_files: List[Path], 
        output_path: Path,
        batch_size: int = 10000
    ) -> None:
        """스트리밍 방식으로 JSONL을 Parquet으로 변환 (메모리 효율적)
        
        Args:
            jsonl_files: JSONL 파일 경로 리스트
            output_path: 출력 Parquet 파일 경로
            batch_size: 배치 크기 (Row Group 크기)
        """
        print(f"🔄 스트리밍 변환 시작 (배치 크기: {batch_size:,})")
        
        # Row Group 기반 접근법 사용
        self._write_parquet_with_row_groups(jsonl_files, output_path, batch_size)
        
        print(f"\n✅ 스트리밍 변환 완료")
    
    def _write_parquet_with_row_groups(
        self, 
        jsonl_files: List[Path], 
        output_path: Path,
        batch_size: int
    ) -> None:
        """Row Group 기반으로 Parquet 파일 작성 (스키마 호환성 해결)"""
        
        # 1단계: 제한된 샘플 데이터를 수집하여 통합 스키마 생성 (메모리 안전 모드)
        print("🔍 통합 스키마 생성 중...")
        all_data = []
        total_chunks = 0
        processed_files = 0
        max_sample_for_schema = 30000  # 스키마 생성을 위한 최대 샘플 수
        
        print(f"📊 총 {len(jsonl_files):,}개 파일에서 최대 {max_sample_for_schema:,}개 샘플로 스키마 생성 (메모리 안전 모드)")
        
        for file_idx, file_path in enumerate(jsonl_files):
            # 샘플 수집 제한
            if total_chunks >= max_sample_for_schema:
                print(f"✅ 샘플 수집 완료: {total_chunks:,}개 (목표: {max_sample_for_schema:,}개)")
                break
                
            processed_files += 1
            
            # 파일 처리 진행 상황 출력
            if processed_files % 100 == 0 or processed_files <= 10:
                print(f"  📁 파일 {processed_files:,}/{len(jsonl_files):,}: {file_path.name} (현재 샘플: {total_chunks:,}개)")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                file_chunks = 0
                for line_no, line in enumerate(f, 1):
                    if total_chunks >= max_sample_for_schema:
                        break
                        
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                        total_chunks += 1
                        file_chunks += 1
                        
                        # 1,000개마다 진행 상황 출력
                        if total_chunks % 1000 == 0:
                            print(f"    📈 샘플 수집 진행: {total_chunks:,}/{max_sample_for_schema:,}개")
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️  {file_path.name} Line {line_no} JSON 파싱 실패: {e}")
                        continue
                
                if file_chunks > 0 and processed_files <= 10:
                    print(f"    ✅ {file_path.name}: {file_chunks:,}개 청크 처리됨")
        
        print(f"✅ 스키마 샘플 수집 완료: {total_chunks:,}개 청크")
        
        # 통합 스키마 생성 (효율적인 샘플 기반)
        if all_data:
            # 이미 제한된 샘플이므로 그대로 사용
            sample_data = all_data
            
            print(f"🔄 스키마 생성용 샘플 DataFrame 생성 중... ({len(sample_data):,}개 샘플)")
            df_sample = pd.DataFrame(sample_data)
            print(f"✅ 샘플 DataFrame 생성 완료: {len(df_sample):,}행 x {len(df_sample.columns)}열")
            
            print("🔄 데이터 타입 최적화 중...")
            df_sample = self._optimize_dtypes(df_sample)
            print("✅ 데이터 타입 최적화 완료")
            
            print("🔄 PyArrow 스키마 생성 중...")
            unified_schema = pa.Schema.from_pandas(df_sample)
            print(f"✅ 통합 스키마 생성 완료 (샘플: {len(sample_data):,}개)")
            
            # 메모리 정리
            del all_data, sample_data, df_sample
            import gc
            gc.collect()
            print("🧹 메모리 정리 완료")
        else:
            print("❌ 스키마 생성 실패: 데이터 없음")
            return
        
        # 2단계: 통합 스키마로 ParquetWriter 생성
        writer = pq.ParquetWriter(
            str(output_path),
            unified_schema,
            compression=self.compression,
            use_dictionary=True,
            write_statistics=True
        )
        
        total_processed = 0
        processed_files = 0
        
        print(f"\n🔄 데이터 변환 시작 (총 {len(jsonl_files):,}개 파일)")
        
        try:
            for file_idx, file_path in enumerate(jsonl_files):
                processed_files += 1
                
                # 파일 처리 진행 상황 출력
                if processed_files % 50 == 0 or processed_files <= 10:
                    print(f"  📁 파일 {processed_files:,}/{len(jsonl_files):,}: {file_path.name}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    batch_data = []
                    file_chunks = 0
                    
                    for line_no, line in enumerate(f, 1):
                        try:
                            data = json.loads(line)
                            batch_data.append(data)
                            file_chunks += 1
                            
                            # 배치 크기에 도달하면 Row Group으로 작성
                            if len(batch_data) >= batch_size:
                                df = pd.DataFrame(batch_data)
                                df = self._optimize_dtypes(df)
                                
                                # 스키마 호환성을 위해 통합 스키마 적용
                                table = pa.Table.from_pandas(df, schema=unified_schema)
                                writer.write_table(table)
                                
                                total_processed += len(batch_data)
                                
                                # 10,000개마다 진행 상황 출력
                                if total_processed % 10000 == 0:
                                    print(f"    📈 전체 진행: {total_processed:,}개 청크 처리됨")
                                
                                batch_data = []
                                
                        except json.JSONDecodeError as e:
                            print(f"⚠️  {file_path.name} Line {line_no} JSON 파싱 실패: {e}")
                            continue
                        except Exception as e:
                            print(f"⚠️  스키마 오류: {e}")
                            # 스키마 오류 시 해당 배치 건너뛰기
                            batch_data = []
                            continue
                    
                    # 파일 끝의 남은 데이터 처리
                    if batch_data:
                        try:
                            df = pd.DataFrame(batch_data)
                            df = self._optimize_dtypes(df)
                            table = pa.Table.from_pandas(df, schema=unified_schema)
                            writer.write_table(table)
                            
                            total_processed += len(batch_data)
                            
                        except Exception as e:
                            print(f"⚠️  마지막 배치 스키마 오류: {e}")
                    
                    # 파일별 처리 결과 출력 (처음 10개 파일만)
                    if processed_files <= 10:
                        print(f"    ✅ {file_path.name}: {file_chunks:,}개 청크 처리됨")
        
        finally:
            # Writer 닫기
            writer.close()
            print(f"✅ 총 {total_processed:,}개 청크 처리 완료")
    
    def get_parquet_stats(self, parquet_path: Path) -> Optional[Dict[str, Any]]:
        """Parquet 파일의 통계 정보 반환
        
        Args:
            parquet_path: Parquet 파일 경로
            
        Returns:
            통계 정보 딕셔너리
        """
        try:
            parquet_file = pq.ParquetFile(parquet_path)
            metadata = parquet_file.metadata
            
            return {
                'num_rows': metadata.num_rows,
                'num_columns': metadata.num_columns,
                'num_row_groups': metadata.num_row_groups,
                'serialized_size': metadata.serialized_size,
                'created_by': metadata.created_by,
                'schema': metadata.schema.to_arrow_schema()
            }
        except Exception as e:
            print(f"⚠️  통계 정보 조회 실패: {e}")
            return None


def convert_jsonl_to_parquet_streaming(
    input_dir: Path, 
    output_file: Path,
    compression: str = 'snappy',
    batch_size: int = 10000
) -> None:
    """
    JSONL 파일들을 스트리밍 방식으로 Parquet으로 변환
    
    Args:
        input_dir: JSONL 파일들이 있는 디렉토리
        output_file: 출력 Parquet 파일 경로
        compression: 압축 방식
        batch_size: 배치 크기
    """
    print("=" * 80)
    print("JSONL to Parquet Converter - Streaming Mode (Memory Efficient)")
    print("=" * 80)
    print(f"📁 입력 디렉토리: {input_dir}")
    print(f"📁 출력 파일: {output_file}")
    print(f"🗜️  압축 방식: {compression}")
    print(f"📦 배치 크기: {batch_size:,} (배치 크기가 클수록 메모리 사용량 증가, 처리 속도 향상)")
    print("=" * 80)
    print()
    
    # JSONL 파일 목록
    jsonl_files = sorted(input_dir.glob("*_chunks.jsonl"))
    
    if not jsonl_files:
        print("❌ JSONL 파일을 찾을 수 없습니다.")
        return
    
    print(f"📄 처리할 파일 수: {len(jsonl_files):,}개")
    print()
    
    # 출력 디렉토리 생성
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 변환 실행
    converter = ParquetConverter(compression=compression)
    
    # 스트리밍 방식으로 변환
    converter.jsonl_to_parquet_streaming(jsonl_files, output_file, batch_size)
    
    # 통계 정보
    stats = converter.get_parquet_stats(output_file)
    if stats:
        print(f"\n📊 Parquet 통계:")
        print(f"   행 수: {stats['num_rows']:,}")
        print(f"   열 수: {stats['num_columns']}")
        print(f"   Row Groups: {stats['num_row_groups']}")
        print(f"   압축 후 크기: {stats['serialized_size'] / (1024**2):.2f} MB")


def process_to_single_parquet(
    input_dir: Path, 
    output_file: Path,
    compression: str = 'snappy'
) -> None:
    """
    전체 JSONL 파일을 하나의 Parquet 파일로 변환
    
    Args:
        input_dir: JSONL 파일들이 있는 디렉토리
        output_file: 출력 Parquet 파일 경로
        compression: 압축 방식
    """
    print("=" * 80)
    print("JSONL to Parquet Converter - Single File Mode")
    print("=" * 80)
    print(f"📁 입력 디렉토리: {input_dir}")
    print(f"📁 출력 파일: {output_file}")
    print(f"🗜️  압축 방식: {compression}")
    print("=" * 80)
    print()
    
    # JSONL 파일 목록
    jsonl_files = sorted(input_dir.glob("*_chunks.jsonl"))
    
    if not jsonl_files:
        print("❌ JSONL 파일을 찾을 수 없습니다.")
        return
    
    print(f"📄 처리할 파일 수: {len(jsonl_files):,}개")
    print()
    
    # 출력 디렉토리 생성
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 변환 실행
    converter = ParquetConverter(compression=compression)
    
    # DataFrame으로 변환 후 저장
    df = converter.jsonl_to_dataframe(jsonl_files)
    converter.save_parquet(df, output_file)
    
    # 통계 정보
    stats = converter.get_parquet_stats(output_file)
    if stats:
        print(f"\n📊 Parquet 통계:")
        print(f"   행 수: {stats['num_rows']:,}")
        print(f"   열 수: {stats['num_columns']}")
        print(f"   Row Groups: {stats['num_row_groups']}")
        print(f"   압축 후 크기: {stats['serialized_size'] / (1024**2):.2f} MB")


def process_with_partitioning(
    input_dir: Path, 
    output_dir: Path,
    partition_col: str,
    compression: str = 'snappy'
) -> None:
    """
    파티셔닝을 적용하여 Parquet 파일들로 변환
    
    Args:
        input_dir: JSONL 파일들이 있는 디렉토리
        output_dir: 출력 디렉토리
        partition_col: 파티션 기준 컬럼
        compression: 압축 방식
    """
    print("=" * 80)
    print(f"JSONL to Parquet Converter - Partitioned Mode ({partition_col})")
    print("=" * 80)
    print(f"📁 입력 디렉토리: {input_dir}")
    print(f"📁 출력 디렉토리: {output_dir}")
    print(f"🗜️  압축 방식: {compression}")
    print(f"📊 파티션 기준: {partition_col}")
    print("=" * 80)
    print()
    
    # JSONL 파일 목록
    jsonl_files = sorted(input_dir.glob("*_chunks.jsonl"))
    
    if not jsonl_files:
        print("❌ JSONL 파일을 찾을 수 없습니다.")
        return
    
    print(f"📄 처리할 파일 수: {len(jsonl_files):,}개")
    print()
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 변환 실행
    converter = ParquetConverter(compression=compression)
    
    # DataFrame으로 변환 후 파티셔닝하여 저장
    df = converter.jsonl_to_dataframe(jsonl_files)
    
    if partition_col not in df.columns:
        print(f"❌ 파티션 컬럼 '{partition_col}'이 데이터에 없습니다.")
        print(f"사용 가능한 컬럼: {list(df.columns)}")
        return
    
    converter.save_parquet(df, output_dir, partition_cols=[partition_col])
    
    print(f"\n✅ 파티셔닝 완료: {output_dir}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="JSONL 파일들을 Parquet으로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 단일 Parquet 파일 생성
  python jsonl_to_parquet.py
  
  # 기업별로 파티셔닝
  python jsonl_to_parquet.py --partition corp_name
  
  # 압축 방식 변경
  python jsonl_to_parquet.py --compression zstd
        """
    )
    
    parser.add_argument(
        '--partition',
        type=str,
        choices=['none', 'corp_name', 'doc_type', 'data_category'],
        default='none',
        help='파티션 기준 (기본값: none)'
    )
    
    parser.add_argument(
        '--compression',
        type=str,
        choices=['snappy', 'gzip', 'brotli', 'zstd'],
        default='snappy',
        help='압축 방식 (기본값: snappy)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50000,
        help='배치 크기 (메모리 사용량 조절, 기본값: 50000)'
    )
    
    parser.add_argument(
        '--streaming',
        action='store_true',
        default=True,
        help='스트리밍 모드 사용 (메모리 효율적, 기본값: True)'
    )
    
    args = parser.parse_args()
    
    # 경로 설정
    script_dir = Path(__file__).parent  # service/etl/loader
    etl_dir = script_dir.parent  # service/etl
    service_dir = etl_dir.parent  # service
    project_root = service_dir.parent  # project root
    data_dir = project_root / "data"
    
    # 입력/출력 경로
    input_dir = data_dir / "transform" / "final"
    parquet_dir = data_dir / "parquet"
    
    # 파티셔닝 처리
    if args.partition == 'none':
        # 단일 파일로 변환
        output_file = parquet_dir / "chunks.parquet"
        
        if args.streaming:
            convert_jsonl_to_parquet_streaming(
                input_dir, 
                output_file, 
                args.compression, 
                args.batch_size
            )
        else:
            process_to_single_parquet(
                input_dir, 
                output_file, 
                args.compression
            )
    else:
        # 파티셔닝하여 변환
        output_dir = parquet_dir / f"chunks_partitioned_{args.partition}"
        process_with_partitioning(
            input_dir, 
            output_dir, 
            args.partition, 
            args.compression
        )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Parquet to PostgreSQL Loader

Parquet 파일의 청크 데이터를 PostgreSQL의 vector_db 스키마로 로드하는 도구
- document_sources: 문서 메타데이터
- document_chunks: 청크 데이터 (natural_text, metadata 등)
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import sql
import json
from datetime import datetime


class ParquetToPostgresLoader:
    """Parquet 파일을 PostgreSQL로 로드하는 클래스"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        user: str = "postgres",
        password: str = "postgres"
    ):
        """
        Args:
            host: PostgreSQL 호스트
            port: PostgreSQL 포트
            database: 데이터베이스 이름
            user: 사용자 이름
            password: 비밀번호
        """
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.conn = None
        self.cursor = None

    def connect(self) -> bool:
        """PostgreSQL 연결"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.cursor = self.conn.cursor()
            print(f"✅ PostgreSQL 연결 성공: {self.connection_params['host']}:{self.connection_params['port']}/{self.connection_params['database']}")
            return True
        except Exception as e:
            print(f"❌ PostgreSQL 연결 실패: {e}")
            return False

    def disconnect(self):
        """PostgreSQL 연결 종료"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("✅ PostgreSQL 연결 종료")

    def load_parquet(self, parquet_path: Path) -> Optional[pd.DataFrame]:
        """Parquet 파일 로드

        Args:
            parquet_path: Parquet 파일 경로

        Returns:
            DataFrame 또는 None
        """
        try:
            print(f"📖 Parquet 파일 로드 중: {parquet_path}")
            df = pd.read_parquet(parquet_path, engine='pyarrow')
            print(f"✅ {len(df):,}개 청크 로드 완료")
            print(f"   컬럼: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"❌ Parquet 파일 로드 실패: {e}")
            return None

    def extract_document_sources(self, df: pd.DataFrame) -> pd.DataFrame:
        """청크 데이터프레임에서 문서 소스 정보 추출

        Args:
            df: 청크 DataFrame

        Returns:
            문서 소스 DataFrame (unique doc_id)
        """
        # metadata 컬럼에서 필요한 정보 추출
        metadata_df = pd.json_normalize(df['metadata'])

        # doc_id 기준으로 그룹화하여 unique 문서 생성
        sources_data = {
            'doc_id': df['doc_id'],
            'corp_name': metadata_df['corp_name'] if 'corp_name' in metadata_df.columns else None,
            'document_name': metadata_df['document_name'] if 'document_name' in metadata_df.columns else None,
            'rcept_dt': metadata_df['rcept_dt'] if 'rcept_dt' in metadata_df.columns else None,
            'doc_type': metadata_df['doc_type'] if 'doc_type' in metadata_df.columns else None,
            'data_category': metadata_df['data_category'] if 'data_category' in metadata_df.columns else None,
            'fiscal_year': metadata_df['fiscal_year'] if 'fiscal_year' in metadata_df.columns else None,
        }

        sources_df = pd.DataFrame(sources_data)

        # doc_id 기준으로 중복 제거 (첫 번째 값 유지)
        sources_df = sources_df.drop_duplicates(subset=['doc_id'], keep='first')

        print(f"📄 문서 소스: {len(sources_df):,}개 unique 문서")

        return sources_df

    def insert_document_sources(self, sources_df: pd.DataFrame, batch_size: int = 1000) -> Dict[str, int]:
        """문서 소스를 vector_db.document_sources 테이블에 삽입

        Args:
            sources_df: 문서 소스 DataFrame
            batch_size: 배치 크기

        Returns:
            doc_id → database id 매핑
        """
        print("\n" + "=" * 80)
        print("📝 문서 소스 삽입 중...")
        print("=" * 80)

        doc_id_map = {}

        try:
            # 기존 doc_id 조회
            self.cursor.execute("SELECT doc_id, id FROM vector_db.document_sources")
            existing_docs = {row[0]: row[1] for row in self.cursor.fetchall()}
            print(f"   기존 문서: {len(existing_docs):,}개")

            # 신규 문서만 필터링
            new_sources = sources_df[~sources_df['doc_id'].isin(existing_docs.keys())]
            print(f"   신규 문서: {len(new_sources):,}개")

            if len(new_sources) == 0:
                print("✅ 모든 문서가 이미 존재합니다")
                return existing_docs

            # 배치 삽입
            insert_query = """
                INSERT INTO vector_db.document_sources
                (doc_id, corp_name, document_name, rcept_dt, doc_type, data_category, fiscal_year, metadata)
                VALUES %s
                ON CONFLICT (doc_id) DO NOTHING
                RETURNING id, doc_id
            """

            total_inserted = 0

            for i in range(0, len(new_sources), batch_size):
                batch = new_sources.iloc[i:i+batch_size]

                # 데이터 준비
                values = [
                    (
                        row['doc_id'],
                        row['corp_name'],
                        row['document_name'],
                        row['rcept_dt'],
                        row['doc_type'],
                        row['data_category'],
                        int(row['fiscal_year']) if pd.notna(row['fiscal_year']) else None,
                        json.dumps({})  # 기본 빈 메타데이터
                    )
                    for _, row in batch.iterrows()
                ]

                # 삽입 및 ID 반환
                execute_values(self.cursor, insert_query, values)
                inserted = self.cursor.fetchall()

                # doc_id → id 매핑 추가
                for db_id, doc_id in inserted:
                    doc_id_map[doc_id] = db_id

                total_inserted += len(inserted)

                if (i + batch_size) % 10000 == 0:
                    print(f"   진행: {i + batch_size:,}/{len(new_sources):,}")

            # 커밋
            self.conn.commit()
            print(f"✅ {total_inserted:,}개 문서 소스 삽입 완료")

            # 기존 + 신규 매핑 병합
            doc_id_map.update(existing_docs)

            return doc_id_map

        except Exception as e:
            self.conn.rollback()
            print(f"❌ 문서 소스 삽입 실패: {e}")
            raise

    def insert_document_chunks(
        self,
        chunks_df: pd.DataFrame,
        batch_size: int = 1000,
        skip_duplicates: bool = True
    ) -> int:
        """문서 청크를 vector_db.document_chunks 테이블에 삽입

        Args:
            chunks_df: 청크 DataFrame
            batch_size: 배치 크기
            skip_duplicates: 중복 스킵 여부

        Returns:
            삽입된 청크 수
        """
        print("\n" + "=" * 80)
        print("📝 문서 청크 삽입 중...")
        print("=" * 80)

        try:
            # 기존 chunk_id 조회 (skip_duplicates가 True인 경우)
            existing_chunks = set()
            if skip_duplicates:
                self.cursor.execute("SELECT chunk_id FROM vector_db.document_chunks")
                existing_chunks = {row[0] for row in self.cursor.fetchall()}
                print(f"   기존 청크: {len(existing_chunks):,}개")

            # 신규 청크만 필터링
            if skip_duplicates:
                new_chunks = chunks_df[~chunks_df['chunk_id'].isin(existing_chunks)]
                print(f"   신규 청크: {len(new_chunks):,}개")
            else:
                new_chunks = chunks_df

            if len(new_chunks) == 0:
                print("✅ 모든 청크가 이미 존재합니다")
                return 0

            # 배치 삽입
            insert_query = """
                INSERT INTO vector_db.document_chunks
                (chunk_id, doc_id, chunk_type, section_path, structured_data, natural_text,
                 corp_name, document_name, rcept_dt, next_context, doc_type, data_category,
                 fiscal_year, keywords, token_count, metadata)
                VALUES %s
                ON CONFLICT (chunk_id) DO NOTHING
            """

            total_inserted = 0

            for i in range(0, len(new_chunks), batch_size):
                batch = new_chunks.iloc[i:i+batch_size]

                # 데이터 준비
                values = []
                for _, row in batch.iterrows():
                    # metadata에서 필요한 정보 추출
                    metadata = row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata'])

                    values.append((
                        row['chunk_id'],
                        row['doc_id'],
                        row['chunk_type'],
                        row.get('section_path', ''),
                        json.dumps(row.get('structured_data', {})),
                        row['natural_text'],
                        metadata.get('corp_name'),
                        metadata.get('document_name'),
                        metadata.get('rcept_dt'),
                        metadata.get('next_context'),
                        metadata.get('doc_type'),
                        metadata.get('data_category'),
                        int(metadata['fiscal_year']) if metadata.get('fiscal_year') and pd.notna(metadata.get('fiscal_year')) else None,
                        metadata.get('keywords', []),
                        int(metadata['token_count']) if metadata.get('token_count') else None,
                        json.dumps(metadata)
                    ))

                # 삽입
                execute_values(self.cursor, insert_query, values)
                total_inserted += len(values)

                if (i + batch_size) % 10000 == 0 or i == 0:
                    print(f"   진행: {i + batch_size:,}/{len(new_chunks):,} 청크 삽입됨")

            # 커밋
            self.conn.commit()
            print(f"✅ {total_inserted:,}개 청크 삽입 완료")

            return total_inserted

        except Exception as e:
            self.conn.rollback()
            print(f"❌ 청크 삽입 실패: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_statistics(self):
        """데이터베이스 통계 조회"""
        print("\n" + "=" * 80)
        print("📊 데이터베이스 통계")
        print("=" * 80)

        try:
            # 문서 소스 수
            self.cursor.execute("SELECT COUNT(*) FROM vector_db.document_sources")
            doc_count = self.cursor.fetchone()[0]
            print(f"   문서 소스: {doc_count:,}개")

            # 청크 수
            self.cursor.execute("SELECT COUNT(*) FROM vector_db.document_chunks")
            chunk_count = self.cursor.fetchone()[0]
            print(f"   청크: {chunk_count:,}개")

            # 기업별 통계
            self.cursor.execute("""
                SELECT corp_name, COUNT(*) as chunk_count
                FROM vector_db.document_chunks
                GROUP BY corp_name
                ORDER BY chunk_count DESC
                LIMIT 10
            """)
            print("\n   기업별 청크 수 (TOP 10):")
            for corp_name, count in self.cursor.fetchall():
                print(f"      {corp_name}: {count:,}개")

            # 문서 유형별 통계
            self.cursor.execute("""
                SELECT doc_type, COUNT(*) as chunk_count
                FROM vector_db.document_chunks
                GROUP BY doc_type
                ORDER BY chunk_count DESC
            """)
            print("\n   문서 유형별 청크 수:")
            for doc_type, count in self.cursor.fetchall():
                print(f"      {doc_type}: {count:,}개")

        except Exception as e:
            print(f"❌ 통계 조회 실패: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Parquet 파일을 PostgreSQL vector_db 스키마로 로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 로드
  python parquet_to_postgres.py

  # 커스텀 데이터베이스
  python parquet_to_postgres.py --host localhost --port 5432 --database mydb --user myuser --password mypass

  # 배치 크기 조정
  python parquet_to_postgres.py --batch-size 5000
        """
    )

    parser.add_argument('--host', type=str, default='localhost', help='PostgreSQL 호스트')
    parser.add_argument('--port', type=int, default=5432, help='PostgreSQL 포트')
    parser.add_argument('--database', type=str, default='postgres', help='데이터베이스 이름')
    parser.add_argument('--user', type=str, default='postgres', help='사용자 이름')
    parser.add_argument('--password', type=str, default='postgres', help='비밀번호')
    parser.add_argument('--batch-size', type=int, default=1000, help='배치 크기')
    parser.add_argument('--parquet-file', type=str, help='Parquet 파일 경로 (기본값: data/parquet/chunks.parquet)')

    args = parser.parse_args()

    # 경로 설정
    if args.parquet_file:
        parquet_path = Path(args.parquet_file)
    else:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent.parent
        parquet_path = project_root / "data" / "parquet" / "chunks.parquet"

    if not parquet_path.exists():
        print(f"❌ Parquet 파일을 찾을 수 없습니다: {parquet_path}")
        print(f"   먼저 jsonl_to_parquet.py를 실행하여 Parquet 파일을 생성하세요")
        sys.exit(1)

    # 로더 생성
    loader = ParquetToPostgresLoader(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password
    )

    try:
        # 연결
        if not loader.connect():
            sys.exit(1)

        # Parquet 파일 로드
        df = loader.load_parquet(parquet_path)
        if df is None:
            sys.exit(1)

        # 문서 소스 추출 및 삽입
        sources_df = loader.extract_document_sources(df)
        doc_id_map = loader.insert_document_sources(sources_df, batch_size=args.batch_size)

        # 청크 삽입
        inserted_count = loader.insert_document_chunks(df, batch_size=args.batch_size)

        # 통계 출력
        loader.get_statistics()

        print("\n" + "=" * 80)
        print("✅ 로딩 완료!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        loader.disconnect()


if __name__ == "__main__":
    main()

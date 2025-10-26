#!/usr/bin/env python3
"""
JSONL to PostgreSQL Loader - JSONL 파일을 직접 PostgreSQL에 로드

JSONL 파일들을 PostgreSQL 데이터베이스에 직접 로드하는 도구
- JSONL 파일 직접 읽기
- 임베딩 생성 및 저장
- pgvector를 통한 벡터 검색 지원
"""

import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import os
from datetime import datetime

# 로깅 설정 - 간단하게 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


class JSONLToPostgresLoader:
    """JSONL 파일을 PostgreSQL에 로드하는 클래스"""
    
    def __init__(
        self, 
        db_config: Dict[str, str],
        batch_size: int = 1000
    ):
        """
        Args:
            db_config: PostgreSQL 연결 설정
            batch_size: 배치 크기
        """
        self.db_config = db_config
        self.batch_size = batch_size
        self.conn = None
        
    def _connect_db(self):
        """데이터베이스 연결"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = True
            logger.info("PostgreSQL 연결 성공")
        except Exception as e:
            logger.error(f"데이터베이스 연결 실패: {e}")
            raise
    
    
    def _create_tables(self):
        """필요한 테이블 생성"""
        cursor = self.conn.cursor()
        
        # pgvector 확장 활성화
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # 청크 테이블 생성
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                chunk_id VARCHAR(255) UNIQUE NOT NULL,
                doc_id VARCHAR(255),
                chunk_type VARCHAR(50),
                section_path TEXT,
                natural_text TEXT,
                structured_data JSONB,
                metadata JSONB,
                token_count INTEGER,
                merged_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        
        # 인덱스 생성
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_corp_name ON chunks ((metadata->>'corp_name'));")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON chunks ((metadata->>'doc_type'));")
        
        cursor.close()
        logger.info("테이블 및 인덱스 생성 완료")
    
    # 파일별 진행 상황 추적, 진행률 표시
    def _read_jsonl_files(self, jsonl_dir: Path) -> Generator[Tuple[Dict[str, Any], Path, int, int], None, None]:
        """JSONL 파일들을 순차적으로 읽기"""
        jsonl_files = sorted(jsonl_dir.glob("*_chunks.jsonl"))
        total_files = len(jsonl_files)
        print(f"📂 처리할 JSONL 파일 수: {total_files}개")
        logger.info(f"📂 처리할 JSONL 파일 수: {total_files}개")
        import sys
        sys.stdout.flush()
        
        for file_idx, file_path in enumerate(jsonl_files, 1):
            print(f"({file_idx}/{total_files}): {file_path.name}")
            sys.stdout.flush()
            chunk_count = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        chunk_count += 1
                        
                        
                        yield data, file_path, file_idx, total_files
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ JSON 파싱 실패 {file_path.name}:{line_no}: {e}")
                        continue
    
    def _insert_chunks_batch(self, chunks_batch: List[Dict[str, Any]]):
        """청크 배치를 데이터베이스에 삽입"""
        cursor = self.conn.cursor()
        
        # 중복 제거: chunk_id 기준으로 중복 제거
        seen_chunk_ids = set()
        unique_chunks = []
        for chunk in chunks_batch:
            chunk_id = chunk.get('chunk_id')
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                unique_chunks.append(chunk)
        
        logger.info(f"중복 제거: {len(chunks_batch)} → {len(unique_chunks)} 청크")
        
        # 청크 데이터 준비
        chunk_data = []
        for chunk in unique_chunks:
            chunk_data.append((
                chunk.get('chunk_id'),
                chunk.get('doc_id'),
                chunk.get('chunk_type'),
                chunk.get('section_path'),
                chunk.get('natural_text'),
                json.dumps(chunk.get('structured_data', {})),
                json.dumps(chunk.get('metadata', {})),
                chunk.get('token_count'),
                chunk.get('merged_count')
            ))
        
        # 청크 삽입 (중복 시 무시) - 개별 INSERT로 처리
        insert_query = """
            INSERT INTO chunks (chunk_id, doc_id, chunk_type, section_path, natural_text, 
                              structured_data, metadata, token_count, merged_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO NOTHING
        """
        
        for chunk_row in chunk_data:
            try:
                cursor.execute(insert_query, chunk_row)
            except Exception as e:
                logger.warning(f"청크 삽입 실패 (무시): {e}")
                continue
        cursor.close()
    
    
    
    def load_jsonl_to_postgres(self, jsonl_dir: Path):
        """JSONL 파일들을 PostgreSQL에 로드"""
        logger.info("JSONL to PostgreSQL 로딩 시작 (청크 데이터만)")
        logger.info(f"📁 대상 디렉토리: {jsonl_dir}")
        logger.info(f"🔧 배치 크기: {self.batch_size}")
        
        # 초기화
        print("🔌 데이터베이스 연결 중...")
        logger.info("🔌 데이터베이스 연결 중...")
        self._connect_db()
        print("📋 테이블 생성 중...")
        logger.info("📋 테이블 생성 중...")
        self._create_tables()
        
        # 통계 변수
        total_chunks = 0
        start_time = time.time()
        
        # 배치 처리
        chunks_batch = []
        
        try:
            print("📖 JSONL 파일 읽기 시작...")
            logger.info("📖 JSONL 파일 읽기 시작...")
            import sys
            sys.stdout.flush()
            last_file_idx = 0
            total_files_count = 0
            # tqdm 대신 직접 진행 상황 표시
            for chunk, file_path, file_idx, total_files in self._read_jsonl_files(jsonl_dir):
                chunks_batch.append(chunk)
                total_chunks += 1
                last_file_idx = file_idx
                total_files_count = total_files
                
                # 1,000개마다 전체 진행 상황 로그
                if total_chunks % 5000 == 0:
                    elapsed_time = time.time() - start_time
                    rate = total_chunks / elapsed_time if elapsed_time > 0 else 0
                    print(f"    전체 진행: {total_chunks:,}개 청크 처리됨 (속도: {rate:.1f} 청크/초)")
                    import sys
                    sys.stdout.flush()
                
                # 배치 크기에 도달하면 처리
                if len(chunks_batch) >= self.batch_size:
                    try:
                        self._insert_chunks_batch(chunks_batch)
                        
                        # 배치 커밋
                        self.conn.commit()
                        
                        # 파일 처리 진행률 계산
                        current_file_idx = last_file_idx
                        file_progress = (current_file_idx / total_files_count) * 100 if total_files_count else 0
                        
                        # 전체 청크 수 추정 (파일당 평균 1000개 청크로 추정)
                        estimated_total_chunks = total_files_count * 1000
                        chunk_progress = (total_chunks / estimated_total_chunks) * 100 if estimated_total_chunks else 0
                        
                        print(f"📊 파일 진행: {current_file_idx}/{total_files_count} ({file_progress:.1f}%) | 청크: ({total_chunks:,}/{estimated_total_chunks:,}) {chunk_progress:.1f}%")
                        import sys
                        sys.stdout.flush()
                        
                    except Exception as batch_error:
                        self.conn.rollback()
                        logger.error(f"❌ 배치 처리 실패 (롤백): {batch_error}")
                        raise
                    
                    chunks_batch = []
            
            # 남은 배치 처리
            if chunks_batch:
                batch_start_time = time.time()
                try:
                    logger.info(f"🔄 마지막 배치 처리 시작: {len(chunks_batch)}개 청크")
                    
                    logger.info("💾 청크 데이터베이스 삽입 중...")
                    self._insert_chunks_batch(chunks_batch)
                    
                    # 마지막 배치 커밋
                    self.conn.commit()
                    
                except Exception as batch_error:
                    self.conn.rollback()
                    logger.error(f"❌ 마지막 배치 처리 실패 (롤백): {batch_error}")
                    raise
            
            total_time = time.time() - start_time
            print(f"    로딩 완료: 총 {total_chunks:,}개 청크 처리 (총 소요시간: {total_time/60:.1f}분)")
            logger.info(f"   로딩 완료: 총 {total_chunks:,}개 청크 처리 (총 소요시간: {total_time/60:.1f}분)")
            import sys
            sys.stdout.flush()
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"로딩 중 오류 발생: {e}")
            raise
        finally:
            if self.conn:
                self.conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """로딩 통계 조회"""
        if not self.conn:
            self._connect_db()
        
        cursor = self.conn.cursor()
        
        # 청크 수 조회
        cursor.execute("SELECT COUNT(*) FROM chunks;")
        chunk_count = cursor.fetchone()[0]
        
        # 모델별 임베딩 수 조회
        model_name = self.embedding_model.replace("/", "_").replace("-", "_")
        cursor.execute(f"SELECT COUNT(*) FROM embeddings_{model_name};")
        embedding_count = cursor.fetchone()[0]
        
        # 기업별 통계
        cursor.execute("""
            SELECT metadata->>'corp_name' as corp_name, COUNT(*) as count
            FROM chunks 
            WHERE metadata->>'corp_name' IS NOT NULL
            GROUP BY metadata->>'corp_name'
            ORDER BY count DESC
            LIMIT 10;
        """)
        corp_stats = cursor.fetchall()
        
        cursor.close()
        
        return {
            'total_chunks': chunk_count,
            'total_embeddings': embedding_count,
            'model_name': self.embedding_model,
            'top_corporations': corp_stats
        }


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="JSONL 파일들을 PostgreSQL에 로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 사용
  python jsonl_to_postgres.py --jsonl-dir data/transform/final
  
  # 배치 크기 조정
  python jsonl_to_postgres.py --jsonl-dir data/transform/final --batch-size 500
  
  # 청크 데이터만 로드 (임베딩은 별도로 생성)
  python jsonl_to_postgres.py --jsonl-dir data/transform/final
        """
    )
    
    parser.add_argument(
        '--jsonl-dir',
        type=Path,
        required=True,
        help='JSONL 파일들이 있는 디렉토리'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='배치 크기 (기본값: 1000)'
    )
    
    
    parser.add_argument(
        '--db-host',
        type=str,
        default=os.getenv('POSTGRES_HOST', 'localhost'),
        help='PostgreSQL 호스트'
    )
    
    parser.add_argument(
        '--db-port',
        type=str,
        default=os.getenv('POSTGRES_PORT', '5432'),
        help='PostgreSQL 포트'
    )
    
    parser.add_argument(
        '--db-name',
        type=str,
        default=os.getenv('POSTGRES_DB', 'skn_project'),
        help='데이터베이스 이름'
    )
    
    parser.add_argument(
        '--db-user',
        type=str,
        default=os.getenv('POSTGRES_USER', 'postgres'),
        help='데이터베이스 사용자'
    )
    
    parser.add_argument(
        '--db-password',
        type=str,
        default=os.getenv('POSTGRES_PASSWORD', 'post1234'),
        help='데이터베이스 비밀번호'
    )
    
    args = parser.parse_args()
    
    # 데이터베이스 설정
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    # 로더 생성 및 실행
    loader = JSONLToPostgresLoader(
        db_config=db_config,
        batch_size=args.batch_size
    )
    
    try:
        loader.load_jsonl_to_postgres(args.jsonl_dir)
        
        # 통계 출력
        stats = loader.get_stats()
        print(f"\n📊 로딩 통계:")
        print(f"   총 청크 수: {stats['total_chunks']:,}")
        print(f"   총 임베딩 수: {stats['total_embeddings']:,}")
        print(f"   사용 모델: {stats['model_name']}")
        print(f"   상위 기업들:")
        for corp_name, count in stats['top_corporations']:
            print(f"     - {corp_name}: {count:,}개")
            
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        return 1
    
    return 0


# CLI 유틸리티 함수들
def load_jsonl_files(jsonl_dir: str, batch_size: int = 1000):
    """JSONL 파일 로딩 (CLI용) - 직접 호출"""
    from pathlib import Path
    
    print(f"📁 JSONL 파일 로딩: {jsonl_dir}")
    logger.info(f"📁 JSONL 파일 로딩: {jsonl_dir}")
    
    try:
        # 기본 DB 설정
        db_config = {
            'host': 'localhost',
            'port': '5432',
            'database': 'skn_project',
            'user': 'postgres',
            'password': 'post1234'
        }
        
        # 직접 클래스 인스턴스 생성하여 로딩
        loader = JSONLToPostgresLoader(db_config=db_config, batch_size=batch_size)
        loader.load_jsonl_to_postgres(Path(jsonl_dir))
        print("✅ JSONL 파일 로딩 완료")
        logger.info("✅ JSONL 파일 로딩 완료")
        return True
    except Exception as e:
        print(f"❌ JSONL 파일 로딩 실패: {e}")
        logger.error(f"❌ JSONL 파일 로딩 실패: {e}")
        return False


def get_loading_stats():
    """로딩 통계 조회 (CLI용)"""
    import subprocess
    
    logger.info("📊 로딩 통계 조회")
    
    try:
        # 청크 수 조회
        result = subprocess.run(
            ["docker", "exec", "SKN18-3rd", "psql", "-U", "postgres", "-d", "skn_project", "-c", 
             "SELECT COUNT(*) as chunk_count FROM chunks;"],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("📋 청크 통계:")
        logger.info(result.stdout)
        
        # 임베딩 테이블별 통계
        embedding_tables = ["embeddings_multilingual_e5_small", "embeddings_kakaobank", "embeddings_fine5"]
        
        for table in embedding_tables:
            try:
                result = subprocess.run(
                    ["docker", "exec", "SKN18-3rd", "psql", "-U", "postgres", "-d", "skn_project", "-c", 
                     f"SELECT COUNT(*) as {table}_count FROM {table};"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"📋 {table} 통계:")
                logger.info(result.stdout)
            except subprocess.CalledProcessError:
                logger.info(f"📋 {table}: 테이블이 존재하지 않거나 비어있음")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 통계 조회 실패: {e.stderr}")
        return False


def clear_data():
    """데이터 삭제 (CLI용)"""
    import subprocess
    
    logger.warning("⚠️  데이터 삭제 (모든 데이터가 삭제됩니다)")
    
    try:
        # 임베딩 테이블 삭제
        embedding_tables = ["embeddings_multilingual_e5_small", "embeddings_kakaobank", "embeddings_fine5"]
        
        for table in embedding_tables:
            result = subprocess.run(
                ["docker", "exec", "SKN18-3rd", "psql", "-U", "postgres", "-d", "skn_project", "-c", 
                 f"TRUNCATE TABLE {table} CASCADE;"],
                capture_output=True,
                text=True,
                check=True
            )
        
        # 청크 테이블 삭제
        result = subprocess.run(
            ["docker", "exec", "SKN18-3rd", "psql", "-U", "postgres", "-d", "skn_project", "-c", 
             "TRUNCATE TABLE chunks CASCADE;"],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("✅ 데이터 삭제 완료")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 데이터 삭제 실패: {e.stderr}")
        return False


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
RAG JSONL CLI - JSONL 기반 RAG 시스템 명령줄 인터페이스

JSONL 파일을 직접 사용하는 RAG 시스템의 CLI 도구입니다.
Parquet 변환 없이 JSONL 파일에서 직접 임베딩을 생성하고 검색합니다.
"""

import argparse
import sys
import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGJSONLSystem:
    """JSONL 기반 RAG 시스템"""
    
    def __init__(
        self,
        db_config: Dict[str, str],
        embedding_model: str = "intfloat/multilingual-e5-small"
    ):
        """
        Args:
            db_config: PostgreSQL 연결 설정
            embedding_model: 임베딩 모델명
        """
        self.db_config = db_config
        self.embedding_model = embedding_model
        self.encoder = None
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
    
    def _load_embedding_model(self):
        """임베딩 모델 로드"""
        try:
            self.encoder = SentenceTransformer(self.embedding_model)
            logger.info(f"임베딩 모델 로드 완료: {self.embedding_model}")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            raise
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        min_similarity: float = 0.0,
        corp_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """문서 검색"""
        if not self.conn:
            self._connect_db()
        if not self.encoder:
            self._load_embedding_model()
        
        # 쿼리 임베딩 생성
        query_embedding = self.encoder.encode([query])[0]
        
        # 모델명으로 테이블명 생성 (실제 테이블명에 맞게)
        if "multilingual-e5-small" in self.embedding_model:
            model_name = "multilingual_e5_small"
        elif "kakaobank" in self.embedding_model:
            model_name = "kakaobank"
        elif "fine5" in self.embedding_model:
            model_name = "fine5"
        else:
            model_name = self.embedding_model.replace("/", "_").replace("-", "_")
        
        # 벡터를 문자열로 변환
        embedding_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
        
        # 검색 쿼리 구성
        base_query = f"""
            SELECT 
                c.chunk_id,
                c.doc_id,
                c.chunk_type,
                c.natural_text,
                c.metadata,
                c.token_count,
                1 - (e.embedding <=> %s::vector) as similarity
            FROM chunks c
            JOIN embeddings_{model_name} e ON c.chunk_id = e.chunk_id
        """
        
        where_conditions = ["1 - (e.embedding <=> %s::vector) >= %s"]
        params = [embedding_str, embedding_str, min_similarity]
        
        if corp_filter:
            where_conditions.append("c.metadata->>'corp_name' = %s")
            params.append(corp_filter)
        
        query_sql = f"""
            {base_query}
            WHERE {' AND '.join(where_conditions)}
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
        """
        params.extend([embedding_str, top_k])
        
        
        # 검색 실행
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query_sql, params)
        results = cursor.fetchall()
        cursor.close()
        
        # 결과 변환
        search_results = []
        for row in results:
            search_results.append({
                'chunk_id': row['chunk_id'],
                'doc_id': row['doc_id'],
                'chunk_type': row['chunk_type'],
                'natural_text': row['natural_text'],
                'metadata': row['metadata'],
                'token_count': row['token_count'],
                'similarity': float(row['similarity'])
            })
        
        return search_results
    
    def get_stats(self) -> Dict[str, Any]:
        """시스템 통계 조회"""
        if not self.conn:
            self._connect_db()
        
        cursor = self.conn.cursor()
        
        # 청크 수 조회
        cursor.execute("SELECT COUNT(*) FROM chunks;")
        chunk_count = cursor.fetchone()[0]
        
        # 모델별 임베딩 수 조회 (실제 테이블명에 맞게)
        if "multilingual-e5-small" in self.embedding_model:
            model_name = "multilingual_e5_small"
        elif "kakaobank" in self.embedding_model:
            model_name = "kakaobank"
        elif "fine5" in self.embedding_model:
            model_name = "fine5"
        else:
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


def get_db_config() -> Dict[str, str]:
    """데이터베이스 설정 가져오기"""
    return {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'skn_project'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'post1234')
    }


def test_connection() -> bool:
    """데이터베이스 연결 테스트"""
    try:
        db_config = get_db_config()
        conn = psycopg2.connect(**db_config)
        conn.close()
        return True
    except Exception as e:
        logger.error(f"DB 연결 테스트 실패: {e}")
        return False


def run_search_command(args, db_config: Dict[str, str]):
    """검색 명령어 실행"""
    logger.info(f"검색 시작: '{args.query}'")
    
    rag_system = RAGJSONLSystem(
        db_config=db_config,
        embedding_model=args.model
    )
    
    try:
        results = rag_system.search(
            query=args.query,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
            corp_filter=args.corp_filter
        )
        
        print(f"\n🔍 검색 결과: '{args.query}'")
        print(f"📊 총 {len(results)}개 결과")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 청크 ID: {result['chunk_id']}")
            print(f"   기업: {result['metadata'].get('corp_name', 'N/A')}")
            print(f"   유사도: {result['similarity']:.4f}")
            print(f"   텍스트: {result['natural_text'][:200]}...")
            print(f"   토큰 수: {result['token_count']}")
        
        # 결과 저장
        if args.save_results:
            output_file = f"search_results_{args.query.replace(' ', '_')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"검색 결과 저장: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"검색 실패: {e}")
        return False


def run_stats_command(args, db_config: Dict[str, str]):
    """통계 명령어 실행"""
    logger.info("시스템 통계 조회")
    
    rag_system = RAGJSONLSystem(
        db_config=db_config,
        embedding_model=args.model
    )
    
    try:
        stats = rag_system.get_stats()
        
        print("\n📊 RAG 시스템 통계")
        print("=" * 50)
        print(f"총 청크 수: {stats['total_chunks']:,}")
        print(f"총 임베딩 수: {stats['total_embeddings']:,}")
        print(f"사용 모델: {stats['model_name']}")
        print(f"\n상위 기업들:")
        for corp_name, count in stats['top_corporations']:
            print(f"  - {corp_name}: {count:,}개")
        
        return True
        
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="JSONL 기반 RAG 시스템 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 검색
  python rag_cli.py search --query "삼성전자 매출" --top-k 5
  
  # 기업 필터링 검색
  python rag_cli.py search --query "매출 증가" --corp-filter "삼성전자"
  
  # 통계 조회
  python rag_cli.py stats
  
  # 다른 모델 사용
  python rag_cli.py search --query "AI 기술" --model sentence-transformers/all-MiniLM-L6-v2
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 로그 출력'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')
    
    # 검색 명령어
    search_parser = subparsers.add_parser('search', help='문서 검색')
    search_parser.add_argument('--query', required=True, help='검색 쿼리')
    search_parser.add_argument('--top-k', type=int, default=5, help='반환할 결과 수')
    search_parser.add_argument('--min-similarity', type=float, default=0.0, help='최소 유사도')
    search_parser.add_argument('--corp-filter', help='기업명 필터')
    search_parser.add_argument('--model', default="intfloat/multilingual-e5-small", help='임베딩 모델')
    search_parser.add_argument('--save-results', action='store_true', help='검색 결과 저장')
    
    # 통계 명령어
    stats_parser = subparsers.add_parser('stats', help='시스템 통계 조회')
    stats_parser.add_argument('--model', default="intfloat/multilingual-e5-small", help='임베딩 모델')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # DB 연결 테스트
    logger.info("DB 연결 테스트 중...")
    if not test_connection():
        logger.error("DB 연결 실패")
        sys.exit(1)
    logger.info("DB 연결 성공")
    
    db_config = get_db_config()
    success = False
    
    # 명령어 실행
    if args.command == 'search':
        success = run_search_command(args, db_config)
    elif args.command == 'stats':
        success = run_stats_command(args, db_config)
    
    if success:
        logger.info("명령어 실행 완료")
        sys.exit(0)
    else:
        logger.error("명령어 실행 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()

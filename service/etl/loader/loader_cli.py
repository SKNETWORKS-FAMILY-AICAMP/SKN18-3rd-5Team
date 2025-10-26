#!/usr/bin/env python3
"""
JSONL Loader CLI - JSONL 파일 로딩 및 임베딩 생성 통합 CLI

전체 파이프라인을 단계별로 실행할 수 있는 명령줄 인터페이스입니다.
"""

import argparse
import sys
import logging
import time
from pathlib import Path

# 로컬 모듈 import
from system_manager import (
    check_docker_compose, start_docker_compose, stop_docker_compose,
    create_schema, drop_schema, check_schema, system_health_check, reset_system
)
from jsonl_to_postgres import load_jsonl_files, get_loading_stats, clear_data

# RAG 시스템 모듈 import
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from service.rag.vectorstore.pgvector_store import PgVectorStore
from service.rag.models.loader import ModelFactory, EmbeddingModelType
from config.vector_database import get_vector_db_config

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 모든 함수는 이제 별도 모듈에서 import됨

# ============================================================================
# RAG 시스템 관리 함수들
# ============================================================================

def test_database_connection():
    """데이터베이스 연결 테스트"""
    logger.info("🔍 데이터베이스 연결 테스트 중...")
    try:
        vector_store = PgVectorStore()
        if vector_store.is_connected():
            logger.info("✅ 데이터베이스 연결 성공!")
            
            # 테이블 정보 조회
            with vector_store.conn.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name, 
                           (SELECT COUNT(*) FROM information_schema.columns 
                            WHERE table_name = t.table_name) as column_count
                    FROM information_schema.tables t
                    WHERE table_schema = 'public' 
                    AND table_name LIKE '%embedding%' OR table_name = 'chunks'
                    ORDER BY table_name
                """)
                
                tables = cursor.fetchall()
                logger.info("📊 관련 테이블 현황:")
                for table_name, col_count in tables:
                    logger.info(f"  • {table_name}: {col_count}개 컬럼")
            
            return True
        else:
            logger.error("❌ 데이터베이스 연결 실패!")
            return False
    except Exception as e:
        logger.error(f"❌ 데이터베이스 연결 테스트 실패: {e}")
        return False

def truncate_documents():
    """문서 테이블 데이터 삭제 (chunks 테이블)"""
    logger.info("🗑️ 문서 테이블 데이터 삭제 중...")
    try:
        vector_store = PgVectorStore()
        with vector_store.conn.cursor() as cursor:
            # chunks 테이블 데이터 삭제 (CASCADE로 관련 임베딩 테이블도 함께 삭제)
            cursor.execute("TRUNCATE TABLE chunks RESTART IDENTITY CASCADE")
            vector_store.conn.commit()
        
        logger.info("✅ 문서 테이블 데이터 삭제 완료!")
        return True
    except Exception as e:
        logger.error(f"❌ 문서 테이블 데이터 삭제 실패: {e}")
        return False

def truncate_vectors():
    """벡터 테이블 데이터 삭제 (임베딩 테이블들)"""
    logger.info("🗑️ 벡터 테이블 데이터 삭제 중...")
    try:
        vector_store = PgVectorStore()
        config = get_vector_db_config()
        
        with vector_store.conn.cursor() as cursor:
            # 모든 임베딩 테이블 데이터 삭제
            for model_type in EmbeddingModelType:
                table_name = config.get_table_name(model_type)
                cursor.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE")
                logger.info(f"  • {table_name} 테이블 데이터 삭제")
            
            vector_store.conn.commit()
        
        logger.info("✅ 벡터 테이블 데이터 삭제 완료!")
        return True
    except Exception as e:
        logger.error(f"❌ 벡터 테이블 데이터 삭제 실패: {e}")
        return False

def load_documents(jsonl_dir: Path, batch_size: int = 1000):
    """문서 로드 (JSONL 파일을 chunks 테이블에 로드)"""
    logger.info(f"📄 문서 로드 시작: {jsonl_dir}")
    try:
        success = load_jsonl_files(jsonl_dir, batch_size)
        if success:
            logger.info("✅ 문서 로드 완료!")
        else:
            logger.error("❌ 문서 로드 실패!")
        return success
    except Exception as e:
        logger.error(f"❌ 문서 로드 실패: {e}")
        return False

def download_models(model: str = "all"):
    """임베딩 모델 다운로드"""
    logger.info(f"📥 모델 다운로드 시작: {model}")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        models_to_download = []
        if model == "all":
            models_to_download = ["e5", "kakaobank", "fine5"]
        else:
            models_to_download = [model]
        
        for model_name in models_to_download:
            if model_name == "e5":
                logger.info("📥 E5 모델 다운로드 중...")
                SentenceTransformer('intfloat/multilingual-e5-small')
                logger.info("✅ E5 모델 다운로드 완료")
            elif model_name == "kakaobank":
                logger.info("📥 KakaoBank 모델 다운로드 중...")
                SentenceTransformer('kakaobank/kf-deberta-base')
                logger.info("✅ KakaoBank 모델 다운로드 완료")
            elif model_name == "fine5":
                logger.info("📥 FinE5 모델 설정 중...")
                logger.info("ℹ️ FinE5는 API 전용 모델입니다. AbaciNLP API 키가 필요합니다.")
                
                # API 키 확인
                import os
                from dotenv import load_dotenv
                load_dotenv()
                
                api_key = os.getenv('FIN_E5_API_KEY')
                if not api_key:
                    logger.warning("⚠️ FIN_E5_API_KEY 환경변수가 설정되지 않았습니다.")
                    logger.info("ℹ️ .env 파일에 FIN_E5_API_KEY=your_api_key 를 추가해주세요.")
                else:
                    logger.info(f"✅ API 키 확인됨: {api_key[:10]}...")
                    
                    # API 연결 테스트
                    try:
                        from service.rag.models.fine5_api_encoder import FinE5APIEncoder
                        encoder = FinE5APIEncoder(api_key=api_key, model_name='abacinlp-text-v1')
                        test_embedding = encoder.encode_query("테스트")
                        logger.info(f"✅ FinE5 API 연결 성공! 임베딩 차원: {len(test_embedding)}")
                    except Exception as e:
                        logger.error(f"❌ FinE5 API 연결 실패: {e}")
                        logger.info("ℹ️ API 키를 확인하고 AbaciNLP 서비스 상태를 확인해주세요.")
                
                logger.info("✅ FinE5 모델 설정 완료")
        
        logger.info("🎉 모든 모델 다운로드 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 모델 다운로드 실패: {e}")
        return False


def load_vectors(model: str = "e5", batch_size: int = 100, limit: int = None):
    """벡터 로드 (임베딩 생성 및 저장)"""
    logger.info(f"🧠 벡터 로드 시작: {model} 모델")
    try:
        # 모델 타입 설정
        if model.lower() == "e5":
            model_type = EmbeddingModelType.MULTILINGUAL_E5_SMALL
        elif model.lower() == "kakaobank":
            model_type = EmbeddingModelType.KAKAOBANK_DEBERTA
        elif model.lower() == "fine5":
            model_type = EmbeddingModelType.FINE5_FINANCE
        else:
            logger.error(f"❌ 지원하지 않는 모델: {model}")
            return False
        
        # 임베딩 생성기 실행
        import subprocess
        from pathlib import Path
        
        # 모델명을 짧은 형태로 변환
        model_short = model.lower()
        
        command = f"""
            cd {Path(__file__).parent} && 
            python embeddings.py --model {model_short} --batch-size {batch_size}
        """
        
        if limit:
            command += f" --limit {limit}"
        
        logger.info(f"🚀 임베딩 생성 명령어: {command}")
        
        # 실시간 로그를 위해 capture_output=False로 설정
        result = subprocess.run(command, shell=True, check=True)
        logger.info("✅ 벡터 로드 완료!")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 벡터 로드 실패: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ 벡터 로드 실패: {e}")
        return False

def show_vector_stats():
    """벡터 통계 조회"""
    logger.info("📊 벡터 통계 조회 중...")
    try:
        vector_store = PgVectorStore()
        config = get_vector_db_config()
        
        with vector_store.conn.cursor() as cursor:
            # 전체 청크 수
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
            
            logger.info(f"📄 전체 청크 수: {total_chunks:,}")
            
            # 각 모델별 임베딩 수
            for model_type in EmbeddingModelType:
                table_name = config.get_table_name(model_type)
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    embedding_count = cursor.fetchone()[0]
                    logger.info(f"🧠 {model_type.value}: {embedding_count:,}개 임베딩")
                except Exception as e:
                    logger.info(f"🧠 {model_type.value}: 테이블 없음")
        
        return True
    except Exception as e:
        logger.error(f"❌ 벡터 통계 조회 실패: {e}")
        return False




def generate_embeddings(model: str = "intfloat/multilingual-e5-small"):
    """임베딩 생성 (CLI용)"""
    import subprocess
    from pathlib import Path
    
    logger.info(f"🧠 임베딩 생성: {model}")
    
    # 임베딩 생성기 실행
    command = f"""
        cd {Path(__file__).parent} && 
        python embeddings.py --model {model}
    """
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ 임베딩 생성 완료")
        if result.stdout:
            logger.info(f"출력: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 임베딩 생성 실패: {e.stderr}")
        return False


def run_full_pipeline(
    batch_size: int = 1000,
    embedding_model: str = "intfloat/multilingual-e5-small",
    skip_embeddings: bool = False
):
    """전체 파이프라인 실행"""
    logger.info("🚀 JSONL RAG 파이프라인 시작")
    start_time = time.time()
    
    # 1단계: Docker Compose 확인
    if not check_docker_compose():
        logger.error("Docker Compose 확인 실패")
        return False
    
    # 2단계: 스키마 생성
    if not create_schema():
        logger.error("스키마 생성 실패")
        return False
    
    # 3단계: JSONL 파일 로딩
    jsonl_dir = Path(__file__).parent.parent.parent.parent / "data" / "transform" / "final"
    if not load_jsonl_files(jsonl_dir, batch_size):
        logger.error("JSONL 파일 로딩 실패")
        return False
    
    # 4단계: 임베딩 생성 (선택적)
    if not skip_embeddings:
        if not generate_embeddings(embedding_model):
            logger.error("임베딩 생성 실패")
            return False
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"🎉 JSONL 로딩 파이프라인 완료!")
    logger.info(f"⏱️  총 소요 시간: {elapsed_time/60:.1f}분")
    
    return True


def main():
    """메인 CLI 함수"""
    
    parser = argparse.ArgumentParser(
        description='JSONL Loader CLI - JSONL 파일 로딩 및 임베딩 생성',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 데이터베이스 관리
  python loader_cli.py db test                     # 데이터베이스 연결 테스트
  python loader_cli.py db list                     # 테이블 목록 조회
  python loader_cli.py db create                   # 스키마 생성
  
        # 모델 다운로드
        python loader_cli.py download                    # 모든 모델 다운로드
        python loader_cli.py download --model e5         # E5 모델만 다운로드
        python loader_cli.py download --model kakaobank  # KakaoBank 모델만 다운로드
        python loader_cli.py download --model fine5      # FinE5 모델 설정 (API 전용)
        
        # 데이터 삭제
        python loader_cli.py truncate doc                # 문서 테이블 데이터 삭제
        python loader_cli.py truncate vector             # 임베딩 테이블 데이터 삭제
        python loader_cli.py truncate all                # 모든 테이블 데이터 삭제
  
        # 데이터 로드
        python loader_cli.py load doc                    # 문서 로드
        python loader_cli.py load vector --model e5      # 벡터 로드 (E5 모델)
        python loader_cli.py load vector --model kakaobank --limit 1000  # KakaoBank 모델로 1000개 제한
        
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 로그 출력'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')
    
    # 데이터베이스 관리
    db_parser = subparsers.add_parser('db', help='데이터베이스 관리')
    db_subparsers = db_parser.add_subparsers(dest='db_command', help='데이터베이스 명령어')
    db_subparsers.add_parser('test', help='데이터베이스 연결 테스트')
    db_subparsers.add_parser('list', help='테이블 목록 조회')
    db_subparsers.add_parser('create', help='스키마 생성')
    
    # truncate 명령어
    truncate_parser = subparsers.add_parser('truncate', help='데이터 삭제 (TRUNCATE)')
    truncate_subparsers = truncate_parser.add_subparsers(dest='truncate_command', help='삭제 명령어')
    truncate_subparsers.add_parser('doc', help='문서 테이블 데이터 삭제')
    truncate_subparsers.add_parser('vector', help='임베딩 테이블 데이터 삭제')
    truncate_subparsers.add_parser('all', help='모든 테이블 데이터 삭제')
    
    # download 명령어
    download_parser = subparsers.add_parser('download', help='임베딩 모델 다운로드')
    download_parser.add_argument('--model', choices=['e5', 'kakaobank', 'fine5', 'all'], default='all', help='다운로드할 모델')
    
    # 로드 명령어
    load_parser = subparsers.add_parser('load', help='데이터 로드')
    load_subparsers = load_parser.add_subparsers(dest='load_command', help='로드 명령어')
    
    # 문서 로드
    load_doc_parser = load_subparsers.add_parser('doc', help='문서 로드')
    load_doc_parser.add_argument('--jsonl-dir', type=Path, help='JSONL 파일 디렉토리')
    load_doc_parser.add_argument('--batch-size', type=int, default=1000, help='배치 크기')
    
    # 벡터 로드 (임베딩 생성)
    load_vector_parser = load_subparsers.add_parser('vector', help='벡터 로드 (임베딩 생성)')
    load_vector_parser.add_argument('--model', choices=['e5', 'kakaobank', 'fine5'], default='e5', help='임베딩 모델')
    load_vector_parser.add_argument('--batch-size', type=int, default=100, help='배치 크기')
    load_vector_parser.add_argument('--limit', type=int, help='처리할 청크 수 제한')
    
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.command == 'db':
            if args.db_command == 'test':
                success = test_database_connection()
            elif args.db_command == 'list':
                success = show_vector_stats()
            elif args.db_command == 'create':
                success = create_schema()
            else:
                db_parser.print_help()
                return
        elif args.command == 'truncate':
            if args.truncate_command == 'doc':
                success = truncate_documents()
            elif args.truncate_command == 'vector':
                success = truncate_vectors()
            elif args.truncate_command == 'all':
                # 모든 테이블 데이터 삭제
                success1 = truncate_documents()
                success2 = truncate_vectors()
                success = success1 and success2
            else:
                truncate_parser.print_help()
                return
        elif args.command == 'download':
            success = download_models(args.model)
        elif args.command == 'load':
            if args.load_command == 'doc':
                # JSONL 디렉토리 설정
                if args.jsonl_dir:
                    jsonl_dir = args.jsonl_dir
                else:
                    jsonl_dir = Path(__file__).parent.parent.parent.parent / "data" / "transform" / "final"
                success = load_documents(jsonl_dir, args.batch_size)
            elif args.load_command == 'vector':
                success = load_vectors(args.model, args.batch_size, args.limit)
            else:
                load_parser.print_help()
                return
        else:
            parser.print_help()
            return
            
    except Exception as e:
        logger.error(f"명령어 실행 실패: {e}")
        sys.exit(1)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

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

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 모든 함수는 이제 별도 모듈에서 import됨

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
  # 전체 파이프라인 실행
  python loader_cli.py run --jsonl-dir ../../../data/transform/final
  
  # 시스템 관리
  python loader_cli.py system health
  python loader_cli.py system reset
  
  # Docker 관리
  python loader_cli.py docker check
  python loader_cli.py docker start
  python loader_cli.py docker stop
  
  # 스키마 관리
  python loader_cli.py schema create
  python loader_cli.py schema check
  python loader_cli.py schema drop
  
  # 데이터 로딩
  python loader_cli.py load data --jsonl-dir ../../../data/transform/final
  python loader_cli.py load stats
  python loader_cli.py load clear
  
  # 임베딩 생성
  python loader_cli.py embed --model intfloat/multilingual-e5-small
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 로그 출력'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')
    
    # 전체 파이프라인 실행
    run_parser = subparsers.add_parser('run', help='전체 파이프라인 실행')
    run_parser.add_argument('--batch-size', type=int, default=1000, help='배치 크기')
    run_parser.add_argument('--embedding-model', default='intfloat/multilingual-e5-small', help='임베딩 모델')
    run_parser.add_argument('--skip-embeddings', action='store_true', help='임베딩 생성 생략')
    
    # 시스템 관리
    system_parser = subparsers.add_parser('system', help='시스템 관리')
    system_subparsers = system_parser.add_subparsers(dest='system_command', help='시스템 명령어')
    system_subparsers.add_parser('health', help='시스템 상태 확인')
    system_subparsers.add_parser('reset', help='시스템 초기화')
    
    # Docker 관리
    docker_parser = subparsers.add_parser('docker', help='Docker 관리')
    docker_subparsers = docker_parser.add_subparsers(dest='docker_command', help='Docker 명령어')
    docker_subparsers.add_parser('check', help='Docker 상태 확인')
    docker_subparsers.add_parser('start', help='Docker 시작')
    docker_subparsers.add_parser('stop', help='Docker 중지')
    
    # 스키마 관리
    schema_parser = subparsers.add_parser('schema', help='스키마 관리')
    schema_subparsers = schema_parser.add_subparsers(dest='schema_command', help='스키마 명령어')
    schema_subparsers.add_parser('create', help='스키마 생성')
    schema_subparsers.add_parser('drop', help='스키마 삭제')
    schema_subparsers.add_parser('check', help='스키마 상태 확인')
    
    # 데이터 로딩
    load_parser = subparsers.add_parser('load', help='데이터 로딩')
    load_subparsers = load_parser.add_subparsers(dest='load_command', help='로딩 명령어')
    
    load_data_parser = load_subparsers.add_parser('data', help='JSONL 파일 로딩')
    load_data_parser.add_argument('--batch-size', type=int, default=1000, help='배치 크기')
    
    load_subparsers.add_parser('stats', help='로딩 통계 조회')
    load_subparsers.add_parser('clear', help='데이터 삭제')
    
    # 임베딩 생성
    embed_parser = subparsers.add_parser('embed', help='임베딩 생성')
    embed_parser.add_argument('--model', default='intfloat/multilingual-e5-small', help='임베딩 모델')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.command == 'run':
                success = run_full_pipeline(
                    batch_size=args.batch_size,
                    embedding_model=args.embedding_model,
                    skip_embeddings=args.skip_embeddings
                )
        elif args.command == 'system':
            if args.system_command == 'health':
                success = system_health_check()
            elif args.system_command == 'reset':
                success = reset_system()
            else:
                system_parser.print_help()
                return
        elif args.command == 'docker':
            if args.docker_command == 'check':
                success = check_docker_compose()
            elif args.docker_command == 'start':
                success = start_docker_compose()
            elif args.docker_command == 'stop':
                success = stop_docker_compose()
            else:
                docker_parser.print_help()
                return
        elif args.command == 'schema':
            if args.schema_command == 'create':
                success = create_schema()
            elif args.schema_command == 'drop':
                success = drop_schema()
            elif args.schema_command == 'check':
                success = check_schema()
            else:
                schema_parser.print_help()
                return
        elif args.command == 'load':
            if args.load_command == 'data':
                # 고정 경로 사용
                jsonl_dir = Path(__file__).parent.parent.parent.parent / "data" / "transform" / "final"
                success = load_jsonl_files(jsonl_dir, args.batch_size)
            elif args.load_command == 'stats':
                success = get_loading_stats()
            elif args.load_command == 'clear':
                success = clear_data()
            else:
                load_parser.print_help()
                return
        elif args.command == 'embed':
            success = generate_embeddings(args.model)
        else:
            parser.print_help()
            return
            
    except Exception as e:
        logger.error(f"명령어 실행 실패: {e}")
        sys.exit(1)
    
    if success:
        logger.info("🎉 명령어 실행 완료")
        sys.exit(0)
    else:
        logger.error("❌ 명령어 실행 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()

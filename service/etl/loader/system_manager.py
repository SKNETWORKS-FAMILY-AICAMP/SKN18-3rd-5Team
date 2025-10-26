#!/usr/bin/env python3
"""
시스템 관리 모듈

Docker 및 PostgreSQL 스키마 관리 기능을 통합 제공합니다.
"""

import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Docker 관리 기능
# =============================================================================

def check_docker_compose():
    """Docker Compose 상태 확인"""
    logger.info("🔍 Docker Compose 상태 확인")
    
    # 컨테이너 상태 확인
    result = subprocess.run("docker ps | grep SKN18", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("✅ SKN18-3rd 컨테이너 실행 중")
        return True
    else:
        logger.error("❌ SKN18-3rd 컨테이너가 실행되지 않음")
        return False


def start_docker_compose():
    """Docker Compose 시작"""
    logger.info("🚀 Docker Compose 시작")
    
    command = "docker-compose up -d"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info("✅ Docker Compose 시작 완료")
        if result.stdout:
            logger.info(f"출력: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Docker Compose 시작 실패: {e.stderr}")
        return False


def stop_docker_compose():
    """Docker Compose 중지"""
    logger.info("🛑 Docker Compose 중지")
    
    command = "docker-compose down"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info("✅ Docker Compose 중지 완료")
        if result.stdout:
            logger.info(f"출력: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Docker Compose 중지 실패: {e.stderr}")
        return False


def get_container_status():
    """컨테이너 상태 정보 조회"""
    logger.info("📊 컨테이너 상태 정보 조회")
    
    try:
        # 컨테이너 목록
        result = subprocess.run("docker ps -a", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("컨테이너 목록:")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"컨테이너 목록 조회 실패: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"컨테이너 상태 조회 중 오류: {e}")
        return False


# =============================================================================
# PostgreSQL 스키마 관리 기능
# =============================================================================

def create_schema():
    """스키마 생성"""
    logger.info("📋 PostgreSQL 스키마 생성")
    
    # 스키마 파일 경로
    schema_file = Path(__file__).parent / "schema_jsonl.sql"
    
    if not schema_file.exists():
        logger.error(f"❌ 스키마 파일을 찾을 수 없습니다: {schema_file}")
        return False
    
    # 스키마 로딩
    try:
        with open(schema_file, 'r') as f:
            result = subprocess.run(
                ["docker", "exec", "-i", "SKN18-3rd", "psql", "-U", "postgres", "-d", "skn_project"],
                input=f.read(),
                text=True,
                capture_output=True,
                check=True
            )
        
        logger.info("✅ 스키마 생성 완료")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 스키마 생성 실패: {e.stderr}")
        return False


def drop_schema():
    """스키마 삭제 (주의: 모든 데이터가 삭제됩니다)"""
    logger.warning("⚠️  스키마 삭제 (모든 데이터가 삭제됩니다)")
    
    try:
        # 테이블 삭제
        drop_commands = [
            "DROP TABLE IF EXISTS embeddings_multilingual_e5_small CASCADE;",
            "DROP TABLE IF EXISTS embeddings_kakaobank CASCADE;",
            "DROP TABLE IF EXISTS embeddings_fine5 CASCADE;",
            "DROP TABLE IF EXISTS chunks CASCADE;",
            "DROP EXTENSION IF EXISTS vector CASCADE;"
        ]
        
        for command in drop_commands:
            result = subprocess.run(
                ["docker", "exec", "SKN18-3rd", "psql", "-U", "postgres", "-d", "skn_project", "-c", command],
                capture_output=True,
                text=True,
                check=True
            )
        
        logger.info("✅ 스키마 삭제 완료")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 스키마 삭제 실패: {e.stderr}")
        return False


def check_schema():
    """스키마 상태 확인"""
    logger.info("🔍 스키마 상태 확인")
    
    try:
        # 테이블 목록 조회
        result = subprocess.run(
            ["docker", "exec", "SKN18-3rd", "psql", "-U", "postgres", "-d", "skn_project", "-c", 
             "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;"],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("📋 현재 테이블 목록:")
        logger.info(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 스키마 상태 확인 실패: {e.stderr}")
        return False


def get_table_info(table_name: str):
    """특정 테이블 정보 조회"""
    logger.info(f"📊 테이블 정보 조회: {table_name}")
    
    try:
        # 테이블 구조 조회
        result = subprocess.run(
            ["docker", "exec", "SKN18-3rd", "psql", "-U", "postgres", "-d", "skn_project", "-c", 
             f"\\d {table_name}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"📋 {table_name} 테이블 구조:")
        logger.info(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 테이블 정보 조회 실패: {e.stderr}")
        return False


# =============================================================================
# 통합 시스템 관리 기능
# =============================================================================

def system_health_check():
    """시스템 전체 상태 확인"""
    logger.info("🏥 시스템 전체 상태 확인")
    
    checks = [
        ("Docker 상태", check_docker_compose),
        ("스키마 상태", check_schema),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            logger.error(f"❌ {check_name} 확인 실패: {e}")
            results[check_name] = False
    
    # 결과 요약
    logger.info("📊 시스템 상태 요약:")
    for check_name, result in results.items():
        status = "✅ 정상" if result else "❌ 문제"
        logger.info(f"  {check_name}: {status}")
    
    return all(results.values())


def reset_system():
    """시스템 초기화 (주의: 모든 데이터가 삭제됩니다)"""
    logger.warning("⚠️  시스템 초기화 (모든 데이터가 삭제됩니다)")
    
    try:
        # 1. 스키마 삭제
        if not drop_schema():
            logger.error("스키마 삭제 실패")
            return False
        
        # 2. 스키마 재생성
        if not create_schema():
            logger.error("스키마 재생성 실패")
            return False
        
        logger.info("✅ 시스템 초기화 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 시스템 초기화 실패: {e}")
        return False

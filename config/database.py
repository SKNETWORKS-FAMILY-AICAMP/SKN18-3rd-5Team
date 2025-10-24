"""
SQLite 데이터베이스 설정
"""
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Generator
import logging

# 데이터베이스 파일 경로
DB_PATH = Path("data/app_database.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

def init_database():
    """데이터베이스 초기화 및 테이블 생성"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # 채팅 세션 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # 채팅 메시지 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
            )
        """)
        
        # 인덱스 생성
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session_id 
            ON chat_messages (session_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
            ON chat_messages (timestamp)
        """)
        
        conn.commit()
        logger.info("데이터베이스 초기화 완료")

@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """데이터베이스 연결 컨텍스트 매니저"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
    try:
        yield conn
    finally:
        conn.close()
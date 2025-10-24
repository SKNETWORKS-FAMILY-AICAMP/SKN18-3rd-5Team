"""
채팅 세션 관리를 위한 SQLite 서비스
"""
from typing import List, Dict, Optional, Any
from datetime import datetime
import sqlite3
from config.database import get_db_connection, init_database
import logging

logger = logging.getLogger(__name__)

class ChatService:
    """채팅 세션 및 메시지 관리 서비스"""
    
    def __init__(self):
        init_database()
    
    def create_session(self, session_id: str, title: str) -> bool:
        """새 채팅 세션 생성"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute("""
                    INSERT INTO chat_sessions (id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (session_id, title, now, now))
                
                conn.commit()
                logger.info(f"새 채팅 세션 생성: {session_id}")
                return True
        except sqlite3.Error as e:
            logger.error(f"세션 생성 실패: {e}")
            return False
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """모든 채팅 세션 조회 (최신순)"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, created_at, updated_at
                    FROM chat_sessions
                    ORDER BY updated_at DESC
                """)
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        'id': row['id'],
                        'title': row['title'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    })
                
                return sessions
        except sqlite3.Error as e:
            logger.error(f"세션 조회 실패: {e}")
            return []
    
    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """특정 세션의 메시지들 조회"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT role, content, timestamp
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """, (session_id,))
                
                messages = []
                for row in cursor.fetchall():
                    messages.append({
                        'role': row['role'],
                        'content': row['content'],
                        'timestamp': row['timestamp']
                    })
                
                return messages
        except sqlite3.Error as e:
            logger.error(f"메시지 조회 실패: {e}")
            return []
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """세션에 새 메시지 추가"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                timestamp = datetime.now().isoformat()
                
                # 메시지 추가
                cursor.execute("""
                    INSERT INTO chat_messages (session_id, role, content, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (session_id, role, content, timestamp))
                
                # 세션 업데이트 시간 갱신
                cursor.execute("""
                    UPDATE chat_sessions
                    SET updated_at = ?
                    WHERE id = ?
                """, (timestamp, session_id))
                
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"메시지 추가 실패: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """채팅 세션 삭제 (메시지도 함께 삭제됨)"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 메시지 먼저 삭제
                cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
                
                # 세션 삭제
                cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
                
                conn.commit()
                logger.info(f"채팅 세션 삭제: {session_id}")
                return True
        except sqlite3.Error as e:
            logger.error(f"세션 삭제 실패: {e}")
            return False
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        """세션 제목 업데이트"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                timestamp = datetime.now().isoformat()
                
                cursor.execute("""
                    UPDATE chat_sessions
                    SET title = ?, updated_at = ?
                    WHERE id = ?
                """, (title, timestamp, session_id))
                
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"세션 제목 업데이트 실패: {e}")
            return False
    
    def session_exists(self, session_id: str) -> bool:
        """세션 존재 여부 확인"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM chat_sessions WHERE id = ?", (session_id,))
                return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"세션 존재 확인 실패: {e}")
            return False
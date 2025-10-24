"""
기존 JSON 파일 기반 채팅 데이터를 SQLite로 마이그레이션하는 스크립트
"""
import json
from pathlib import Path
from service.chat_service import ChatService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_json_to_sqlite():
    """JSON 파일들을 SQLite로 마이그레이션"""
    chat_service = ChatService()
    
    # 기존 JSON 파일 경로
    json_dir = Path("data/chat_sessions")
    
    if not json_dir.exists():
        logger.info("기존 JSON 파일이 없습니다. 마이그레이션을 건너뜁니다.")
        return
    
    migrated_count = 0
    error_count = 0
    
    for json_file in json_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            session_id = json_file.stem
            title = session_data.get('title', f'대화 {session_id}')
            messages = session_data.get('messages', [])
            
            # SQLite에 세션이 이미 존재하는지 확인
            if chat_service.session_exists(session_id):
                logger.info(f"세션 {session_id}는 이미 존재합니다. 건너뜁니다.")
                continue
            
            # 새 세션 생성
            if chat_service.create_session(session_id, title):
                # 메시지들 추가
                for message in messages:
                    role = message.get('role', 'user')
                    content = message.get('content', '')
                    
                    if content:  # 빈 메시지는 건너뛰기
                        chat_service.add_message(session_id, role, content)
                
                migrated_count += 1
                logger.info(f"세션 {session_id} 마이그레이션 완료 ({len(messages)}개 메시지)")
            else:
                error_count += 1
                logger.error(f"세션 {session_id} 생성 실패")
                
        except Exception as e:
            error_count += 1
            logger.error(f"파일 {json_file.name} 마이그레이션 실패: {e}")
    
    logger.info(f"마이그레이션 완료: {migrated_count}개 세션 성공, {error_count}개 실패")
    
    # 마이그레이션 완료 후 JSON 파일들을 백업 폴더로 이동
    if migrated_count > 0:
        backup_dir = Path("data/chat_sessions_backup")
        backup_dir.mkdir(exist_ok=True)
        
        for json_file in json_dir.glob("*.json"):
            backup_file = backup_dir / json_file.name
            json_file.rename(backup_file)
        
        logger.info(f"기존 JSON 파일들을 {backup_dir}로 백업했습니다.")

if __name__ == "__main__":
    migrate_json_to_sqlite()
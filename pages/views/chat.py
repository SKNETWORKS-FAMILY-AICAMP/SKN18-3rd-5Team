from __future__ import annotations
import streamlit as st
import json
from datetime import datetime
from pathlib import Path

# 예상 질문 목록
SUGGESTED_QUESTIONS = [
    "📈 오늘 코스피 상황은 어떤가요?",
    "💰 초보자에게 추천하는 투자 방법은?",
    "📊 현재 주목받는 섹터는 무엇인가요?",
    "🏦 은행주와 증권주 중 어디에 투자해야 할까요?",
    "⚡ 전기차 관련주 전망은 어떤가요?",
    "🌐 해외 투자 시 주의사항은?",
    "📉 주식이 떨어질 때 대응 방법은?",
    "💎 장기투자 vs 단기투자 어떤 게 좋을까요?"
]

# 대화 저장 경로
CHAT_SAVE_DIR = Path("data/chat_sessions")
CHAT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def render_chat_panel() -> None:
    """Render interactive chat interface for Q&A."""
    _init_state()
    
    # 대화창 관리 사이드바를 맨 먼저 렌더링 (상단에 위치)
    _render_chat_sessions_sidebar()
    
    # # 현재 대화창 제목 표시
    current_session = st.session_state.chat_sessions[st.session_state.current_session_id]
    # st.markdown(f"### 💬 {current_session['title']}")
    
    # 예상 질문 버튼들 (채팅 히스토리가 초기 상태일 때만 표시)
    current_history = current_session['messages']
    if len(current_history) <= 1:
        _render_suggested_questions()

    # 채팅 메시지 표시
    chat_container = st.container()
    for message in current_history:
        role = message["role"]
        with chat_container:
            st.chat_message(name=role, avatar=_avatar_for(role)).write(message["content"])

    # 채팅 입력창
    user_input = st.chat_input("질문을 입력하세요.")
    if user_input:
        _handle_user_input(user_input)


def _render_suggested_questions() -> None:
    """예상 질문 버튼들을 렌더링"""
    st.markdown("### 💡 자주 묻는 질문들")
    st.markdown("궁금한 내용을 클릭해보세요!")
    
    # 2열로 버튼 배치
    cols = st.columns(2)
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        col = cols[i % 2]
        with col:
            if st.button(question, key=f"suggested_{i}"):
                _handle_user_input(question)
    
    st.divider()


def _init_state() -> None:
    """세션 상태 초기화"""
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
        st.session_state.current_session_id = None
        _load_saved_sessions()
    
    # 현재 세션이 없으면 새로 생성
    if not st.session_state.current_session_id or st.session_state.current_session_id not in st.session_state.chat_sessions:
        _create_new_session()


def _render_chat_sessions_sidebar() -> None:
    """대화창 관리 사이드바 - 상단에 위치"""
    with st.sidebar:
        # 대화창 관리를 맨 위로 이동
        st.markdown("## 💬 대화창 관리")
        
        # 기존 대화 목록
        if st.session_state.chat_sessions:
            for session_id, session in st.session_state.chat_sessions.items():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if st.button(
                        f"{'🔊' if session_id == st.session_state.current_session_id else ' '} {session['title'][:20]}...",
                        key=f"session_{session_id}",
                        use_container_width=True
                    ):
                        st.session_state.current_session_id = session_id
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"delete_{session_id}", help="대화 삭제"):
                        _delete_session(session_id)
                        st.rerun()

        # 새 대화 버튼
        if st.button("➕ 새 대화", use_container_width=True):
            _create_new_session()
            st.rerun()
        
        # 구분선 추가
        st.markdown("---")


def _create_new_session() -> None:
    """새 대화 세션 생성"""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    title = f"대화 {len(st.session_state.chat_sessions) + 1}"
    
    st.session_state.chat_sessions[session_id] = {
        "title": title,
        "created_at": datetime.now().isoformat(),
        "messages": [
            {
                "role": "assistant",
                "content": "안녕하세요! 투자 관련 궁금한 점을 언제든 물어보세요. 아래 버튼을 클릭하거나 직접 질문을 입력해주세요! 😊",
                "timestamp": datetime.now().isoformat()
            }
        ]
    }
    st.session_state.current_session_id = session_id
    _save_session(session_id)


def _delete_session(session_id: str) -> None:
    """대화 세션 삭제"""
    if session_id in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[session_id]
        
        # 삭제된 세션이 현재 세션이면 다른 세션으로 변경
        if st.session_state.current_session_id == session_id:
            if st.session_state.chat_sessions:
                st.session_state.current_session_id = list(st.session_state.chat_sessions.keys())[0]
            else:
                _create_new_session()
        
        # 파일에서도 삭제
        session_file = CHAT_SAVE_DIR / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()


def _handle_user_input(user_input: str) -> None:
    """사용자 입력 처리 (버튼 클릭 또는 직접 입력)"""
    _append_message("user", user_input)
    # TODO: 백엔드 응답 연동
    _append_message("assistant", "현재는 데모 상태입니다. LLM 응답을 연결해 주세요.")
    
    # 첫 번째 사용자 메시지로 대화 제목 업데이트
    current_session = st.session_state.chat_sessions[st.session_state.current_session_id]
    if len(current_session['messages']) == 3:  # 초기 메시지 + 사용자 질문 + 봇 응답
        current_session['title'] = user_input[:30] + ("..." if len(user_input) > 30 else "")
    
    _save_session(st.session_state.current_session_id)
    st.rerun()


def _append_message(role: str, content: str) -> None:
    """현재 세션에 메시지 추가"""
    current_session = st.session_state.chat_sessions[st.session_state.current_session_id]
    current_session['messages'].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })


def _save_session(session_id: str) -> None:
    """세션을 파일에 저장"""
    session_data = st.session_state.chat_sessions[session_id]
    session_file = CHAT_SAVE_DIR / f"{session_id}.json"
    
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)


def _load_saved_sessions() -> None:
    """저장된 세션들을 로드"""
    if not CHAT_SAVE_DIR.exists():
        return
    
    for session_file in CHAT_SAVE_DIR.glob("*.json"):
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                session_id = session_file.stem
                st.session_state.chat_sessions[session_id] = session_data
        except Exception as e:
            st.error(f"세션 로드 실패: {session_file.name} - {e}")


def _avatar_for(role: str) -> str:
    return "🧑‍💻" if role == "user" else "🤖"

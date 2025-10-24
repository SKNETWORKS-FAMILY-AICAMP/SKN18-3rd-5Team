from __future__ import annotations
import streamlit as st
from datetime import datetime
from service.chat_service import ChatService

def change_chat_theme() -> None:
    st.markdown("""
    <style>
    div.stButton > button {
        background: #FFFFFF;        /* 흰색 배경 */
        color: #3B82F6;             /* 버튼 글자 색 (예: 파란색) */
        border: 1px solid #E5E7EB;  /* 테두리 약하게 */
        padding: 0.6rem 1.2rem;     
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    div.stButton > button:hover {
        background: #F9FAFB;        /* hover 시 살짝 회색톤 */
        border-color: #D1D5DB;
    }
    
    /* 채팅 입력창 스타일 */
    .stChatInput > div > div > div > div {
        border: 2px solid #E5E7EB !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
        transition: all 0.2s ease !important;
    }
    
    .stChatInput > div > div > div > div:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1), 0 2px 8px rgba(0,0,0,0.12) !important;
    }
    
    /* 입력창 내부 텍스트 영역 */
    .stChatInput textarea {
        border: 1px solid #D1D5DB !important;
        border-radius: 8px !important;
        outline: none !important;
        padding: 12px 16px !important;
        background: #FFFFFF !important;
        transition: border-color 0.2s ease !important;
    }
    
    .stChatInput textarea:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* 전송 버튼 스타일 */
    .stChatInput button {
        background: #3B82F6 !important;
        border: none !important;
        border-radius: 8px !important;
        margin: 4px !important;
    }
    
    .stChatInput button:hover {
        background: #2563EB !important;
    }
    </style>
    """, unsafe_allow_html=True)
    


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




# 채팅 서비스 초기화
chat_service = ChatService()

def render_chat_panel() -> None:
    """Render interactive chat interface for Q&A."""
    # 버튼 스타일 적용
    change_chat_theme()
    
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
                        width='stretch'
                    ):
                        st.session_state.current_session_id = session_id
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"delete_{session_id}", help="대화 삭제"):
                        _delete_session(session_id)
                        st.rerun()

        # 새 대화 버튼
        if st.button("➕ 새 대화", width='stretch'):
            _create_new_session()
            st.rerun()
        
        st.write("---")
        st.caption("© 2025 SKN18-3rd-5Team")


def _create_new_session() -> None:
    """새 대화 세션 생성"""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    title = f"대화 {len(st.session_state.chat_sessions) + 1}"
    
    # SQLite에 세션 생성
    if chat_service.create_session(session_id, title):
        # 초기 메시지 추가
        chat_service.add_message(
            session_id, 
            "assistant", 
            "안녕하세요! 투자 관련 궁금한 점을 언제든 물어보세요. 위 버튼을 클릭하거나 직접 질문을 입력해주세요! 😊"
        )
        
        # 세션 상태에 추가
        st.session_state.chat_sessions[session_id] = {
            "title": title,
            "created_at": datetime.now().isoformat(),
            "messages": [
                {
                    "role": "assistant",
                    "content": "안녕하세요! 투자 관련 궁금한 점을 언제든 물어보세요. 위 버튼을 클릭하거나 직접 질문을 입력해주세요! 😊",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
        st.session_state.current_session_id = session_id


def _delete_session(session_id: str) -> None:
    """대화 세션 삭제"""
    # SQLite에서 세션 삭제
    if chat_service.delete_session(session_id):
        # 세션 상태에서도 삭제
        if session_id in st.session_state.chat_sessions:
            del st.session_state.chat_sessions[session_id]
        
        # 삭제된 세션이 현재 세션이면 다른 세션으로 변경
        if st.session_state.current_session_id == session_id:
            if st.session_state.chat_sessions:
                st.session_state.current_session_id = list(st.session_state.chat_sessions.keys())[0]
            else:
                _create_new_session()


def _handle_user_input(user_input: str) -> None:
    """사용자 입력 처리 (버튼 클릭 또는 직접 입력)"""
    _append_message("user", user_input)
    # TODO: 백엔드 응답 연동
    _append_message("assistant", "현재는 데모 상태입니다. LLM 응답을 연결해 주세요.")
    
    # 첫 번째 사용자 메시지로 대화 제목 업데이트
    current_session = st.session_state.chat_sessions[st.session_state.current_session_id]
    if len(current_session['messages']) == 3:  # 초기 메시지 + 사용자 질문 + 봇 응답
        new_title = user_input[:30] + ("..." if len(user_input) > 30 else "")
        current_session['title'] = new_title
        # SQLite에도 제목 업데이트
        chat_service.update_session_title(st.session_state.current_session_id, new_title)
    
    st.rerun()


def _append_message(role: str, content: str) -> None:
    """현재 세션에 메시지 추가"""
    session_id = st.session_state.current_session_id
    timestamp = datetime.now().isoformat()
    
    # SQLite에 메시지 추가
    chat_service.add_message(session_id, role, content)
    
    # 세션 상태에도 추가
    current_session = st.session_state.chat_sessions[session_id]
    current_session['messages'].append({
        "role": role,
        "content": content,
        "timestamp": timestamp
    })


def _load_saved_sessions() -> None:
    """SQLite에서 저장된 세션들을 로드"""
    try:
        sessions = chat_service.get_all_sessions()
        
        for session in sessions:
            session_id = session['id']
            messages = chat_service.get_session_messages(session_id)
            
            st.session_state.chat_sessions[session_id] = {
                "title": session['title'],
                "created_at": session['created_at'],
                "messages": messages
            }
    except Exception as e:
        st.error(f"세션 로드 실패: {e}")


def _avatar_for(role: str) -> str:
    return "🧑‍💻" if role == "user" else "🤖"

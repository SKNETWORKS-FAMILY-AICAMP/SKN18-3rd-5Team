from __future__ import annotations
import streamlit as st

# 예시 화면 -> 수정 

def render_chat_panel() -> None:
    """Render interactive chat interface for Q&A."""
    _init_state()

    chat_container = st.container()
    for message in st.session_state.chat_history:
        role = message["role"]
        with chat_container:
            st.chat_message(name=role, avatar=_avatar_for(role)).write(message["content"])

    user_input = st.chat_input("질문을 입력하세요.")
    if user_input:
        _append_message("user", user_input)
        # TODO: 백엔드 응답 연동
        _append_message("assistant", "현재는 데모 상태입니다. LLM 응답을 연결해 주세요.")
        st.experimental_rerun()


def _init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "안녕하세요! 무엇을 도와드릴까요?",
            }
        ]


def _append_message(role: str, content: str) -> None:
    st.session_state.chat_history.append({"role": role, "content": content})


def _avatar_for(role: str) -> str:
    return "🧑‍💻" if role == "user" else "🤖"

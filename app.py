from __future__ import annotations
from datetime import datetime
import streamlit as st
from pages.app_bootstrap import render_sidebar, render_page_title  # 필수
from pages.views.user_level import render_user_level

# ---------------------------
# 기본 설정
# ---------------------------
st.set_page_config(
    page_title="Investment Q&A System",
    page_icon="🤖",
    layout="wide",
)
render_sidebar()

# ---------------------------
# UTILS
# ---------------------------
def _format_timestamp(timestamp: datetime) -> str:
    return timestamp.strftime("%Y-%m-%d %H:%M")

# ---------------------------
# VIEW
# ---------------------------
def render_top():
    # 시스템 이름 등 구현
    pass
    
# 대시보드 구현
def render_status_overview() -> None:
    ## 예시 입니다 -> 수정
    
    """Display current RAG / LLM training status metrics."""
    st.subheader("RAG/LLM 학습 현황")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("임베딩 문서 수", value="1,248", delta="+32")
    with col2:
        st.metric("마지막 학습", value=_format_timestamp(datetime.now()))
    with col3:
        st.metric("평균 응답 정확도", value="92%", delta="+3%")


# 실행
def _render():
    # 시스템 정보
    render_top()
    # 시스템 상황판
    render_status_overview()
    # 사용자 투자 지식 수준 판별
    render_user_level()

    st.write("---")
    st.caption("© 2025 · SK Networks Family AI Camp 18기 - 3rd - 5Team")

if __name__== "__main__":
    _render()
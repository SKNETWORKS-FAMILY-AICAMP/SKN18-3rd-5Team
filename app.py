from __future__ import annotations
from datetime import datetime
import streamlit as st
from pages.app_bootstrap import render_sidebar, render_page_title  # 필수

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
    st.progress(0.6, text="재학습 파이프라인 진행률 (예시)")
    st.caption("실제 값은 백엔드 연동 후 갱신하세요.")

# 나의 투자 수준 선택하기
def render_user_level():
    ## 예시 입니다 -> 수정
    st.subheader("투자지식수준설문영역")

# 실행
def _render():
    render_top()
    render_status_overview()
    render_user_level()

    st.write("---")
    st.caption("© 2025 · SK Networks Family AI Camp 18기 - 3rd - 5Team")

if __name__== "__main__":
    _render()
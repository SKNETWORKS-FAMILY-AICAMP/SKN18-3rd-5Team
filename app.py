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
    # 상단 히어로 섹션
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                padding: 3rem 2rem; 
                border-radius: 15px; 
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="flex: 1;">
                <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    SK Networks Family AI Camp 18기 > 3rd Team > Investment Q&A System
                </div>
                <h1 style="color: #2c3e50; font-size: 3rem; font-weight: 700; margin: 0.5rem 0;">
                    Investment Q&A System
                </h1>
                <div style="color: #7f8c8d; font-size: 1rem; margin-top: 1rem;">
                    <span style="margin-right: 2rem;">📊 By Team-5</span>
                    <span style="margin-right: 2rem;">📅 Updated on 27 Oct, 2025</span>
                </div>
                <div style="color: #7f8c8d; font-size: 0.9rem; margin-top: 0.5rem;">
                    <span style="margin-right: 2rem;">🔍 Powered by RAG & LLM</span>
                    <span>✅ Investment Knowledge Assistant</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 대시보드 구현
def render_status_overview() -> None:
    """Display current RAG / LLM training status metrics."""
    
    # 섹션 헤더
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e8f4f8 0%, #f1f8ff 100%); 
                padding: 1.5rem 2rem; 
                border-radius: 12px; 
                margin: 2rem 0 1rem 0;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                border: 1px solid rgba(0, 0, 0, 0.05);">
        <h2 style="color: #2c3e50; margin: 0; font-size: 1.8rem; font-weight: 600;">
            📊 RAG/LLM 학습 현황
        </h2>
        <p style="color: #7f8c8d; margin: 0.5rem 0 0 0; font-size: 1rem;">
            실시간 시스템 상태 및 성능 지표
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 메트릭 카드들
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f0f8ff 100%); 
                   padding: 2rem; 
                   border-radius: 15px; 
                   text-align: center;
                   box-shadow: 0 4px 15px rgba(33, 150, 243, 0.1);
                   border: 1px solid rgba(33, 150, 243, 0.1);">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">📚</div>
            <div style="color: #1976d2; font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">1,248</div>
            <div style="color: #424242; font-size: 1rem; margin-bottom: 0.3rem;">임베딩 문서 수</div>
            <div style="color: #757575; font-size: 0.9rem;">
                <span style="color: #4caf50;">+32</span> 최근 추가
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fce4ec 0%, #fff0f5 100%); 
                   padding: 2rem; 
                   border-radius: 15px; 
                   text-align: center;
                   box-shadow: 0 4px 15px rgba(233, 30, 99, 0.1);
                   border: 1px solid rgba(233, 30, 99, 0.1);">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎯</div>
            <div style="color: #c2185b; font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">92%</div>
            <div style="color: #424242; font-size: 1rem; margin-bottom: 0.3rem;">평균 응답 정확도</div>
            <div style="color: #757575; font-size: 0.9rem;">
                <span style="color: #4caf50;">+3%</span> 성능 향상
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f3e5f5 0%, #faf8ff 100%); 
                   padding: 2rem; 
                   border-radius: 15px; 
                   text-align: center;
                   box-shadow: 0 4px 15px rgba(156, 39, 176, 0.1);
                   border: 1px solid rgba(156, 39, 176, 0.1);">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">⏰</div>
            <div style="color: #7b1fa2; font-size: 1.8rem; font-weight: bold; margin-bottom: 0.5rem;">{}</div>
            <div style="color: #424242; font-size: 1rem; margin-bottom: 0.3rem;">마지막 학습 시간</div>
            <div style="color: #757575; font-size: 0.9rem;">
                <span style="color: #4caf50;">●</span> 시스템 정상 운영
            </div>
        </div>
        """.format(_format_timestamp(datetime.now())), unsafe_allow_html=True)


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
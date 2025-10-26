from __future__ import annotations
from datetime import datetime
import streamlit as st
from pages.app_bootstrap import render_sidebar, render_page_title  # 필수
from pages.views.user_level import render_user_level
from config.database import init_database

# ---------------------------
# 기본 설정
# ---------------------------
st.set_page_config(
    page_title="StockBuddy: Investment Q&A System",
    page_icon="🤖",
    layout="wide",
)

# 데이터베이스 초기화
init_database()
render_sidebar() # 사이드바 나오기

# ---------------------------
# UTILS
# ---------------------------
def _format_timestamp(timestamp: datetime) -> str:
    return timestamp.strftime("%Y-%m-%d %H:%M")


# ---------------------------
# VIEW
# ---------------------------


######################
# 1. 상단 히어로 섹션
#####################
def render_top():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #D8B4FE 0%, #818CF8 100%);
                padding: 2rem 3rem; 
                border-radius: 15px; 
                margin-bottom: 0rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="flex: 1;">
                <h1 style="color: #FAFAFA; font-size: 3rem; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.2); font-weight: 700; margin: 0.5rem 0;">
                    Investment Q&A System
                </h1>
                <div style="color: #FAFAFA; font-size: 1rem; margin-top: 1rem; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);">
                    <span style="margin-right: 1rem;">💻 By Team-5</span>
                    <span style="margin-right: 1rem;">📅 Updated on 27 Oct, 2025</span>
                    <span>✅ Investment Knowledge Assistant</span>
                </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

####################
# 2. 대시보드 구현
###################

def render_status_overview(docs="5,076", acc="≈89%", last="2025-10-20"):
    st.markdown(f"""
    <div style="margin: 2rem 0 1rem 0;">
        <h2 style="color: #1f2d3d; font-size: 1.6rem; font-weight: 700; margin-bottom: 0.3rem;">
            📊 RAG/LLM 학습 현황
        </h2>
        <div style="height: 2px; width: 60px; background-color: #1f4fd6; border-radius: 2px; margin-bottom: 0.8rem;"></div>
        <p style="color: #5f6b7a; font-size: 1rem; margin: 0;">
            실시간 시스템 상태 및 성능 지표를 요약한 정보입니다.
        </p>
    </div>
    """, unsafe_allow_html=True)




###################
# 3. 메트릭 카드들
##################

    col1, col2, col3 = st.columns(3)

    # 공통 스타일
    card_style = """
    border-radius:10px; 
    padding:1rem; 
    box-shadow:0 2px 6px rgba(0,0,0,0.05); 
    text-align:center;
"""

    with col1:
        st.markdown(
            f"""
            <div style="background-color:#E9F2FF; {card_style}">
            <div style="display:flex; justify-content:center;">
                <div style="display:flex; align-items:center; gap:1rem; width:max-content;">
                <div style="font-size:2.5rem;">📚</div>
                <div style="text-align:left;">
                    <div style="font-size:1.6rem; font-weight:700; color:#1976D2;">{docs}</div>
                    <div style="font-size:0.9rem; color:#555;">임베딩 문서 수</div>
                </div>
                </div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="background-color:#E9F2FF; {card_style}">
            <div style="display:flex; justify-content:center;">
                <div style="display:flex; align-items:center; gap:1rem; width:max-content;">
                <div style="font-size:2.5rem;">🎯</div>
                <div style="text-align:left;">
                    <div style="font-size:1.6rem; font-weight:700; color:#D32F2F;">{acc}</div>
                    <div style="font-size:0.9rem; color:#555;">평균 유사도</div>
                </div>
                </div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style="background-color:#E9F2FF; {card_style}">
            <div style="display:flex; justify-content:center;">
                <div style="display:flex; align-items:center; gap:1rem; width:max-content;">
                <div style="font-size:2.5rem;">⏰</div>
                <div style="text-align:left;">
                    <div style="font-size:1.6rem; font-weight:700; color:#6A1B9A;">{last}</div>
                    <div style="font-size:0.9rem; color:#555;">마지막 학습 일자</div>
                </div>
                </div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

#############
# copyright 사이드바 표시
#############
st.sidebar.write("---")
st.sidebar.caption("© 2025 SKN18-3rd-5Team")


#-----------------
# 실행
#---------------

# 헤더 섹션 (페이지 최상단)
render_top()

def _render():
    render_status_overview()    # 시스템 상황판
    render_user_level()       # 사용자 투자 지식 수준 판별

if __name__== "__main__":
    _render()
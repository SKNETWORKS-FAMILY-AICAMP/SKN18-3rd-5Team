from __future__ import annotations
import streamlit as st
from collections import namedtuple

PageInfo = namedtuple('PageInfo', ['link', 'label', 'icon'])
PAGE_INFO = {
    "P0": PageInfo(link="app.py", label="홈", icon="🏠"),
    "P1": PageInfo(link="pages/page1.py", label="Q&A", icon="🔎"),
#     "P2": PageInfo(link="", label="", icon=""),
#     "P3": PageInfo(link="", label="", icon=""),
    "PDT": PageInfo(link="pages/data_tool.py", label="데이터 도구", icon="🧰"),
}

def _hide_builtin_nav():
    """Streamlit 기본 멀티페이지 네비(상단 자동 목록) 숨김 + 사이드바 정돈"""
    st.markdown(
        """
        <style>
          [data-testid="stSidebarNav"] { display: none !important; }
          section[data-testid="stSidebar"] { padding-top: .5rem; }
          /* 사이드바 링크 간격/호버 */
          [data-testid="stSidebar"] a { padding: .35rem .25rem !important; border-radius: 8px; }
          [data-testid="stSidebar"] a:hover { background: rgba(255,255,255,.06); }
        </style>
    """,
        unsafe_allow_html=True,
    )

def _inject_common_styles():
    """페이지 전역에서 재사용할 공통 스타일"""
    st.markdown(
        """
        <style>
          .app-title { font-size: 1.9rem; font-weight: 700; margin: 0 0 .35rem; }
          .app-title--compact { font-size: 36px !important; line-height: 1.3; }
          .stApp { background-color: #ffffff !important; }
          [data-testid="stAppViewContainer"],
          [data-testid="stSidebar"],
          [data-testid="stMarkdownContainer"],
          [data-testid="stHeader"] {
            background-color: #ffffff !important;
            color: #000000 !important;
          }
          .stApp button {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #d0d0d0 !important;
            box-shadow: none !important;
            border-radius: 10px !important;
          }
          .stApp button:hover,
          .stApp button:focus {
            background-color: #f5f5f5 !important;
            color: #000000 !important;
            border-color: #b0b0b0 !important;
            box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.05) !important;
          }
          .stApp button:disabled {
            background-color: #f0f0f0 !important;
            color: rgba(0, 0, 0, 0.4) !important;
            border-color: #d0d0d0 !important;
            box-shadow: none !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_page_title(page_info: Any, *, variant: str = "default") -> None:
    """페이지 타이틀 렌더링 (기본/compact)"""
    title_class = "app-title"
    if variant == "compact":
        title_class += " app-title--compact"
    # st.markdown(f'<h1 class="{title_class}">{text}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="{title_class}">{page_info.icon}&nbsp;{page_info.label}</h1>', unsafe_allow_html=True)

def render_sidebar():
    st.markdown("""
    <style>
    style="position: relative; user-select: auto; width: 400px; height: auto; box-sizing: border-box; flex-shrink: 0;"
    """, unsafe_allow_html=True)
    _inject_common_styles()
    
    # 기본 네비게이션 숨김
    _hide_builtin_nav()

    """공통 사이드바"""
    with st.sidebar:
        st.header("투자 Q&A 시스템")
        st.markdown("투자에 대해 무엇이든 물어보세요!")
        
        # 페이지 네비게이션을 하단으로 이동
        # 대화창 관리는 chat.py에서 직접 렌더링됨
        
        # 하단에 페이지 링크들 배치
        st.markdown("---")
        st.markdown("# 📍 페이지 이동")
        st.page_link(PAGE_INFO["P0"].link, label=PAGE_INFO["P0"].label, icon=PAGE_INFO["P0"].icon)
        st.page_link(PAGE_INFO["P1"].link, label=PAGE_INFO["P1"].label, icon=PAGE_INFO["P1"].icon)
        # st.page_link(PAGE_INFO["P2"].link, label=PAGE_INFO["P2"].label, icon=PAGE_INFO["P2"].icon)
        # st.page_link(PAGE_INFO["P3"].link, label=PAGE_INFO["P3"].label, icon=PAGE_INFO["P3"].icon)
        st.page_link(PAGE_INFO["PDT"].link, label=PAGE_INFO["PDT"].label, icon=PAGE_INFO["PDT"].icon)
        st.write("---")
        st.caption("© 2025 SKN18-3rd-5Team")

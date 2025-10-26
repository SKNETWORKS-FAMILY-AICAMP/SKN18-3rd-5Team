from __future__ import annotations
import streamlit as st
from collections import namedtuple

PageInfo = namedtuple('PageInfo', ['link', 'label', 'icon'])
PAGE_INFO = {
    "P0": PageInfo(link="app.py", label="홈", icon="🏠"),
    "P1": PageInfo(link="pages/page1.py", label="스톡버디에게 물어보기", icon="🙋"),
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
            color: #000000 !important;
          }
          .stApp button {
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
    # 공통 페이지 설정
    st.set_page_config(
        page_title="StockBuddy: Investment Q&A System",
        page_icon="🤖",
        layout="wide",
    )
    
    st.markdown("""
    <style>
    /* 🌟 사이드바 고정 크기 및 정렬 */
    [data-testid="stSidebar"] {
        width: 400px !important;
        min-width: 400px !important;
        max-width: 400px !important;
        background-color: #f9fafc !important;
        color: #111827 !important;
    }

    /* 🌈 사이드바 내부 중앙 정렬 */
    [data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        align-items: center;     /* 가로 중앙 */
        text-align: center;      /* 텍스트 중앙 */
        padding: 1.5rem 1rem;
    }

    /* 🖼️ 이미지 중앙 정렬 */
    [data-testid="stSidebar"] .stImage img {
        position: relative;
        left: calc(50% - 67px);
        width: 30px;
        height: 216px; # 우측 배너 위치와 맞추기
        object-fit: contain; /* or cover */
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    /* 💬 헤더 스타일 */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        text-align: center !important;
        color: #1e293b !important;
    }

    /* 📎 구분선과 링크 스타일 */
    [data-testid="stSidebar"] hr {
        border: 0;
        border-top: 1px solid #e2e8f0;
        width: 80%;
        margin: 1.2rem auto;
    }

    [data-testid="stSidebar"] a {
        text-decoration: none;
        color: #2563eb !important;
        font-weight: 600;
    }

    [data-testid="stSidebar"] a:hover {
        color: #1d4ed8 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _inject_common_styles()
    _hide_builtin_nav()

    """공통 사이드바"""
    with st.sidebar:
        st.image("assets/img/StockBuddy3.png")
        # st.image("data/StockBuddy3.png")
        st.subheader("스톡버디와 함께 투자를 이야기해보세요 💬")

        st.markdown("---")
        st.markdown("### 📍 페이지 이동")
        st.page_link(PAGE_INFO["P0"].link, label=PAGE_INFO["P0"].label, icon=PAGE_INFO["P0"].icon)
        st.page_link(PAGE_INFO["P1"].link, label=PAGE_INFO["P1"].label, icon=PAGE_INFO["P1"].icon)
        st.page_link(PAGE_INFO["PDT"].link, label=PAGE_INFO["PDT"].label, icon=PAGE_INFO["PDT"].icon)


        
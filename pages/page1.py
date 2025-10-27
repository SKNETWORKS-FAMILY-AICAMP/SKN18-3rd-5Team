import streamlit as st
from pages.views import render_chat_panel, render_user_level_summary, render_chat_lgrp_test
from pages.app_bootstrap import render_sidebar, render_page_title, PAGE_INFO  # 필수

# =========================
# 공통 페이지 설정
# =========================
st.set_page_config(
    page_title="StockBuddy: Investment Q&A System",
    page_icon="🤖",
    layout="wide",
)

render_sidebar()
render_page_title(PAGE_INFO["P1"], variant="compact")


# =========================
# Views
# =========================
render_user_level_summary()

#랭그레프 테스트 (chat_panel에 추가 예정)
# render_chat_lgrp_test()

render_chat_panel()

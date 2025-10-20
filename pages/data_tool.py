from __future__ import annotations
import os, sys, io, time, contextlib
from pathlib import Path
from datetime import datetime
import streamlit as st
from pages.app_bootstrap import render_sidebar, render_page_title, PAGE_INFO  # 필수

# =========================
# 공통 페이지 설정
# =========================
render_sidebar()
render_page_title(PAGE_INFO["PDT"], variant="compact")

# 루트 경로 
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# =========================
# ETL
# =========================
ETL_STEPS = ("Extract", "Transform", "Load")

def render() -> None:
    """Render ETL orchestration dashboard."""
    st.subheader("ETL 파이프라인")

    _init_state()
    _render_pipeline_overview()
    st.divider()
    _render_action_buttons()
    st.divider()
    _render_logs()


def _init_state() -> None:
    if "etl_logs" not in st.session_state:
        st.session_state.etl_logs = []


def _render_pipeline_overview() -> None:
    st.markdown(
        """
        **파이프라인 순서**
        1. 데이터 수집 (Extract)
        2. 정제 및 전처리 (Transform)
        3. 적재 (Load)
        """
    )


def _render_action_buttons() -> None:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Extract 실행", use_container_width=True):
            _log_step("Extract")
    with col2:
        if st.button("Transform 실행", use_container_width=True):
            _log_step("Transform")
    with col3:
        if st.button("Load 실행", use_container_width=True):
            _log_step("Load")
    with col4:
        if st.button("전체 파이프라인 실행", type="primary", use_container_width=True):
            for step in ETL_STEPS:
                _log_step(step)


def _render_logs() -> None:
    st.subheader("실행 로그")
    if not st.session_state.etl_logs:
        st.info("아직 실행된 작업이 없습니다. 버튼을 눌러 파이프라인을 실행해 보세요.")
        return

    for entry in reversed(st.session_state.etl_logs):
        st.code(entry, language="bash")


def _log_step(step: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.etl_logs.append(f"[{timestamp}] {step} 단계가 완료되었습니다. (샘플)")
    st.toast(f"{step} 단계 실행 완료!", icon="✅")


if __name__ == "__main__":
    render()
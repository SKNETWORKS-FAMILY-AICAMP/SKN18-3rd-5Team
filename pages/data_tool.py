from __future__ import annotations
import os, sys, io, time, contextlib
import asyncio
from pathlib import Path
from datetime import datetime
import streamlit as st
from pages.app_bootstrap import render_sidebar, render_page_title, PAGE_INFO  # 필수
from service.crawling.report_crawling import crawl_shinhan_reports
from service.crawling.kospi_top_crawling import do_crawl

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
PIPELINES = (
    ("RAG 구축용 ETL 파이프라인", "rag"),
    ("FineTuning용 ETL 파이프라인", "finetune"),
)

def render() -> None:
    """Render ETL orchestration dashboard."""
    st.subheader("ETL 파이프라인")

    _init_state()
    _render_pipeline_overview()
    st.divider()
    _render_pipeline_controls()
    st.divider()
    _render_logs()


def _init_state() -> None:
    if "etl_logs" not in st.session_state:
        st.session_state.etl_logs = []


def _render_pipeline_overview() -> None:
    st.markdown(
        """
        **공통 파이프라인 순서**
        1. 데이터 수집 (Extract)
        2. 정제 및 전처리 (Transform)
        3. 적재 (Load)
        """
    )

def _render_pipeline_controls() -> None:
    for label, key_prefix in PIPELINES:
        st.markdown(f"### {label}")

        report_count = None
        if key_prefix == "finetune":
            with st.container(border=True):
                st.markdown("**[Extract용 옵션 패널]**")
                report_count = st.number_input(
                    "금융 리포트 추출 개수",
                    min_value=1,
                    max_value=1000,
                    value=10,
                    step=1,
                    key=f"{key_prefix}_report_count",
                )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Extract 실행", key=f"{key_prefix}_extract", use_container_width=True):
                _handle_step("Extract", label, key_prefix, report_count)
        with col2:
            if st.button("Transform 실행", key=f"{key_prefix}_transform", use_container_width=True):
                _handle_step("Transform", label, key_prefix, report_count)
        with col3:
            if st.button("Load 실행", key=f"{key_prefix}_load", use_container_width=True):
                _handle_step("Load", label, key_prefix, report_count)
        with col4:
            if st.button(
                "전체 파이프라인 실행",
                key=f"{key_prefix}_run_all",
                type="primary",
                use_container_width=True,
            ):
                for step in ETL_STEPS:
                    if not _handle_step(step, label, key_prefix, report_count):
                        break
        st.markdown("")  # spacing


def _render_logs() -> None:
    st.subheader("실행 로그")
    if not st.session_state.etl_logs:
        st.info("아직 실행된 작업이 없습니다. 버튼을 눌러 파이프라인을 실행해 보세요.")
        return

    for entry in reversed(st.session_state.etl_logs):
        st.code(entry, language="bash")


def _handle_step(
    step: str,
    pipeline_label: str,
    key_prefix: str,
    report_count: int | None = None,
) -> bool:
    if step == "Extract":
        if key_prefix == "finetune":
            count = int(report_count) if report_count is not None else 10
            if not _run_finetune_extract(pipeline_label, count):
                return False
        elif key_prefix == "rag":
            if not _run_rag_extract(pipeline_label):
                return False
    _log_step(step, pipeline_label)
    return True


def _run_rag_extract(pipeline_label: str) -> bool:
    try:
        _log_info(pipeline_label, "KOSPI 상위 종목 데이터를 수집 중입니다...")
        with st.spinner("KOSPI 상위 종목 데이터를 수집 중입니다..."):
            do_crawl()
    except Exception as exc:  # noqa: BLE001
        _log_error("Extract", pipeline_label, exc)
        st.error(f"RAG Extract 단계 실행 중 오류가 발생했습니다: {exc}")
        return False
    return True


def _run_finetune_extract(pipeline_label: str, report_count: int = 10) -> bool:
    try:
        _log_info(pipeline_label, f"금융 리포트 {report_count}건을 수집 중입니다...")
        with st.spinner(f"금융 리포트 {report_count}건을 수집 중입니다..."):
            asyncio.run(crawl_shinhan_reports(report_count))
    except Exception as exc:  # noqa: BLE001
        _log_error("Extract", pipeline_label, exc)
        st.error(f"FineTuning Extract 단계 실행 중 오류가 발생했습니다: {exc}")
        return False
    return True


def _log_step(step: str, pipeline_label: str) -> None:
    _log_info(pipeline_label, f"{step} 단계가 완료되었습니다.")
    st.toast(f"{pipeline_label}: {step} 단계 실행 완료!", icon="✅")


def _log_info(pipeline_label: str, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.etl_logs.append(f"[{timestamp}] [{pipeline_label}] {message}")


def _log_error(step: str, pipeline_label: str, error: Exception) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.etl_logs.append(
        f"[{timestamp}] [{pipeline_label}] {step} 단계 실행 중 오류 발생: {error}"
    )
    st.toast(f"{pipeline_label}: {step} 단계 실행 실패", icon="❌")


if __name__ == "__main__":
    render()

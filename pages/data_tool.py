from __future__ import annotations
import os, sys, io, time, contextlib
import asyncio
from pathlib import Path
from datetime import datetime
import streamlit as st
from pages.app_bootstrap import render_sidebar, render_page_title, PAGE_INFO  # 필수
from service.crawling.report_crawling import crawl_shinhan_reports
from service.crawling.kospi_top_crawling import do_crawl
from service.fine_tuning.data_cleansing import do_cleansing
from service.fine_tuning.data_chunking import do_chunking
from service.fine_tuning.csv2json import convert
from service.fine_tuning.llama_factory.split_test_train_data import split

# =========================
# 공통 페이지 설정
# =========================
st.set_page_config(
    page_title="StockBuddy: Investment Q&A System",
    page_icon="🤖",
    layout="wide",
)
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
# 파이프라인 표시 이름과 내부 키를 함께 정의한다.
PIPELINES = (
    ("1️⃣ RAG 구축용 ETL 파이프라인", "rag"),
    ("2️⃣ FineTuning용 ETL 파이프라인", "finetune"),
)

PIPELINE_STEP_OUTPUTS: dict[str, dict[str, list[str]]] = {
    "rag": {
        "Extract": ["kospi_top100.txt", "kospi_top100_map.json", "data/zip/", "data/xml/", "data/markdown/*"],
        "Transform": ["data/transform/*"],
        "Load": ["pgvector"],
    },
    "finetune": {
        "Extract": ["shinhan_research_2025_playwright.csv"],
        "Transform": ["clean_data.csv", "chunked_data.csv", "csv2json.json"],
        "Load": ["service/fine_tuning/llama_factory/dataset/test.json, train.json"],
    },
}

STEP_OUTPUT_LABELS = {
    "Extract": "Extract 결과 파일/경로",
    "Transform": "Transform 결과 파일/경로",
    "Load": "Load 결과 ▶️ 이제 LLaMA Factory에서 학습을 진행하세요!",
}

STEP_OUTPUT_ICONS = {
    "Extract": "📄",
    "Transform": "🛠️",
    "Load": "🚀",
}

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
        **공통 파이프라인 순서** : 1. 데이터 수집 (Extract) ▶️  2. 정제 및 전처리 (Transform)  ▶️  3. 적재 (Load)
        """
    )

def _render_pipeline_controls() -> None:
    for label, key_prefix in PIPELINES:
        st.markdown(f"### {label}")

        report_count = None
        if key_prefix == "finetune":
            # FineTuning Extract에 필요한 매개변수만 옵션 패널로 노출한다.
            st.markdown(
                """
                <style>
                .finetune-report-label {
                    padding-top: 6px;
                }
                div[data-testid="stNumberInput"] input[aria-label="금융 리포트 추출 개수"] {
                    border: 1px solid #d0d0d0;
                    border-radius: 6px;
                    padding: 6px 8px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                st.markdown("**[Extract용 옵션 패널]**")
                label_col, input_col, _spacer = st.columns([1, 1, 6])
                with label_col:
                    st.markdown(
                        '<div class="finetune-report-label">금융 리포트 추출 개수</div>',
                        unsafe_allow_html=True,
                    )
                with input_col:
                    report_count = st.number_input(
                        "금융 리포트 추출 개수",
                        min_value=1,
                        max_value=999999,
                        value=10,
                        step=1,
                        key=f"{key_prefix}_report_count",
                        label_visibility="collapsed",
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
    success = True
    # Extract 단계는 파이프라인 유형에 따라 실행 경로가 다르다.
    if step == "Extract":
        if key_prefix == "finetune":
            count = int(report_count) if report_count is not None else 10
            success = _run_finetune_extract(pipeline_label, count)
        elif key_prefix == "rag":
            success = _run_rag_extract(pipeline_label)
    elif step == "Transform":
        if key_prefix == "finetune":
            success = _run_finetune_transform(pipeline_label)
    elif step == "Load":
        if key_prefix == "finetune":
            success = _run_finetune_load(pipeline_label)
    if not success:
        return False
    _log_step(step, pipeline_label)
    _render_step_outputs(key_prefix, step)
    return True


def _run_rag_extract(pipeline_label: str) -> bool:
    try:
        # RAG 용 ETL은 KOSPI 상위 종목 크롤링을 수행한다.
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
        # FineTuning 용 ETL은 입력 받은 수만큼 신한 금융 리포트를 수집한다.
        _log_info(pipeline_label, f"금융 리포트 {report_count}건을 수집 중입니다...")
        with st.spinner(f"금융 리포트 {report_count}건을 수집 중입니다..."):
            asyncio.run(crawl_shinhan_reports(report_count))
    except Exception as exc:  # noqa: BLE001
        _log_error("Extract", pipeline_label, exc)
        st.error(f"FineTuning Extract 단계 실행 중 오류가 발생했습니다: {exc}")
        return False
    return True


def _run_finetune_transform(pipeline_label: str) -> bool:
    try:
        _log_info(
            pipeline_label,
            "정제 → 청크 분할 → JSON 변환 작업을 순차적으로 실행 중입니다...",
        )
        with st.spinner("FineTuning Transform 단계 작업을 진행 중입니다..."):
            do_cleansing()
            do_chunking()
            convert()
    except Exception as exc:  # noqa: BLE001
        _log_error("Transform", pipeline_label, exc)
        st.error(f"FineTuning Transform 단계 실행 중 오류가 발생했습니다: {exc}")
        return False
    return True


def _run_finetune_load(pipeline_label: str) -> bool:
    try:
        _log_info(
            pipeline_label,
            "학습/평가 데이터 분할 및 LLaMA Factory 적재 준비를 진행 중입니다...",
        )
        with st.spinner("FineTuning Load 단계 작업을 진행 중입니다..."):
            split()
    except Exception as exc:  # noqa: BLE001
        _log_error("Load", pipeline_label, exc)
        st.error(f"FineTuning Load 단계 실행 중 오류가 발생했습니다: {exc}")
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


def _render_step_outputs(key_prefix: str, step: str) -> None:
    outputs = PIPELINE_STEP_OUTPUTS.get(key_prefix, {}).get(step)
    if not outputs:
        return

    icon = STEP_OUTPUT_ICONS.get(step, "📁")
    label = STEP_OUTPUT_LABELS.get(step, "결과 파일/경로")
    bullet_lines = "\n".join(f"- `{item}`" for item in outputs)
    st.markdown(f"{icon} **{label}**\n{bullet_lines}")


if __name__ == "__main__":
    render()

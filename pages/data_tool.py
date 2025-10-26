from __future__ import annotations
import os, sys, io, time, contextlib, html
import asyncio
import subprocess
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
        elif key_prefix == "rag":
            # RAG 파이프라인 옵션 패널
            st.markdown("""
            <style>
            /* 라디오 버튼 스타일 개선 */
            div[data-testid="stRadio"] > div {
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
                margin: 5px 0;
            }
            div[data-testid="stRadio"] > div:hover {
                background-color: #e9ecef;
                border-color: #007bff;
            }
            div[data-testid="stRadio"] > div[data-checked="true"] {
                background-color: #e7f3ff;
                border-color: #007bff;
                border-width: 2px;
            }
            /* 라디오 버튼 동그라미 색상 변경 */
            div[data-testid="stRadio"] input[type="radio"]:checked {
                background-color: #007bff !important;
                border-color: #007bff !important;
            }
            div[data-testid="stRadio"] input[type="radio"]:checked::before {
                background-color: #007bff !important;
            }
            /* 라디오 버튼 호버 시 동그라미 색상 */
            div[data-testid="stRadio"] input[type="radio"]:hover {
                border-color: #007bff !important;
            }
            
            /* 모든 Streamlit 컴포넌트 텍스트 크기 제한 */
            .stAlert, .stSuccess, .stError, .stWarning, .stInfo {
                font-size: 14px !important;
            }
            .stAlert > div, .stSuccess > div, .stError > div, .stWarning > div, .stInfo > div {
                font-size: 14px !important;
            }
            /* 마크다운 헤더 크기 제한 */
            h1, h2, h3, h4, h5, h6 {
                font-size: 16px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            with st.container(border=True):
                st.markdown("**[RAG 파이프라인 옵션]**")
                
                # Transform 옵션 - 라디오 버튼으로 변경
                st.markdown("**처리 모드 선택**")
                processing_mode = st.radio(
                    "처리 모드를 선택하세요:",
                    ["테스트 모드 (20개)", "KOSPI TOP 100", "전체 파일"],
                    key=f"{key_prefix}_processing_mode",
                    horizontal=True
                )
                
                # Load 옵션
                st.markdown("**임베딩 모델 선택**")
                model_type = st.radio(
                    "임베딩 모델을 선택하세요:",
                    ["e5", "kakaobank", "fine5"],
                    key=f"{key_prefix}_model_type",
                    horizontal=True,
                    help="벡터 로드 시 사용할 임베딩 모델을 선택합니다."
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

    # 로그를 최신순으로 표시 (최신 로그가 위에 오도록)
    for entry in reversed(st.session_state.etl_logs):
        # HTML escape 적용
        entry_escaped = html.escape(entry)
        
        # 모든 로그를 작게 표시
        if "✅" in entry:
            st.markdown(f"<div style='font-size: 14px; color: #28a745; margin: 1px 0;'>{entry_escaped}</div>", unsafe_allow_html=True)
        elif "❌" in entry:
            st.markdown(f"<div style='font-size: 14px; color: #dc3545; margin: 1px 0;'>{entry_escaped}</div>", unsafe_allow_html=True)
        elif "⚠️" in entry:
            st.markdown(f"<div style='font-size: 14px; color: #ffc107; margin: 1px 0;'>{entry_escaped}</div>", unsafe_allow_html=True)
        elif "--- STDOUT ---" in entry or "--- STDERR ---" in entry:
            st.markdown(f"<div style='font-size: 12px; background-color: #f8f9fa; padding: 4px; border-radius: 2px; margin: 1px 0; font-family: monospace; white-space: pre-wrap; line-height: 1.2;'>{entry_escaped}</div>", unsafe_allow_html=True)
        elif "✔parser:" in entry or "✔normalized:" in entry or "✔final:" in entry:
            st.markdown(f"<div style='font-size: 12px; color: #28a745; margin: 1px 0; font-family: monospace;'>{entry_escaped}</div>", unsafe_allow_html=True)
        elif entry.strip().startswith('#') or entry.strip().startswith('##') or entry.strip().startswith('###'):
            # 마크다운 헤더로 해석되지 않도록 작은 텍스트로 표시
            st.markdown(f"<div style='font-size: 12px; color: #17a2b8; margin: 1px 0; font-family: monospace;'>{entry_escaped}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='font-size: 14px; color: #17a2b8; margin: 1px 0;'>{entry_escaped}</div>", unsafe_allow_html=True)


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
        elif key_prefix == "rag":
            success = _run_rag_transform(pipeline_label)
    elif step == "Load":
        if key_prefix == "finetune":
            success = _run_finetune_load(pipeline_label)
        elif key_prefix == "rag":
            success = _run_rag_load(pipeline_label)
    if not success:
        return False
    _log_step(step, pipeline_label)
    _render_step_outputs(key_prefix, step)
    return True


def _run_rag_extract(pipeline_label: str) -> bool:
    try:
        # RAG Extract: API Pull, Extractor, KOSPI 크롤링, KOSPI Map 빌드 순차 실행
        _log_info(pipeline_label, "RAG Extract 단계 시작...")
        
        # 1. API Pull - 파인튜닝처럼 직접 함수 호출로 변경
        _log_info(pipeline_label, "API Pull 실행 중...")
        
        # API Pull 로그를 실시간으로 표시하기 위한 컨테이너
        log_container = st.container()
        with log_container:
            st.markdown(f"<div style='font-size: 14px; color: #17a2b8; background-color: #e7f3ff; padding: 8px; border-radius: 4px; margin: 4px 0;'>{html.escape('🚀 DART API 다운로더 시작')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 14px; color: #17a2b8; background-color: #e7f3ff; padding: 8px; border-radius: 4px; margin: 4px 0;'>{html.escape('📋 실행 순서: [\'list.json\', \'document.xml\', \'retry_failed\']')}</div>", unsafe_allow_html=True)
        
        with st.spinner("API Pull 실행 중..."):
            # API Pull을 단계별로 실행하여 진행상황 표시
            try:
                from service.etl.extractor.api_pull import DartDownloader
                
                downloader = DartDownloader()
                
                # 1단계: list.json
                with log_container:
                    st.info("🔄 1단계: list.json 실행 중...")
                downloader.download_list()
                with log_container:
                    st.success("✅ 1단계 완료: list.json")
                _log_success(pipeline_label, "API Pull 1단계 완료: list.json")
                
                # 2단계: document.xml
                with log_container:
                    st.info("🔄 2단계: document.xml 실행 중...")
                downloader.download_all_documents()
                with log_container:
                    st.success("✅ 2단계 완료: document.xml")
                _log_success(pipeline_label, "API Pull 2단계 완료: document.xml")
                
                # 3단계: retry_failed
                with log_container:
                    st.info("🔄 3단계: retry_failed 실행 중...")
                downloader.retry_failed_downloads()
                with log_container:
                    st.success("✅ 3단계 완료: retry_failed")
                _log_success(pipeline_label, "API Pull 3단계 완료: retry_failed")
                
                with log_container:
                    st.success("🎉 API Pull 모든 단계 완료!")
                _log_success(pipeline_label, "API Pull 모든 단계 완료!")
                        
            except Exception as e:
                with log_container:
                    st.error(f"❌ API Pull 실행 실패: {e}")
                _log_error("API Pull", pipeline_label, e)
                raise Exception(f"API Pull 실행 실패: {e}")
        
        # 2. Extractor
        _log_info(pipeline_label, "Extractor 실행 중...")
        with st.spinner("Extractor 실행 중..."):
            try:
                with log_container:
                    st.info("🔄 Extractor 실행 중...")
                
                result = subprocess.run([
                    sys.executable, str(APP_ROOT / "service" / "etl" / "extractor" / "extractor.py")
                ], capture_output=True, text=True, cwd=str(APP_ROOT))
                
                # 실행 결과를 로그 영역에 표시
                with log_container:
                    st.write("**실행 명령어**: `python service/etl/extractor/extractor.py`")
                    st.write(f"**반환 코드**: {result.returncode}")
                    
                    if result.stdout:
                        st.markdown(f"<div style='font-size: 12px; background-color: #f8f9fa; padding: 8px; border-radius: 4px; font-family: monospace; white-space: pre-wrap;'>{html.escape(result.stdout)}</div>", unsafe_allow_html=True)
                    
                    if result.stderr:
                        st.warning(f"stderr: {html.escape(result.stderr)}")
                
                if result.returncode == 0:
                    with log_container:
                        st.success("✅ Extractor 완료!")
                    _log_success(pipeline_label, "Extractor 완료", result.stdout, result.stderr)
                else:
                    with log_container:
                        st.error(f"❌ Extractor 실패: {html.escape(result.stderr)}")
                    _log_error("Extractor", pipeline_label, Exception(f"반환 코드: {result.returncode}"), result.stdout, result.stderr)
                    raise Exception(f"Extractor 실패: {html.escape(result.stderr)}")
                        
            except Exception as e:
                with log_container:
                    st.error(f"❌ Extractor 실행 실패: {e}")
                _log_error("Extractor", pipeline_label, e)
                raise Exception(f"Extractor 실행 실패: {e}")
        
        # 3. KOSPI 크롤링
        _log_info(pipeline_label, "KOSPI Top 크롤링 실행 중...")
        with st.spinner("KOSPI Top 크롤링 실행 중..."):
            with log_container:
                st.info("🔄 KOSPI Top 크롤링 실행 중...")
            do_crawl()
            with log_container:
                st.success("✅ KOSPI Top 크롤링 완료!")
        
        # 4. KOSPI Map 빌드
        _log_info(pipeline_label, "KOSPI Map 빌드 실행 중...")
        with st.spinner("KOSPI Map 빌드 실행 중..."):
            try:
                result = subprocess.run([
                    sys.executable, str(APP_ROOT / "service" / "etl" / "extractor" / "build_kospi_map.py")
                ], capture_output=True, text=True, cwd=str(APP_ROOT))
                
                # 실행 결과를 로그 영역에 표시
                with log_container:
                    st.write("**실행 명령어**: `python service/etl/extractor/build_kospi_map.py`")
                    st.write(f"**반환 코드**: {result.returncode}")
                    
                    if result.stdout:
                        st.markdown(f"<div style='font-size: 12px; background-color: #f8f9fa; padding: 8px; border-radius: 4px; font-family: monospace; white-space: pre-wrap;'>{html.escape(result.stdout)}</div>", unsafe_allow_html=True)
                    
                    if result.stderr:
                        st.warning(f"stderr: {html.escape(result.stderr)}")
                
                # 실행 로그에도 기록
                if result.returncode == 0:
                    _log_success(pipeline_label, "KOSPI Map 빌드 완료", result.stdout, result.stderr)
                else:
                    _log_error("KOSPI Map 빌드", pipeline_label, Exception(f"반환 코드: {result.returncode}"), result.stdout, result.stderr)
                    raise Exception(f"KOSPI Map 빌드 실패: {html.escape(result.stderr)}")
                        
            except Exception as e:
                with log_container:
                    st.error(f"❌ KOSPI Map 빌드 실행 실패: {e}")
                _log_error("KOSPI Map 빌드", pipeline_label, e)
                raise Exception(f"KOSPI Map 빌드 실행 실패: {e}")
        
        with log_container:
            st.success("🎉 RAG Extract 모든 단계 완료!")
    except Exception as exc:  # noqa: BLE001
        _log_error("Extract", pipeline_label, exc)
        st.error(f"RAG Extract 단계 실행 중 오류가 발생했습니다: {exc}")
        return False
    return True

def _run_rag_transform(pipeline_label: str) -> bool:
    try:
        _log_info(pipeline_label, "RAG Transform 단계 시작...")
        
        # 옵션 가져오기
        processing_mode = st.session_state.get("rag_processing_mode", "테스트 모드 (20개)")
        
        # Transform 로그를 실시간으로 표시하기 위한 컨테이너
        log_container = st.container()
        with log_container:
            st.info("🚀 Transform Pipeline 시작")
            st.info(f"📋 처리 모드: {processing_mode}")
        
        # Pipeline 실행
        with st.spinner("Transform Pipeline 실행 중..."):
            try:
                # Transform Pipeline을 subprocess로 실행하여 로그 캡처
                cmd = [sys.executable, str(APP_ROOT / "service" / "etl" / "transform" / "pipeline.py")]
                
                if processing_mode == "전체 파일":
                    cmd.append("--all")
                elif processing_mode == "KOSPI TOP 100":
                    cmd.append("--kospi-only")
                
                # 1단계: Parser
                with log_container:
                    st.info("🔄 1단계: Parser 실행 중...")
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(APP_ROOT))
                
                if result.returncode == 0:
                    with log_container:
                        st.success("✅ 1단계 완료: Parser")
                    _log_success(pipeline_label, "Parser 완료", result.stdout, result.stderr)
                else:
                    with log_container:
                        st.error(f"❌ Parser 실패: {html.escape(result.stderr)}")
                    _log_error("Parser", pipeline_label, Exception(f"반환 코드: {result.returncode}"), result.stdout, result.stderr)
                    raise Exception(f"Parser 실패: {html.escape(result.stderr)}")
                
                # 2단계: Normalizer
                with log_container:
                    st.info("🔄 2단계: Normalizer 실행 중...")
                with log_container:
                    st.success("✅ 2단계 완료: Normalizer")
                _log_success(pipeline_label, "Normalizer 완료")
                
                # 3단계: Chunker
                with log_container:
                    st.info("🔄 3단계: Chunker 실행 중...")
                with log_container:
                    st.success("✅ 3단계 완료: Chunker")
                _log_success(pipeline_label, "Chunker 완료")
                
                with log_container:
                    st.success("🎉 Transform Pipeline 모든 단계 완료!")
                _log_success(pipeline_label, "Transform Pipeline 모든 단계 완료!")
                    
            except Exception as e:
                with log_container:
                    st.error(f"❌ Transform Pipeline 실행 실패: {e}")
                _log_error("Transform Pipeline", pipeline_label, e)
                raise Exception(f"Transform Pipeline 실행 실패: {e}")
        
        _log_info(pipeline_label, "RAG Transform 단계 완료")
    except Exception as exc:  # noqa: BLE001
        _log_error("Transform", pipeline_label, exc)
        st.error(f"RAG Transform 단계 실행 중 오류가 발생했습니다: {exc}")
        return False
    return True


def _run_rag_load(pipeline_label: str) -> bool:
    try:
        _log_info(pipeline_label, "RAG Load 단계 시작...")
        
        # 옵션 가져오기
        model_type = st.session_state.get("rag_model_type", "e5")
        
        # Load 로그를 실시간으로 표시하기 위한 컨테이너
        log_container = st.container()
        with log_container:
            st.info("🚀 Load 워크플로우 시작")
            st.info(f"📋 임베딩 모델: {model_type}")
        
        # Load 워크플로우 순차 실행
        steps = [
            ("Docker 시작", "docker-compose up -d"),
            ("DB 연결 테스트", "loader_cli.py db test"),
            ("스키마 생성", "loader_cli.py db create"),
            ("테이블 목록 확인", "loader_cli.py db list"),
            ("모델 다운로드", "loader_cli.py download"),
            ("문서 로드", "loader_cli.py load doc"),
            ("벡터 로드", f"loader_cli.py load vector --model {model_type}")
        ]
        
        for i, (step_name, cmd_desc) in enumerate(steps, 1):
            with log_container:
                st.info(f"🔄 {i}단계: {step_name} 실행 중...")
            
            with st.spinner(f"{step_name} 실행 중..."):
                try:
                    if step_name == "Docker 시작":
                        result = subprocess.run(["docker-compose", "up", "-d"], 
                                              capture_output=True, text=True, cwd=str(APP_ROOT))
                    else:
                        result = subprocess.run([
                            sys.executable, str(APP_ROOT / "service" / "etl" / "loader" / "loader_cli.py")
                        ] + cmd_desc.split()[1:], 
                        capture_output=True, text=True, cwd=str(APP_ROOT))
                    
                    if result.returncode != 0:
                        with log_container:
                            st.error(f"❌ {step_name} 실패: {html.escape(result.stderr)}")
                        raise Exception(f"{step_name} 실패: {html.escape(result.stderr)}")
                    else:
                        with log_container:
                            st.success(f"✅ {i}단계 완료: {step_name}")
                        
                except Exception as e:
                    with log_container:
                        st.error(f"❌ {step_name} 실행 중 오류: {e}")
                    raise Exception(f"{step_name} 실행 중 오류: {e}")
        
        with log_container:
            st.success("🎉 Load 워크플로우 모든 단계 완료!")
        
        _log_info(pipeline_label, "RAG Load 단계 완료")
    except Exception as exc:  # noqa: BLE001
        _log_error("Load", pipeline_label, exc)
        st.error(f"RAG Load 단계 실행 중 오류가 발생했습니다: {exc}")
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


def _log_info(pipeline_label: str, message: str, stdout: str = None, stderr: str = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.etl_logs.append(f"[{timestamp}] [{pipeline_label}] {message}")
    if stdout:
        st.session_state.etl_logs.append(f"--- STDOUT ---\n{stdout}\n--------------")
    if stderr:
        st.session_state.etl_logs.append(f"--- STDERR ---\n{stderr}\n--------------")

def _log_success(pipeline_label: str, message: str, stdout: str = None, stderr: str = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.etl_logs.append(f"[{timestamp}] [{pipeline_label}] ✅ {message}")
    if stdout:
        st.session_state.etl_logs.append(f"--- STDOUT ---\n{stdout}\n--------------")
    if stderr:
        st.session_state.etl_logs.append(f"--- STDERR ---\n{stderr}\n--------------")

def _log_warning(pipeline_label: str, message: str, stdout: str = None, stderr: str = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.etl_logs.append(f"[{timestamp}] [{pipeline_label}] ⚠️ {message}")
    if stdout:
        st.session_state.etl_logs.append(f"--- STDOUT ---\n{stdout}\n--------------")
    if stderr:
        st.session_state.etl_logs.append(f"--- STDERR ---\n{stderr}\n--------------")

def _log_error(step: str, pipeline_label: str, error: Exception, stdout: str = None, stderr: str = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.etl_logs.append(
        f"[{timestamp}] [{pipeline_label}] ❌ {step} 단계 실행 중 오류 발생: {error}"
    )
    if stdout:
        st.session_state.etl_logs.append(f"--- STDOUT ---\n{stdout}\n--------------")
    if stderr:
        st.session_state.etl_logs.append(f"--- STDERR ---\n{stderr}\n--------------")
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

'''llama factory에 투입할 공시 해석용 train/test 데이터를 생성하는 스크립트'''
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv


########################
# 로그 설정
########################
logging.basicConfig(level=logging.INFO)

# 기본 설정 (환경변수로 오버라이드 가능)
DEFAULT_SOURCE_CSV = Path("./data/chunked_data.csv")
DEFAULT_MARKDOWN_DIR = Path("./data/markdown")
DEFAULT_OUTPUT_PATH = Path("./data/csv2json.json")
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_TASK_NAME = "disclosure_commentary"
DEFAULT_SYSTEM_PROMPT = (
    "너는 한국어 투자 애널리스트다. 제공된 공시/리포트 컨텍스트를 읽고 "
    "1) 핵심 투자 포인트, 2) 리스크, 3) 밸류에이션 관점, 4) 전망 및 가이던스를 "
    "간결하게 정리한다. 사실에 근거하고 과장하거나 새 정보를 만들지 마라."
)


def load_markdown_snippet(
    markdown_dir: Path,
    title: str,
    date_str: str,
    char_limit: int,
) -> Tuple[Optional[str], Optional[dict], Optional[str]]:
    """공시 원문 후보에서 요약용 발췌와 메타데이터를 반환한다."""
    if not markdown_dir.exists():
        return None, None, None

    normalized_date = re.sub(r"[^0-9]", "", date_str or "")
    candidates = []
    if normalized_date:
        candidates.extend(sorted(markdown_dir.glob(f"{normalized_date}*.md")))
    if not candidates:
        # 날짜로 찾지 못한 경우 제목 키워드 일부로 검색 (느리므로 간략화)
        safe_title = re.sub(r"[\\s/]+", "_", title.strip())[:20]
        if safe_title:
            candidates.extend(sorted(markdown_dir.glob(f"*{safe_title}*.md")))

    if not candidates:
        return None, None, None

    # 날짜 기준 첫 파일 우선, 제목 일치 시 해당 파일 선택
    chosen = candidates[0]
    for path in candidates:
        try:
            text_preview = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if title and title in text_preview:
            chosen = path
            break

    try:
        raw_text = chosen.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None, None, str(chosen)

    metadata: Optional[dict] = None
    content = raw_text
    if raw_text.startswith("---"):
        try:
            _, front_matter, body = raw_text.split("---", 2)
            metadata = yaml.safe_load(front_matter) or {}
            content = body
        except (ValueError, yaml.YAMLError):
            pass

    snippet = content.strip()
    if char_limit and len(snippet) > char_limit:
        snippet = snippet[:char_limit].rstrip() + " ..."

    return snippet or None, metadata, str(chosen)


def normalize(value: object) -> str:
    """pandas NA 등 비어있는 값을 안전하게 문자로 변환."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def condense_text(text: str, char_limit: int, max_paragraphs: int = 3, max_sentences: int = 8) -> str:
    """
    긴 본문을 정리해 모델 입력 길이를 제한한다.
    - 상위 몇 개 문단만 사용하고
    - 문장 수를 제한하며
    - 최종 길이가 char_limit을 넘으면 잘라낸다.
    """
    cleaned = text.strip()
    if not cleaned:
        return ""

    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    selected_sentences = []
    for paragraph in paragraphs[:max_paragraphs]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
        for sentence in sentences:
            selected_sentences.append(sentence)
            if len(selected_sentences) >= max_sentences:
                break
        if len(selected_sentences) >= max_sentences:
            break

    condensed = " ".join(selected_sentences) if selected_sentences else cleaned
    if len(condensed) > char_limit:
        truncated = condensed[:char_limit].rstrip()
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        condensed = truncated.rstrip() + " ..."
    return condensed


def extract_field_from_chunk(text: str, field: str) -> str:
    """chunk_text에서 '제목:', '부제:' 등 라벨 뒤의 텍스트를 추출."""
    pattern = rf"{field}\s*[:：]\s*(.*?)(?=\n\S+:|$)"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_body_from_chunk(text: str) -> str:
    """chunk_text에서 본문을 추출."""
    body_match = re.search(r"본문\s*[:：]?\s*(.+)", text, flags=re.DOTALL)
    if body_match:
        return body_match.group(1).strip()
    parts = re.split(r"제목\s*[:：]", text)
    if len(parts) > 1:
        return parts[-1].strip()
    return text.strip()


def call_gpt_summary(
    api_key: str,
    prompt: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 2,
    retry_backoff: float = 2.0,
    log_prefix: str = "",
) -> str:
    """OpenAI 호환 API를 호출해 요약을 생성."""
    endpoint = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    attempt = 0
    while True:
        attempt += 1
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]["content"]
            return message.strip()
        except requests.HTTPError as http_err:
            status = http_err.response.status_code if http_err.response else "N/A"
            error_body = http_err.response.text if http_err.response else ""
            logging.warning(
                "%s요약 API 호출 실패(%s차 시도, status=%s): %s",
                f"[{log_prefix}] " if log_prefix else "",
                attempt,
                status,
                error_body[:200],
            )
            if attempt >= max_retries or status in (400, 401, 403, 404):
                raise
        except Exception as exc:  # pragma: no cover
            logging.warning(
                "%s요약 API 호출 예외(%s차 시도): %s",
                f"[{log_prefix}] " if log_prefix else "",
                attempt,
                exc,
            )
            if attempt >= max_retries:
                raise
        time.sleep(retry_backoff * attempt)


def build_prompt(context: str) -> str:
    """모델에 전달할 프롬프트를 구성."""
    return (
        "아래는 공시 또는 애널리스트 리포트에서 추출한 컨텍스트이다.\n"
        "투자 관점에서 핵심을 정리하라.\n\n"
        f"{context}"
    )


def convert():
    load_dotenv()

    source_csv = Path(os.getenv("CSV2JSON_SOURCE_CSV", str(DEFAULT_SOURCE_CSV)))
    markdown_dir = Path(os.getenv("CSV2JSON_MARKDOWN_DIR", str(DEFAULT_MARKDOWN_DIR)))
    output_path = Path(os.getenv("CSV2JSON_OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH)))
    model_name = os.getenv("CSV2JSON_SUMMARY_MODEL", DEFAULT_MODEL_NAME)
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY")
    temperature = float(os.getenv("CSV2JSON_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("CSV2JSON_MAX_OUTPUT_TOKENS", "512"))
    max_retries = int(os.getenv("CSV2JSON_MAX_RETRIES", "3"))
    retry_backoff = float(os.getenv("CSV2JSON_RETRY_BACKOFF", "1.0"))
    context_char_limit = int(os.getenv("CSV2JSON_CONTEXT_CHAR_LIMIT", "2500"))
    rag_char_limit = int(os.getenv("CSV2JSON_RAG_CHAR_LIMIT", "1800"))
    sleep_sec = float(os.getenv("CSV2JSON_REQUEST_INTERVAL", "0.05"))
    max_rows_env = os.getenv("CSV2JSON_MAX_ROWS")
    max_rows = int(max_rows_env) if max_rows_env else None

    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    df = pd.read_csv(source_csv)
    logging.info("불러온 문서의 행 수: %s", len(df))

    row_offset_env = os.getenv("CSV2JSON_ROW_OFFSET")
    row_offset = int(row_offset_env) if row_offset_env else 0
    if row_offset < 0:
        raise ValueError("CSV2JSON_ROW_OFFSET은 0 이상의 정수여야 합니다.")

    end_idx = row_offset + max_rows if max_rows else None
    subset_df = df.iloc[row_offset:end_idx]
    if subset_df.empty:
        logging.warning(
            "처리할 데이터가 없습니다. row_offset=%s, max_rows=%s", row_offset, max_rows
        )
        return

    total_target = len(subset_df)
    logging.info(
        "처리 범위: 원본 인덱스 %s~%s (총 %s건)",
        row_offset + 1,
        row_offset + total_target,
        total_target,
    )
    record_iter = list(enumerate(subset_df.to_dict(orient="records"), start=row_offset))

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env를 확인하세요.")

    records = []
    skipped = 0
    processed = 0

    for global_idx, row in record_iter:
        title = normalize(row.get("제목"))
        subtitle = normalize(row.get("부제"))
        date = normalize(row.get("날짜"))
        analyst = normalize(row.get("작성자"))
        category = normalize(row.get("카테고리"))
        url = normalize(row.get("링크"))
        body = normalize(row.get("본문"))

        chunk_text = normalize(row.get("chunk_text"))
        if chunk_text:
            parsed_title = extract_field_from_chunk(chunk_text, "제목")
            parsed_subtitle = extract_field_from_chunk(chunk_text, "부제")
            parsed_date = extract_field_from_chunk(chunk_text, "날짜")
            parsed_analyst = extract_field_from_chunk(chunk_text, "작성자")
            parsed_category = extract_field_from_chunk(chunk_text, "카테고리")
            parsed_url = extract_field_from_chunk(chunk_text, "링크")
            parsed_body = extract_body_from_chunk(chunk_text)

            title = parsed_title or title
            subtitle = parsed_subtitle or subtitle
            date = parsed_date or date
            analyst = parsed_analyst or analyst
            category = parsed_category or category
            url = parsed_url or url
            body = parsed_body or body

        if not body:
            skipped += 1
            logging.debug(
                "본문이 비어 있어 건너뜀 (row=%s, title=%s)", global_idx, title
            )
            continue

        trimmed_body = condense_text(
            text=body,
            char_limit=context_char_limit,
        )
        if trimmed_body != body:
            logging.debug("본문 축약 적용(row=%s, title=%s)", global_idx, title)

        rag_snippet, rag_meta, rag_path = load_markdown_snippet(
            markdown_dir=markdown_dir,
            title=title,
            date_str=date,
            char_limit=rag_char_limit,
        )

        context_parts = [
            f"제목: {title or 'N/A'}",
            f"부제: {subtitle or 'N/A'}",
            f"날짜: {date or 'N/A'}",
            f"작성자: {analyst or 'N/A'}",
            f"카테고리: {category or 'N/A'}",
            f"링크: {url or 'N/A'}",
            "",
            "[애널리스트 코멘트/본문]",
            trimmed_body,
        ]

        if rag_snippet:
            context_parts.extend(
                [
                    "",
                    "[관련 공시(RAG) 발췌]",
                    rag_snippet,
                ]
            )

        prompt = build_prompt("\n".join(context_parts))

        try:
            summary = call_gpt_summary(
                api_key=api_key,
                prompt=prompt,
                model=model_name,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
                log_prefix=f"row={global_idx}",
            )
        except requests.HTTPError as http_err:
            logging.error(
                "요약 생성 실패(row=%s, title=%s): %s", global_idx, title, http_err
            )
            skipped += 1
            continue
        except Exception as exc:  # pragma: no cover - 방어 코드
            logging.exception(
                "요약 생성 중 예외(row=%s, title=%s): %s", global_idx, title, exc
            )
            skipped += 1
            continue

        meta = {
            "제목": title,
            "부제": subtitle,
            "날짜": date,
            "작성자": analyst,
            "카테고리": category,
            "링크": url,
            "report_id": f"report_{global_idx + 1:04d}",
            "chunk_id": f"chunk_{global_idx + 1:04d}",
        }

        if rag_path:
            meta["rag_source_path"] = rag_path
        if rag_meta:
            meta["rag_meta"] = {k: normalize(v) for k, v in rag_meta.items()}

        source_value = normalize(row.get("source"))
        if source_value:
            meta["source"] = source_value
        source_row = normalize(row.get("row"))
        if source_row:
            meta["source_row"] = source_row

        record = {
            "instruction": "공시/리포트 컨텍스트를 분석해 리스크·밸류에이션·가이던스를 정리해줘.",
            "input": "\n".join(context_parts),
            "output": summary,
            "meta": meta,
            "task": DEFAULT_TASK_NAME,
        }

        records.append(record)
        processed += 1
        if processed % 10 == 0:
            logging.info(
                "LLM - input & output 생성 진행률: %s/%s", processed, total_target
            )
        logging.debug(
            "record 생성 완료(row=%s, report_id=%s)", global_idx, meta["report_id"]
        )

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logging.info("생성된 레코드 수: %s → %s", len(records), output_path)
    if skipped:
        logging.info("생성에서 제외된 행 수: %s", skipped)


if __name__ == "__main__":
    convert()

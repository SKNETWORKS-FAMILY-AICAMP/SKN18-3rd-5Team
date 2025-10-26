import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, TypedDict

from graph.state import QAState
from service.rag.query.temporal_parser import TemporalQueryParser, TemporalInfo

# 사용자의 자연어 질문을 받아서, 내부적으로 검색 효율을 높이고자
# 시점(날짜), 티커(종목코드), 회사명 등과 같은 메타데이터를 추출/보강하여
# 개선된 질의문을 생성하는 QueryRewrite 노드의 핵심 로직을 담고 있습니다.
#
# 주요 역할:
# - 사용자의 입력 질문에서 기업명/티커/보고서 정보 추출 및 alias 변환
# - 데이터 기반으로 최신 보고서의 날짜 및 종목 코드 정보를 결합하여 질의를 증강
# - 재작성된 질의를 QA 파이프라인의 다음 단계로 전달
#
# 구현 참고:
# - 상단 run(state)는 LangGraph 노드의 입구 역할로 state 딕셔너리의 "question"을 받아 재작성 후 반환합니다.
# - _rewrite_with_metadata 및 관련 유틸 함수들은 질의에서의 정보 인식/결합, 날짜/회사명 처리에 특화되어 있습니다.
# - 실제 기업 메타정보는 data/ 디렉토리의 JSON 파일로 관리되며, 데이터 갱신이 발생하면 가장 최신 일자를 자동 감지합니다.

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_DATE_FILE_PATTERN = re.compile(r"(\d{8})\.json$")
_TIME_HINT_PATTERN = re.compile(
    r"\b\d{4}\s*년|\b\d{2}\s*년|\b\d{4}[-./]\d{1,2}|\b\d{8}\b|분기|반기|최근|yoy|qoq",
    re.IGNORECASE,
)


class AliasEntry(TypedDict):
    info: Dict[str, str]
    no_space: str


@lru_cache(maxsize=1)
def _get_temporal_parser() -> TemporalQueryParser:
    """TemporalQueryParser 인스턴스를 캐싱해 재사용"""
    return TemporalQueryParser()


def run(state: QAState) -> QAState:
    """질문에서 기업/시간 정보를 추출해 재작성된 질의를 state에 저장"""
    q = (state.get("question") or "").strip()
    temporal_info: Optional[TemporalInfo] = None
    if q:
        temporal_info = _get_temporal_parser().parse(q)
    rewritten = _rewrite_with_metadata(q, temporal_info)
    state["rewritten_query"] = rewritten
    return state


def _rewrite_with_metadata(question: str, temporal_info: Optional[TemporalInfo]) -> str:
    """기업/시간 메타데이터를 덧붙여 검색 효율이 높은 질의를 생성"""
    if not question:
        return ""

    matched = _match_companies(question)
    temporal_chunks = _temporal_chunks(temporal_info)

    meta_chunks: List[str] = []
    if temporal_chunks:
        meta_chunks.extend(temporal_chunks)

    if not matched and not meta_chunks:
        return question

    for info in matched:
        ticker = info.get("stock_code")
        report_date = _format_report_date(info.get("rcept_dt", ""))
        report_name = info.get("report_nm", "")

        details: List[str] = []
        if ticker:
            details.append(f"티커 {ticker}")
        if report_date:
            details.append(f"최신 공시일 {report_date}")
        if report_name:
            details.append(report_name)

        if details:
            meta_chunks.append(f"{info['name']} ({', '.join(details)})")
        else:
            meta_chunks.append(info["name"])

    if not meta_chunks:
        return question

    terminal = "" if question.endswith((".", "?", "!")) else "."
    has_time_hint = bool(_TIME_HINT_PATTERN.search(question))
    prefix = "최신 공시 기준 " if not has_time_hint else ""
    rewritten = (
        f"{question}{terminal} {prefix}관련 티커·시점 정보: {' / '.join(meta_chunks)}."
    )
    return rewritten.strip()


def _temporal_chunks(temporal_info: Optional[TemporalInfo]) -> List[str]:
    """TemporalInfo 데이터를 사람이 읽을 수 있는 요약 문자열 리스트로 변환"""
    if temporal_info is None:
        return []

    chunks: List[str] = []
    if temporal_info.years:
        years = ", ".join(str(y) for y in sorted(set(temporal_info.years)))
        chunks.append(f"연도 {years}")
    if temporal_info.quarters:
        quarters = ", ".join(f"{q}분기" for q in sorted(set(temporal_info.quarters)))
        chunks.append(f"분기 {quarters}")
    if temporal_info.relative:
        chunks.append(f"상대 시점 {temporal_info.relative}")
    if temporal_info.date_range:
        start = temporal_info.date_range.get("start")
        end = temporal_info.date_range.get("end")
        if start or end:
            chunks.append(f"기간 {start or '?'}~{end or '?'}")
    return chunks


def _match_companies(question: str) -> List[Dict[str, str]]:
    """질문에서 기업명·별칭·티커를 추출해 최신 공시 정보와 매핑"""
    alias_map, stock_map = _corp_lookup()
    question_cf = question.casefold()
    question_compact = re.sub(r"\s+", "", question_cf)

    matches: List[Dict[str, str]] = []
    seen: Set[str] = set()

    # 6자리 숫자 티커 직접 매칭
    for code in set(re.findall(r"\b\d{6}\b", question)):
        info = stock_map.get(code)
        if not info:
            continue
        key = info.get("stock_code") or info.get("name")
        if key and key not in seen:
            matches.append(info)
            seen.add(key)

    # 회사명/별칭 매칭
    for alias_cf, payload in alias_map.items():
        info = payload["info"]
        alias_no_space = payload["no_space"]
        key = info.get("stock_code") or info.get("name")
        if not key or key in seen:
            continue
        if alias_cf and alias_cf in question_cf:
            matches.append(info)
            seen.add(key)
            continue
        if alias_no_space and alias_no_space in question_compact:
            matches.append(info)
            seen.add(key)

    return matches


@lru_cache(maxsize=1)
def _corp_lookup() -> Tuple[Dict[str, AliasEntry], Dict[str, Dict[str, str]]]:
    """기업 alias/티커 맵을 최신 JSON 데이터에서 로딩해 캐싱"""
    alias_map: Dict[str, AliasEntry] = {}
    stock_map: Dict[str, Dict[str, str]] = {}

    if not _DATA_DIR.exists():
        return alias_map, stock_map

    latest_file = _latest_corp_file()
    if latest_file is None:
        return alias_map, stock_map

    try:
        with latest_file.open(encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return alias_map, stock_map

    corp_records: Dict[str, Dict[str, str]] = {}
    for item in payload.get("list", []):
        corp_name = (item.get("corp_name") or "").strip()
        if not corp_name:
            continue
        corp_code = (item.get("corp_code") or "").strip()
        key = corp_code or corp_name
        stock_code = (item.get("stock_code") or "").strip()
        rcept_dt = (item.get("rcept_dt") or "").strip()
        report_nm = (item.get("report_nm") or "").strip()

        candidate = {
            "name": corp_name,
            "stock_code": stock_code,
            "rcept_dt": rcept_dt,
            "report_nm": report_nm,
        }
        existing = corp_records.get(key)
        if existing is None or _is_newer(rcept_dt, existing.get("rcept_dt", "")):
            corp_records[key] = candidate

    records = sorted(corp_records.values(), key=lambda r: r["name"])

    for record in records:
        aliases = _generate_aliases(record["name"])
        record_entry = dict(record)

        if record_entry["stock_code"]:
            stock_map.setdefault(record_entry["stock_code"], record_entry)

        for alias in aliases:
            alias_cf = alias.casefold()
            alias_map.setdefault(
                alias_cf,
                {
                    "info": record_entry,
                    "no_space": re.sub(r"\s+", "", alias_cf),
                },
            )

    return alias_map, stock_map


def _latest_corp_file() -> Optional[Path]:
    """data 디렉터리에서 날짜가 가장 최신인 기업 메타 JSON 파일 반환"""
    latest_path: Optional[Path] = None
    latest_dt: Optional[datetime] = None

    for path in _DATA_DIR.glob("*.json"):
        match = _DATE_FILE_PATTERN.match(path.name)
        if not match:
            continue
        try:
            file_dt = datetime.strptime(match.group(1), "%Y%m%d")
        except ValueError:
            continue
        if latest_dt is None or file_dt > latest_dt:
            latest_dt = file_dt
            latest_path = path

    return latest_path


def _generate_aliases(name: str) -> List[str]:
    """기업명을 공백/법적 표기 제거 등으로 정규화한 alias 목록 생성"""
    base = name.replace("\u3000", " ").strip()
    variants = {base}

    cleaned = base
    for token in ("주식회사", "(주)", "㈜"):
        cleaned = cleaned.replace(token, "")
    variants.add(cleaned.strip())

    no_space_base = re.sub(r"\s+", "", base)
    variants.add(no_space_base)

    no_space_cleaned = re.sub(r"\s+", "", cleaned)
    variants.add(no_space_cleaned)

    variants = {variant for variant in variants if variant}
    return sorted(variants, key=lambda v: (-len(v), v))


def _is_newer(candidate: str, baseline: str) -> bool:
    """두 날짜 문자열을 비교해 candidate가 기준보다 최신인지 판단"""
    candidate_dt = _parse_date(candidate)
    baseline_dt = _parse_date(baseline)

    if candidate_dt and baseline_dt:
        return candidate_dt >= baseline_dt
    if candidate_dt and not baseline_dt:
        return True
    return False


def _parse_date(date_str: str) -> Optional[datetime]:
    """YYYYMMDD 문자열을 datetime 객체로 파싱"""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y%m%d")
    except (ValueError, TypeError):
        return None


def _format_report_date(date_str: str) -> str:
    """YYYYMMDD 공시일을 사람이 읽기 쉬운 YYYY-MM-DD 형태로 변환"""
    parsed = _parse_date(date_str)
    return parsed.strftime("%Y-%m-%d") if parsed else date_str

#!/usr/bin/env python3
"""
- data/kospi_top100.txt 를 읽어 각 줄의 종목명 추출
- data/20251017.json 을 읽어 corp_name ↔ corp_code 매핑 생성
- 종목명을 corp_name 과 매칭해 { 종목명: corp_code } 형태로
  data/kospi_top100_map.json 파일을 생성
"""

import json
from pathlib import Path
from typing import Dict
import re
from datetime import datetime


def normalize(name: str) -> str:
    # Basic normalization for matching (trim + remove BOM)
    if not name:
        return ""
    name = name.replace("\ufeff", "")
    return name.strip()


def load_corp_map(json_path: Path) -> Dict[str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping: Dict[str, str] = {}
    for item in data.get("list", []):
        corp_name = normalize(item.get("corp_name", ""))
        corp_code = item.get("corp_code")
        if corp_name and corp_code and corp_name not in mapping:
            mapping[corp_name] = corp_code
    return mapping


def main() -> None:
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    src_names = data_dir / "kospi_top100.txt"
    
    # 최신 JSON 파일 찾기
    json_files = [f for f in data_dir.glob("*.json") if not f.name.startswith('.')]
    if not json_files:
        print("❌ JSON 파일이 없습니다.")
        return
    
    # 날짜 형식 파일 중 가장 최근 파일 선택
    date_files = []
    for f in json_files:
        match = re.match(r'^(\d{8})\.json$', f.name)
        if match:
            try:
                date_str = match.group(1)
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                date_files.append((date_obj, f))
            except ValueError:
                continue
    
    if date_files:
        date_files.sort(key=lambda x: x[0])
        json_src = date_files[-1][1]
    else:
        json_files.sort(key=lambda x: x.name)
        json_src = json_files[-1]
    
    print(f"📁 사용할 JSON 파일: {json_src.name}")
    
    out_file = data_dir / "kospi_top100_map.json"

    corp_map = load_corp_map(json_src)
    
    # ETF/펀드 필터링 키워드
    etf_keywords = ['TIGER', 'KODEX', 'ACE', 'HANARO', 'ARIRANG', 'KBSTAR', 'TIMEFOLIO']
    
    # 회사명 매핑 테이블 (kospi_top100.txt → DART corp_name)
    name_mappings = {
        "현대차": "현대자동차",
        "한국전력": "한국전력공사",
        "KT&G": "케이티앤지",
        "LS ELECTRIC": "LS일렉트릭",
        "삼성화재": "삼성화재해상보험",
        "KT": "케이티",
        "LG화학": "LG화학",
        "SK텔레콤": "SK텔레콤",
        "POSCO홀딩스": "POSCO홀딩스",
        "NAVER": "네이버",
        "카카오": "카카오",
        "LG전자": "LG전자",
        "현대모비스": "현대모비스",
        "SK": "SK",
        "LG": "LG",
        "CJ제일제당": "CJ제일제당",
        "한국가스공사": "한국가스공사",
        "현대글로비스": "현대글로비스",
        "SK이노베이션": "SK이노베이션",
        "LG생활건강": "LG생활건강"
    }

    result: Dict[str, str] = {}
    missing: Dict[str, str] = {}

    with open(src_names, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            # skip header line
            if "종목명" in raw and "현재가" in raw:
                continue

            # capture first column name: prefer tab split; fallback to 2+ spaces
            if "\t" in raw:
                name = raw.split("\t", 1)[0].strip()
            else:
                parts = re.split(r"\s{2,}", raw, maxsplit=1)
                name = parts[0].strip() if parts else raw.strip()

            # ETF/펀드 필터링
            if any(keyword in name for keyword in etf_keywords):
                continue
                
            key_orig = name
            key_norm = normalize(name)
            
            # 1. 직접 매칭
            code = corp_map.get(key_norm)
            if code:
                result[key_orig] = code
                continue
                
            # 2. 매핑 테이블 사용
            mapped_name = name_mappings.get(key_norm)
            if mapped_name:
                code = corp_map.get(mapped_name)
                if code:
                    result[key_orig] = code
                    continue
            
            # 3. 공백 제거 매칭
            nospace_map = {k.replace(" ", ""): v for k, v in corp_map.items()}
            code2 = nospace_map.get(key_norm.replace(" ", ""))
            if code2:
                result[key_orig] = code2
                continue
                
            # 4. 우선주 처리 ('...우' 접미사 제거)
            if key_norm.endswith("우"):
                base = key_norm[:-1]
                code3 = corp_map.get(base) or nospace_map.get(base.replace(" ", ""))
                if code3:
                    result[key_orig] = code3
                    continue
                    
            # 5. 매핑 테이블 + 공백 제거
            if mapped_name:
                code4 = nospace_map.get(mapped_name.replace(" ", ""))
                if code4:
                    result[key_orig] = code4
                    continue
            
            missing[key_orig] = ""

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Written: {out_file} (matched {len(result)} names, missing {len(missing)})")
    if missing:
        print("Missing examples:")
        for k in list(missing.keys())[:10]:
            print(" -", k)


if __name__ == "__main__":
    main()


 
 
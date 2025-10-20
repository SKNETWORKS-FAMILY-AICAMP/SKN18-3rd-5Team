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
    json_src = data_dir / "20251017.json"
    out_file = data_dir / "kospi_top100_map.json"

    corp_map = load_corp_map(json_src)

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

            key_orig = name
            key_norm = normalize(name)
            code = corp_map.get(key_norm)
            if code:
                result[key_orig] = code
            else:
                # try loose match: exact after removing spaces
                nospace_map = {k.replace(" ", ""): v for k, v in corp_map.items()}
                code2 = nospace_map.get(key_norm.replace(" ", ""))
                if code2:
                    result[key_orig] = code2
                else:
                    # handle preferred stocks '...우' by stripping suffix and retry
                    if key_norm.endswith("우"):
                        base = key_norm[:-1]
                        code3 = corp_map.get(base) or nospace_map.get(base.replace(" ", ""))
                        if code3:
                            result[key_orig] = code3
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


 
 
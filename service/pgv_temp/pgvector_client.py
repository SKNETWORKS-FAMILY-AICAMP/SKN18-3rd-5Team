import csv, os
from typing import List, Dict

""" 
실제 PG 접속 대신 로컬 CSV(cleaned_shinhan_example.csv) 존재 시 그 데이터를, 
없으면 샘플 더미 데이터를 검색 결과로 돌려줍니다.

*** 실제 pgvector 연동 후 제거 ***

"""

DATA_P = "../data/cleaned_shinhan_example.csv"

DEMO_ROWS = [
  {
    "report_id": "20250315000123",
    "date": "2025-03-15",
    "title": "2025년 2차전지 산업 전망",
    "url": "https://shinhan.com/report/20250315",
    "chunk_id": "rep_demo_001",
    "chunk_text": "전기차 수요 둔화가 예상보다 완만하며 2025년 2분기부터 실적 반등 가능성이 있다."
  },
  {
    "report_id": "20250228000077",
    "date": "2025-02-28",
    "title": "반도체 업황 점검",
    "url": "https://shinhan.com/report/20250228",
    "chunk_id": "rep_demo_002",
    "chunk_text": "메모리 가격이 하반기부터 상승 전환될 것으로 예상된다. AI 서버 수요도 강세다."
  },
]

def _load_csv_if_any():
    if not os.path.exists(DATA_P):
        return DEMO_ROWS
    rows = []
    with open(DATA_P, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader, start=1):
            content = (r.get("content") or "").strip()
            if not content:
                continue
            rows.append({
                "report_id": r.get("report_id") or f"rep_{i:06d}",
                "date": r.get("date") or "2025-01-01",
                "title": r.get("title") or f"Report {i}",
                "url": r.get("url") or "https://example.com",
                "chunk_id": f"rep_{i:06d}_001",
                "chunk_text": content[:1200]
            })
    return rows

async def fetch_similar(query_vec, k=4) -> List[Dict]:
    rows = _load_csv_if_any()
    return rows[:k]

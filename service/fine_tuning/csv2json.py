'''llama factory 사용하기 전에 csv -> json으로 변환하는 파일'''
########################
# 필요한 패키지 설치
########################
import pandas as pd             # 데이터 분석
from pathlib import Path        # 파일 경로 설정
import json                     # json 형식 사용
import logging                  # 로그 출력
import re                       # 정규표현식 사용


########################
# 로그 설정
########################
logging.basicConfig(level=logging.DEBUG)  # INFO 이상만 보이게 설정


########################
# 청크 데이터 불러오기
########################
input_path = Path("./data/chunked_data.csv")
output_path =  Path("./data/csv2json.json")
df = pd.read_csv(input_path)
logging.info(f"불러온 문서의 행의 크기: {len(df)}") # 클렌징 데이터 숫자 파악


########################
# 정규식 기반 파싱 함수
########################
def extract_field(text, field):
    """
    '제목:', '부제:' 등 라벨 뒤의 텍스트를 추출
    """
    pattern = rf"{field}\s*[:：]\s*(.*?)(?=\n\S+:|$)"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_body(text):
    """
    '본문:' 이후 전체 텍스트를 추출
    - '본문' 라벨이 없거나 줄바꿈이 이상할 때도 최대한 복원
    """
    # 본문 태그를 기준으로 나누기
    body_match = re.search(r"본문\s*[:：]?\s*(.+)", text, flags=re.DOTALL)
    if body_match:
        return body_match.group(1).strip()

    # 예외: "본문:" 태그가 없을 때
    # 제목 이후 전체를 본문으로 간주
    parts = re.split(r"제목\s*[:：]", text)
    if len(parts) > 1:
        return parts[-1].strip()
    return text.strip()


########################
# csv -> json
########################
records = []

for i, row in df.iterrows():
    text = str(row.get("chunk_text", "")).strip()

    title = extract_field(text, "제목")
    subtitle = extract_field(text, "부제")
    date = extract_field(text, "날짜")
    analyst = extract_field(text, "작성자")
    category = extract_field(text, "카테고리")
    url = extract_field(text, "링크")
    body = extract_body(text)

    # 디버깅용 — 본문 누락 시 출력
    if not body:
        logging.debug(f"본문 누락 감지: {title} (row {i})")

    record = {
        "instruction": "이 리포트의 주요 내용을 요약해줘.",
        "input": f"아래는 신한리서치 리포트의 일부입니다. 이를 기반으로 핵심 내용을 요약해주세요.\n\n[컨텍스트]\n{body}",
        "output": "",
        "meta": {
            "제목": title,
            "부제": subtitle,
            "날짜": date,
            "작성자": analyst,
            "카테고리": category,
            "링크": url,
            "report_id": f"report_{i+1:04d}",
            "chunk_id": f"chunk_{i+1:04d}"
        },
        "task": "qa"
    }

    records.append(record)


########################
# 결과 저장
########################
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

logging.info(f"{len(records)}개 QA 레코드 저장됨 → {output_path}")
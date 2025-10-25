##############################
# 신한리서치 데이터 정제
##############################
import pandas as pd
import re
from pathlib import Path

# 데이터 경로
data_path = Path(__file__).resolve().parents[2] / "data" / "shinhan_research_2025_playwright.csv"
save_path = Path(__file__).resolve().parents[2] / "data" / "clean_data.csv"

##############################
# 텍스트 정제 함수
##############################
def clean_text(text):
    if pd.isna(text):
        return ""

    # 1. HTML 태그 제거
    text = re.sub(r"<[^>]+>", " ", str(text))

    # 2. 중복 공백 / 개행 제거
    text = re.sub(r"\s+", " ", text).strip()

    # 3. 특수문자 / 불필요한 기호 제거
    text = re.sub(r"[^\w\s가-힣.,!?()~\-]", "", text)

    return text.strip()

def do_cleansing():
    # 데이터 불러오기
    df = pd.read_csv(data_path)
    print(f"원본 데이터: {len(df)}개 문서")

    ##############################
    # 컬럼 정제
    ##############################
    df["본문"] = df["본문"].apply(clean_text)
    df["제목"] = df["제목"].apply(lambda x: str(x).strip())
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    ##############################
    # 결측치 / 중복 / 짧은 텍스트 제거
    ##############################
    df = df.dropna(subset=["본문"])
    df = df[df["본문"].str.len() > 100]
    df = df.drop_duplicates(subset=["제목"])

    ##############################
    # 저장
    ##############################
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"정제 완료: {len(df)}개 문서 → 저장 경로: {save_path}")


if __name__ == "__main__":
    do_cleansing()
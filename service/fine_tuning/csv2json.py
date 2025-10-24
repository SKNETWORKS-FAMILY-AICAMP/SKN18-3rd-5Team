'''llama factory 사용하기 전에 csv -> json으로 변환하는 파일'''
########################
# 필요한 패키지 설치
########################
import pandas as pd             # 데이터 분석
from pathlib import Path        # 파일 경로 설정
import json                     # json 형식 사용
import logging

########################
# 로그 설정
########################
logging.basicConfig(level=logging.INFO)  # INFO 이상만 보이게 설정


########################
# 청크 데이터 불러오기
########################
csv_path = Path("./data/chunked_data.csv")
df = pd.read_csv(csv_path)
logging.info(f"불러온 문서의 행의 크기: {len(df)}") # 클렌징 데이터 숫자 파악



########################
# csv -> JSON 구조로 변환
########################
json_data = [] # 리스트 형식으로 받음

'''
llama factory 사용하기 전에 test 데이터와 train 데이터 나누는 파일
train:test = 8:2 비율로 나눔
'''

########################
# 필요한 패키지 설치
########################
import json
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging


########################
# 로그 설정
########################
logging.basicConfig(level=logging.INFO)  # INFO 이상만 보이게 설정

########################
# 파일 경로 설정
########################
origin_path = Path("./data/csv2json.json")
train_path = Path("./service/fine_tuning/llama_factory/dataset/train.json")
test_path = Path("./service/fine_tuning/llama_factory/dataset/test.json")


# JSON 파일 열기
with open(origin_path, "r", encoding="utf-8") as f:
    data = json.load(f)

train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
with open(test_path, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

logging.info(f"train.json: {len(train_data)}, test.json: {len(test_data)} 저장 완료!")

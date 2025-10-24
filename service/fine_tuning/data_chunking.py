'''신한리서치 데이터 청크 분할 코드'''

##############################
# 필요한 라이브러리 설치
##############################
from pathlib import Path # 파일 경로 설정
from langchain_community.document_loaders import CSVLoader # LangChain에서 csv 파일 불러오기
import logging # 출력 로그 설정을 위함
from langchain.text_splitter import RecursiveCharacterTextSplitter # 청크 나누기
import pandas as pd # 청크된 데이터 저장하기 위함


##############################
# 로그 설정
##############################
logging.basicConfig(level=logging.INFO)  # INFO 이상만 보이게 설정


##############################
# 파일 경로 지정 및 파일 읽기
##############################
file_path = Path("./data/clean_data.csv") # 클렌징 파일 경로 지정
loader = CSVLoader(file_path, encoding="utf-8-sig") # csv 파일 읽기
docs = loader.load()


##############################
# 파일 데이터 확인
##############################
logging.info(f"정제된 문서의 수: {len(docs)}") # 클렌징 데이터 숫자 파악
logging.debug(f"정제된 문서의 메타데이터: {docs[0].metadata}") # 문서의 메타데이터 확인
logging.debug(f"정제된 문서의 텍스트: {docs[0].page_content[:300]}")


##############################
# RecursiveCharacterTextSplitter를 사용한 청크 나누기
##############################
csv_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=True,      # 정규식 사용 설정
    separators=[                  # 정규식 기반 구분자 목록
        r'(?<=[.!?])\s+',         # 문장 단위 구분: 마침표/물음표/느낌표 뒤 공백
        r'\n{2,}',                # 두 줄 이상 공백 (단락)
        r'\n',                    # 한 줄 개행
        r' ',                     # 일반 공백
        ''                        # 마지막 fallback (못 나눴을 때)
    ]
)

docs_with_splitter = csv_splitter.split_documents(docs)
logging.info(f"청크된 문서의 수: {len(docs_with_splitter)}")


##############################
# 청크된 데이터 확인
##############################
logging.debug(f"청크로 나눈 문서의 메타데이터: {docs_with_splitter[0].metadata}")
logging.debug(f"청크로 나눈 텍스트: {docs_with_splitter[0].page_content}")


##############################
# 청크된 데이터 저장
##############################

# Document 객체 리스트를 DataFrame으로 변환
chunk_data = pd.DataFrame([
    {
        "chunk_text": doc.page_content,
        "source": doc.metadata.get("source", ""),
        "row": doc.metadata.get("row", "")
    }
    for doc in docs_with_splitter
])

output_path = Path("./data/chunked_data.csv") # 저장할 파일 위치 설정
chunk_data.to_csv(output_path, index=False, encoding="utf-8-sig") # csv로 저장

logging.info(f"청크된 데이터를 저장하였습니다. 저장경로: {output_path}")
logging.info(f"저장된 청크의 수: {chunk_data.shape[0]}")

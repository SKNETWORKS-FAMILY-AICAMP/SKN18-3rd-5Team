"""
ETL (Extract) 서비스 모듈

데이터 추출(Extract) 단계를 담당하는 모듈들로 구성됩니다.
DART API에서 공시 데이터를 다운로드하고, XML을 구조화된 형태로 변환합니다.

모듈 구성:
- api_pull: DART API에서 공시 데이터 다운로드 및 ZIP 압축해제
- extractor: XML을 마크다운으로 변환 (구조화된 문서 생성)
- build_kospi_map: KOSPI 종목명과 기업코드 매핑 테이블 생성

사용법:
    from service.etl.extractor import DartDownloader, DartConfig, extractor_main, kospi_mapper_main
    
    # 1. DART API에서 데이터 다운로드
    downloader = DartDownloader()
    downloader.download_list()  # 공시 목록 다운로드
    downloader.download_all_documents()  # XML 파일 다운로드
    
    # 2. XML을 마크다운으로 변환
    extractor_main()  # XML → Markdown 변환 실행
    
    # 3. KOSPI 종목 매핑 생성
    kospi_mapper_main()  # KOSPI 종목 매핑 테이블 생성
"""

# 주요 클래스들을 import하여 모듈 레벨에서 접근 가능하게 함
from .api_pull import DartDownloader, DartConfig
from .extractor import main as extractor_main
from .build_kospi_map import main as kospi_mapper_main

__all__ = [
    'DartDownloader', 
    'DartConfig', 
    'extractor_main',
    'kospi_mapper_main'
]

# 모듈 버전 정보
__version__ = "1.0.0"
__author__ = "SKN18-3rd-5Team"
__description__ = "ETL Extract Service Module for DART API Data Processing"


import requests
import json
import os
import zipfile
from datetime import datetime, timedelta
from dotenv import load_dotenv


# ==========================================
# 파라미터 설정
# ==========================================
load_dotenv()

class DartConfig:
    API_KEY = os.getenv('DART_API_KEY', '')
    
    URL = 'list.json'           # 'list.json' 또는 'document.xml'

    # list.json 파라미터
    CORP_CLS = 'Y'              # Y(유가/코스피), K(코스닥), N(코넥스), E(기타)
    
    # 다운받을 보고서 유형 
    REPORT_TYPES = [
        'A001',     # 사업보고서
        'A003',     # 분기보고서 (1분기, 3분기)
        'F001',     # 주요사항보고서
        'I003',     # 자기주식취득결정
    ]
    # 감사보고서는 사업보고서에 포함되어 있음
    
    DOWNLOAD_YEAR = True        # 자동 1년치 다운로드
    
    # document.xml 파라미터
    RCEPT_NO = '' 

    AUTO_EXTRACT = True         # ZIP 자동 압축해제


# ==========================================
# 다운로더 클래스
# ==========================================
class DartDownloader:
    def __init__(self):
        self.base_url = 'https://opendart.fss.or.kr/api'
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(self.data_dir, exist_ok=True)
    
    # ==========================================
    # 공시검색
    # ==========================================
    def download_list(self):
        print("=" * 60)
        print("공시검색 시작")
        print("=" * 60)
        
        self._download_year_data()
    
    # ==========================================
    # 1년치 다운로드 (3개월씩 4회)
    # ==========================================
    def _download_year_data(self):
        all_data = []
        seen_rcept_nos = set()
        today = datetime.now()
        
        for i in range(4):
            end_date = today - timedelta(days=i * 90)
            begin_date = end_date - timedelta(days=89)
            
            begin_str = begin_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
            
            print(f"\n[기간 {i+1}/4] {begin_str} ~ {end_str}")
            
            period_data = self._download_all_pages(begin_str, end_str)
            
            # 중복 제거하면서 추가
            for item in period_data:
                rcept_no = item.get('rcept_no')
                if rcept_no and rcept_no not in seen_rcept_nos:
                    all_data.append(item)
                    seen_rcept_nos.add(rcept_no)
        
        self._save_combined_data(all_data)
    
    # ==========================================
    # 전체 페이지 수집
    # ==========================================
    def _download_all_pages(self, begin_date, end_date):
        all_data = []
        seen_rcept_nos = set()
        
        # 각 보고서 유형별로 다운로드
        for report_type in DartConfig.REPORT_TYPES:
            print(f"\n  [{report_type}] 다운로드 중...")
            type_data = self._download_by_type(begin_date, end_date, report_type)
            
            # 중복 제거하면서 추가
            for item in type_data:
                rcept_no = item.get('rcept_no')
                if rcept_no and rcept_no not in seen_rcept_nos:
                    all_data.append(item)
                    seen_rcept_nos.add(rcept_no)
        
        return all_data
    
    # ==========================================
    # 특정 유형의 공시 다운로드
    # ==========================================
    def _download_by_type(self, begin_date, end_date, report_type):
        type_data = []
        page_no = 1
        
        while True:
            params = {
                'crtfc_key': DartConfig.API_KEY,
                'bgn_de': begin_date,
                'end_de': end_date,
                'page_no': str(page_no),
                'page_count': '100',
                'corp_cls': DartConfig.CORP_CLS,
                'pblntf_detail_ty': report_type  # 공시유형 파라미터 수정!
            }
            
            url = f'{self.base_url}/list.json'
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"    HTTP 오류: {response.status_code}")
                break
            
            data = response.json()
            
            if data.get('status') != '000':
                if data.get('status') == '013':  # 데이터 없음
                    break
                print(f"    API 오류: {data.get('message')}")
                break
            
            items = data.get('list', [])
            if not items:
                break
            
            type_data.extend(items)
            
            if page_no == 1:
                total_count = data.get('total_count', 0)
                print(f"총 {total_count:,}건")
            
            print(f"페이지 {page_no}: {len(items)}건 (누적: {len(type_data)})")

            if len(type_data) >= total_count:
                print(f"    ✅ {report_type} 완료: {len(type_data)}건")
                break
            
            page_no += 1
        
        return type_data
    
    # ==========================================
    # 데이터 저장
    # ==========================================
    def _save_combined_data(self, all_data):
        today = datetime.now().strftime('%Y%m%d')
        file_name = f'{today}.json'
        file_path = os.path.join(self.data_dir, file_name)
        
        output_data = {
            'total_count': len(all_data),
            'download_date': today,
            'report_types': DartConfig.REPORT_TYPES,
            'list': all_data
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 60)
        print(f"저장 완료: {file_path}")
        print(f"총 {len(all_data):,}건")
        print("=" * 60)
    
    # ==========================================
    # 공시서류 원본 다운로드
    # ==========================================
    def download_document(self):
        print("=" * 60)
        print("공시서류 원본 다운로드")
        print("=" * 60)
        
        params = {
            'crtfc_key': DartConfig.API_KEY,
            'rcept_no': DartConfig.RCEPT_NO,
        }
        
        url = f'{self.base_url}/document.xml'
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"HTTP 오류: {response.status_code}")
            return
        
        file_name = f'dart_document_{DartConfig.RCEPT_NO}.zip'
        file_path = os.path.join(self.data_dir, file_name)
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        file_size = len(response.content) / 1024
        print(f"다운로드 완료: {file_size:.2f} KB")
        
        if DartConfig.AUTO_EXTRACT:
            self.extract_zip(file_path)
    
    # ==========================================
    # ZIP 압축 해제
    # ==========================================
    def extract_zip(self, zip_file_path):
        zip_filename = os.path.basename(zip_file_path)
        extract_folder_name = zip_filename.replace('.zip', '')
        extract_folder = os.path.join(self.data_dir, extract_folder_name)
        
        os.makedirs(extract_folder, exist_ok=True)
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        
        print(f"압축 해제 완료: {extract_folder}")


# ==========================================
# 메인 실행
# ==========================================
def main():
    downloader = DartDownloader()
    
    if DartConfig.URL == 'list.json':
        downloader.download_list()
    elif DartConfig.URL == 'document.xml':
        downloader.download_document()


if __name__ == '__main__':
    main()

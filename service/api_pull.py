import requests
import json
import os
import zipfile
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import re 
from datetime import datetime

# ==========================================
# 파라미터 설정
# ==========================================
load_dotenv()

class DartConfig:
    API_KEY = os.getenv('DART_API_KEY', '')
    
    # 실행 선택
    URL = 'document.xml'        # 'list.json', 'document.xml', 'retry_failed'

    # 공시 검색 설정 (1년치 고정)
    CORP_CLS = 'Y'              # Y(유가/코스피), K(코스닥), N(코넥스), E(기타)
    
    REPORT_TYPES = [
        'A001',     # 사업보고서 (감사보고서 포함되어 있음)
        'A003',     # 분기보고서
        'F001',     # 주요사항보고서
        'I003',     # 자기주식취득결정
    ]
    
    # ZIP 다운로드 설정
    DOWNLOAD_DELAY = 1          # API 호출 간 대기 시간(초)
    AUTO_EXTRACT = True         # ZIP 자동 압축해제

# ==========================================
# 다운로더 클래스 (공시목록 다운_json -> rcept_no 파라미터 입력 -> 원본파일 다운_xml)
# ==========================================
class DartDownloader:
    def __init__(self):
        self.base_url = 'https://opendart.fss.or.kr/api'
        
        # 폴더 구조 설정
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.zip_dir = os.path.join(self.data_dir, 'zip')
        self.xml_dir = os.path.join(self.data_dir, 'xml')
        self.log_dir = os.path.join(self.data_dir, 'logs')
        
        # 폴더 생성
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.zip_dir, exist_ok=True)
        os.makedirs(self.xml_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 실패한 접수번호 저장 파일
        self.failed_file = os.path.join(self.log_dir, 'failed_rcept_nos.txt')
        self.success_file = os.path.join(self.log_dir, 'success_rcept_nos.txt')
    
    # ==========================================
    # 1년치 공시 목록 다운로드
    # ==========================================
    def download_list(self):
        print("=" * 60)
        print("1년치 공시 목록 다운로드")
        print("=" * 60)
        
        all_data = []
        seen_rcept_nos = set()
        today = datetime.now()
        
        # 3개월씩 4번 (총 1년) -> API 3개월씩 다운 가능
        for i in range(4):
            end_date = today - timedelta(days=i * 90)
            begin_date = end_date - timedelta(days=89)
            
            begin_str = begin_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
            
            print(f"\n[기간 {i+1}/4] {begin_str} ~ {end_str}")
            
            period_data = self._download_all_pages(begin_str, end_str)
            
            # 중복 제거
            for item in period_data:
                rcept_no = item.get('rcept_no')
                if rcept_no and rcept_no not in seen_rcept_nos:
                    all_data.append(item)
                    seen_rcept_nos.add(rcept_no)
        
        # 저장
        self._save_json(all_data)
    
    # ==========================================
    # 전체 페이지 수집
    # ==========================================
    def _download_all_pages(self, begin_date, end_date):
        all_data = []
        seen_rcept_nos = set()
        
        # 보고서 순차적으로 다운로드
        for report_type in DartConfig.REPORT_TYPES: 
            print(f"    [{report_type}] 다운로드 중...")
            type_data = self._download_by_type(begin_date, end_date, report_type)
            
            for item in type_data:
                rcept_no = item.get('rcept_no')
                if rcept_no and rcept_no not in seen_rcept_nos:
                    all_data.append(item)
                    seen_rcept_nos.add(rcept_no)
        
        return all_data
    
    # ==========================================
    # 파라미터 설정 
    # ==========================================
    def _download_by_type(self, begin_date, end_date, report_type):
        type_data = []
        page_no = 1
        total_count = 0
        
        while True:
            params = {
                'crtfc_key': DartConfig.API_KEY,
                'bgn_de': begin_date,
                'end_de': end_date,
                'page_no': str(page_no),
                'page_count': '100',
                'corp_cls': DartConfig.CORP_CLS,
                'pblntf_detail_ty': report_type
            }
            
            response = requests.get(f'{self.base_url}/list.json', params=params)
            
            if response.status_code != 200: # 200:성공
                break
            
            data = response.json()
            
            if data.get('status') != '000': # 000:정상
                if data.get('status') == '013': # 013:조회된 데이타가 없습니다.
                    break
                print(f"    오류: {data.get('message')}")
                break
            
            items = data.get('list', [])
            if not items:
                break
            
            type_data.extend(items)
            
            if page_no == 1:
                total_count = data.get('total_count', 0)
                print(f"    총 {total_count:,}건")
            
            print(f"    페이지 {page_no}: {len(items)}건 (누적: {len(type_data)}/{total_count})")

            if len(type_data) >= total_count:
                break

            page_no += 1
        
        return type_data
    
    # ==========================================
    # JSON 저장
    # ==========================================
    def _save_json(self, all_data):
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
        print(f"✅ 저장 완료: {file_path}")
        print(f"📊 총 {len(all_data):,}건")
        print("=" * 60)
    
    # ==========================================
    # 실패한 접수번호 저장
    # ==========================================
    def _save_failed_rcept_no(self, rcept_no):
        with open(self.failed_file, 'a', encoding='utf-8') as f:
            f.write(f"{rcept_no}\n")
    
    # ==========================================
    # 성공한 접수번호 저장
    # ==========================================
    def _save_success_rcept_no(self, rcept_no):
        with open(self.success_file, 'a', encoding='utf-8') as f:
            f.write(f"{rcept_no}\n")
    
    # ==========================================
    # 실패한 접수번호 로드
    # ==========================================
    def _load_failed_rcept_nos(self):
        if os.path.exists(self.failed_file):
            with open(self.failed_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        return []
    
    # ==========================================
    # 성공한 접수번호 로드
    # ==========================================
    def _load_success_rcept_nos(self):
        if os.path.exists(self.success_file):
            with open(self.success_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        return []
    
    # ==========================================
    # 전체 ZIP 다운로드 로직 구현
    # ==========================================
    def download_all_documents(self):
        print("=" * 60)
        print("전체 ZIP 파일 다운로드")
        print("=" * 60)
        
        # 날짜 형식의 JSON 파일 중 가장 최근 파일 찾기
        json_files = [f for f in os.listdir(self.data_dir) 
                     if f.endswith('.json') and not f.startswith('.')]
        
        if not json_files:
            print("     ❌ JSON 파일이 없습니다. 먼저 list.json을 실행하세요.")
            return
        
        # 날짜 형식 파일 필터링 (YYYYMMDD.json)
        date_files = []
        for f in json_files:
            match = re.match(r'^(\d{8})\.json$', f)
            if match:
                try:
                    date_str = match.group(1)
                    date_obj = datetime.strptime(date_str, '%Y%m%d')
                    date_files.append((date_obj, f))
                except ValueError:
                    continue
        
        if date_files:
            # 날짜순 정렬 후 가장 최근 파일 선택
            date_files.sort(key=lambda x: x[0])
            json_file = date_files[-1][1]
        file_path = os.path.join(self.data_dir, json_file)
        print(f"📁 파일: {json_file}")
        
        # JSON 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = data.get('list', [])
        total = len(items)
        
        # 이미 성공한 접수번호들 로드
        success_rcept_nos = set(self._load_success_rcept_nos())
        
        print(f"📊 총 {total:,}건 다운로드 시작")
        print(f"🔄 이미 성공한 파일: {len(success_rcept_nos):,}건 건너뛰기")
        
        success = 0
        failed = 0
        skipped = 0
        
        for idx, item in enumerate(items, 1):
            rcept_no = item.get('rcept_no')
            corp_name = item.get('corp_name')
            report_nm = item.get('report_nm', '')
            
            # 이미 성공한 파일은 건너뛰기
            if rcept_no in success_rcept_nos:
                print(f"\n[{idx}/{total}] {corp_name} - {report_nm} (건너뛰기)")
                skipped += 1
                continue
            
            print(f"\n[{idx}/{total}] {corp_name} - {report_nm}")
            
            # rcept_no를 파라미터로 삽입
            if self._download_zip(rcept_no):
                success += 1
                self._save_success_rcept_no(rcept_no)
            else:
                failed += 1
                self._save_failed_rcept_no(rcept_no)
            
            # API 부하 방지
            if idx < total:
                time.sleep(DartConfig.DOWNLOAD_DELAY)
        
        print("\n" + "=" * 60)
        print(f"✅ 완료: 성공 {success:,}건, 실패 {failed:,}건, 건너뛰기 {skipped:,}건")
        print(f"📁 실패한 접수번호: {self.failed_file}")
        print(f"📁 성공한 접수번호: {self.success_file}")
        print("=" * 60)
    
    # ==========================================
    # ZIP 다운로드
    # ==========================================
    def _download_zip(self, rcept_no):
        params = {
            'crtfc_key': DartConfig.API_KEY,
            'rcept_no': rcept_no,
        }
        
        try:
            response = requests.get(
                f'{self.base_url}/document.xml',
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"  ❌ HTTP 오류: {response.status_code}")
                return False
            
            # ZIP 파일은 zip 폴더에 저장
            file_name = f'document.zip'
            file_path = os.path.join(self.zip_dir, file_name)
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            file_size = len(response.content) / 1024
            print(f"  ✅ ZIP: {file_size:.2f} KB", end='')
            
            if DartConfig.AUTO_EXTRACT:
                if self._extract_zip(file_path, rcept_no):
                    print(" → XML 추출 완료")
                else:
                    print(" → XML 추출 실패")
            else:
                print()
            
            return True
            
        except Exception as e:
            print(f"  ❌ 오류: {e}")
            return False
    
    # ==========================================
    # ZIP 압축 해제 (xml 폴더에 직접 저장)
    # ==========================================
    def _extract_zip(self, zip_file_path, rcept_no):
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # ZIP 안의 모든 파일 목록
                file_list = zip_ref.namelist()
                
                # 각 파일을 xml 폴더에 직접 저장
                for file_name in file_list:
                    base_name = os.path.basename(file_name)
                    
                    # xml 폴더에 직접 저장
                    output_path = os.path.join(self.xml_dir, base_name)
                    
                    # 파일 읽기 및 저장
                    with zip_ref.open(file_name) as source:
                        with open(output_path, 'wb') as target:
                            target.write(source.read())
            
            return True
            
        except Exception as e:
            print(f" (오류: {e})", end='')
            return False


    # ==========================================
    # 실패한 파일들만 재다운로드
    # ==========================================
    def retry_failed_downloads(self):
        print("=" * 60)
        print("실패한 파일들 재다운로드")
        print("=" * 60)
        
        failed_rcept_nos = self._load_failed_rcept_nos()
        
        if not failed_rcept_nos:
            print("❌ 실패한 파일이 없습니다.")
            return
        
        print(f"📊 실패한 파일: {len(failed_rcept_nos):,}건")
        print("=" * 60)
        
        success = 0
        failed = 0
        
        for idx, rcept_no in enumerate(failed_rcept_nos, 1):
            print(f"\n[{idx}/{len(failed_rcept_nos)}] 접수번호: {rcept_no}")
            
            if self._download_zip(rcept_no):
                success += 1
                self._save_success_rcept_no(rcept_no)
                # 실패 목록에서 제거
                self._remove_from_failed_list(rcept_no)
            else:
                failed += 1
            
            # API 부하 방지
            if idx < len(failed_rcept_nos):
                time.sleep(DartConfig.DOWNLOAD_DELAY)
        
        print("\n" + "=" * 60)
        print(f"✅ 재다운로드 완료: 성공 {success:,}건, 실패 {failed:,}건")
        print("=" * 60)
    
    # ==========================================
    # 실패 목록에서 제거
    # ==========================================
    def _remove_from_failed_list(self, rcept_no):
        if os.path.exists(self.failed_file):
            with open(self.failed_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            with open(self.failed_file, 'w', encoding='utf-8') as f:
                for line in lines:
                    if line.strip() != rcept_no:
                        f.write(line)

# ==========================================
# 메인 실행
# ==========================================
def main():
    downloader = DartDownloader()
    
    if DartConfig.URL == 'list.json':
        # 1단계: 1년치 공시 목록 다운로드
        downloader.download_list()
        
    elif DartConfig.URL == 'document.xml':
        # 2단계: 전체 ZIP 다운로드
        downloader.download_all_documents()
    
    elif DartConfig.URL == 'retry_failed':
        # 3단계: 실패한 파일들만 재다운로드
        downloader.retry_failed_downloads()
    
    else:
        print("❌ URL을 'list.json', 'document.xml', 'retry_failed' 중 하나로 설정하세요")


if __name__ == '__main__':
    main()
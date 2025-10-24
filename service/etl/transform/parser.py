#!/usr/bin/env python3
"""
====================================================================================
Transform Pipeline - Step 1: 구조화 및 1차 청킹
====================================================================================

[파이프라인 순서]
1. parser.py      → 마크다운을 구조화된 청크로 변환
2. normalizer.py → 데이터 정규화 및 자연어 품질 개선
3. chunker.py         → 스마트 청킹 및 메타데이터 강화

[이 파일의 역할]
- 마크다운 파일을 테이블/텍스트 단위로 파싱
- 테이블 행을 기본 자연어로 변환 (최소한의 처리만)
- 섹션 경로 및 기본 메타데이터 추출
- 원본 데이터 보존 (structured_data)

[입력]
- data/markdown/*.md (XML에서 변환된 마크다운)

[출력]
- data/transform/parser/*_chunks.jsonl (1차 청크)
====================================================================================
"""

import re
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# 공통 모듈
from utils import Chunk, write_jsonl, get_transform_paths


# ==========================================
# Table -> Natural Language (Convert)
# ==========================================


class TableRowConverter:
    """테이블 행을 자연어로 변환"""
    
    @staticmethod
    def convert(headers: List[str], row: List[str], section_path: str) -> str:
        """
        테이블 행을 자연어로 변환
        섹션 컨텍스트를 고려한 변환
        """
        
        # 빈 데이터 체크
        non_empty_values = [v for v in row if v and v.strip() and v != '-']
        if not non_empty_values:
            return ""
        
        # 보고서(사업보고서, 분기보고서, 주요사항보고서(+ 자기취득주식결정보고서)) -> 테이블 유형 파악 필요.

        # 섹션별 특화 변환
        if "주식의 총수" in section_path:
            return TableRowConverter._convert_stock_table(headers, row) # 주식 테이블
        elif "재무" in section_path or "손익" in section_path:
            return TableRowConverter._convert_financial_table(headers, row) # 재무 테이블
        else:
            return TableRowConverter._convert_generic_table(headers, row) # 일반 테이블
    
    # ==========================================
    # 1. 주식 열 테이블
    # ==========================================
    @staticmethod
    def _convert_stock_table(headers: List[str], row: List[str]) -> str:
        """주식 테이블 전용 변환 (기본 형태만)"""
        
        # 구 분 열 찾기
        gubun_indices = [i for i, h in enumerate(headers) if '구 분' in h]
        
        # 카테고리 추출
        categories = []
        for idx in gubun_indices:
            if idx < len(row) and row[idx] and row[idx] != '-':
                categories.append(row[idx])
        
        # 데이터 추출
        data_map = {}
        for i, header in enumerate(headers):
            if '구 분' not in header and i < len(row) and row[i] and row[i] != '-':
                value = row[i].strip()
                header_clean = header.strip()
                
                if value:
                    data_map[header_clean] = value
        
        # 자연어 생성 (간결하게, 조사 없이)
        if categories:
            category_text = " - ".join(categories)
            
            if data_map:
                # "키: 값" 형태로 간결하게 (조사 제거)
                data_items = [f"{k} {v}" for k, v in data_map.items()]
                return f"{category_text}: {', '.join(data_items)}"
            else:
                return category_text
        
        return ""
    

    # ==========================================
    # 2. 재무 열 테이블
    # ==========================================
    @staticmethod
    def _convert_financial_table(headers: List[str], row: List[str]) -> str:
        """재무 테이블 전용 변환 (기본 형태만, 단위 변환은 normalizer에서)"""
        
        # 첫 번째 열은 보통 항목명
        if not row or not row[0] or row[0] == '-':
            return ""
        
        item_name = row[0].strip()
        
        # 나머지 열은 값 (간결하게, 조사 없이)
        values = []
        for i in range(1, min(len(headers), len(row))):
            if row[i] and row[i] != '-':
                header = headers[i].strip()
                value = row[i].strip()
                # "항목: 값" 형태로 간결하게
                values.append(f"{header} {value}")
        
        if values:
            # "과목명 - 항목1 값1, 항목2 값2" 형태
            return f"{item_name} - {', '.join(values)}"
        else:
            return item_name
    
    # ==========================================
    # 3. 일반 테이블
    # ==========================================
    @staticmethod
    def _convert_generic_table(headers: List[str], row: List[str]) -> str:
        """일반 테이블 변환 (간결하고 자연스럽게)"""
        
        items = []
        for i in range(min(len(headers), len(row))):
            if row[i] and row[i] != '-' and row[i].strip():
                header = headers[i].strip()
                value = row[i].strip()
                
                if header and value:
                    # "키: 값" 형태로 간결하게
                    items.append(f"{header}: {value}")
        
        if items:
            return ", ".join(items)
        else:
            return ""

# ==========================================
# Markdown Chunking
# ==========================================
class MarkdownChunker:
    """마크다운을 청크로 분할"""
    
    def __init__(self, markdown_content: str, doc_metadata: Dict[str, Any]):
        self.content = markdown_content
        self.doc_metadata = doc_metadata
        self.chunks = []
        self.current_section_path = []
    
    def process(self) -> List[Chunk]:
        """마크다운 처리"""
        
        lines = self.content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # YAML 헤더 건너뛰기
            if i == 0 and line.startswith('---'):
                while i < len(lines) and (i == 0 or not lines[i].startswith('---')):
                    i += 1
                i += 1
                continue
            
            # 섹션 헤더
            if line.startswith('#'):
                self._update_section(line)
                i += 1
                continue
            
            # 테이블
            if line.startswith('|'):
                i = self._process_table(lines, i)
                continue
            
            # 텍스트
            if line.strip() and not line.startswith('#'):
                i = self._process_text(lines, i)
                continue
            
            i += 1
        
        return self.chunks
    
    def _update_section(self, header_line: str):
        """섹션 업데이트"""
        level = header_line.count('#')
        title = header_line.lstrip('#').strip()
        
        # 레벨에 맞게 조정
        while len(self.current_section_path) >= level:
            self.current_section_path.pop()
        
        self.current_section_path.append(title)
    
    def _process_table(self, lines: List[str], start_idx: int) -> int:
        """테이블 처리 - 복잡한 구조 감지 및 처리"""
        
        # 테이블 위의 단위 정보 수집 (테이블 바로 위 3줄 확인)
        table_unit = None
        for check_idx in range(max(0, start_idx - 3), start_idx):
            if check_idx < len(lines):
                line = lines[check_idx].strip()
                # "단위: 천원", "(단위: 백만원)", "-(단위: 천원)", "(단위 : 천원/Ton)" 등
                # 공백 포함 패턴 지원: (단위 : 백만원)
                unit_match = re.search(r'[\(\-]?\s*단위\s*[:：\s]\s*([^\)/\)]+)', line, re.IGNORECASE)
                if unit_match:
                    table_unit = unit_match.group(1).strip()
                    # 비화폐 단위는 제외 (Ton, %, 리터, 개 등)
                    if any(non_monetary in table_unit for non_monetary in ['Ton', '%', '리터', '개', '주', '건', '명', '회']):
                        table_unit = None
                        continue
                    # 화폐 단위만 추출 (원, 천원, 백만원, 억원)
                    if '원' in table_unit:
                        break
                    else:
                        table_unit = None
        
        # 테이블 추출
        table_lines = []
        i = start_idx
        while i < len(lines) and (lines[i].startswith('|') or lines[i].strip() == ''):
            if lines[i].startswith('|'):
                table_lines.append(lines[i])
            i += 1
        
        if len(table_lines) < 3:
            return i
        
        # 테이블 구조 분석
        table_type = self._analyze_table_structure(table_lines)
        
        if table_type == 'vertical':
            # 수직 구조 테이블 (항목명이 첫 번째 열에 있음)
            self._parse_vertical_table(table_lines, table_unit)
        else:
            # 일반 테이블 (0행이 헤더)
            self._parse_normal_table(table_lines, table_unit)
        
        return i
    
    def _analyze_table_structure(self, table_lines: List[str]) -> str:
        """테이블 구조 판단"""
        
        if len(table_lines) < 3:
            return 'normal'
        
        headers = [h.strip() for h in table_lines[0].split('|')[1:-1]]
        first_row = [c.strip() for c in table_lines[2].split('|')[1:-1]]
        
        # 패턴 1: 헤더 0행이 "1.", "2." 같은 번호로 시작
        if headers and re.match(r'^\d+\.', headers[0]):
            return 'vertical'
        
        # 패턴 2: 헤더 0행이 "제 XX 기" 패턴
        if headers and re.match(r'^제\s*\d+', headers[0]):
            return 'vertical'
        
        # 패턴 3: 첫 번째 데이터 행이 "1.", "2." 같은 번호로 시작
        if first_row and re.match(r'^\d+\.', first_row[0]):
            return 'vertical'
        
        # 패턴 4: 첫 행의 값들이 "말", "초", "부터", "까지" 같은 키워드
        if len(first_row) > 1:
            keyword_count = sum(1 for v in first_row if v in ['말', '초', '부터', '까지', '시작일', '종료일'])
            if keyword_count >= 2:
                return 'vertical'
        
        # 패턴 5: 헤더에 동일한 값이 반복되면 수직 테이블
        # 예: "항 목", "항 목", "금 액"
        if len(headers) >= 2:
            header_counts = {}
            for h in headers:
                if h:
                    header_counts[h] = header_counts.get(h, 0) + 1
            # 동일한 헤더가 2번 이상 나오면
            if any(count >= 2 for count in header_counts.values()):
                return 'vertical'
        
        # 패턴 6: 열 개수가 2-3개이고 첫 번째 열이 항목명처럼 보임
        if 2 <= len(headers) <= 3:
            # 두 번째 행부터 첫 번째 열 값들 확인
            first_col_values = []
            for row_idx in range(2, min(len(table_lines), 7)):  # 최대 5개 행 확인
                row = [c.strip() for c in table_lines[row_idx].split('|')[1:-1]]
                if row:
                    first_col_values.append(row[0])
            
            # 대부분이 항목명처럼 보이면 수직 테이블
            item_like = sum(1 for v in first_col_values if 
                          ('일' in v or '자' in v or '액' in v or '명' in v or '율' in v or '목' in v or v.endswith('여부')))
            if item_like >= len(first_col_values) * 0.5:
                return 'vertical'
        
        return 'normal'
    
    def _parse_vertical_table(self, table_lines: List[str], table_unit: str = None):
        """수직 구조 테이블 파싱 (항목명이 첫 번째 열)
        
        예시 구조 1 (6열):
        | 1. 계약금액 | 1. 계약금액 | 1. 계약금액 | 1,000,000,000 | 1,000,000,000 | 1,000,000,000 |
        | 2. 계약기간 | 2. 계약기간 | 시작일      | 2025-01-10    | 2025-01-10    | 2025-01-10    |
        
        예시 구조 2 (3열):
        | 항 목(1) | 항 목(2) | 금 액 |
        | 1. 배당가능이익 | 가. 순자산액 | 193,082,198,582 |
        
        앞쪽 열들은 항목명 (보통 중복), 뒤쪽 열들은 데이터
        """
        
        section_path = ' > '.join(self.current_section_path)
        
        # 0행 파싱
        headers = [h.strip() for h in table_lines[0].split('|')[1:-1]]
        num_cols = len(headers)
        
        # 2행부터 각 행 처리
        for row_idx in range(2, len(table_lines)):
            row = [c.strip() for c in table_lines[row_idx].split('|')[1:-1]]
            
            if not row:
                continue
            
            # 데이터 열 감지 (마지막에서부터 역순 탐색)
            data_col_idx = -1
            for i in range(len(row) - 1, -1, -1):
                cell = row[i]
                
                # 마크다운 볼드, 숫자, 날짜 패턴 = 데이터 열
                if '**' in cell or re.match(r'^\d{4}년', cell) or re.match(r'^\d+,\d+', cell) or re.match(r'^\d+\.\d+', cell) or (cell and cell.replace(',', '').replace('-', '').isdigit()):
                    data_col_idx = i
                    break
            
            # 데이터 열이 없으면 스킵
            if data_col_idx == -1:
                continue
            
            # 항목명 수집 (데이터 열 이전의 모든 열, 중복 제거)
            item_parts = []
            seen_items = set()
            
            for i in range(data_col_idx):
                cell = row[i]
                if cell and cell != '-':
                    # 중복 제거 (대소문자 구분)
                    if cell not in seen_items:
                        item_parts.append(cell)
                        seen_items.add(cell)
            
            # 항목명이 없으면 스킵
            if not item_parts:
                continue
            
            # 항목명 조합
            item_name = ' > '.join(item_parts)
            
            # 데이터 추출
            value = row[data_col_idx].replace('**', '').strip()
            
            # 빈 값이면 스킵 (키와 값이 같은 경우는 유지)
            if not value or value == '-':
                continue
            
            # 구조화 데이터 생성
            structured_data = {item_name: value}
            
            # 단위 적용하여 natural_text 생성
            formatted_value = self._apply_unit_to_value(value, table_unit)
            
            # 키에 포함된 숫자도 단위 변환
            formatted_item_name = self._apply_unit_to_text(item_name, table_unit)
            natural_text = f"{formatted_item_name}: {formatted_value}"
            
            # 청크 생성
            chunk = Chunk(
                chunk_id=self._generate_id(natural_text, 'table_row', section_path),
                doc_id=f"{self.doc_metadata.get('rcept_dt', '')}_{self.doc_metadata.get('corp_code', '')}",
                chunk_type='table_row',
                section_path=section_path,
                structured_data=structured_data,
                natural_text=natural_text,
                metadata={
                    'corp_name': self.doc_metadata.get('corp_name', ''),
                    'document_name': self.doc_metadata.get('document_name', ''),
                    'rcept_dt': self.doc_metadata.get('rcept_dt', ''),
                }
            )
            
            self.chunks.append(chunk)
    
    def _parse_normal_table(self, table_lines: List[str], table_unit: str = None):
        """일반 테이블 파싱 (0행이 헤더)"""
        
        section_path = ' > '.join(self.current_section_path)
        headers = [h.strip() for h in table_lines[0].split('|')[1:-1]]
        
        # 데이터 행 처리
        for row_idx in range(2, len(table_lines)):
            row = [c.strip() for c in table_lines[row_idx].split('|')[1:-1]]
            
            if len(row) != len(headers):
                continue
            
            # 구조화
            structured_data = {}
            for j, header in enumerate(headers):
                if j < len(row):
                    structured_data[header] = row[j]
            
            # 자연어 변환 (단위 적용)
            natural_text = TableRowConverter.convert(headers, row, section_path)
            
            if not natural_text:
                continue
            
            # 단위 적용
            if table_unit:
                natural_text = self._apply_unit_to_text(natural_text, table_unit)
            
            # 청크 생성
            chunk = Chunk(               
                chunk_id=self._generate_id(natural_text, 'table_row', section_path),
                doc_id=f"{self.doc_metadata.get('rcept_dt', '')}_{self.doc_metadata.get('corp_code', '')}",
                chunk_type='table_row',                
                section_path=section_path,
                structured_data=structured_data,
                natural_text=natural_text,
                metadata={
                    'corp_name': self.doc_metadata.get('corp_name', ''),
                    'document_name': self.doc_metadata.get('document_name', ''),
                    'rcept_dt': self.doc_metadata.get('rcept_dt', ''),
                }
            )
            
            self.chunks.append(chunk)
        
    def _apply_unit_to_value(self, value: str, table_unit: str) -> str:
        """값에 단위 적용하여 숫자로 변환 (콤마 없는 0 추가)
        
        Args:
            value: 원본 값 (예: "838,319" 또는 "32,711,600천원")
            table_unit: 단위 정보 (예: "천원", "백만원")
        
        Returns:
            단위가 적용된 숫자 (예: "838319000" 또는 "32711600000")
        """
        # 1. 먼저 값 자체에 단위가 있는지 확인하고 변환
        value = self._convert_inline_units(value)
        
        # 2. 테이블 단위 적용
        if not table_unit:
            return value
        
        # 단위 승수 결정
        unit_multiplier = 1
        if table_unit:
            if '천원' in table_unit or '천 원' in table_unit:
                unit_multiplier = 1000
            elif '백만원' in table_unit or '백만 원' in table_unit:
                unit_multiplier = 1000000
            elif '억원' in table_unit or '억 원' in table_unit:
                unit_multiplier = 100000000
        
        if unit_multiplier == 1:
            return value
        
        # 숫자 추출 및 변환
        try:
            num_str = value.replace(',', '').replace(' ', '').strip()
            if not num_str.replace('.', '').replace('-', '').isdigit():
                return value
            
            num = int(float(num_str) * unit_multiplier)
            
            # 단위 적용된 숫자 반환 (콤마 없는 0 추가)
            return str(num)
        except:
            return value
    
    def _convert_inline_units(self, text: str) -> str:
        """셀 내 단위(천원, 백만원, 억원)를 raw 숫자로 변환
        
        예: "32,711,600천원" → "32711600000"
        예: "1,234백만원" → "1234000000"
        예: "500억원" → "50000000000"
        """
        
        def convert_with_unit(match):
            num_str = match.group(1)  # 숫자 부분
            unit = match.group(2)      # 단위 부분
            
            try:
                # 콤마 제거하고 숫자로 변환
                num = float(num_str.replace(',', ''))
                
                # 단위별 승수
                multiplier = 1
                if '천원' in unit or '천 원' in unit:
                    multiplier = 1000
                elif '백만원' in unit or '백만 원' in unit:
                    multiplier = 1_000_000
                elif '억원' in unit or '억 원' in unit:
                    multiplier = 100_000_000
                elif '조원' in unit or '조 원' in unit:
                    multiplier = 1_000_000_000_000
                
                # 변환된 숫자 (콤마 없는 raw 숫자)
                result = int(num * multiplier)
                return str(result)
                
            except (ValueError, AttributeError):
                # 변환 실패 시 원본 반환
                return match.group(0)
        
        # 패턴: 숫자 + 단위
        # 예: 32,711,600천원, 1234백만원, 500억원
        pattern = r'([\d,]+)(천원|천 원|백만원|백만 원|억원|억 원|조원|조 원)'
        text = re.sub(pattern, convert_with_unit, text)
        
        return text
    
    def _apply_unit_to_text(self, text: str, table_unit: str) -> str:
        """텍스트 내 모든 숫자에 단위 적용 (콤마 없는 0 추가)
        
        Args:
            text: 원본 텍스트 (예: "매출: 838,319, 자산: 1,200,000")
            table_unit: 단위 정보
        
        Returns:
            단위가 적용된 텍스트 (예: "매출: 838319000, 자산: 1200000000")
        """
        
        # 1. 먼저 셀 내 단위(천원, 백만원, 억원) 처리
        text = self._convert_inline_units(text)
        
        # 2. 테이블 단위가 있으면 적용
        if not table_unit:
            return text
        
        # 콤마로 구분된 숫자 패턴 찾기
        def replace_number(match):
            return self._apply_unit_to_value(match.group(0), table_unit)
        
        # 숫자 패턴 (콤마 포함, 단위 없음)
        text = re.sub(
            r'\b\d{1,3}(?:,\d{3})+\b(?!\s*[원주%억만천])',
            replace_number,
            text
        )
        
        return text
    
    def _clean_header(self, header: str) -> str:
        """헤더 정리 (번호, 기호 제거)"""
        # "1. 계약금액(원)" → "계약금액"
        # "제 78 기" → "제78기"
        
        cleaned = header.strip()
        
        # 앞의 번호 제거 ("1.", "2." 등)
        cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
        
        # 괄호 안 내용 제거 (단위 등)
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        
        # 공백 정리
        cleaned = re.sub(r'\s+', '', cleaned)
        
        return cleaned
    
    def _process_text(self, lines: List[str], start_idx: int) -> int:
        """텍스트 처리"""
        
        text_lines = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i]
            
            if line.startswith('|') or line.startswith('#'):
                break
            
            if not line.strip():
                if i + 1 < len(lines) and not lines[i + 1].strip():
                    break
            
            text_lines.append(line)
            i += 1
        
        text = '\n'.join(text_lines).strip()
        
        if len(text) > 10:
            section_path = ' > '.join(self.current_section_path)
            chunk = Chunk(
                chunk_id=self._generate_id(text, 'text', section_path),
                doc_id=f"{self.doc_metadata.get('rcept_dt', '')}_{self.doc_metadata.get('corp_code', '')}",
                chunk_type='text',
                section_path=section_path,
                structured_data={},
                natural_text=text,
                metadata={
                    'corp_name': self.doc_metadata.get('corp_name', ''),
                    'document_name': self.doc_metadata.get('document_name', ''),
                    'rcept_dt': self.doc_metadata.get('rcept_dt', ''),
                }
            )
            self.chunks.append(chunk)
        
        return i
    
    def _generate_id(self, content: str, chunk_type: str, section_path: str = "") -> str:
        """ID 생성 - 의미있는 ID로 개선"""
        # 문서 ID 기반 + 타입 + 순번으로 생성
        doc_id = f"{self.doc_metadata.get('rcept_dt', '')}_{self.doc_metadata.get('corp_code', '')}"
        
        # 섹션 경로에서 마지막 부분 추출
        section_name = section_path.split(' > ')[-1] if section_path else "unknown"
        section_clean = re.sub(r'[^\w가-힣]', '', section_name)[:10]  # 특수문자 제거, 10자 제한
        
        # 타입별 순번 계산
        type_count = sum(1 for chunk in self.chunks if chunk.chunk_type == chunk_type)
        
        # 의미있는 ID 생성: doc_id + type + section + 순번
        return f"{doc_id}_{chunk_type}_{section_clean}_{type_count:03d}"


def main(process_all=False):
    """
    Step 1: 마크다운 → 1차 청크 변환
    
    입력: data/markdown/*.md
    출력: data/transform/parser/*_chunks.jsonl
    """
    
    # 처리 모드 설정
    if process_all:
        max_files = None  # 전체 처리
        print("🔧 전체 파일 처리 모드")
    else:
        max_files = 20  # 테스트용 20개만
        print("🔧 테스트 모드 (20개 파일만 처리)")

    # 경로 설정 (transform 폴더 기준)
    paths = get_transform_paths(__file__)
    markdown_dir = paths['markdown_dir']
    output_dir = paths['parser_dir']

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # 모든 마크다운 파일 찾기
    all_markdown_files = list(markdown_dir.glob("*.md"))
    
    if not all_markdown_files:
        print("❌ 마크다운 파일이 없습니다.")
        return

    # 파일 수 제한 적용
    if max_files:
        markdown_files = all_markdown_files[:max_files]
        print(f"📄 전체 {len(all_markdown_files)}개 중 {len(markdown_files)}개 파일 처리")
    else:
        markdown_files = all_markdown_files
        print(f"📄 전체 {len(markdown_files)}개 파일 처리")

    print("=" * 80)
    print("Transform Pipeline - Step 1: 구조화 및 1차 청킹")
    print("=" * 80)
    print(f"📁 입력: {markdown_dir}")
    print(f"📁 출력: {output_dir}")
    print(f"📄 처리할 파일 수: {len(markdown_files)}개")
    if max_files:
        print(f"💡 전체 처리하려면: python {Path(__file__).name} --all")
    print(f"\n다음 단계: normalizer.py로 정규화 수행")
    print("=" * 80)
    print()

    # 전체 통계
    total_chunks = 0
    processed_files = 0
    failed_files = 0
    all_chunks = []  # 모든 청크를 저장할 리스트

    # 각 마크다운 파일 처리
    for i, md_file in enumerate(markdown_files, 1):
        print(f"[{i}/{len(markdown_files)}] 처리 중: {md_file.name}")
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            # YAML 헤더에서 메타데이터 추출
            doc_metadata = {}
            if markdown_content.startswith('---'):
                yaml_end = markdown_content.find('---', 3)
                if yaml_end != -1:
                    yaml_header = markdown_content[3:yaml_end]
                    for line in yaml_header.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            doc_metadata[key.strip()] = value.strip()

            # 기본값 설정
            if not doc_metadata.get('corp_code'):
                doc_metadata = {
                    'corp_name': 'Unknown',
                    'document_name': 'Unknown'
                }

            print(f"  📊 문서 정보: {doc_metadata.get('corp_name')} ({doc_metadata.get('stock_code')})")

            # 청크 생성
            chunker = MarkdownChunker(markdown_content, doc_metadata)
            chunks = chunker.process()

            print(f"  ✅ {len(chunks)}개 청크 생성")

            # JSONL 저장 (파일명: 마크다운 파일명 그대로 사용)
            output_filename = f"{md_file.stem}_chunks.jsonl"
            output_file = output_dir / output_filename

            # 공통 write_jsonl 사용
            write_jsonl(output_file, [chunk.to_dict() for chunk in chunks])

            print(f"  💾 저장: {output_file.name}")

            # 통계 업데이트
            total_chunks += len(chunks)
            processed_files += 1
            all_chunks.extend(chunks)  # 모든 청크를 리스트에 추가

            # 청크 타입별 통계
            chunk_types = {}
            for chunk in chunks:
                chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1

            print(f"  📈 청크 타입: {', '.join([f'{k}({v})' for k, v in chunk_types.items()])}")
            print()

        except Exception as e:
            print(f"  ❌ 오류: {e}")
            failed_files += 1
            print()

    # 전체 결과 출력
    print("=" * 60)
    print("     처리 완료")
    print("=" * 60)

    # 테이블 청크 샘플
    table_chunks = [c for c in all_chunks if c.chunk_type == 'table_row'][:3]

    for i, chunk in enumerate(table_chunks):
        print(f"\n[{i+1}] {chunk.section_path}")
        print(f"타입: {chunk.chunk_type}")
        print(f"자연어: {chunk.natural_text}")
        print(f"구조화: {json.dumps(chunk.structured_data, ensure_ascii=False, indent=2)}")

    # 텍스트 청크 샘플
    text_chunks = [c for c in all_chunks if c.chunk_type == 'text'][:2]

    if text_chunks:
        print("\n" + "=" * 60)
        print("텍스트 청크 샘플:")
        print("=" * 60)

        for i, chunk in enumerate(text_chunks):
            print(f"\n[{i+1}] {chunk.section_path}")
            print(f"내용: {chunk.natural_text[:200]}...")

    # 통계
    print("\n" + "=" * 60)
    print("청크 통계:")
    print("=" * 60)

    chunk_types = {}
    for chunk in all_chunks:
        chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1

    for chunk_type, count in chunk_types.items():
        print(f"  {chunk_type}: {count}개")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("사용법:")
        print(f"  python {Path(__file__).name}        # 테스트 모드 (10개 파일만)")
        print(f"  python {Path(__file__).name} --all  # 전체 파일 처리")
        print(f"  python {Path(__file__).name} --help # 도움말")
        sys.exit(0)
    
    # --all 옵션 확인
    process_all = len(sys.argv) > 1 and sys.argv[1] == "--all"
    main(process_all=process_all)
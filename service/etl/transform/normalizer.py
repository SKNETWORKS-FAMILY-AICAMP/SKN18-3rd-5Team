#!/usr/bin/env python3
"""
====================================================================================
Transform Pipeline - Step 2: 데이터 정규화 및 자연어 품질 개선
====================================================================================

[파이프라인 순서]
1. parser.py      → 마크다운을 구조화된 청크로 변환
2. normalizer.py  → 데이터 정규화 및 자연어 품질 개선 (현재 파일)
3. chunker.py     → 스마트 청킹 및 메타데이터 강화

[이 파일의 역할]
- 날짜 형식 통일 (YYYY-MM-DD, "2022주-10-31" → "2022-10-31")
- 통화 단위 표준화 (억원 변환, 콤마 제거)
- 숫자/비율 정규화 ("2,361,968,545주" → "2361968545주", "비율(%): 28.1%" → "비율: 28.1%")
- 표 경로 단순화 (검색 최적화)
- 목차 및 불필요한 내용 제거
- 마크다운 문법 정리
- 자연어 품질 개선 (반복 축소, 문맥 추가)

[입력]
- data/transform/parser/*_chunks.jsonl (parser.py 출력)

[출력]
- data/transform/normalized/*_chunks.jsonl (정규화된 청크)
====================================================================================
"""

import re
import unicodedata
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

# 공통 모듈
from utils import read_jsonl, write_jsonl, get_file_list, ensure_output_dir, get_transform_paths


@dataclass
class NormalizationConfig:
    """정규화 설정"""
    # 날짜 형식 (ISO 8601)
    date_format: str = "%Y-%m-%d"
    
    # 통화 단위 기준 (억원)
    currency_unit_threshold: int = 100_000_000
    
    # 제거할 패턴
    remove_patterns: List[str] = None
    
    def __post_init__(self):
        if self.remove_patterns is None:
            self.remove_patterns = [
                # 목차 관련 패턴들
                r'\*\*목\s*차\*\*.*?(?=\n\n|\Z)',  # **목 차** 패턴
                r'목\s*차.*?(?=\n\n|\Z)',  # 목차로 시작하는 모든 내용
                r'페\s*이\s*지.*?(?=\n\n|\Z)',  # 페이지 관련 내용
                r'독립된\s*감사인의\s*감사보고서.*?\d+\s*~\s*\d+',  # 감사보고서 목차
                r'연\s*결\s*재\s*무\s*제\s*표.*?\d+',  # 재무제표 목차
                r'연\s+결\s+재\s+무\s+제\s+표\s+에\s+대\s+한',  # "연 결 재 무 제 표 에 대 한" 패턴
                r'ㆍ\s*연\s*결.*?\d+\s*~\s*\d+',  # 연결 관련 목차
                r'ㆍ',  # 단독 ㆍ 문자 제거
                r'외부감사\s*실시내용.*?\d+\s*~\s*\d+',  # 외부감사 목차
                r'주석.*?\d+\s*~\s*\d+',  # 주석 목차
                r'I+\.\s+[가-힣\s]+‥+\s*\d+',  # 로마 숫자 목차
                r'제\s*\d+\s*\([전당]*\)\s*기',  # 중복 기수 표시
                # 상세한 목차 패턴
                r'【[^】]*】\s*-+\s*\d+',  # 【 제목 】 -------- 페이지번호
                # 목차 블록 전체 제거 (로마숫자/아라비아숫자 + 제목 + 점선 + 페이지번호)
                r'(?:^|\n)(?:[IVX]+\.|[0-9\-\.]+)\s+[가-힣\s\(\)]+\s+-+\s*\d+(?:\n[IVX0-9\-\.]+\s+[가-힣\s\(\)]+\s+-+\s*\d+)*',
                # 점선 패턴
                r'-{3,}.*?(?=\n\n|\Z)',  # 3개 이상의 점선
                r'‥+.*?(?=\n\n|\Z)',  # 점선 패턴
            ]


class DataNormalizer:
    """데이터 정규화 처리"""
    
    def __init__(self, config: NormalizationConfig = None):
        self.config = config or NormalizationConfig()
    
    def normalize_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """청크 단위 정규화
        
        Returns:
            정규화된 단일 chunk (텍스트 분할은 chunker.py에서 처리)
        """

        # structured_data 정규화 먼저 (natural_text 생성에 사용됨)
        if chunk.get('structured_data'):
            chunk['structured_data'] = self._normalize_structured_data(
                chunk['structured_data']
            )

        # structured_data 재구성 (빈 값이 많은 경우)
        if chunk.get('chunk_type') == 'table_row' and chunk.get('structured_data'):
            section = chunk.get('section_path', '')
            if any(kw in section.replace(' ', '') for kw in ['재무', '손익', '자산', '재무제표']):
                has_financial_data = any(
                    v and v != '-'
                    for k, v in chunk['structured_data'].items()
                    if isinstance(v, str) and ('기말' in k or '기초' in k or '년' in k or '분기' in k)
                )
                if not has_financial_data and chunk.get('natural_text'):
                    chunk['structured_data'] = self._reconstruct_structured_from_text(chunk['natural_text'])
        
        # natural_text 개선
        if chunk.get('natural_text'):
            chunk['natural_text'] = self._improve_natural_text(
                chunk['natural_text'],
                chunk.get('chunk_type'),
                chunk.get('structured_data', {}),
                chunk.get('metadata', {}),
                chunk.get('section_path', '')
            )
        
        return chunk
    
    
    def _improve_natural_text(
        self, 
        text: str, 
        chunk_type: str,
        structured_data: Dict,
        metadata: Dict,
        section_path: str = ""
    ) -> str:
        """자연어 품질 개선 (검색 최적화 강화)"""
        
        # 1. 목차 및 불필요 패턴 제거
        text = self._remove_unnecessary_content(text)
        
        # 2. 마크다운 제거
        text = self._clean_markdown(text)
        
        # 3. 날짜 정규화 (검색 최적화)
        text = self._normalize_dates_in_text(text)
        
        # 4. 타입별 처리
        if chunk_type == 'table_row':
            text = self._improve_table_text(text, structured_data, metadata, section_path)
            # 5. 표 데이터 경로 단순화 (검색 최적화)
            text = self._simplify_table_paths(text)
        elif chunk_type == 'text':
            text = self._improve_text_content(text, metadata, section_path)
        
        # 6. 공백 정리
        text = self._normalize_whitespace(text)
        
        return text
    
    def _remove_unnecessary_content(self, text: str) -> str:
        """불필요한 내용 제거"""
        
        # 목차 제거
        for pattern in self.config.remove_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # 페이지 번호 제거
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # 과도한 점선 제거
        text = re.sub(r'‥{3,}', '', text)
        
        return text
    
    def _clean_markdown(self, text: str) -> str:
        """마크다운 정리"""
        
        # Bold 제거
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        
        # Italic 제거
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        return text
    
    def _improve_table_text(
        self, 
        text: str, 
        structured_data: Dict,
        metadata: Dict,
        section_path: str = ""
    ) -> str:
        """테이블 자연어 개선 - 자연스러운 문장 생성
        
        단위는 parser.py에서 이미 적용되어 있음
        """

        section = section_path

        # 1. structured_data 기반 자연어 문장 생성 (우선순위 1)
        natural_sentence = self._generate_natural_sentence(structured_data, section, metadata)
        if natural_sentence:
            return natural_sentence

        # 2. 반복 제거 및 기본 개선
        text = self._reduce_repetition(text)
        
        # 3. 큰 raw 숫자 처리 (10자리 이상) - 섹션 무관하게 항상 적용
        text = self._process_large_raw_numbers(text)
        
        # 4. 재무제표 특화 처리 (공백 제거 후 체크)
        if any(kw in section.replace(' ', '') for kw in ['재무', '손익', '자산', '재무제표', '위험관리', '파생거래']):
            text = self._improve_financial_text(text, structured_data)
        
        # 5. 주식 정보 특화 처리
        elif '주식' in section:
            text = self._improve_stock_text(text, structured_data)
        
        # 6. 날짜 정규화
        text = self._normalize_dates_in_text(text)
        
        return text
    
    def _reconstruct_structured_from_text(self, text: str) -> Dict[str, str]:
        """natural_text에서 structured_data 재구성"""
        data = {}

        # "키: 값, 키: 값" 형태 파싱
        pairs = text.split(', ')
        for pair in pairs:
            if ': ' in pair:
                key, value = pair.split(': ', 1)
                data[key.strip()] = value.strip()

        return data
    
    def _generate_natural_sentence(self, data: Dict, section: str, metadata: Dict) -> str:
        """섹션별 자연스러운 문장 생성"""

        if not data:
            return ""

        # 0. 자기주식 취득 (최우선)
        if '자기주식' in section and ('취득' in section or '신탁' in section):
            return self._generate_treasury_stock_sentence(data, metadata)

        # 1. 신용평가 (키 기반 감지 추가)
        elif '신용평가' in section or '신용등급' in data or '신용평가기관' in data:
            return self._generate_credit_rating_sentence(data, metadata)

        # 2. 사업연도 정보 (새로 추가) - 키 검색으로 수정
        elif any('사업연도' in k for k in data.keys()) or any('년' in k and '월' in k and '일' in k for k in data.keys()):
            sentence = self._generate_fiscal_period_sentence(data, metadata)
            if sentence:
                return sentence

        # 3. 중소/중견기업 여부 (새로 추가) - 키와 값 모두 검색
        elif (any('중소기업' in k or '중견기업' in k or '벤처기업' in k for k in data.keys()) or
              any('중소기업' in v or '중견기업' in v or '벤처기업' in v for v in data.values() if isinstance(v, str))):
            sentence = self._generate_company_classification_sentence(data, metadata)
            if sentence:
                return sentence

        # 4. 재무제표 (공백 제거 후 체크)
        elif any(keyword in section.replace(' ', '') for keyword in ['재무', '손익', '자산', '부채', '재무제표']):
            return self._generate_financial_sentence(data, metadata)

        # 5. 주식 정보
        elif '주식' in section:
            return self._generate_stock_sentence(data, metadata)

        # 6. 배당 정보
        elif '배당' in section:
            return self._generate_dividend_sentence(data, metadata)

        # 7. 임원 정보
        elif '임원' in section or '이사' in section:
            return self._generate_executive_sentence(data, metadata)

        return ""
    
    def _generate_treasury_stock_sentence(self, data: Dict, metadata: Dict) -> str:
        """자기주식 취득 자연어 문장 생성
        
        structured_data 예시:
        - "1. 계약금액": "1,000,000,000"
        - "2. 계약기간 > 시작일": "2025-01-10"
        - "3. 계약목적": "주주가치 제고"
        """
        
        # 데이터가 단일 항목인 경우, structured.py에서 이미 자연어 생성됨
        # 여기서는 그대로 반환하되 날짜/숫자 정규화만 수행
        if len(data) == 1:
            key, value = next(iter(data.items()))
            
            # 값 정규화
            value_normalized = value
            
            # 날짜 정규화
            if re.match(r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일', value):
                value_normalized = self._normalize_dates_in_text(value)
            
            # 숫자 단위 추가
            if '주식' in key and value.replace(',', '').replace('주', '').isdigit():
                num = value.replace(',', '').replace('주', '')
                value_normalized = f"{int(num):,}주"
            elif '금액' in key and value.replace(',', '').isdigit():
                value_normalized = self._format_financial_amount(value)
            elif '비율' in key and value.replace(',', '').replace('%', '').replace('.', '').isdigit():
                if '%' not in value:
                    value_normalized = f"{value}%"
            
            # 자연어 문장 반환
            return f"{key}: {value_normalized}"
        
        # 다중 항목인 경우 (레거시 대응)
        company = metadata.get('corp_name', '회사')
        contract_amount = None
        start_date = None
        end_date = None
        purpose = None
        broker = None
        decision_date = None
        shares_before = None
        shares_ratio = None
        
        # 키 매칭 (다양한 표현 대응)
        for key, value in data.items():
            # 시작일
            if '시작일' in key or '개시' in key:
                start_date = self._normalize_dates_in_text(str(value))
            
            # 종료일
            elif '종료일' in key or '만료' in key:
                end_date = self._normalize_dates_in_text(str(value))
            
            # 목적
            elif '목적' in key:
                purpose = value
            
            # 증권사/중개업자
            elif '체결기관' in key or '중개업자' in key:
                broker = value
            
            # 결정일
            elif '결정일' in key or '예정일자' in key:
                decision_date = self._normalize_dates_in_text(str(value))
            
            # 계약금액
            elif '계약금액' in key or ('금액' in key and '계약' in key):
                if value and value != '-' and value.replace(',', '').isdigit():
                    contract_amount = self._format_financial_amount(value)
            
            # 보유 주식수
            elif '보통주식' in key and value and value.replace(',', '').replace('주', '').isdigit():
                shares_before = self._format_number_with_unit(value.replace('주', ''), '주')
            
            # 보유 비율
            elif '비율' in key and value and value != '-':
                shares_ratio = value if '%' in str(value) else f"{value}%"
        
        # 자연어 문장 생성
        parts = []
        
        # 조사 선택 (받침 유무)
        josa = self._get_josa(company, '은', '는')
        
        # 기본 문장
        if decision_date:
            parts.append(f"{company}{josa} {decision_date}에 자기주식 취득 신탁계약 체결을 결정했습니다.")
        else:
            parts.append(f"{company}{josa} 자기주식 취득 신탁계약을 체결했습니다.")
        
        # 계약금액
        if contract_amount:
            parts.append(f"계약금액은 {contract_amount}입니다.")
        
        # 계약기간
        if start_date and end_date:
            parts.append(f"계약기간은 {start_date}부터 {end_date}까지입니다.")
        elif start_date:
            parts.append(f"계약 시작일은 {start_date}입니다.")
        
        # 목적
        if purpose:
            parts.append(f"취득 목적은 '{purpose}'입니다.")
        
        # 증권사
        if broker:
            parts.append(f"계약체결기관은 {broker}입니다.")
        
        # 보유현황
        if shares_before or shares_ratio:
            holding_info = []
            if shares_before:
                holding_info.append(f"{shares_before}")
            if shares_ratio:
                holding_info.append(f"비율 {shares_ratio}")
            parts.append(f"계약 전 자기주식 보유현황은 {', '.join(holding_info)}입니다.")
        
        return " ".join(parts)
    
    def _generate_credit_rating_sentence(self, data: Dict, metadata: Dict) -> str:
        """신용평가 자연어 문장 생성"""
        parts = []
        
        # 회사명
        company = metadata.get('corp_name', '회사')
        
        # 평가일
        eval_date = data.get('평가일', '')
        if eval_date:
            eval_date = self._normalize_dates_in_text(eval_date)
            parts.append(f"{eval_date}에")
        
        # 신용평가기관
        agency = data.get('신용평가기관', '')
        if agency:
            parts.append(f"{agency}로부터")
        
        # 신용등급
        rating = data.get('신용등급', '')
        if rating:
            parts.append(f"{rating} 등급을")
        
        # 평가목적
        purpose = data.get('평가목적', '')
        if purpose:
            parts.append(f"{purpose} 목적으로")
        
        # 평가구분
        eval_type = data.get('평가구분', '')
        if eval_type:
            parts.append(f"({eval_type})")
        
        if parts:
            return f"{company}는 {' '.join(parts)} 받았습니다."
        
        return ""
    
    def _generate_financial_sentence(self, data: Dict, metadata: Dict) -> str:
        """재무 데이터 자연어 문장 생성"""
        parts = []

        # 과목명 찾기
        item_name = None
        for key in ['과목', '과 목', '항목', '구분']:
            if key in data and data[key] and data[key] != '-':
                item_name = data[key]
                break

        if not item_name:
            # 첫 번째 값이 있는 키를 과목명으로
            for key, value in data.items():
                if value and value != '-' and not any(x in key for x in ['기말', '기초', '년', '주석', '주 석']):
                    item_name = value
                    break

        if item_name:
            parts.append(f"{item_name}")

        # 주석 정보
        footnote = None
        for key in ['주석', '주 석']:
            if key in data and data[key] and data[key] != '-':
                footnote = data[key]
                break

        # 연도별/기간별 금액
        amounts = []
        for key, value in data.items():
            # 과목명, 주석은 건너뜀
            if key in ['과목', '과 목', '항목', '구분', '주석', '주 석']:
                continue

            if not value or value == '-' or not value.replace(',', '').replace('.', '').isdigit():
                continue

            # 기말/기초 패턴
            period_label = ""
            if '기말' in key or '기초' in key:
                # "제 27(당) 기말" -> "제27기 말"
                period_match = re.search(r'제\s*(\d+)', key)
                if period_match:
                    period_num = period_match.group(1)
                    if '당' in key:
                        period_label = f"제{period_num}기(당기)"
                    elif '전' in key:
                        period_label = f"제{period_num}기(전기)"
                    else:
                        period_label = f"제{period_num}기"

                    if '기말' in key:
                        period_label += " 말"
                    elif '기초' in key:
                        period_label += " 초"

            # 연도 패턴
            elif '년' in key or re.match(r'\d{4}', key):
                year = re.search(r'\d{4}', key)
                if year:
                    period_label = f"{year.group(0)}년"

            # 금액 변환
            amount_text = self._format_financial_amount(value)

            if period_label:
                amounts.append(f"{period_label} {amount_text}")

        # 문장 조합
        if amounts:
            if parts:
                result = f"{parts[0]}은(는) {', '.join(amounts)}입니다"
            else:
                result = f"{', '.join(amounts)}"

            # 주석 추가
            if footnote:
                result += f" (주석 {footnote})"

            return result

        return ""
    
    def _generate_stock_sentence(self, data: Dict, metadata: Dict) -> str:
        """주식 정보 자연어 문장 생성"""
        parts = []
        company = metadata.get('corp_name', '회사')
        
        # 주식 종류
        stock_type = None
        for key in ['구 분', '구분', '주식종류']:
            if key in data:
                stock_type = data[key]
                break
        
        if stock_type:
            parts.append(f"{company}의 {stock_type}은(는)")
        
        # 주식수
        shares = data.get('주식수', data.get('발행주식수', ''))
        if shares:
            shares_formatted = self._format_number_with_unit(shares, '주')
            parts.append(f"{shares_formatted}이며")
        
        # 금액
        amount = data.get('금액', data.get('발행금액', ''))
        if amount:
            amount_formatted = self._format_financial_amount(amount)
            parts.append(f"금액은 {amount_formatted}입니다")
        
        return " ".join(parts) if parts else ""
    
    def _generate_dividend_sentence(self, data: Dict, metadata: Dict) -> str:
        """배당 정보 자연어 문장 생성"""
        parts = []
        company = metadata.get('corp_name', '회사')
        
        # 결산기
        fiscal_year = data.get('결산기', data.get('사업연도', ''))
        if fiscal_year:
            parts.append(f"{company}는 {fiscal_year} 결산기에")
        
        # 주당배당금
        dividend = data.get('주당배당금', data.get('배당금', ''))
        if dividend:
            dividend_formatted = self._format_number_with_unit(dividend, '원')
            parts.append(f"주당 {dividend_formatted}을")
        
        # 배당률
        dividend_rate = data.get('배당률', data.get('배당수익률', ''))
        if dividend_rate:
            parts.append(f"배당률 {dividend_rate}%로")
        
        if parts:
            parts.append("지급했습니다")
        
        return " ".join(parts) if len(parts) > 1 else ""
    
    def _generate_executive_sentence(self, data: Dict, metadata: Dict) -> str:
        """임원 정보 자연어 문장 생성"""
        parts = []

        # 성명
        name = data.get('성명', data.get('이름', ''))
        if name:
            parts.append(f"{name}은(는)")

        # 직위
        position = data.get('직위', data.get('직책', ''))
        if position:
            parts.append(f"{position}으로")

        # 취임일
        appointment_date = data.get('취임일', data.get('선임일', ''))
        if appointment_date:
            date_formatted = self._normalize_dates_in_text(appointment_date)
            parts.append(f"{date_formatted}에 취임했습니다")

        return " ".join(parts) if len(parts) > 1 else ""

    def _generate_fiscal_period_sentence(self, data: Dict, metadata: Dict) -> str:
        """사업연도 정보 자연어 문장 생성"""
        company = metadata.get('corp_name', '회사')

        # 데이터 추출
        fiscal_year = None
        start_date = None
        end_date = None

        for key, value in data.items():
            if not value or value == '-':
                continue

            # 사업연도 패턴
            if '사업연도' in key and value not in ['부터', '까지', '사업연도']:
                fiscal_year = value

            # 케이스 1: 키와 값 모두 날짜인 경우 (예: "2024년 01월 01일": "2024년 09월 30일")
            if '년' in key and '월' in key and '일' in key and '년' in value and '월' in value:
                start_date = self._normalize_dates_in_text(key)
                end_date = self._normalize_dates_in_text(value)

            # 케이스 2: 값이 "까지"/"부터"이고 키에 날짜가 있는 경우
            elif value == '까지' and '년' in key and '월' in key:
                end_date = self._normalize_dates_in_text(key)
            elif value == '부터' and '년' in key and '월' in key:
                start_date = self._normalize_dates_in_text(key)

            # 케이스 3: 키가 "부터"/"까지"이고 값에 날짜가 있는 경우
            elif ('부터' in key or '시작' in key) and '년' in value:
                start_date = self._normalize_dates_in_text(value)
            elif ('까지' in key or '종료' in key or '만료' in key) and '년' in value:
                end_date = self._normalize_dates_in_text(value)

        # 문장 생성
        if start_date and end_date:
            year = start_date[:4] if start_date else ''
            return f"{company}의 {year}년도 사업연도는 {start_date}부터 {end_date}까지입니다."
        elif fiscal_year:
            return f"{company}의 사업연도는 {fiscal_year}입니다."

        return ""

    def _generate_company_classification_sentence(self, data: Dict, metadata: Dict) -> str:
        """중소/중견기업 분류 정보 자연어 문장 생성"""
        company = metadata.get('corp_name', '회사')

        classifications = []

        # 데이터 분석 - 키와 값 모두에서 기업 유형 찾기
        for key, value in data.items():
            if not value or value == '-':
                continue

            key_clean = key.strip()
            value_clean = value.strip()

            # 키 또는 값에서 기업 유형 감지
            enterprise_type = None
            status = None

            # 값에서 기업 유형 찾기 (잘못된 파싱 대응)
            if '중소기업' in value_clean:
                enterprise_type = '중소기업'
            elif '중견기업' in value_clean:
                enterprise_type = '중견기업'
            elif '벤처기업' in value_clean:
                enterprise_type = '벤처기업'

            # 키에서도 찾기
            if not enterprise_type:
                if '중소기업' in key_clean:
                    enterprise_type = '중소기업'
                elif '중견기업' in key_clean:
                    enterprise_type = '중견기업'
                elif '벤처기업' in key_clean:
                    enterprise_type = '벤처기업'

            # 해당/미해당 판단 (키 또는 값에서)
            if '해당' in value_clean:
                if '미해당' in value_clean:
                    status = False
                else:
                    status = True
            elif '해당' in key_clean:
                if '미해당' in key_clean:
                    status = False
                else:
                    status = True
            elif '미해당' in value_clean or '미해당' in key_clean:
                status = False

            if enterprise_type and status is not None:
                classifications.append((enterprise_type, status))

        # 중복 제거
        classifications = list(dict.fromkeys(classifications))

        # 문장 생성
        if not classifications:
            return ""

        parts = []
        positive_classifications = [name for name, is_positive in classifications if is_positive]
        negative_classifications = [name for name, is_positive in classifications if not is_positive]

        if positive_classifications:
            parts.append(f"{company}는 {', '.join(positive_classifications)}에 해당합니다")

        if negative_classifications:
            if parts:
                parts.append(f"{', '.join(negative_classifications)}에는 해당하지 않습니다")
            else:
                parts.append(f"{company}는 {', '.join(negative_classifications)}에 해당하지 않습니다")

        return ". ".join(parts) + "." if parts else ""
    
    def _format_financial_amount(self, value: str, context: str = "") -> str:
        """금액을 억원/만원 단위로 포맷팅 (연도 제외, 안전한 콤마 추가)"""
        try:
            # 문자열 정리 (콤마 제거)
            cleaned = str(value).replace(',', '').replace(' ', '')
            num = int(cleaned)
            
            
            # 금액 변환 (안전한 콤마 추가)
            if abs(num) >= 100_000_000:
                eok = num / 100_000_000
                return f"{eok:,.1f}억원" if eok != int(eok) else f"{int(eok):,}억원"
            elif abs(num) >= 10_000:
                man = num / 10_000
                return f"{man:,.1f}만원" if man != int(man) else f"{int(man):,}만원"
            else:
                return f"{num:,}원"
        except (ValueError, TypeError):
            return str(value)
    
    def _format_number_with_unit(self, value: str, unit: str) -> str:
        """숫자에 단위 추가 (안전한 콤마 추가)"""
        try:
            # 콤마 제거 후 숫자 변환
            clean_value = str(value).replace(',', '').replace(' ', '')
            num = int(clean_value)
            return f"{num:,}{unit}"
        except (ValueError, TypeError):
            return f"{value}{unit}"
    
    def _get_josa(self, word: str, josa_with_final: str, josa_without_final: str) -> str:
        """받침에 따라 적절한 조사 선택
        
        Args:
            word: 단어
            josa_with_final: 받침 있을 때 조사 (예: '은', '이', '을')
            josa_without_final: 받침 없을 때 조사 (예: '는', '가', '를')
        """
        if not word:
            return josa_without_final
        
        # 마지막 글자
        last_char = word[-1]
        
        # 한글이 아니면 기본값
        if not ('가' <= last_char <= '힣'):
            # 영어/숫자 등: 발음 기준 (간단히 모음으로 끝나면 받침 없음)
            if last_char.lower() in ['a', 'e', 'i', 'o', 'u', '0', '2', '4', '6', '8']:
                return josa_without_final
            return josa_with_final
        
        # 한글: 받침 확인
        # 유니코드: '가'(0xAC00) + (초성*21*28) + (중성*28) + 종성
        # 종성이 0이면 받침 없음
        code = ord(last_char) - 0xAC00
        jongseong = code % 28
        
        if jongseong == 0:
            return josa_without_final  # 받침 없음
        else:
            return josa_with_final  # 받침 있음
    
    def _reduce_repetition(self, text: str) -> str:
        """과도한 반복 축소 및 자연어 개선
        
        주의: 숫자 내 콤마는 절대 분리하지 않음
        """
        
        import re
        
        # 숫자 내 콤마를 보호하기 위해 최소한의 처리만 수행
        # "키: 키" 패턴 제거 (동일한 키와 값, 단 의미있는 데이터는 제외)
        # 단, "신규설립", "신규연결" 등은 유지
        # 주의: 숫자만으로 구성된 키는 제외 (예: "6: 6"은 "596: 6,333"의 일부)
        text = re.sub(
            r'([^,:]+):\s*\1(?=,|$)', 
            lambda m: m.group(0) if (
                any(keyword in m.group(1) for keyword in ['신규', '연결', '설립', '합병', '분할']) or
                m.group(1).replace(' ', '').isdigit()  # 숫자만 있는 경우 제외
            ) else '', 
            text
        )

        # 3. "은(는)" 제거
        text = re.sub(r'은\(는\)', '', text)

        # 4. 빈 항목 정리 (": ," 또는 시작/끝의 콤마)
        # 주의: ": 숫자"는 제외 (예: "596: 6"은 정상 데이터)
        text = re.sub(r':\s*,(?!\d)', ',', text)  # ": ," 제거 (단, 뒤에 숫자가 아닐 때만)
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'^,\s*|\s*,$', '', text)

        # 5. 불필요한 공백 정리
        text = re.sub(r'\s{2,}', ' ', text)

        # 6. 콤마 뒤 공백 통일 (숫자 내 콤마는 제외)
        # 숫자 패턴이 아닌 콤마만 공백 추가
        text = re.sub(r'(?<!\d),(?=\s*[^0-9])', ', ', text)

        # 7. 콜론 뒤 공백 통일
        text = re.sub(r':\s+', ': ', text)

        return text.strip()
    
    def _process_large_raw_numbers(self, text: str) -> str:
        """큰 raw 숫자 처리 (10자리 이상)
        
        Parser에서 raw number로 변환된 큰 숫자에 단위 추가
        예: 1782278000000 → 1.8조원
        """
        
        def add_unit_to_large_number(match):
            num_str = match.group(1)  # 그룹 1 사용
            try:
                num = int(num_str)
                
                # 비정상적으로 큰 숫자 (20자리 이상)는 데이터 오류로 표시
                if len(num_str) >= 20:
                    jo = num / 1_000_000_000_000
                    if jo > 1000:  # 1000조원 이상은 비정상
                        return f"[데이터오류] {jo:.1f}조원"
                    else:
                        return f"{jo:.1f}조원"
                
                # 1조 이상 → 조원
                elif abs(num) >= 1_000_000_000_000:
                    jo = num / 1_000_000_000_000
                    return f"{jo:.1f}조원"
                
                # 1억 이상 → 억원
                elif abs(num) >= 100_000_000:
                    eok = num / 100_000_000
                    return f"{eok:.1f}억원" if eok != int(eok) else f"{int(eok):,}억원"
                
                # 1만 이상 → 만원
                elif abs(num) >= 10_000:
                    man = num / 10_000
                    return f"{man:.1f}만원" if man != int(man) else f"{int(man):,}만원"
                
                else:
                    return f"{num:,}원"
                    
            except (ValueError, AttributeError, OverflowError):
                # 오버플로우 발생 시 원본 반환
                return match.group(0)
        
        # 10자리 이상 숫자만 처리
        # 단어 경계가 아닌 위치에서도 매치 (콜론, 공백 등 뒤)
        # 단, 이미 단위가 있거나 소수점 일부인 경우 제외
        text = re.sub(r'(?:^|[^0-9.])(\d{10,})(?![억만조원%주0-9.])(?!\s*년)', add_unit_to_large_number, text)
        
        return text
    
    def _improve_financial_text(self, text: str, data: Dict) -> str:
        """재무 데이터 자연어 개선 - 일관된 단위 추가"""
        
        # 1. 먼저 퍼센트 처리 (소수점이 있는 작은 숫자)
        # 0.XX 형태 → XX%
        text = re.sub(
            r'\b0\.(\d{1,2})\b',
            lambda m: f"{int(m.group(1))}%" if m.group(1)[0] != '0' else f"{float(m.group(0))*100:.1f}%",
            text
        )
        
        # 2. 큰 숫자에 단위 추가 (억원) - 안전한 콤마 처리
        def add_currency_unit(match):
            # 콤마 제거 후 숫자 변환
            num_str = match.group(0).replace(',', '').replace(' ', '')
            try:
                num = int(num_str)
                
                # 1억 이상 → 억원
                if abs(num) >= 100_000_000:
                    eok = num / 100_000_000
                    if eok == int(eok):
                        return f"{int(eok):,}억원"
                    else:
                        return f"{eok:,.1f}억원"
                
                # 1만 이상 1억 미만 → 만원 (선택적)
                elif abs(num) >= 10_000:
                    man = num / 10_000
                    if man == int(man):
                        return f"{int(man):,}만원"
                    else:
                        return f"{man:,.1f}만원"
                
                # 그 외 → 원
                else:
                    return f"{num:,}원"
                        
            except (ValueError, AttributeError):
                return match.group(0)
        
        # 8자리 이상 숫자 (이미 단위 없는 경우만, 연도 제외)
        text = re.sub(r'\b(\d{8,})(?![억만원%주])(?!\s*년)', add_currency_unit, text)
        
        # 콤마가 있는 숫자 (이미 단위 없는 경우만, 연도 제외)
        text = re.sub(r'\b(\d{1,3}(?:,\d{3})+)(?![억만원%주])(?!\s*년)', add_currency_unit, text)
        
        # 4자리 이상 숫자 (단위 없는 경우, 연도 제외)
        text = re.sub(r'\b(\d{4,7})(?![억만원%주,])(?!\s*년)', add_currency_unit, text)
        
        return text
    
    def _improve_stock_text(self, text: str, data: Dict) -> str:
        """주식 데이터 자연어 개선 - 일관된 단위 추가"""
        
        # 1. 비율 처리 먼저 (퍼센트)
        # 0.XX 형태 → XX%
        text = re.sub(
            r'\b0\.(\d{1,2})\b',
            lambda m: f"{int(m.group(1))}%" if m.group(1)[0] != '0' else f"{float(m.group(0))*100:.1f}%",
            text
        )
        
        # "비율", "율" 키워드가 있으면 숫자에 % 추가
        if '비율' in text or '율' in text or '지분' in text:
            text = re.sub(r'\b(\d+)(?!%|주|,)', r'\1%', text)
            text = re.sub(r'\b(\d+\.\d+)(?!%)', r'\1%', text)
        
        # 2. "주" 단위 추가 (이미 단위가 없는 경우만, 연도 제외)
        # 콤마가 있는 숫자 (연도 패턴 제외)
        text = re.sub(r'\b(\d{1,3}(?:,\d{3})+)(?![억만원%주])(?!\s*년)', r'\1주', text)
        
        # 4자리 이상 숫자 (단위 없는 경우, 연도 패턴 제외)
        text = re.sub(r'\b(\d{4,})(?![억만원%주,])(?!\s*년)', r'\1주', text)
        
        return text
    
    def _improve_text_content(self, text: str, metadata: Dict, section_path: str = "") -> str:
        """텍스트 콘텐츠 개선 - 가독성 향상 및 스마트 분할"""
        
        # 1. 날짜 정규화
        text = self._normalize_dates_in_text(text)
        
        # 2. 숫자에 단위 추가 (컨텍스트 기반)
        text = self._add_units_to_numbers(text, section_path)

        # 3. 불필요한 공백 제거
        # "회 사 명" → "회사명"
        text = re.sub(r'(\S)\s+(\S)(?=\s*:)', r'\1\2', text)

        # 4. 콜론 앞뒤 공백 정리
        # "회사명 :" → "회사명:"
        text = re.sub(r'\s*:\s*', ': ', text)

        # 5. 괄호 앞뒤 공백 정리
        # "회사명 ( 123 )" → "회사명(123)"
        text = re.sub(r'\s*\(\s*', '(', text)
        text = re.sub(r'\s*\)\s*', ') ', text)

        # 6. 과도한 줄바꿈 정리
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 7. 과도한 공백 정리
        text = re.sub(r' {2,}', ' ', text)
        
        return text
    
    def _add_units_to_numbers(self, text: str, section_path: str = "") -> str:
        """컨텍스트 기반 숫자 단위 추가
        
        주의: 
        - 날짜 패턴은 제외
        - "단위: 천원", "(단위: 백만원)" 등의 컨텍스트 반영
        """
        
        # 이미 단위가 있는지 체크
        has_unit_pattern = r'(?:[억만천백십]|원|주|%|건|개|명|년|월|일)'
        
        # 텍스트에서 단위 정보 추출
        unit_multiplier = 1
        if '단위' in text or '單位' in text:
            if '천원' in text or '천 원' in text:
                unit_multiplier = 1000
            elif '백만원' in text or '백만 원' in text:
                unit_multiplier = 1000000
            elif '억원' in text or '억 원' in text:
                unit_multiplier = 100000000
        
        # 1. 재무/금액 관련
        if any(keyword in text for keyword in ['매출', '자산', '부채', '자본', '금액', '가격', '원']):
            # 8자리 이상 → 억원 (날짜 제외, 단위 적용, 안전한 콤마 처리)
            def format_large_number(match):
                # 콤마 제거 후 숫자 변환
                clean_num_str = match.group(1).replace(',', '').replace(' ', '')
                num = int(clean_num_str) * unit_multiplier
                return f"{num/100_000_000:,.0f}억원"
            
            text = re.sub(
                r'\b(\d{8,})(?!' + has_unit_pattern + r'|[-./]\d)',
                format_large_number,
                text
            )
            
            # 콤마 숫자 → 적절한 단위 (날짜 아닌 경우만, 단위 적용, 안전한 콤마 처리)
            def add_currency_if_not_date(match):
                num_str = match.group(1)
                # 주변 문맥 확인 (날짜 하이픈/점이 앞뒤에 있으면 스킵)
                start = match.start()
                end = match.end()
                if start > 0 and text[start-1] in '-./':
                    return num_str
                if end < len(text) and text[end] in '-./':
                    return num_str
                # 콤마 제거 후 포맷팅
                return self._format_currency(num_str, unit_multiplier)
            
            text = re.sub(
                r'\b(\d{1,3}(?:,\d{3})+)(?!' + has_unit_pattern + ')',
                add_currency_if_not_date,
                text
            )
        
        # 2. 주식 관련
        elif any(keyword in text for keyword in ['주식', '주수', '보통주', '우선주']):
            text = re.sub(
                r'\b(\d{1,3}(?:,\d{3})+)(?!' + has_unit_pattern + r')(?!\s*년)',
                r'\1주',
                text
            )
            # 날짜 패턴 제외: YYYY-MM-DD, YYYY.MM.DD, YYYY년 MM월 DD일
            text = re.sub(r'\b(\d{4,})(?!' + has_unit_pattern + r')(?!\s*년)(?![-./]\d)', r'\1주', text)
        
        # 3. 비율 관련
        elif any(keyword in text for keyword in ['비율', '율', '지분', '점유']):
            # 0.XX → XX%
            text = re.sub(r'\b0\.(\d+)\b', lambda m: f"{float(m.group(0))*100:.1f}%", text)
        
        return text
    33
    def _format_currency(self, num_str: str, unit_multiplier: int = 1) -> str:
        """금액 포맷팅 (안전한 콤마 추가)
        
        Args:
            num_str: 숫자 문자열 (예: "838319" 또는 "838,319")
            unit_multiplier: 단위 승수
                - 1: 원 (기본)
                - 1000: 천원
                - 1000000: 백만원
                - 100000000: 억원
        """
        try:
            # 콤마 제거 후 숫자 변환
            clean_num_str = num_str.replace(',', '').replace(' ', '')
            num = int(clean_num_str) * unit_multiplier
            
            if num >= 100_000_000:
                eok = num / 100_000_000
                return f"{eok:,.0f}억원" if eok == int(eok) else f"{eok:,.1f}억원"
            elif num >= 10_000:
                return f"{num//10_000:,}만원"
            else:
                return f"{num:,}원"
        except:
            return num_str
    
    def _normalize_dates_in_text(self, text: str) -> str:
        """텍스트 내 날짜 정규화 (검색 최적화)"""
        
        # 1. "YYYY년 MM월 DD일" → "YYYY-MM-DD"
        def korean_date_to_iso(match):
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        text = re.sub(
            r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',
            korean_date_to_iso,
            text
        )
        
        # 2. "YYYY.MM.DD" → "YYYY-MM-DD"
        text = re.sub(
            r'(\d{4})\.(\d{2})\.(\d{2})',
            r'\1-\2-\3',
            text
        )
        
        # 3. ❌ 문제: "2022주-10-31" → "2022-10-31" (검색 최적화)
        text = re.sub(
            r'(\d{4})주-(\d{2})-(\d{2})',
            r'\1-\2-\3',
            text
        )
        
        # 4. 기수 표현 정규화 "제 54 기" → "제54기"
        text = re.sub(r'제\s*(\d+)\s*기', r'제\1기', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """공백 정규화"""
        
        # 여러 공백을 하나로
        text = re.sub(r' {2,}', ' ', text)
        
        # 줄바꿈 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def _simplify_table_paths(self, text: str) -> str:
        """표 데이터의 장황한 경로 단순화 (검색 최적화)
        
        예시:
        "1. 직전사업연도 말 현재 상법상 배당가능이익 한도 > 가. 순자산액: 3,954.7억원"
        → "배당가능이익 한도 - 순자산액: 3,954.7억원"
        """
        
        # 패턴 1: "숫자. 긴 제목 > 하위항목 > 값" → "제목 - 하위항목: 값"
        # "1. 직전사업연도 말 현재 상법상 배당가능이익 한도 > 가. 순자산액: 3,954.7억원"
        pattern1 = r'\d+\.\s+(?:[가-힣\s]+?)\s+([가-힣]+)\s+(?:한도|내역|현황|상황)\s*>\s*(?:[가-힣]{1,2}\.\s*)?([^:]+):'
        text = re.sub(pattern1, r'\1 한도 - \2:', text)
        
        # 패턴 2: "숫자. 제목 > 숫자. 하위항목 > 값" → "제목 - 하위항목: 값"
        # "7. 해지 전 자기주식 보유현황 > 배당가능범위 내취득(주) > 보통주식 > 5,947,889 > 비율(%): 1.2"
        pattern2 = r'\d+\.\s+([^>]+?)\s+>\s+([^>]+?)\s+>\s+([^>]+?)\s+>\s+[\d,]+\s+>\s+'
        text = re.sub(pattern2, r'\1 - \2 - \3 ', text)
        
        # 패턴 3: 단순화 - "숫자. 제목 > 하위항목: 값" → "제목 - 하위항목: 값"
        pattern3 = r'\d+\.\s+([^>]+?)\s+>\s+([^:]+):'
        text = re.sub(pattern3, r'\1 - \2:', text)
        
        return text
    
    def _normalize_structured_data(self, data: Dict) -> Dict:
        """구조화 데이터 정규화"""
        
        normalized = {}
        
        for key, value in data.items():
            # 키 정리
            clean_key = self._clean_markdown(key.strip())
            
            # 값 정리
            if isinstance(value, str):
                clean_value = self._clean_markdown(value.strip())
                
                # 날짜 정규화
                clean_value = self._normalize_date_value(clean_value)
                
                # 숫자 정규화 (단위 정보 추가)
                clean_value = self._normalize_number_value(clean_value, clean_key)
                
                normalized[clean_key] = clean_value
            else:
                normalized[clean_key] = value
        
        return normalized
    
    def _normalize_date_value(self, value: str) -> str:
        """날짜 값 정규화"""
        
        # "YYYY년 MM월 DD일" 형식
        match = re.match(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', value)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # "YYYY.MM.DD" 형식
        match = re.match(r'(\d{4})\.(\d{2})\.(\d{2})', value)
        if match:
            return '-'.join(match.groups())
        
        return value
    
    def _normalize_number_value(self, value: str, key: str) -> str:
        """숫자 값 정규화 (단위 추가 + 검색 최적화)"""
        
        # 1. ❌ 문제: "2,361,968,545주" → 콤마 제거
        # 숫자 뒤에 단위가 붙은 경우 콤마 제거
        value = re.sub(r'(\d{1,3}(?:,\d{3})+)(주|원|억|건)', r'\1\2', value)
        value = value.replace(',', '')  # 모든 콤마 제거
        
        # 2. ❌ 문제: "비율(%): 28.1%" → "비율: 28.1%"
        value = re.sub(r'비율\(%\):\s*(\d+\.?\d*)%?', r'비율: \1%', value)
        
        # 3. % 중복 제거
        value = re.sub(r'(\d+\.?\d*)%%', r'\1%', value)
        
        # 이미 단위가 있으면 그대로
        if any(unit in value for unit in ['원', '주', '억', '%', '건']):
            return value
        
        # 콤마로 구분된 큰 숫자 (안전한 콤마 처리)
        if re.match(r'^[\d,]+$', value.replace(',', '')):
            # 콤마 제거 후 숫자 변환
            clean_value = value.replace(',', '').replace(' ', '')
            try:
                num = int(clean_value)
                
                # 문맥에서 단위 추론 (주석 필드 제외)
                if ('주식' in key or '주' in key) and '주석' not in key and '주 석' not in key:
                    return f"{num:,}주"
                elif '자산' in key or '부채' in key or '자본' in key or '매출' in key or '금액' in key:
                    if abs(num) >= self.config.currency_unit_threshold:
                        eok = num / 100_000_000
                        if eok == int(eok):
                            return f"{int(eok):,}억원"
                        else:
                            return f"{eok:,.1f}억원"
                    return f"{num:,}원"
                
            except:
                pass
        
        return value


def process_jsonl_file(input_file: Path, output_file: Path):
    """
    Step 2: JSONL 파일 정규화
    
    입력: step1_parser의 JSONL 파일
    출력: step2_normalized의 JSONL 파일
    """
    
    normalizer = DataNormalizer()
    
    processed_chunks = []
    error_count = 0
    
    for chunk in read_jsonl(input_file):
        try:
            normalized_chunk = normalizer.normalize_chunk(chunk)
            processed_chunks.append(normalized_chunk)
        except Exception as e:
            print(f"⚠️  청크 처리 실패: {e}")
            error_count += 1
            continue
    
    # 정규화된 청크 저장
    write_jsonl(output_file, processed_chunks)
    
    print(f"✅ {len(processed_chunks)}개 청크 정규화 완료")
    if error_count > 0:
        print(f"⚠️  {error_count}개 청크 처리 실패")


def process_directory(input_dir: Path, output_dir: Path):
    """
    Step 2: 디렉토리 내 모든 JSONL 파일 정규화
    
    입력: data/transform/parser/
    출력: data/transform/normalized/
    """
    
    ensure_output_dir(output_dir)
    
    jsonl_files = get_file_list(input_dir)
    
    if not jsonl_files:
        print("❌ JSONL 파일이 없습니다.")
        return
    
    print("=" * 80)
    print("Transform Pipeline - Step 2: 데이터 정규화 및 품질 개선")
    print("=" * 80)
    print(f"📁 입력: {input_dir}")
    print(f"📁 출력: {output_dir}")
    print(f"📄 처리할 파일 수: {len(jsonl_files)}개")
    print(f"\n처리 내용: 마크다운 제거, 숫자 단위 변환, 날짜 정규화, 품질 개선")
    print(f"다음 단계: chunker.py로 스마트 청킹 수행")
    print("=" * 80)
    print()
    
    for i, input_file in enumerate(jsonl_files, 1):
        print(f"[{i}/{len(jsonl_files)}] 처리 중: {input_file.name}")
        output_file = output_dir / input_file.name
        process_jsonl_file(input_file, output_file)
        print(f"  💾 저장: {output_file.name}")
        print()
    
    print("=" * 80)
    print("Step 2 완료!")
    print("=" * 80)


def main():
    """Normalizer 메인 함수"""
    import sys
    
    # 디렉토리 모드 (권장)
    if len(sys.argv) == 1:
        # 기본 경로 사용
        paths = get_transform_paths(__file__)
        input_dir = paths['parser_dir']
        output_dir = paths['normalized_dir']
        
        process_directory(input_dir, output_dir)
    
    # 단일 파일 모드
    elif len(sys.argv) == 3:
        input_file = Path(sys.argv[1])
        output_file = Path(sys.argv[2])
        process_jsonl_file(input_file, output_file)
    
    else:
        print("사용법:")
        print("  1. 디렉토리 모드 (권장): python normalizer.py")
        print("  2. 단일 파일 모드:      python normalizer.py <input.jsonl> <output.jsonl>")
        sys.exit(1)


if __name__ == "__main__":
    main()
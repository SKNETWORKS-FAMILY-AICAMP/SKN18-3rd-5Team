#!/usr/bin/env python3
"""
====================================================================================
Transform Pipeline - Step 2: 데이터 정규화 및 자연어 품질 개선
====================================================================================

[파이프라인 순서]
1. structured.py      → 마크다운을 구조화된 청크로 변환
2. data_normalizer.py → 데이터 정규화 및 자연어 품질 개선 (현재 파일)
3. chunker.py         → 스마트 청킹 및 메타데이터 강화

[이 파일의 역할]
- 날짜 형식 통일 (YYYY-MM-DD)
- 통화 단위 표준화 (억원 변환)
- 목차 및 불필요한 내용 제거
- 마크다운 문법 정리
- 자연어 품질 개선 (반복 축소, 문맥 추가)

[입력]
- data/transform/structured/*_chunks.jsonl (structured.py 출력)

[출력]
- data/transform/normalized/*_chunks.jsonl (정규화된 청크)
====================================================================================
"""

import re
import json
import math
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import unicodedata

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        RecursiveCharacterTextSplitter = None
        print("⚠️  LangChain not available. Text chunks will not be split further.")


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
                r'\*\*목\s+차\*\*.*?(?=\n\n|\Z)',  # 목차 (마크다운)
                r'I+\.\s+[가-힣\s]+‥+\s*\d+',  # 로마 숫자 목차
                r'제\s*\d+\s*\([전당]*\)\s*기',  # 중복 기수 표시
                # 상세한 목차 패턴
                r'【[^】]*】\s*-+\s*\d+',  # 【 제목 】 -------- 페이지번호
                # 목차 블록 전체 제거 (로마숫자/아라비아숫자 + 제목 + 점선 + 페이지번호)
                r'(?:^|\n)(?:[IVX]+\.|[0-9\-\.]+)\s+[가-힣\s\(\)]+\s+-+\s*\d+(?:\n[IVX0-9\-\.]+\s+[가-힣\s\(\)]+\s+-+\s*\d+)*',
            ]


class DataNormalizer:
    """데이터 정규화 처리"""
    
    def __init__(self, config: NormalizationConfig = None):
        self.config = config or NormalizationConfig()
    
    def normalize_chunk(self, chunk: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """청크 단위 정규화

        Returns:
            단일 chunk 또는 text 타입인 경우 분할된 chunk 리스트
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

        # text 타입 청크는 LangChain splitter로 분할
        if chunk.get('chunk_type') == 'text' and LANGCHAIN_AVAILABLE:
            return self._split_text_chunk(chunk)

        return chunk

    def _split_text_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """text 타입 청크를 LangChain splitter로 분할

        적응형 chunk_size 사용: max(300, min(1000, ceil(total_length // 30)))
        """
        text = chunk.get('natural_text', '')
        if not text or len(text) < 200:
            # 짧은 텍스트는 분할하지 않음
            return [chunk]

        # 적응형 chunk_size 계산
        total_length = len(text)
        chunk_size = max(300, min(1000, math.ceil(total_length / 30)))
        chunk_overlap = min(50, chunk_size // 5)

        # LangChain splitter 생성
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        # 텍스트 분할
        split_texts = text_splitter.split_text(text)

        # 분할된 텍스트로 chunk 생성
        result_chunks = []
        base_chunk_id = chunk.get('chunk_id', '')

        for idx, split_text in enumerate(split_texts):
            new_chunk = chunk.copy()
            new_chunk['natural_text'] = split_text
            # chunk_id에 분할 인덱스 추가
            if '_split_' in base_chunk_id:
                # 이미 분할된 경우 새로운 인덱스로 교체
                new_chunk['chunk_id'] = re.sub(r'_split_\d+$', f'_split_{idx}', base_chunk_id)
            else:
                new_chunk['chunk_id'] = f"{base_chunk_id}_split_{idx}"

            # metadata에 분할 정보 추가
            if 'metadata' not in new_chunk:
                new_chunk['metadata'] = {}
            new_chunk['metadata']['split_index'] = idx
            new_chunk['metadata']['total_splits'] = len(split_texts)
            new_chunk['metadata']['chunk_size'] = chunk_size

            result_chunks.append(new_chunk)

        return result_chunks

    def _improve_natural_text(
        self, 
        text: str, 
        chunk_type: str,
        structured_data: Dict,
        metadata: Dict,
        section_path: str = ""
    ) -> str:
        """자연어 품질 개선"""
        
        # 1. 목차 및 불필요 패턴 제거
        text = self._remove_unnecessary_content(text)
        
        # 2. 마크다운 제거
        text = self._clean_markdown(text)
        
        # 3. 타입별 처리
        if chunk_type == 'table_row':
            text = self._improve_table_text(text, structured_data, metadata, section_path)
        elif chunk_type == 'text':
            text = self._improve_text_content(text, metadata, section_path)
        
        # 4. 공백 정리
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
        """테이블 자연어 개선 - 자연스러운 문장 생성"""

        section = section_path

        # 1. structured_data 기반 자연어 문장 생성 (우선순위 1)
        natural_sentence = self._generate_natural_sentence(structured_data, section, metadata)
        if natural_sentence:
            return natural_sentence

        # 2. 반복 제거 및 기본 개선
        text = self._reduce_repetition(text)

        # 3. 재무제표 특화 처리 (공백 제거 후 체크)
        if any(kw in section.replace(' ', '') for kw in ['재무', '손익', '자산', '재무제표']):
            text = self._improve_financial_text(text, structured_data)

        # 4. 주식 정보 특화 처리
        elif '주식' in section:
            text = self._improve_stock_text(text, structured_data)

        # 5. 날짜 정규화
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
        """자기주식 취득 자연어 문장 생성"""
        company = metadata.get('corp_name', '회사')
        doc_name = metadata.get('document_name', '')
        
        # 데이터 추출
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
            key_lower = key.lower().replace(' ', '')
            
            # 계약금액
            if '계약금액' in key or '금액' in key:
                if value and value != '-' and value.replace(',', '').isdigit():
                    contract_amount = self._format_financial_amount(value)
            
            # 시작일
            elif '시작일' in key or '개시' in key:
                start_date = self._normalize_dates_in_text(str(value))
            
            # 종료일
            elif '종료일' in key or '만료' in key:
                end_date = self._normalize_dates_in_text(str(value))
            
            # 목적
            elif '목적' in key or '용도' in key:
                purpose = value
            
            # 증권사/중개업자
            elif '체결기관' in key or '중개업자' in key or '증권' in key:
                broker = value
            
            # 결정일
            elif '결정일' in key or '예정일자' in key:
                decision_date = self._normalize_dates_in_text(str(value))
            
            # 보유 주식수
            elif '보통주식' in key and value and value.replace(',', '').isdigit():
                shares_before = self._format_number_with_unit(value, '주')
            
            # 보유 비율
            elif '비율' in key and value and value != '-':
                shares_ratio = value if '%' in str(value) else f"{value}%"
        
        # 자연어 문장 생성
        parts = []
        
        # 기본 문장
        if decision_date:
            parts.append(f"{company}는 {decision_date}에 자기주식 취득 신탁계약 체결을 결정했습니다.")
        else:
            parts.append(f"{company}는 자기주식 취득 신탁계약을 체결했습니다.")
        
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
    
    def _format_financial_amount(self, value: str) -> str:
        """금액을 억원/만원 단위로 포맷팅"""
        try:
            num = int(str(value).replace(',', '').replace(' ', ''))
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
        """숫자에 단위 추가"""
        try:
            num = int(str(value).replace(',', '').replace(' ', ''))
            return f"{num:,}{unit}"
        except (ValueError, TypeError):
            return f"{value}{unit}"
    
    def _reduce_repetition(self, text: str) -> str:
        """과도한 반복 축소 및 자연어 개선"""

        # 1. 동일한 키-값 반복 제거
        # "중소기업 해당 여부: 중견기업 해당 여부, 중소기업 해당 여부: 중견기업 해당 여부"
        # → "중소기업 해당 여부: 중견기업 해당 여부"
        parts = text.split(', ')
        seen = set()
        unique_parts = []
        for part in parts:
            part_clean = part.strip()
            if part_clean and part_clean not in seen:
                unique_parts.append(part_clean)
                seen.add(part_clean)
        text = ', '.join(unique_parts)

        # 2. "키: 키" 패턴 제거 (동일한 키와 값)
        # "사업연도: 사업연도" → ""
        text = re.sub(r'([^,:]+):\s*\1(?=,|$)', '', text)

        # 3. "은(는)" 제거
        text = re.sub(r'은\(는\)', '', text)

        # 4. 빈 항목 정리 (": ," 또는 시작/끝의 콤마)
        text = re.sub(r':\s*,', ',', text)
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'^,\s*|\s*,$', '', text)

        # 5. 불필요한 공백 정리
        text = re.sub(r'\s{2,}', ' ', text)

        # 6. 콤마 뒤 공백 통일
        text = re.sub(r',\s*', ', ', text)

        # 7. 콜론 뒤 공백 통일
        text = re.sub(r':\s+', ': ', text)

        return text.strip()
    
    def _improve_financial_text(self, text: str, data: Dict) -> str:
        """재무 데이터 자연어 개선 - 일관된 단위 추가"""
        
        # 1. 먼저 퍼센트 처리 (소수점이 있는 작은 숫자)
        # 0.XX 형태 → XX%
        text = re.sub(
            r'\b0\.(\d{1,2})\b',
            lambda m: f"{int(m.group(1))}%" if m.group(1)[0] != '0' else f"{float(m.group(0))*100:.1f}%",
            text
        )
        
        # 2. 큰 숫자에 단위 추가 (억원)
        def add_currency_unit(match):
            num_str = match.group(0).replace(',', '')
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
        
        # 8자리 이상 숫자 (이미 단위 없는 경우만)
        text = re.sub(r'\b(\d{8,})(?![억만원%주])', add_currency_unit, text)
        
        # 콤마가 있는 숫자 (이미 단위 없는 경우만)
        text = re.sub(r'\b(\d{1,3}(?:,\d{3})+)(?![억만원%주])', add_currency_unit, text)
        
        # 4자리 이상 숫자 (단위 없는 경우)
        text = re.sub(r'\b(\d{4,7})(?![억만원%주,])', add_currency_unit, text)
        
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
        
        # 2. "주" 단위 추가 (이미 단위가 없는 경우만)
        # 콤마가 있는 숫자
        text = re.sub(r'\b(\d{1,3}(?:,\d{3})+)(?![억만원%주])', r'\1주', text)
        
        # 4자리 이상 숫자 (단위 없는 경우)
        text = re.sub(r'\b(\d{4,})(?![억만원%주,])', r'\1주', text)
        
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
        """컨텍스트 기반 숫자 단위 추가"""
        
        # 이미 단위가 있는지 체크
        has_unit_pattern = r'(?:[억만천백십]|원|주|%|건|개|명|년|월|일)'
        
        # 1. 재무/금액 관련
        if any(keyword in text for keyword in ['매출', '자산', '부채', '자본', '금액', '가격', '원']):
            # 8자리 이상 → 억원
            text = re.sub(
                r'\b(\d{8,})(?!' + has_unit_pattern + ')',
                lambda m: f"{int(m.group(1))/100_000_000:,.0f}억원",
                text
            )
            # 콤마 숫자 → 적절한 단위
            text = re.sub(
                r'\b(\d{1,3}(?:,\d{3})+)(?!' + has_unit_pattern + ')',
                lambda m: self._format_currency(m.group(1)),
                text
            )
        
        # 2. 주식 관련
        elif any(keyword in text for keyword in ['주식', '주수', '보통주', '우선주']):
            text = re.sub(
                r'\b(\d{1,3}(?:,\d{3})+)(?!' + has_unit_pattern + ')',
                r'\1주',
                text
            )
            text = re.sub(r'\b(\d{4,})(?!' + has_unit_pattern + ')', r'\1주', text)
        
        # 3. 비율 관련
        elif any(keyword in text for keyword in ['비율', '율', '지분', '점유']):
            # 0.XX → XX%
            text = re.sub(r'\b0\.(\d+)\b', lambda m: f"{float(m.group(0))*100:.1f}%", text)
        
        return text
    
    def _format_currency(self, num_str: str) -> str:
        """금액 포맷팅"""
        try:
            num = int(num_str.replace(',', ''))
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
        """텍스트 내 날짜 정규화"""
        
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
        
        # 3. 기수 표현 정규화 "제 54 기" → "제54기"
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
        """숫자 값 정규화 (단위 추가)"""
        
        # 이미 단위가 있으면 그대로
        if any(unit in value for unit in ['원', '주', '억', '%', '건']):
            return value
        
        # 콤마로 구분된 큰 숫자
        if re.match(r'^[\d,]+$', value.replace(',', '')):
            num_str = value.replace(',', '')
            try:
                num = int(num_str)
                
                # 문맥에서 단위 추론 (주석 필드 제외)
                if ('주식' in key or '주' in key) and '주석' not in key and '주 석' not in key:
                    return f"{value}주"
                elif '자산' in key or '부채' in key or '자본' in key or '매출' in key or '금액' in key:
                    if abs(num) >= self.config.currency_unit_threshold:
                        eok = num / 100_000_000
                        if eok == int(eok):
                            return f"{int(eok):,}억원"
                        else:
                            return f"{eok:,.1f}억원"
                    return f"{value}원"
                
            except:
                pass
        
        return value


def process_jsonl_file(input_file: str, output_file: str):
    """
    Step 2: JSONL 파일 정규화
    
    입력: step1_structured의 JSONL 파일
    출력: step2_normalized의 JSONL 파일
    """
    
    normalizer = DataNormalizer()
    
    processed_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_no, line in enumerate(infile, 1):
            try:
                chunk = json.loads(line)
                result = normalizer.normalize_chunk(chunk)

                # normalize_chunk은 단일 chunk 또는 chunk 리스트를 반환
                if isinstance(result, list):
                    # 분할된 경우 모든 chunk 저장
                    for normalized_chunk in result:
                        outfile.write(json.dumps(normalized_chunk, ensure_ascii=False) + '\n')
                        processed_count += 1
                else:
                    # 단일 chunk 저장
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    processed_count += 1
            except Exception as e:
                print(f"⚠️  Line {line_no} 처리 실패: {e}")
                error_count += 1
                continue
    
    print(f"✅ {processed_count}개 청크 정규화 완료")
    if error_count > 0:
        print(f"⚠️  {error_count}개 청크 처리 실패")


def process_directory(input_dir: str, output_dir: str):
    """
    Step 2: 디렉토리 내 모든 JSONL 파일 정규화
    
    입력: data/transform/structured/
    출력: data/transform/normalized/
    """
    from pathlib import Path
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    jsonl_files = list(input_path.glob("*_chunks.jsonl"))
    
    if not jsonl_files:
        print("❌ JSONL 파일이 없습니다.")
        return
    
    print("=" * 80)
    print("Transform Pipeline - Step 2: 데이터 정규화 및 품질 개선")
    print("=" * 80)
    print(f"📁 입력: {input_path}")
    print(f"📁 출력: {output_path}")
    print(f"📄 처리할 파일 수: {len(jsonl_files)}개")
    print(f"\n처리 내용: 마크다운 제거, 숫자 단위 변환, 날짜 정규화, 품질 개선")
    print(f"다음 단계: chunker.py로 스마트 청킹 수행")
    print("=" * 80)
    print()
    
    for i, input_file in enumerate(jsonl_files, 1):
        print(f"[{i}/{len(jsonl_files)}] 처리 중: {input_file.name}")
        output_file = output_path / input_file.name
        process_jsonl_file(str(input_file), str(output_file))
        print(f"  💾 저장: {output_file.name}")
        print()
    
    print("=" * 80)
    print("Step 2 완료!")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # 디렉토리 모드 (권장)
    if len(sys.argv) == 1:
        # 기본 경로 사용
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent.parent.parent / "data"
        input_dir = data_dir / "transform" / "structured"
        output_dir = data_dir / "transform" / "normalized"
        
        process_directory(str(input_dir), str(output_dir))
    
    # 단일 파일 모드
    elif len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        process_jsonl_file(input_file, output_file)
    
    else:
        print("사용법:")
        print("  1. 디렉토리 모드 (권장): python data_normalizer.py")
        print("  2. 단일 파일 모드:      python data_normalizer.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
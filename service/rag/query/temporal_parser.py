#!/usr/bin/env python3
"""
시간 표현 파싱 및 필터 생성
금융 문서의 시계열 쿼리 처리
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class TemporalInfo:
    """시간 정보 데이터 클래스"""
    years: List[int]
    quarters: List[int]
    relative: Optional[str]  # 'recent', 'last_year', 'trend', etc.
    date_range: Optional[Dict[str, str]]  # {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}
    filters: Dict[str, Any]  # SQL 필터용


class TemporalQueryParser:
    """시간 표현 파싱 및 SQL 필터 생성"""
    
    def __init__(self):
        """초기화"""
        self.current_year = datetime.now().year
        self.current_quarter = (datetime.now().month - 1) // 3 + 1
    
    def parse(self, query: str) -> TemporalInfo:
        """
        쿼리에서 시간 정보 추출
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            TemporalInfo: 파싱된 시간 정보
            
        예시:
            "삼성전자의 2024년 3분기 매출액은?" 
            → years=[2024], quarters=[3]
            
            "최근 3개 분기 실적 추이"
            → relative='trend', quarters=[현재-2, 현재-1, 현재]
        """
        temporal_info = TemporalInfo(
            years=[],
            quarters=[],
            relative=None,
            date_range=None,
            filters={}
        )
        
        # 1. 명시적 연도 추출
        temporal_info.years = self._extract_years(query)
        
        # 2. 명시적 분기 추출
        temporal_info.quarters = self._extract_quarters(query)
        
        # 3. 상대적 시간 표현
        temporal_info.relative = self._extract_relative_time(query)
        
        # 4. 상대적 표현을 구체적인 연도/분기로 변환
        if temporal_info.relative:
            self._resolve_relative_time(temporal_info)
        
        # 5. 날짜 범위 추출
        temporal_info.date_range = self._extract_date_range(query)
        
        # 6. SQL 필터 생성
        temporal_info.filters = self._generate_filters(temporal_info)
        
        return temporal_info
    
    def _extract_years(self, query: str) -> List[int]:
        """연도 추출"""
        years = []
        
        # "2024년", "2024", "20년" 형태
        patterns = [
            r'(20\d{2})년?',  # 2024년, 2024
            r'(\d{2})년',     # 24년
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                year = int(match)
                # 2자리 연도를 4자리로 변환
                if year < 100:
                    year = 2000 + year
                if 2000 <= year <= 2100:
                    years.append(year)
        
        return sorted(set(years))
    
    def _extract_quarters(self, query: str) -> List[int]:
        """분기 추출"""
        quarters = []
        
        # "1분기", "Q1", "3분기", "제1사분기" 형태
        patterns = [
            r'([1-4])분기',
            r'Q([1-4])',
            r'제([1-4])분기',
            r'제([1-4])사분기'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                quarter = int(match)
                if 1 <= quarter <= 4:
                    quarters.append(quarter)
        
        return sorted(set(quarters))
    
    def _extract_relative_time(self, query: str) -> Optional[str]:
        """상대적 시간 표현 추출"""
        
        # 최근 / 최신
        if re.search(r'최근|최신|현재', query):
            # 추이/변화/트렌드가 함께 있으면 'trend'
            if re.search(r'추이|변화|트렌드|증감|성장|감소|증가', query):
                return 'trend'
            # 숫자가 있으면 'recent_n'
            match = re.search(r'최근\s*(\d+)\s*(분기|년|개월)', query)
            if match:
                return f"recent_{match.group(1)}_{match.group(2)}"
            return 'recent'
        
        # 작년 / 전년
        if re.search(r'작년|전년|지난해', query):
            return 'last_year'
        
        # 전 분기
        if re.search(r'전\s*분기|이전\s*분기', query):
            return 'last_quarter'
        
        # 비교
        if re.search(r'비교|대비|차이', query):
            return 'comparison'
        
        return None
    
    def _resolve_relative_time(self, temporal_info: TemporalInfo):
        """상대적 시간 표현을 구체적인 연도/분기로 변환"""
        
        if temporal_info.relative == 'recent':
            # 최근 = 현재 연도, 현재 분기
            if not temporal_info.years:
                temporal_info.years = [self.current_year]
            if not temporal_info.quarters:
                temporal_info.quarters = [self.current_quarter]
        
        elif temporal_info.relative == 'last_year':
            # 작년
            temporal_info.years = [self.current_year - 1]
        
        elif temporal_info.relative == 'last_quarter':
            # 전 분기
            if self.current_quarter == 1:
                temporal_info.years = [self.current_year - 1]
                temporal_info.quarters = [4]
            else:
                temporal_info.years = [self.current_year]
                temporal_info.quarters = [self.current_quarter - 1]
        
        elif temporal_info.relative == 'trend':
            # 추이 = 최근 3개 분기
            if not temporal_info.quarters:
                current_q = self.current_quarter
                current_y = self.current_year
                
                quarters = []
                years = []
                for i in range(3):
                    quarters.append(current_q)
                    years.append(current_y)
                    
                    current_q -= 1
                    if current_q < 1:
                        current_q = 4
                        current_y -= 1
                
                temporal_info.quarters = sorted(set(quarters))
                temporal_info.years = sorted(set(years))
        
        elif temporal_info.relative and temporal_info.relative.startswith('recent_'):
            # "최근 3분기", "최근 2년" 등
            parts = temporal_info.relative.split('_')
            if len(parts) == 3:
                _, n, unit = parts
                n = int(n)
                
                if unit == '분기':
                    # 최근 n개 분기
                    current_q = self.current_quarter
                    current_y = self.current_year
                    
                    quarters = []
                    years = []
                    for i in range(n):
                        quarters.append(current_q)
                        years.append(current_y)
                        
                        current_q -= 1
                        if current_q < 1:
                            current_q = 4
                            current_y -= 1
                    
                    temporal_info.quarters = sorted(set(quarters))
                    temporal_info.years = sorted(set(years))
                
                elif unit == '년':
                    # 최근 n년
                    years = [self.current_year - i for i in range(n)]
                    temporal_info.years = sorted(years)
    
    def _extract_date_range(self, query: str) -> Optional[Dict[str, str]]:
        """날짜 범위 추출"""
        
        # "YYYY-MM-DD부터 YYYY-MM-DD까지" 형태
        pattern = r'(\d{4}-\d{2}-\d{2})\s*(?:부터|에서|~)\s*(\d{4}-\d{2}-\d{2})\s*(?:까지|까지|까지)'
        match = re.search(pattern, query)
        
        if match:
            return {
                'start': match.group(1),
                'end': match.group(2)
            }
        
        return None
    
    def _generate_filters(self, temporal_info: TemporalInfo) -> Dict[str, Any]:
        """SQL 필터 생성"""
        filters = {}
        
        # 연도 필터
        if temporal_info.years:
            if len(temporal_info.years) == 1:
                filters['fiscal_year'] = temporal_info.years[0]
            else:
                filters['fiscal_year__in'] = temporal_info.years
        
        # 분기 필터 (메타데이터에 'fiscal_quarter' 필드가 있다고 가정)
        if temporal_info.quarters:
            if len(temporal_info.quarters) == 1:
                filters['fiscal_quarter'] = temporal_info.quarters[0]
            else:
                filters['fiscal_quarter__in'] = temporal_info.quarters
        
        # 날짜 범위 필터
        if temporal_info.date_range:
            filters['rcept_dt__gte'] = temporal_info.date_range['start'].replace('-', '')
            filters['rcept_dt__lte'] = temporal_info.date_range['end'].replace('-', '')
        
        return filters
    
    def format_query_with_temporal(self, query: str, temporal_info: TemporalInfo) -> str:
        """
        쿼리에 시간 정보를 명시적으로 추가
        
        예시:
            "삼성전자 매출액" + temporal_info(2024, Q3)
            → "삼성전자의 2024년 3분기 매출액"
        """
        parts = [query]
        
        if temporal_info.years:
            year_str = ', '.join([f"{y}년" for y in temporal_info.years])
            parts.append(year_str)
        
        if temporal_info.quarters:
            quarter_str = ', '.join([f"{q}분기" for q in temporal_info.quarters])
            parts.append(quarter_str)
        
        return ' '.join(parts)

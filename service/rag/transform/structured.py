#!/usr/bin/env python3
"""
====================================================================================
Transform Pipeline - Step 1: 구조화 및 1차 청킹
====================================================================================

[파이프라인 순서]
1. structured.py      → 마크다운을 구조화된 청크로 변환
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
- data/transform/structured/*_chunks.jsonl (1차 청크)
====================================================================================
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict, field
import hashlib
import sys


# ==========================================
# Table -> Natural Language (Convert)
# ==========================================
@dataclass
class Chunk:
    """청크 데이터 구조"""
    chunk_id: str
    doc_id: str
    chunk_type: str  # 'text', 'table_row', 'list_item'
    section_path: str
    
    # 구조화된 데이터
    structured_data: Dict[str, Any] = field(default_factory=dict)
    
    # 자연어 변환 (검색용)
    natural_text: str = ""
    
    # 메타데이터 (부가 정보만)
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        """테이블 처리"""
        
        # 테이블 추출
        table_lines = []
        i = start_idx
        while i < len(lines) and (lines[i].startswith('|') or lines[i].strip() == ''):
            if lines[i].startswith('|'):
                table_lines.append(lines[i])
            i += 1
        
        if len(table_lines) < 3:
            return i
        
        # 헤더 파싱 ([0,0]은 '구분'으로 처리)
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
            
            # 자연어 변환
            section_path = ' > '.join(self.current_section_path)
            natural_text = TableRowConverter.convert(headers, row, section_path)
            
            if not natural_text:
                continue
            
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
        
        return i
    
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


def main():
    """
    Step 1: 마크다운 → 1차 청크 변환
    
    입력: data/markdown/*.md
    출력: data/transform/structured/*_chunks.jsonl
    """
    
    # 명령행 인자 처리
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        max_files = None  # 전체 처리
        print("🔧 전체 파일 처리 모드")
    else:
        max_files = 10  # 테스트용 10개만
        print("🔧 테스트 모드 (10개 파일만 처리)")

    # 경로 설정 (transform 폴더 기준)
    script_dir = Path(__file__).parent  # service/rag/transform
    data_dir = script_dir.parent.parent.parent / "data"  # 프로젝트 루트/data
    markdown_dir = data_dir / "markdown"
    output_dir = data_dir / "transform" / "structured"

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

            # JSONL 저장
            output_file = output_dir / f"{md_file.stem}_chunks.jsonl"

            with open(output_file, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(json.dumps(asdict(chunk), ensure_ascii=False) + '\n')

            print(f"  💾 저장: {output_file.name}")

            # 통계 업데이트
            total_chunks += len(chunks)
            processed_files += 1

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
    table_chunks = [c for c in chunks if c.chunk_type == 'table_row'][:3]

    for i, chunk in enumerate(table_chunks):
        print(f"\n[{i+1}] {chunk.section_path}")
        print(f"타입: {chunk.chunk_type}")
        print(f"자연어: {chunk.natural_text}")
        print(f"구조화: {json.dumps(chunk.structured_data, ensure_ascii=False, indent=2)}")

    # 텍스트 청크 샘플
    text_chunks = [c for c in chunks if c.chunk_type == 'text'][:2]

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
    for chunk in chunks:
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
    
    main()
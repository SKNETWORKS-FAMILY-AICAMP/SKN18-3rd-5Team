#!/usr/bin/env python3
"""
====================================================================================
Transform Pipeline - Step 3: 스마트 청킹 및 메타데이터 강화
====================================================================================

[파이프라인 순서]
1. parser.py      → 마크다운을 구조화된 청크로 변환
2. normalizer.py  → 데이터 정규화 및 자연어 품질 개선
3. chunker.py     → 스마트 청킹 및 메타데이터 강화 (현재 파일)

[이 파일의 역할]
- 작은 청크들을 의미 단위로 병합
- 토큰 수 제한 내에서 최적 크기 조정
- 앞뒤 문맥 윈도우 추가
- 메타데이터 강화 (doc_type, data_category, keywords 등)
- 검색 최적화를 위한 토큰 수 계산

[입력]
- data/transform/normalized/*_chunks.jsonl (normalizer.py 출력)

[출력]
- data/transform/final/*_chunks.jsonl (최종 청크, 벡터 DB 저장 준비 완료)

[다음 단계]
- 벡터 DB에 임베딩 및 저장
====================================================================================
"""

import re
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import tiktoken

# 공통 모듈
from utils import read_jsonl, write_jsonl, get_file_list, ensure_output_dir, get_transform_paths

# LangChain text splitter (optional)
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
class ChunkConfig:
    """청킹 설정"""
    # 최대 토큰 수 (OpenAI embedding 기준)
    max_tokens: int = 7000  # 안전한 제한 (8192 - 여유분)
    
    # 청크 오버랩 (문맥 보존)
    overlap_tokens: int = 200
    
    # 최소 청크 크기
    min_tokens: int = 50
    
    # 임베딩 모델
    embedding_model: str = "text-embedding-3-small"


class SmartChunker:
    """스마트 청킹 처리"""
    
    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
        
        # 토크나이저 초기화
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.config.embedding_model)
        except:
            # fallback
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def split_text_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """text 타입 청크를 LangChain splitter로 분할
        
        normalizer.py에서 이동: 텍스트 분할은 chunker.py의 책임
        
        적응형 chunk_size 사용: max(300, min(1000, ceil(total_length // 30)))
        """
        if not LANGCHAIN_AVAILABLE:
            return [chunk]
        
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
    
    def split_table_row_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """table_row 타입 청크를 적응형 크기로 분할
        
        적응형 chunk_size 사용: max(300, min(1000, ceil(total_length // 30)))
        """
        text = chunk.get('natural_text', '')
        if not text or len(text) < 300:
            # 짧은 테이블은 분할하지 않음
            return [chunk]

        # 적응형 chunk_size 계산
        total_length = len(text)
        chunk_size = max(300, min(1000, math.ceil(total_length / 30)))
        
        # 텍스트를 청크 크기로 분할
        split_texts = []
        start = 0
        
        while start < total_length:
            end = start + chunk_size
            
            # 문장 경계에서 자르기 (콤마, 세미콜론, 마침표)
            if end < total_length:
                # 뒤에서부터 문장 구분자 찾기
                for i in range(min(100, chunk_size // 2), 0, -1):
                    if start + i < total_length and text[start + i] in [',', ';', '.', ' ']:
                        end = start + i + 1
                        break
            
            split_text = text[start:end].strip()
            if split_text:
                split_texts.append(split_text)
            
            start = end

        # 분할된 텍스트로 chunk 생성
        result_chunks = []
        base_chunk_id = chunk.get('chunk_id', '')

        for idx, split_text in enumerate(split_texts):
            new_chunk = chunk.copy()
            new_chunk['natural_text'] = split_text
            
            # chunk_id에 분할 인덱스 추가
            if '_split_' in base_chunk_id:
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
    
    def should_merge_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        작은 청크들을 병합
        - 테이블 행이 너무 작으면 관련 행들과 병합
        - 텍스트가 너무 짧으면 앞뒤와 병합
        """
        
        merged = []
        buffer = []
        buffer_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk['natural_text'])
            
            # 버퍼가 비어있으면 추가
            if not buffer:
                buffer.append(chunk)
                buffer_tokens = chunk_tokens
                continue
            
            # 같은 섹션이고 토큰 합이 max 이하면 병합
            if (self._same_section(buffer[-1], chunk) and 
                buffer_tokens + chunk_tokens <= self.config.max_tokens):
                buffer.append(chunk)
                buffer_tokens += chunk_tokens
            else:
                # 버퍼를 병합해서 저장
                merged.append(self._merge_buffer(buffer))
                buffer = [chunk]
                buffer_tokens = chunk_tokens
        
        # 남은 버퍼 처리
        if buffer:
            merged.append(self._merge_buffer(buffer))
        
        return merged
    
    def _same_section(self, chunk1: Dict, chunk2: Dict) -> bool:
        """같은 섹션인지 확인"""
        return (chunk1['section_path'] == chunk2['section_path'] and
                chunk1['chunk_type'] == chunk2['chunk_type'])
    
    def _merge_buffer(self, buffer: List[Dict]) -> Dict:
        """버퍼의 청크들을 하나로 병합"""
        
        if len(buffer) == 1:
            return buffer[0]
        
        # 자연어 텍스트 병합
        natural_texts = [c['natural_text'] for c in buffer]
        merged_text = ' '.join(natural_texts)
        
        # 구조화 데이터 병합 (테이블 행인 경우)
        merged_structured = {}
        if buffer[0]['chunk_type'] == 'table_row':
            for chunk in buffer:
                merged_structured.update(chunk.get('structured_data', {}))
        
        # 첫 번째 청크의 메타데이터 사용 (chunk_id는 새로 생성)
        merged_metadata = buffer[0]['metadata'].copy()
        merged_metadata['merged_count'] = len(buffer)
        
        return {
            'chunk_id': f"{buffer[0]['chunk_id']}_merged",
            'doc_id': buffer[0]['doc_id'],
            'chunk_type': buffer[0]['chunk_type'],
            'section_path': buffer[0]['section_path'],
            'structured_data': merged_structured,
            'natural_text': merged_text,
            'metadata': merged_metadata
        }
    
    def add_context_window(self, chunks: List[Dict]) -> List[Dict]:
        """
        각 청크에 앞뒤 문맥 추가
        - 이전/다음 청크의 일부를 메타데이터에 포함
        """
        
        for i, chunk in enumerate(chunks):
            # 이전 청크 문맥
            if i > 0:
                prev_text = chunks[i-1]['natural_text']
                chunk['metadata']['prev_context'] = self._truncate_text(
                    prev_text, 
                    max_tokens=100
                )
            
            # 다음 청크 문맥
            if i < len(chunks) - 1:
                next_text = chunks[i+1]['natural_text']
                chunk['metadata']['next_context'] = self._truncate_text(
                    next_text,
                    max_tokens=100
                )
        
        return chunks
    
    def enhance_metadata(self, chunk: Dict) -> Dict:
        """메타데이터 강화"""
        
        metadata = chunk['metadata']
        
        # 1. 문서 타입 추론
        metadata['doc_type'] = self._infer_doc_type(metadata.get('document_name', ''))
        
        # 2. 데이터 타입 분류
        metadata['data_category'] = self._classify_data_category(
            chunk['natural_text'],
            chunk.get('section_path', '')
        )
        
        # 3. 회계 연도 추출
        metadata['fiscal_year'] = self._extract_fiscal_year(
            chunk['natural_text'],
            chunk.get('structured_data', {})
        )
        
        # 4. 키워드 추출 (검색 강화)
        metadata['keywords'] = self._extract_keywords(
            chunk['natural_text'],
            chunk.get('structured_data', {})
        )
        
        # 5. 토큰 수 계산
        metadata['token_count'] = self._count_tokens(chunk['natural_text'])
        
        return chunk
    
    def _infer_doc_type(self, doc_name: str) -> str:
        """문서 타입 추론"""
        if '감사보고서' in doc_name:
            return 'audit_report'
        elif '사업보고서' in doc_name:
            return 'business_report'
        elif '분기보고서' in doc_name:
            return 'quarterly_report'
        else:
            return 'other'
    
    def _classify_data_category(self, text: str, section: str) -> str:
        """데이터 카테고리 분류"""
        
        # 재무제표
        if any(keyword in section for keyword in ['재무상태표', '손익계산서', '현금흐름']):
            return 'financial_statement'
        
        # 주식 정보
        elif '주식' in section:
            return 'stock_info'
        
        # 감사 의견
        elif '감사' in section and '의견' in text:
            return 'audit_opinion'
        
        # 주석
        elif '주석' in section:
            return 'footnote'
        
        # 기타
        else:
            return 'general'
    
    def _extract_fiscal_year(self, text: str, structured_data: Dict) -> Optional[int]:
        """회계 연도 추출"""
        
        # 구조화 데이터에서 추출
        for key, value in structured_data.items():
            if '기' in key or '년도' in key:
                match = re.search(r'20\d{2}', str(value))
                if match:
                    return int(match.group())
        
        # 텍스트에서 추출
        years = re.findall(r'20\d{2}', text)
        if years:
            return int(years[0])
        
        return None
    
    def _extract_keywords(self, text: str, structured_data: Dict) -> List[str]:
        """키워드 추출 (간단한 버전)"""
        
        keywords = set()
        
        # 구조화 데이터에서 키 추출
        keywords.update(structured_data.keys())
        
        # 텍스트에서 재무 용어 추출
        financial_terms = [
            '자산', '부채', '자본', '매출', '영업이익', '당기순이익',
            '현금', '유동', '비유동', '감가상각', '이자', '배당',
            '주식', '보통주', '우선주'
        ]
        
        for term in financial_terms:
            if term in text:
                keywords.add(term)
        
        return list(keywords)[:10]  # 최대 10개
    
    def _count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        try:
            return len(self.tokenizer.encode(text))
        except:
            # fallback: 대략 4자당 1토큰
            return len(text) // 4
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """텍스트를 최대 토큰 수로 자르기"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens) + "..."


def process_chunks_with_enhancement(chunks: List[Dict]) -> List[Dict]:
    """청크 강화 처리"""
    
    chunker = SmartChunker()
    
    # 0. 청크 분할 (text: LangChain splitter, table_row: 적응형 분할)
    split_chunks = []
    for chunk in chunks:
        if chunk.get('chunk_type') == 'text':
            split_chunks.extend(chunker.split_text_chunk(chunk))
        elif chunk.get('chunk_type') == 'table_row':
            split_chunks.extend(chunker.split_table_row_chunk(chunk))
        else:
            split_chunks.append(chunk)
    
    print(f"✅ 청크 분할 (text/table_row): {len(chunks)} → {len(split_chunks)}")
    
    # 1. 작은 청크 병합
    merged = chunker.should_merge_chunks(split_chunks)
    print(f"✅ 청크 병합: {len(split_chunks)} → {len(merged)}")
    
    # 2. 문맥 윈도우 추가
    with_context = chunker.add_context_window(merged)
    
    # 3. 메타데이터 강화
    enhanced = [chunker.enhance_metadata(chunk) for chunk in with_context]
    
    # 4. 큰 청크 추가 분할 (7000 토큰 이상)
    final_chunks = []
    oversized_count = 0
    split_count = 0
    
    for chunk in enhanced:
        token_count = chunk['metadata'].get('token_count', 0)
        if token_count > 7000:
            oversized_count += 1
            text = chunk['natural_text']
            
            # 목표: 3500 토큰씩 분할 (안전 마진 포함)
            # 한글 기준: 1토큰 ≈ 1.1자 → 3500토큰 ≈ 3850자
            target_chars = 3850
            num_parts = math.ceil(len(text) / target_chars)
            part_size = len(text) // num_parts
            
            if num_parts > 1:
                split_count += 1
                for idx in range(num_parts):
                    start = idx * part_size
                    end = start + part_size if idx < num_parts - 1 else len(text)
                    
                    # 문장 경계에서 자르기 (콤마, 공백, 줄바꿈)
                    if end < len(text):
                        for i in range(min(300, part_size // 3), 0, -1):
                            pos = start + i
                            if pos < len(text) and text[pos] in [',', ' ', '\n', '.', ':']:
                                end = pos + 1
                                break
                    
                    part_text = text[start:end].strip()
                    if not part_text:
                        continue
                    
                    new_chunk = chunk.copy()
                    new_chunk['natural_text'] = part_text
                    new_chunk['chunk_id'] = f"{chunk['chunk_id']}_oversized_{idx}"
                    new_chunk['metadata'] = chunk['metadata'].copy()
                    new_chunk['metadata']['oversized_split'] = True
                    new_chunk['metadata']['oversized_index'] = idx
                    new_chunk['metadata']['oversized_total'] = num_parts
                    new_chunk['metadata']['token_count'] = chunker._count_tokens(part_text)
                    
                    final_chunks.append(new_chunk)
            else:
                # 분할이 필요하지 않은 경우 (텍스트가 짧음)
                final_chunks.append(chunk)
        else:
            final_chunks.append(chunk)
    
    if oversized_count > 0:
        print(f"✅ 큰 청크 발견: {oversized_count}개 (7000+ 토큰)")
        if split_count > 0:
            print(f"✅ 큰 청크 분할: {split_count}개 청크 → 평균 3500 토큰으로 분할")
    
    return final_chunks


def process_jsonl_file(input_file: Path, output_file: Path):
    """
    Step 3: JSONL 파일 스마트 청킹
    
    입력: step2_normalized의 JSONL 파일
    출력: final의 JSONL 파일
    """
    
    # 모든 청크 읽기
    chunks = list(read_jsonl(input_file))
    
    # 스마트 청킹 적용
    enhanced_chunks = process_chunks_with_enhancement(chunks)
    
    # 저장
    write_jsonl(output_file, enhanced_chunks)
    
    print(f"✅ {len(enhanced_chunks)}개 청크 처리 완료")
    
    # 통계 출력
    token_counts = [c['metadata']['token_count'] for c in enhanced_chunks]
    print(f"  📊 평균 토큰 수: {sum(token_counts) / len(token_counts):.0f}")
    print(f"  📊 최소 토큰 수: {min(token_counts)}")
    print(f"  📊 최대 토큰 수: {max(token_counts)}")


def process_directory(input_dir: Path, output_dir: Path):
    """
    Step 3: 디렉토리 내 모든 JSONL 파일 스마트 청킹
    
    입력: data/transform/normalized/
    출력: data/transform/final/
    """
    
    ensure_output_dir(output_dir)
    
    jsonl_files = get_file_list(input_dir)
    
    if not jsonl_files:
        print("❌ JSONL 파일이 없습니다.")
        return
    
    print("=" * 80)
    print("Transform Pipeline - Step 3: 스마트 청킹 및 메타데이터 강화")
    print("=" * 80)
    print(f"📁 입력: {input_dir}")
    print(f"📁 출력: {output_dir}")
    print(f"📄 처리할 파일 수: {len(jsonl_files)}개")
    print(f"\n처리 내용: 텍스트 분할, 청크 병합, 문맥 윈도우 추가, 메타데이터 강화")
    print(f"다음 단계: 벡터 DB에 임베딩 및 저장")
    print("=" * 80)
    print()
    
    total_input = 0
    total_output = 0
    
    for i, input_file in enumerate(jsonl_files, 1):
        print(f"[{i}/{len(jsonl_files)}] 처리 중: {input_file.name}")
        
        # 입력 청크 수 카운트
        input_count = sum(1 for _ in read_jsonl(input_file))
        total_input += input_count
        
        output_file = output_dir / input_file.name
        process_jsonl_file(input_file, output_file)
        
        # 출력 청크 수 카운트
        output_count = sum(1 for _ in read_jsonl(output_file))
        total_output += output_count
        
        print(f"  💾 저장: {output_file.name}")
        print()
    
    print("=" * 80)
    print("Step 3 완료!")
    print("=" * 80)
    print(f"총 처리: {total_input}개 → {total_output}개 청크")
    if total_input != total_output:
        change_ratio = ((total_output - total_input) / total_input) * 100
        if change_ratio > 0:
            print(f"분할/병합 효과: {change_ratio:+.1f}% 변화")
        else:
            print(f"병합 효과: {abs(change_ratio):.1f}% 감소")


def main():
    """Chunker 메인 함수"""
    import sys
    
    # 디렉토리 모드 (권장)
    if len(sys.argv) == 1:
        # 기본 경로 사용
        paths = get_transform_paths(__file__)
        input_dir = paths['normalized_dir']
        output_dir = paths['final_dir']
        
        process_directory(input_dir, output_dir)
    
    # 단일 파일 모드
    elif len(sys.argv) == 3:
        input_file = Path(sys.argv[1])
        output_file = Path(sys.argv[2])
        process_jsonl_file(input_file, output_file)
    
    # 테스트 모드
    elif len(sys.argv) == 2 and sys.argv[1] == "--test":
        print("=" * 80)
        print("테스트 모드")
        print("=" * 80)
        
        sample_chunks = [
            {
                'chunk_id': 'test_001',
                'doc_id': 'doc_001',
                'chunk_type': 'table_row',
                'section_path': '재무제표 > 재무상태표',
                'natural_text': '유동자산은 14,220억원',
                'structured_data': {'과목': '유동자산', '금액': '1422091558149'},
                'metadata': {
                    'document_name': '감사보고서'
                }
            }
        ]
        
        enhanced = process_chunks_with_enhancement(sample_chunks)
        
        import json
        print(json.dumps(enhanced[0], ensure_ascii=False, indent=2))
    
    else:
        print("사용법:")
        print("  1. 디렉토리 모드 (권장): python chunker.py")
        print("  2. 단일 파일 모드:      python chunker.py <input.jsonl> <output.jsonl>")
        print("  3. 테스트 모드:         python chunker.py --test")
        sys.exit(1)


if __name__ == "__main__":
    main()
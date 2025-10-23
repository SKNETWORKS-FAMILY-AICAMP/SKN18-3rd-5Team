#!/usr/bin/env python3
"""
====================================================================================
Transform Pipeline - Step 3: 스마트 청킹 및 메타데이터 강화
====================================================================================

[파이프라인 순서]
1. structured.py      → 마크다운을 구조화된 청크로 변환
2. data_normalizer.py → 데이터 정규화 및 자연어 품질 개선
3. chunker.py         → 스마트 청킹 및 메타데이터 강화 (현재 파일)

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
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import tiktoken


@dataclass
class ChunkConfig:
    """청킹 설정"""
    # 최대 토큰 수 (OpenAI embedding 기준)
    max_tokens: int = 8000  # 여유있게 설정
    
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
    
    # 1. 작은 청크 병합
    merged = chunker.should_merge_chunks(chunks)
    print(f"✅ 청크 병합: {len(chunks)} → {len(merged)}")
    
    # 2. 문맥 윈도우 추가
    with_context = chunker.add_context_window(merged)
    
    # 3. 메타데이터 강화
    enhanced = [chunker.enhance_metadata(chunk) for chunk in with_context]
    
    return enhanced


def process_jsonl_file(input_file: str, output_file: str):
    """
    Step 3: JSONL 파일 스마트 청킹
    
    입력: step2_normalized의 JSONL 파일
    출력: final의 JSONL 파일
    """
    import json
    
    # 모든 청크 읽기
    chunks = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    # 스마트 청킹 적용
    enhanced_chunks = process_chunks_with_enhancement(chunks)
    
    # 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in enhanced_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"✅ {len(enhanced_chunks)}개 청크 처리 완료")
    
    # 통계 출력
    token_counts = [c['metadata']['token_count'] for c in enhanced_chunks]
    print(f"  📊 평균 토큰 수: {sum(token_counts) / len(token_counts):.0f}")
    print(f"  📊 최소 토큰 수: {min(token_counts)}")
    print(f"  📊 최대 토큰 수: {max(token_counts)}")


def process_directory(input_dir: str, output_dir: str):
    """
    Step 3: 디렉토리 내 모든 JSONL 파일 스마트 청킹
    
    입력: data/transform/normalized/
    출력: data/transform/final/
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
    print("Transform Pipeline - Step 3: 스마트 청킹 및 메타데이터 강화")
    print("=" * 80)
    print(f"📁 입력: {input_path}")
    print(f"📁 출력: {output_path}")
    print(f"📄 처리할 파일 수: {len(jsonl_files)}개")
    print(f"\n다음 단계: 벡터 DB에 임베딩 및 저장")
    print("=" * 80)
    print()
    
    total_input = 0
    total_output = 0
    
    for i, input_file in enumerate(jsonl_files, 1):
        print(f"[{i}/{len(jsonl_files)}] 처리 중: {input_file.name}")
        
        # 입력 청크 수 카운트
        with open(input_file, 'r', encoding='utf-8') as f:
            input_count = sum(1 for _ in f)
        total_input += input_count
        
        output_file = output_path / input_file.name
        process_jsonl_file(str(input_file), str(output_file))
        
        # 출력 청크 수 카운트
        with open(output_file, 'r', encoding='utf-8') as f:
            output_count = sum(1 for _ in f)
        total_output += output_count
        
        print(f"  💾 저장: {output_file.name}")
        print()
    
    print("=" * 80)
    print("Step 3 완료!")
    print("=" * 80)
    print(f"총 처리: {total_input}개 → {total_output}개 청크")
    if total_input > total_output:
        reduction = (1 - total_output/total_input) * 100
        print(f"병합 효과: {reduction:.1f}% 감소")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # 디렉토리 모드 (권장)
    if len(sys.argv) == 1:
        # 기본 경로 사용
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent.parent.parent / "data"
        input_dir = data_dir / "transform" / "normalized"
        output_dir = data_dir / "transform" / "final"
        
        process_directory(str(input_dir), str(output_dir))
    
    # 단일 파일 모드
    elif len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
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
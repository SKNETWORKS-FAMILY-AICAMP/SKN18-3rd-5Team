#!/usr/bin/env python3
"""
====================================================================================
Transform Pipeline - Utility Functions
====================================================================================

파일 I/O, 디렉토리 처리 유틸리티 및 공통 데이터 모델
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Iterator, Tuple
from dataclasses import dataclass, asdict, field


# ==========================================
# 공통 데이터 모델
# ==========================================
@dataclass
class Chunk:
    """청크 데이터 구조 (모든 파이프라인 단계에서 공통 사용)"""
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict로 변환"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Dict에서 생성"""
        return cls(**data)


# ==========================================
# 파일 I/O 유틸리티
# ==========================================
def read_jsonl(file_path: Path) -> Iterator[Dict[str, Any]]:
    """JSONL 파일 읽기 (제너레이터)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_no} JSON 파싱 실패: {e}")
                continue


def write_jsonl(file_path: Path, chunks: List[Dict[str, Any]]):
    """JSONL 파일 쓰기"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


def write_jsonl_stream(file_path: Path, chunk: Dict[str, Any]):
    """JSONL 파일 스트림 쓰기 (append)"""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


# ==========================================
# 디렉토리 처리 유틸리티
# ==========================================
def get_file_list(input_dir: Path, pattern: str = "*_chunks.jsonl") -> List[Path]:
    """디렉토리에서 파일 목록 가져오기"""
    return sorted(input_dir.glob(pattern))


def ensure_output_dir(output_dir: Path):
    """출력 디렉토리 생성"""
    output_dir.mkdir(parents=True, exist_ok=True)


# ==========================================
# 경로 설정 유틸리티
# ==========================================
def get_project_paths(script_file: str = None) -> Tuple[Path, Path, Path, Path, Path]:
    """
    프로젝트 경로들을 반환하는 공통 함수
    
    Args:
        script_file: __file__ 값 (기본값: None이면 현재 파일 기준)
    
    Returns:
        Tuple[script_dir, etl_dir, service_dir, project_root, data_dir]
    """
    if script_file is None:
        script_dir = Path(__file__).parent  # service/etl/transform
    else:
        script_dir = Path(script_file).parent  # service/etl/transform
    
    etl_dir = script_dir.parent         # service/etl
    service_dir = etl_dir.parent        # service
    project_root = service_dir.parent   # project root
    data_dir = project_root / "data"
    
    return script_dir, etl_dir, service_dir, project_root, data_dir


def get_transform_paths(script_file: str = None) -> Dict[str, Path]:
    """
    Transform 파이프라인 관련 경로들을 딕셔너리로 반환
    
    Args:
        script_file: __file__ 값 (기본값: None이면 현재 파일 기준)
    
    Returns:
        Dict with keys: markdown_dir, parser_dir, normalized_dir, final_dir
    """
    script_dir, etl_dir, service_dir, project_root, data_dir = get_project_paths(script_file)
    
    return {
        'markdown_dir': data_dir / "markdown",
        'parser_dir': data_dir / "transform" / "parser",
        'normalized_dir': data_dir / "transform" / "normalized", 
        'final_dir': data_dir / "transform" / "final"
    }

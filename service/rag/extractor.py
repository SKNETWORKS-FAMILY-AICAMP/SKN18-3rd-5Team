#!/usr/bin/env python3
"""
XML to Markdown 변환기 (최종 버전)
- ROWSPAN/COLSPAN 병합 셀을 분할하여 각 행에 값 복사
- 각 행이 독립적인 의미를 가지도록 정규화
- 청크 분할과 벡터 DB 저장을 위한 구조 최적화
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from lxml import etree


# ==========================================
# 1. 메타데이터 추출
# ==========================================

def extract_xml_metadata(root: etree._Element) -> Dict[str, Any]:
    """XML에서 메타데이터 추출"""
    metadata = {}

    doc_name_elem = root.find('.//DOCUMENT-NAME')
    if doc_name_elem is not None:
        metadata['document_name'] = doc_name_elem.text
        metadata['document_acode'] = doc_name_elem.get('ACODE', '')

    company_elem = root.find('.//COMPANY-NAME')
    if company_elem is not None:
        metadata['company_name'] = company_elem.text
        metadata['company_aregcik'] = company_elem.get('AREGCIK', '')

    return metadata


# ==========================================
# 2. 마크다운 헤더 생성
# ==========================================

def generate_markdown_header(
    corp_code: str,
    corp_name: str,
    stock_code: str,
    rcept_dt: str,
    suffix: str,
    xml_metadata: Dict[str, Any]
) -> str:
    """마크다운 파일 상단 메타데이터 헤더 생성"""

    header_lines = [
        "---",
        f"corp_code: {corp_code}",
        f"corp_name: {corp_name}",
        f"stock_code: {stock_code}",
        f"rcept_dt: {rcept_dt}",
    ]

    if suffix:
        header_lines.append(f"suffix: {suffix}")

    if xml_metadata.get('document_name'):
        header_lines.append(f"document_name: {xml_metadata['document_name']}")

    if xml_metadata.get('document_acode'):
        header_lines.append(f"document_acode: {xml_metadata['document_acode']}")

    if xml_metadata.get('formula_version'):
        header_lines.append(f"formula_version: {xml_metadata['formula_version']}")

    if xml_metadata.get('formula_date'):
        header_lines.append(f"formula_date: {xml_metadata['formula_date']}")

    if xml_metadata.get('company_name'):
        header_lines.append(f"company_name_xml: {xml_metadata['company_name']}")

    header_lines.append("---")
    header_lines.append("")

    if xml_metadata.get('document_name'):
        header_lines.append(f"# {xml_metadata['document_name']}")
        header_lines.append("")

        if corp_name:
            header_lines.append(f"**회사명**: {corp_name} ({stock_code})")
            header_lines.append("")

    return "\n".join(header_lines)


# ==========================================
# 3. 셀 처리
# ==========================================

def clean_cell_text(text: str) -> str:
    """셀 텍스트 정리"""
    if not text:
        return ""

    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('|', '\\|')
    return text


def get_cell_value(cell: etree._Element) -> Tuple[str, str]:
    """셀에서 값과 ACODE 추출"""
    cell_text = clean_cell_text(''.join(cell.itertext()))
    acode = cell.get('ACODE', '')
    return cell_text, acode


# ==========================================
# 4. 테이블 정규화 (핵심 로직)
# ==========================================

class Cell:
    """테이블 셀 정보"""
    def __init__(self, text: str = "", is_header: bool = False, acode: str = ""):
        self.text = text
        self.is_header = is_header
        self.acode = acode
    
    def __repr__(self):
        return f"Cell({self.text})"


def normalize_table(table_elem: etree._Element) -> List[List[Cell]]:
    """
    ROWSPAN/COLSPAN을 처리하여 테이블을 정규화
    병합된 셀의 값을 모든 해당 셀에 복사
    """
    
    # 원본 테이블 데이터 수집
    raw_rows = []
    
    tbody = table_elem.find('.//TBODY')
    thead = table_elem.find('.//THEAD')
    
    # THEAD 처리
    if thead is not None:
        for tr in thead.findall('.//TR'):
            row_data = []
            for cell in tr:
                if cell.tag in ['TD', 'TH', 'TE', 'TU']:
                    text, acode = get_cell_value(cell)
                    colspan = int(cell.get('COLSPAN', '1'))
                    rowspan = int(cell.get('ROWSPAN', '1'))
                    is_header = True
                    row_data.append((text, colspan, rowspan, is_header, acode))
            if row_data:
                raw_rows.append(row_data)
    
    # TBODY 처리
    rows_container = tbody if tbody is not None else table_elem
    for tr in rows_container.findall('.//TR'):
        row_data = []
        for cell in tr:
            if cell.tag in ['TD', 'TH', 'TE', 'TU']:
                text, acode = get_cell_value(cell)
                colspan = int(cell.get('COLSPAN', '1'))
                rowspan = int(cell.get('ROWSPAN', '1'))
                is_header = cell.tag in ['TH']
                row_data.append((text, colspan, rowspan, is_header, acode))
        if row_data:
            raw_rows.append(row_data)
    
    if not raw_rows:
        return []
    
    # 최대 열 수 계산
    max_cols = 0
    for row_data in raw_rows:
        cols = sum(colspan for _, colspan, _, _, _ in row_data)
        max_cols = max(max_cols, cols)
    
    # 2D 그리드 생성
    max_rows = len(raw_rows)
    grid = [[None for _ in range(max_cols)] for _ in range(max_rows)]
    
    # 그리드 채우기 (ROWSPAN/COLSPAN 처리)
    for row_idx, row_data in enumerate(raw_rows):
        col_idx = 0
        
        for text, colspan, rowspan, is_header, acode in row_data:
            # 이미 채워진 셀 건너뛰기 (이전 ROWSPAN에 의해)
            while col_idx < max_cols and grid[row_idx][col_idx] is not None:
                col_idx += 1
            
            if col_idx >= max_cols:
                break
            
            # 빈 셀이면서 ROWSPAN이 있는 경우: 상위 행의 값 참조
            if text == "" and rowspan > 1 and row_idx > 0:
                # 바로 위 행의 같은 열(또는 왼쪽 열)에서 값 찾기
                parent_text = ""
                for look_col in range(col_idx, -1, -1):
                    if grid[row_idx - 1][look_col] is not None:
                        parent_cell = grid[row_idx - 1][look_col]
                        if parent_cell.text and parent_cell.text != "":
                            parent_text = parent_cell.text
                            break
                
                # 상위 텍스트를 사용
                if parent_text:
                    text = parent_text
            
            # 셀 객체 생성
            cell = Cell(text, is_header, acode)
            
            # ROWSPAN과 COLSPAN에 걸쳐 동일한 셀 복사
            for r in range(rowspan):
                for c in range(colspan):
                    if row_idx + r < max_rows and col_idx + c < max_cols:
                        # 병합된 모든 위치에 동일한 텍스트 복사
                        grid[row_idx + r][col_idx + c] = Cell(text, is_header, acode)
            
            col_idx += colspan
    
    # None을 빈 셀로 교체
    for row_idx in range(max_rows):
        for col_idx in range(max_cols):
            if grid[row_idx][col_idx] is None:
                grid[row_idx][col_idx] = Cell("", False, "")
    
    return grid


def is_layout_table(table_elem: etree._Element) -> bool:
    """레이아웃용 테이블인지 판단"""
    
    # ACLASS가 EXTRACTION이면 무조건 데이터 테이블
    aclass = table_elem.get('ACLASS', '')
    if aclass == 'EXTRACTION':
        return False
    
    # BORDER가 1이면 데이터 테이블
    border = table_elem.get('BORDER', '1')
    if border == '1':
        return False
    
    # LIBRARY 안에 있으면 레이아웃 테이블 (단, 위 조건들 통과한 경우만)
    parent = table_elem.getparent()
    while parent is not None:
        if parent.tag == 'LIBRARY':
            return True
        parent = parent.getparent()
    
    # BORDER가 0이면 레이아웃 테이블
    if border == '0':
        return True
    
    return False


def create_markdown_table(grid: List[List[Cell]]) -> str:
    """정규화된 그리드를 마크다운 테이블로 변환"""
    
    if not grid or len(grid) == 0:
        return ""
    
    # 헤더 최적화: 다층 헤더를 명확한 단일 행으로 변환
    optimized_grid = optimize_header(grid)
    
    markdown_lines = []
    
    for row_idx, row in enumerate(optimized_grid):
        row_texts = []
        for cell in row:
            text = cell.text
            # ACODE가 있으면 강조 (데이터 셀)
            if cell.acode and text and text != "-":
                text = f"**{text}**"
            row_texts.append(text)
        
        markdown_lines.append("| " + " | ".join(row_texts) + " |")
        
        # 첫 번째 행 후 구분선
        if row_idx == 0:
            markdown_lines.append("| " + " | ".join(["---"] * len(row)) + " |")
    
    return "\n".join(markdown_lines)


def optimize_header(grid: List[List[Cell]]) -> List[List[Cell]]:
    """
    다층 헤더를 단일 행으로 간소화
    - 2행 헤더를 1행으로 병합
    - 두 번째 행의 구체적인 값 우선 사용
    - 중복된 헤더에 라벨 추가
    """
    
    if len(grid) < 2:
        return grid
    
    # 첫 2행이 모두 헤더인지 확인
    first_row_headers = sum(1 for cell in grid[0] if cell.is_header)
    second_row_headers = sum(1 for cell in grid[1] if cell.is_header or cell.acode)
    
    # 헤더가 아니면 원본 반환
    if first_row_headers < len(grid[0]) * 0.5:
        return grid
    if second_row_headers < len(grid[1]) * 0.5:
        return grid
    
    # 헤더 병합 (열 수 유지)
    merged_header = []
    
    for col_idx in range(len(grid[0])):
        first_cell = grid[0][col_idx]
        second_cell = grid[1][col_idx]
        
        # 두 번째 행에 값이 있으면 우선
        if second_cell.text and second_cell.text.strip():
            merged_header.append(second_cell)
        # 아니면 첫 번째 행 값
        elif first_cell.text and first_cell.text.strip():
            merged_header.append(first_cell)
        # 둘 다 없으면 빈 셀
        else:
            merged_header.append(Cell("", True, ""))
    
    # 중복된 헤더 라벨링 (예: "구 분"이 여러 개면 "구 분(대)", "구 분(소)" 등)
    header_counts = {}
    for i, cell in enumerate(merged_header):
        text = cell.text
        if text:
            if text not in header_counts:
                header_counts[text] = 0
            header_counts[text] += 1
    
    # "구 분"이 2개 이상이면 라벨 추가
    if header_counts.get("구 분", 0) > 1 or header_counts.get("구  분", 0) > 1:
        gubun_count = 0
        for i, cell in enumerate(merged_header):
            if cell.text in ["구 분", "구  분"]:
                if gubun_count == 0:
                    merged_header[i] = Cell("구 분(대)", True, cell.acode)
                elif gubun_count == 1:
                    merged_header[i] = Cell("구 분(소)", True, cell.acode)
                gubun_count += 1
    
    # 최적화된 그리드 = 병합된 헤더 + 나머지 데이터 행
    return [merged_header] + grid[2:]


def extract_table_as_markdown(table_elem: etree._Element) -> str:
    """XML TABLE 요소를 마크다운 테이블로 변환"""
    
    # 레이아웃 테이블 처리
    if is_layout_table(table_elem):
        return extract_layout_table(table_elem)
    
    # 데이터 테이블 정규화
    grid = normalize_table(table_elem)
    
    if not grid:
        return ""
    
    # 마크다운 테이블 생성
    return create_markdown_table(grid)


def extract_layout_table(table_elem: etree._Element) -> str:
    """레이아웃 테이블을 간단한 텍스트로 변환"""
    lines = []
    
    tbody = table_elem.find('.//TBODY')
    rows_container = tbody if tbody is not None else table_elem
    
    for tr in rows_container.findall('.//TR'):
        row_text = []
        for cell in tr:
            if cell.tag in ['TD', 'TH', 'TE', 'TU']:
                text, acode = get_cell_value(cell)
                if text and text != "":
                    if acode:
                        text = f"**{text}**"
                    row_text.append(text)
        
        if row_text:
            lines.append(" ".join(row_text))
    
    return "\n".join(lines)


# ==========================================
# 5. 섹션 및 텍스트 변환
# ==========================================

def process_element_to_markdown(elem: etree._Element, level: int = 1, in_library: bool = False) -> List[str]:
    """XML 요소를 재귀적으로 순회하며 마크다운으로 변환"""

    markdown_lines = []
    tag = elem.tag.upper() if isinstance(elem.tag, str) else ''
    current_in_library = in_library or (tag == 'LIBRARY')

    # SECTION 처리
    if tag.startswith('SECTION-'):
        section_level = int(tag.split('-')[1])

        title_elem = elem.find('TITLE')
        if title_elem is not None and title_elem.text:
            title = title_elem.text.strip()
            heading_level = min(section_level + 1, 6)
            markdown_lines.append(f"{'#' * heading_level} {title}")
            markdown_lines.append("")

        for child in elem:
            if child.tag != 'TITLE':
                markdown_lines.extend(process_element_to_markdown(child, level + 1, current_in_library))

    # TABLE 처리
    elif tag == 'TABLE':
        table_md = extract_table_as_markdown(elem)
        if table_md:
            markdown_lines.append(table_md)
            markdown_lines.append("")

    # P (단락) 처리
    elif tag == 'P':
        text = ''.join(elem.itertext()).strip()
        if text:
            text = re.sub(r'\s+', ' ', text)
            
            # "- "로 시작하는 항목들을 분리
            if '- ' in text and text.count('- ') > 1:
                items = re.split(r'(?<=\.)\s*-\s*', text)
                for item in items:
                    if item.strip():
                        if not item.startswith('- '):
                            item = '- ' + item
                        markdown_lines.append(item.strip())
            else:
                markdown_lines.append(text)
            
            markdown_lines.append("")

    # TITLE (독립적인 경우)
    elif tag == 'TITLE':
        text = ''.join(elem.itertext()).strip()
        if text:
            if elem.get('ATOC') == 'Y':
                markdown_lines.append(f"## {text}")
            else:
                markdown_lines.append(f"**{text}**")
            markdown_lines.append("")

    # TABLE-GROUP 처리
    elif tag == 'TABLE-GROUP':
        for child in elem:
            markdown_lines.extend(process_element_to_markdown(child, level, current_in_library))

    # 기타 컨테이너 요소들
    elif tag in ['BODY', 'LIBRARY', 'CORRECTION', 'TBODY', 'THEAD']:
        for child in elem:
            markdown_lines.extend(process_element_to_markdown(child, level, current_in_library))

    # 기타 요소
    else:
        if len(elem) > 0:
            for child in elem:
                markdown_lines.extend(process_element_to_markdown(child, level, current_in_library))

    return markdown_lines


# ==========================================
# 6. XML 파일 처리
# ==========================================

def convert_xml_to_markdown(
    xml_path: Path,
    json_info: Dict[str, Any]
) -> str:
    """XML 파일을 마크다운으로 변환"""

    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()

        parser = etree.XMLParser(recover=True, remove_blank_text=True)
        root = etree.fromstring(xml_content.encode('utf-8'), parser=parser)

        xml_metadata = extract_xml_metadata(root)

        suffix = ''
        if '_' in xml_path.stem:
            suffix = xml_path.stem.split('_', 1)[1]

        markdown_content = generate_markdown_header(
            corp_code=json_info.get('corp_code', ''),
            corp_name=json_info.get('corp_name', ''),
            stock_code=json_info.get('stock_code', ''),
            rcept_dt=json_info.get('rcept_dt', ''),
            suffix=suffix,
            xml_metadata=xml_metadata
        )

        body_lines = process_element_to_markdown(root)
        markdown_content += "\n".join(body_lines)

        return markdown_content

    except Exception as e:
        print(f"❌ 변환 실패: {xml_path.name} - {e}")
        import traceback
        traceback.print_exc()
        return ""


# ==========================================
# 7. 메인 실행
# ==========================================

def main():
    # 경로 설정
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent.parent / "data"
    json_file = data_dir / "20251020.json"
    xml_dir = data_dir / "xml"
    output_dir = data_dir / "markdown"

    print("=" * 60)
    print("XML to Markdown Converter (Final - Normalized)")
    print("=" * 60)
    print(f"JSON file: {json_file}")
    print(f"XML directory: {xml_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # 출력 디렉토리 생성
    output_dir.mkdir(exist_ok=True)

    # JSON 데이터 로드
    print("📖 JSON 메타데이터 로드 중...")
    
    # JSON 파일 로드
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rcept_dict = {}
        for item in data.get('list', []):
            rcept_no = item.get('rcept_no')
            if rcept_no:
                rcept_dict[rcept_no] = item
        
        print(f"✅ {len(rcept_dict)}개 문서 정보 로드 완료")
    except Exception as e:
        print(f"❌ JSON 로드 실패: {e}")
        rcept_dict = {}
    
    print()

    # XML 파일 그룹화
    print("📁 XML 파일 그룹화 중...")
    
    groups = {}
    for xml_file in xml_dir.glob('*.xml'):
        stem = xml_file.stem
        if '_' in stem:
            base_rcept_no = stem.split('_')[0]
        else:
            base_rcept_no = stem
        
        if base_rcept_no not in groups:
            groups[base_rcept_no] = []
        groups[base_rcept_no].append(xml_file)
    
    total_files = sum(len(files) for files in groups.values())
    print(f"✅ {len(groups)}개 그룹, 총 {total_files}개 파일")
    print()

    # 변환 시작
    print("🔄 마크다운 변환 중...")
    print()

    processed_count = 0
    success_count = 0

    for rcept_no, xml_files in groups.items():
        json_info = rcept_dict.get(rcept_no, {})

        for xml_file in xml_files:
            print(f"  ⚙️  {xml_file.name}")

            markdown_content = convert_xml_to_markdown(xml_file, json_info)

            if not markdown_content:
                print(f"    ⚠️  변환 실패, 건너뜀")
                processed_count += 1
                continue

            output_file = output_dir / f"{xml_file.stem}.md"

            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)

                print(f"    ✅ 저장 완료: {output_file.name}")
                success_count += 1
            except Exception as e:
                print(f"    ❌ 저장 실패: {e}")

            processed_count += 1
            print()

    print("=" * 60)
    print("변환 완료!")
    print("=" * 60)
    print(f"처리된 파일: {processed_count}/{total_files}")
    print(f"성공: {success_count}")
    print(f"실패: {processed_count - success_count}")
    print(f"출력 디렉토리: {output_dir}")
    print()


if __name__ == "__main__":
    main()
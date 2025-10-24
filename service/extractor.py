#!/usr/bin/env python3
"""
XML 파일들을 JSONL 형식으로 변환하는 스크립트
- data/20251017.json의 rcept_no와 XML 파일명 매칭
- CSS 스타일 부분 제거
- corp_code, corp_name, stock_code를 앞단에 추가
- 각 XML 파일을 하나의 JSONL 행으로 변환
"""

import json
import os
import xml.etree.ElementTree as ET
from lxml import etree, html
from pathlib import Path
from typing import Dict, Any, List
import re

# ==========================================
# 1. xml/html cleansing
# ========================================== 
def clean_xml_text(text: str) -> str:
    """XML 텍스트를 정리 (공백, 개행 정규화)"""
    if not text:
        return ""
    # 여러 공백을 하나로, 개행 정리
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def remove_css_styles(xml_content: str) -> str:
    """XML 내용에서 CSS 스타일 부분 제거"""
    # CSS 스타일 블록 제거 (/* ... */ 형태)
    xml_content = re.sub(r'/\*.*?\*/', '', xml_content, flags=re.DOTALL)
    
    # <style> 태그와 그 내용 제거
    xml_content = re.sub(r'<style[^>]*>.*?</style>', '', xml_content, flags=re.DOTALL | re.IGNORECASE)
    
    # CSS 클래스 정의 제거 (.xforms 등)
    lines = xml_content.split('\n')
    cleaned_lines = []
    in_css_block = False
    
    for line in lines:
        line = line.strip()
        
        # CSS 블록 시작 감지
        if line.startswith('.xforms') or line.startswith('{') and not in_css_block:
            in_css_block = True
            continue
        
        # CSS 블록 끝 감지
        if in_css_block and line == '}':
            in_css_block = False
            continue
        
        # CSS 블록이 아닌 경우만 추가
        if not in_css_block and line:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def remove_style_attributes(element: ET.Element) -> None:
    """XML 요소에서 style 관련 속성 제거"""
    # style 관련 속성들 제거
    style_attrs = ['style', 'class', 'width', 'height', 'align', 'valign', 'border', 
                   'padding', 'margin', 'font-size', 'font-family', 'color', 'background']
    
    # 현재 요소의 속성에서 style 관련 제거
    attrs_to_remove = []
    for attr in element.attrib:
        if any(style_attr in attr.lower() for style_attr in style_attrs):
            attrs_to_remove.append(attr)
    
    for attr in attrs_to_remove:
        del element.attrib[attr]
    
    # 자식 요소들에 대해 재귀적으로 적용
    for child in element:
        remove_style_attributes(child)

def xml_to_dict(element: ET.Element) -> Dict[str, Any]:
    """XML 요소를 딕셔너리로 변환"""
    result = {}
    
    # 텍스트 내용이 있으면 저장
    if element.text and element.text.strip():
        result['text'] = clean_xml_text(element.text)
    
    # 속성들 저장 (style 관련 제외)
    if element.attrib:
        attrs = {}
        for key, value in element.attrib.items():
            if not any(style_attr in key.lower() for style_attr in 
                      ['style', 'class', 'width', 'height', 'align', 'valign', 'border',
                       'padding', 'margin', 'font-size', 'font-family', 'color', 'background']):
                attrs[key] = value
        if attrs:
            result['attributes'] = attrs
    
    # 자식 요소들 처리
    children = {}
    for child in element:
        child_dict = xml_to_dict(child)
        tag = child.tag
        
        # 같은 태그가 여러 개 있으면 리스트로 저장
        if tag in children:
            if not isinstance(children[tag], list):
                children[tag] = [children[tag]]
            children[tag].append(child_dict)
        else:
            children[tag] = child_dict
    
    if children:
        result.update(children)
    
    return result

def _collect_text(element: ET.Element, skip_tags: List[str]) -> List[str]:
    """트리에서 표시 텍스트만 수집 (태그/속성 제외)."""
    texts: List[str] = []
    tag_lower = element.tag.lower() if isinstance(element.tag, str) else ""
    if tag_lower in skip_tags:
        return texts

    if element.text and element.text.strip():
        texts.append(clean_xml_text(element.text))
    for child in element:
        texts.extend(_collect_text(child, skip_tags))
    if element.tail and element.tail.strip():
        texts.append(clean_xml_text(element.tail))
    return texts

def xml_to_plain_text(root: ET.Element) -> str:
    """XML/HTML 트리에서 가시 텍스트만 추출해 하나의 문자열로 반환."""
    # HTML 성격의 잡음 태그 무시
    skip_tags = [
        'style', 'script', 'head', 'meta', 'link', 'noscript',
    ]
    pieces = _collect_text(root, skip_tags)
    # 공백 정규화 및 중복 공백 축소
    text = ' '.join(pieces)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_xml_content(xml_content: str) -> str:
    """XML 내용을 정리하여 파싱 가능하게 만듦"""
    # CSS 스타일 제거
    cleaned_content = remove_css_styles(xml_content)
    
    # XML/HTML 선언 부분 정리
    if cleaned_content.startswith('<?xml'):
        # XML 형식인 경우 DOCUMENT 태그 찾기
        lines = cleaned_content.split('\n')
        for i, line in enumerate(lines):
            if '<DOCUMENT' in line:
                cleaned_content = '\n'.join(lines[i:])
                break
    elif cleaned_content.startswith('<html'):
        # HTML 형식인 경우 body 태그부터 시작
        lines = cleaned_content.split('\n')
        for i, line in enumerate(lines):
            if '<body' in line.lower():
                cleaned_content = '\n'.join(lines[i:])
                break
    
    # 잘못된 문자 제거 (XML에서 허용되지 않는 문자들)
    # ASCII 제어 문자 제거 (0x00-0x1F, 0x7F 제외)
    cleaned_content = ''.join(char for char in cleaned_content 
                             if ord(char) >= 32 or char in '\t\n\r')
    
    # 잘못된 XML 문자 참조 수정
    cleaned_content = re.sub(r'&(?![a-zA-Z0-9#]+;)', '&amp;', cleaned_content)
    
    return cleaned_content

def process_xml_file(xml_path: str) -> Dict[str, Any]:
    """문자열 정리 + 파싱(lxml 복구 모드) + 후처리 후 간결 텍스트 추출."""
    try:
        # XML 파일 읽기 (여러 인코딩 시도)
        encodings = ['utf-8', 'euc-kr', 'cp949', 'latin1']
        xml_content = None
        
        for encoding in encodings:
            try:
                with open(xml_path, 'r', encoding=encoding) as f:
                    xml_content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if xml_content is None:
            print(f"인코딩 오류 {xml_path}: 모든 인코딩 시도 실패")
            return {}
        
        # XML 내용 정리
        cleaned_content = clean_xml_content(xml_content)
        
        # XML 파싱 시도: lxml 복구 파서 사용
        try:
            parser = etree.XMLParser(recover=True)
            root_xml = etree.fromstring(cleaned_content.encode('utf-8', errors='ignore'), parser=parser)
            # style 속성 제거 (lxml Element도 dict 인터페이스 제공)
            remove_style_attributes(root_xml)
            # 텍스트 추출: XPath로 모든 텍스트 수집 후 정규화
            pieces = root_xml.xpath('//text()')
            plain_text = ' '.join(t for t in pieces if t and t.strip())
            plain_text = re.sub(r'\s+', ' ', plain_text).strip()
            return {"text": plain_text}
        except etree.XMLSyntaxError as e:
            # HTML 파서로 폴백
            print(f"XML 파싱 실패, HTML로 폴백 {xml_path}: {e}")
            try:
                doc = html.fromstring(cleaned_content)
                # head/script/style 등은 lxml가 기본적으로 무시하지 않으므로 XPath로 텍스트만 추출
                pieces = doc.xpath('//text()')
                plain_text = ' '.join(t for t in pieces if t and t.strip())
                plain_text = re.sub(r'\s+', ' ', plain_text).strip()
                return {"text": plain_text}
            except Exception as e2:
                print(f"HTML 파싱 실패 {xml_path}: {e2}")
                return {}
        
    except Exception as e:
        print(f"파일 처리 오류 {xml_path}: {e}")
        return {}
# ==========================================
# 2. json에서 접수번호 딕셔너리 수집
# ========================================== 
def load_json_data(json_path: str) -> Dict[str, Any]:
    """JSON 파일에서 rcept_no 목록 로드"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # rcept_no를 키로 하는 딕셔너리 생성
        rcept_dict = {}
        for item in data.get('list', []):
            rcept_no = item.get('rcept_no')
            if rcept_no:
                rcept_dict[rcept_no] = item
        
        return rcept_dict
    except Exception as e:
        print(f"JSON 파일 로드 오류: {e}")
        return {}

def select_latest_xml_files(xml_dir: Path) -> List[Path]:
    """같은 접수번호(rcept_no)에 대해 접수번호_숫자 중 가장 큰 숫자를 선택.

    규칙:
    - 파일명 형식: <rcept_no>.xml 또는 <rcept_no>_<number>.xml
    - 숫자 접미사가 가장 큰 파일을 채택 (접미사 없으면 0으로 간주)
    """
    candidates: Dict[str, Dict[str, Any]] = {}
    for path in xml_dir.glob("*.xml"):
        stem = path.stem
        if "_" in stem:
            base, suffix = stem.split("_", 1)
            try:
                num = int(suffix)
            except ValueError:
                num = -1  # 숫자 아님 → 우선순위 낮게
        else:
            base = stem
            num = 0

        current = candidates.get(base)
        if current is None or num > current["num"]:
            candidates[base] = {"num": num, "path": path}

    return [entry["path"] for entry in candidates.values()]

# ==========================================
# 3. main()
# ========================================== 
def main():
    # 경로 설정 - 스크립트 파일 기준으로 상대 경로
    script_dir = Path(__file__).parent  # service 폴더
    data_dir = script_dir.parent / "data"  # ../data
    json_file = data_dir / "20251017.json"
    xml_dir = data_dir / "xml"
    output_file = data_dir / "docs.jsonl"
    
    print(f"Data directory: {data_dir}")
    print(f"JSON file: {json_file}")
    print(f"XML directory: {xml_dir}")
    
    print("JSON 데이터 로드 중...")
    rcept_data = load_json_data(json_file)
    print(f"로드된 rcept_no 개수: {len(rcept_data)}")
    
    # XML 파일 목록 가져오기 (정정공시 중 가장 큰 접미사 선택)
    xml_files = select_latest_xml_files(xml_dir)
    print(f"처리할 XML 파일 개수(최신본 선택): {len(xml_files)}")
    
    # JSONL 파일에 쓰기
    processed_count = 0
    matched_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for xml_file in xml_files:
            print(f"처리 중: {xml_file.name}")
            
            # 파일명에서 rcept_no 추출 (확장자 제거)
            rcept_no = xml_file.stem
            
            # JSON 데이터에서 해당 rcept_no 찾기
            json_info = rcept_data.get(rcept_no, {})
            
            # XML 파일 처리
            xml_content = process_xml_file(xml_file)
            
            # JSONL 행 생성 - 회사 정보를 앞단에 추가
            jsonl_row = {
                "corp_code": json_info.get("corp_code", ""),
                "corp_name": json_info.get("corp_name", ""),
                "stock_code": json_info.get("stock_code", ""),
                "xml_content": xml_content
            }
            
            # JSONL 파일에 쓰기
            f.write(json.dumps(jsonl_row, ensure_ascii=False) + '\n')
            
            processed_count += 1
            if json_info:
                matched_count += 1
    
    print(f"\n변환 완료!")
    print(f"처리된 파일: {processed_count}")
    print(f"매칭된 파일: {matched_count}")
    print(f"출력 파일: {output_file}")

if __name__ == "__main__":
    main()
# XML to Markdown 변환 워크플로우

## 개요

DART XML 파일을 마크다운으로 변환하고, 청크로 분할하여 RAG 시스템에 활용하는 전체 워크플로우입니다.

## 데이터 흐름

```
XML 파일 (5,076개)
    ↓
[xml_to_markdown.py]
    ↓
Markdown 파일 (5,076개)
    ↓
[trnsform.py]
    ↓
JSONL 청크 파일
```

---

## 1단계: XML → Markdown 변환

### 스크립트: `xml_to_markdown.py`

#### 주요 기능
1. **메타데이터 헤더 생성**
   - YAML frontmatter 형식
   - corp_code, corp_name, stock_code, rcept_dt
   - document_name, document_acode, formula_version

2. **XML 테이블 → Markdown 테이블 변환**
   - TABLE → 표준 마크다운 테이블
   - 헤더/데이터 행 구분
   - COLSPAN, ROWSPAN 속성 처리
   - AUNIT 단위 정보 보존

3. **섹션 계층 구조 유지**
   - SECTION-1, SECTION-2, SECTION-3 → ##, ###, ####
   - TITLE 태그 → 헤딩
   - P 태그 → 단락

#### 실행 방법

```bash
# 가상환경 활성화
source .venv/bin/activate

# 변환 실행
python service/rag/xml_to_markdown.py
```

#### 입력
- `data/20251020.json`: 회사 메타데이터 (3,956개 문서)
- `data/xml/*.xml`: XML 파일 (5,076개 파일)
  - 본문: `20241114000727.xml`
  - 감사보고서: `20241114000727_00760.xml`
  - 연결감사보고서: `20241114000727_00761.xml`

#### 출력
- `data/markdown/*.md`: 마크다운 파일 (5,076개)
- 총 용량: ~60MB

#### 출력 예시

```markdown
---
corp_code: 00116301
corp_name: NI스틸
stock_code: 008260
rcept_dt: 20241114
document_name: 분기보고서
document_acode: 11013
formula_version: 5.6
formula_date: 20240412
---

# 분기보고서

**회사명**: NI스틸 (008260)

## I. 회사의 개요

### 1. 회사의 개요

| 구분 | 연결대상회사수 | 주요종속회사수 |
| --- | --- | --- |
| 상장 | - | - |
| 비상장 | - | - |
```

---

## 2단계: Markdown → 청크 변환

### 스크립트: `trnsform.py`

#### 주요 기능

1. **테이블 행 → 자연어 변환**
   - 헤더와 데이터 조합
   - 섹션별 특화 변환 (주식, 재무, 일반)
   - 검색 친화적 텍스트 생성

2. **텍스트 청크 분할**
   - 단락 단위 분할
   - 최소 길이 필터링 (10자 이상)
   - 섹션 경로 보존

3. **구조화된 데이터 보존**
   - 원본 마크다운 보존
   - 테이블 셀 값 딕셔너리 형태 저장
   - 메타데이터 추가

#### 실행 방법

```bash
# 가상환경 활성화
source .venv/bin/activate

# 단일 파일 테스트
python service/rag/trnsform.py
```

#### 입력
- `data/markdown/*.md`: 마크다운 파일

#### 출력
- `data/chunks/*_chunks.jsonl`: 청크 파일
- 각 줄은 하나의 청크 (JSON 객체)

#### 출력 예시

```json
{
  "chunk_id": "a3f2b1c4d5e6f7g8",
  "doc_id": "20241114_00116301",
  "chunk_type": "table_row",
  "section_path": "분기보고서 > I. 회사의 개요 > 1. 회사의 개요",
  "raw_markdown": "| 구분 | 연결대상회사수 | 주요종속회사수 |\n| --- | --- | --- |\n| 상장 | - | - |",
  "structured_data": {
    "구분": "상장",
    "연결대상회사수": "-",
    "주요종속회사수": "-"
  },
  "natural_text": "상장: 연결대상회사수 -주, 주요종속회사수 -주",
  "metadata": {
    "corp_name": "NI스틸",
    "document_name": "분기보고서",
    "row_index": 0
  }
}
```

---

## 청크 타입

### 1. `table_row` (테이블 행)
- 테이블의 각 데이터 행
- 헤더와 값을 자연어로 변환
- 구조화된 데이터 보존

### 2. `text` (텍스트)
- 일반 단락
- 10자 이상만 포함
- 섹션 경로 포함

---

## 자연어 변환 로직

### 주식 테이블 변환
```
입력: | 보통주 | 발행주식총수 | 10,000,000 |
출력: "보통주: 발행주식총수 10,000,000주"
```

### 재무 테이블 변환
```
입력: | 매출액 | 2024.09 | 15,000,000,000 |
출력: "매출액은 2024.09에 150.0억원"
```

### 일반 테이블 변환
```
입력: | 대표이사 | 이름 | 홍길동 |
출력: "대표이사은 이름, 이름은 홍길동"
```

---

## 디렉토리 구조

```
data/
├── 20251020.json              # 회사 메타데이터
├── xml/                       # 원본 XML 파일 (5,076개)
│   ├── 20241114000727.xml
│   ├── 20241114000727_00760.xml
│   └── ...
├── markdown/                  # 변환된 마크다운 (5,076개)
│   ├── 20241114000727.md
│   ├── 20241114000727_00760.md
│   └── ...
└── chunks/                    # 청크 파일
    ├── 20241114000727_chunks.jsonl
    └── ...
```

---

## 성능 지표

### XML → Markdown 변환
- **입력**: 5,076개 XML 파일
- **출력**: 5,076개 마크다운 파일 (~60MB)
- **처리 시간**: 약 5-10분 (예상)
- **성공률**: ~100%

### Markdown → 청크 변환
- **입력**: 1개 마크다운 파일 (33KB)
- **출력**: 199개 청크
  - table_row: 159개 (80%)
  - text: 40개 (20%)
- **처리 시간**: < 1초

---

## 문제 해결

### Q1. JSONL 파일 손상 문제
**이전 방식**: `extractor.py` → `docs.jsonl` (JSON 파싱 오류)

**해결 방법**: XML → Markdown → 청크 (2단계 변환)

**이유**:
- XML이 깨끗한 원본 데이터
- 중간 JSON 없이 직접 변환
- 디버깅 용이 (마크다운 확인 가능)

### Q2. 파일 경로 오류
**오류**: `/mnt/user-data/outputs/` 경로 하드코딩

**해결**:
```python
script_dir = Path(__file__).parent
data_dir = script_dir.parent.parent / "data"
```

---

## 다음 단계

### 3단계: 벡터 임베딩 생성
- 청크의 `natural_text` 필드를 임베딩
- ChromaDB 또는 PostgreSQL (pgvector)에 저장

### 4단계: RAG 시스템 통합
- 검색: 자연어 쿼리 → 유사 청크 검색
- 생성: 검색된 청크 → LLM 컨텍스트

---

## 참고

### 관련 파일
- `service/rag/xml_to_markdown.py`: XML → Markdown 변환
- `service/rag/trnsform.py`: Markdown → 청크 변환
- `service/rag/extractor.py`: (레거시) XML → JSONL 변환
- `data/20251020.json`: 회사 메타데이터

### 데이터 품질
- ✅ XML 파일: 표준 형식, 파싱 오류 없음
- ✅ 마크다운: 가독성 높음, 수동 검증 가능
- ✅ 청크: 자연어 변환, 구조화 데이터 보존
- ❌ JSONL (레거시): JSON 파싱 오류 (따옴표 불균형)

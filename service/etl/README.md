# ETL Service Module

DART 공시 데이터를 RAG 시스템에 적합한 형태로 변환하는 전체 ETL 파이프라인입니다.

## 📐 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         ETL Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │   Extract   │ -> │  Transform  │ -> │    Load     │       │
│  │   (추출)     │    │   (변환)     │    │   (적재)     │       │
│  └─────────────┘    └─────────────┘    └─────────────┘       │
│         │                   │                   │              │
│         │                   │                   │              │
│    DART API           Markdown            Parquet              │
│    XML Files          JSONL Chunks        Vector DB            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🗂️ 디렉토리 구조

```
service/etl/
├── extractor/              # Extract: 데이터 추출
│   ├── __init__.py
│   ├── api_pull.py         # DART API 다운로드
│   ├── extractor.py        # XML → Markdown 변환
│   └── build_kospi_map.py  # KOSPI 종목 매핑
│
├── transform/              # Transform: 데이터 변환
│   ├── __init__.py
│   ├── pipeline.py         # 전체 파이프라인 실행
│   ├── parser.py           # Markdown → 구조화된 청크
│   ├── normalizer.py       # 텍스트 정규화 및 품질 개선
│   ├── chunker.py          # 스마트 청킹 및 메타데이터 강화
│   ├── models.py           # 데이터 모델 정의
│   └── utils.py            # 공통 유틸리티
│
└── loader/                 # Load: 데이터 적재
    ├── __init__.py
    ├── jsonl_to_parquet.py # JSONL → Parquet 변환
    ├── loader.py           # Parquet → pgvector 로드
    └── README.md
```

## 🚀 전체 파이프라인 실행

### 1단계: Extract (데이터 추출)

```bash
cd service/etl/extractor

# 1-1. DART API에서 공시 목록 다운로드
python api_pull.py  # DartConfig.URL = 'list.json'

# 1-2. XML 파일 다운로드
python api_pull.py  # DartConfig.URL = 'document.xml'

# 1-3. XML → Markdown 변환
python extractor.py

# 1-4. KOSPI 종목 매핑 생성
python build_kospi_map.py
```

**출력**:

- `data/20251020.json`: 공시 목록
- `data/xml/*.xml`: 원본 XML 파일
- `data/markdown/*.md`: 변환된 마크다운 파일
- `data/kospi_top100_map.json`: 종목 매핑

### 2단계: Transform (데이터 변환)

```bash
cd service/etl/transform

# 전체 파이프라인 실행 (테스트 모드: 20개 파일)
python pipeline.py

# 전체 파일 처리
python pipeline.py --all
```

**파이프라인 단계**:

1. **Parser**: Markdown → 구조화된 청크 (JSONL)
2. **Normalizer**: 텍스트 정규화 및 품질 개선
3. **Chunker**: 스마트 청킹 및 메타데이터 강화

**출력**:

- `data/transform/final/*_chunks.jsonl`: 최종 청크

### 3단계: Load (데이터 적재)

```bash
cd service/etl/loader

# JSONL → Parquet 변환 (단일 파일)
python jsonl_to_parquet.py

# 기업별 파티셔닝
python jsonl_to_parquet.py --partition corp_name

# 압축 방식 변경
python jsonl_to_parquet.py --compression zstd
```

**출력**:

- `data/parquet/chunks.parquet`: 통합 Parquet 파일
- `data/parquet/by_corp_name/`: 기업별 파티션

## 📊 데이터 흐름

```
DART API (공시 데이터)
    ↓
XML (원본 문서)
    ↓
Markdown (구조화된 문서)
    ↓ [Parser]
JSONL (구조화된 청크 - parser/)
    ↓ [Normalizer]
JSONL (정규화된 청크 - normalized/)
    ↓ [Chunker]
JSONL (최종 청크 - final/)
    ↓ [jsonl_to_parquet]
Parquet (컬럼 기반 압축)
    ↓ [Embedding]
Parquet + Embeddings
    ↓ [loader]
pgvector (PostgreSQL)
```

## 🔧 각 모듈의 역할

### Extractor (추출)

| 모듈            | 입력      | 출력      | 설명                              |
| --------------- | --------- | --------- | --------------------------------- |
| api_pull        | DART API  | JSON, XML | 공시 목록 및 XML 다운로드         |
| extractor       | XML       | Markdown  | XML을 읽기 쉬운 마크다운으로 변환 |
| build_kospi_map | TXT, JSON | JSON      | KOSPI 종목명 매핑 테이블 생성     |

### Transform (변환)

| 모듈       | 입력     | 출력  | 설명                            |
| ---------- | -------- | ----- | ------------------------------- |
| parser     | Markdown | JSONL | 마크다운을 구조화된 청크로 파싱 |
| normalizer | JSONL    | JSONL | 날짜/숫자 정규화, 품질 개선     |
| chunker    | JSONL    | JSONL | 토큰 수 기반 스마트 청킹        |

### Loader (적재)

| 모듈             | 입력    | 출력     | 설명                         |
| ---------------- | ------- | -------- | ---------------------------- |
| jsonl_to_parquet | JSONL   | Parquet  | 컬럼 기반 압축 포맷으로 변환 |
| loader           | Parquet | pgvector | 임베딩 및 벡터 DB 저장       |

## 📈 처리 성능

### 테스트 환경

- CPU: Apple M1 Pro
- RAM: 16GB
- 파일 수: 5,000개 공시 문서

### 처리 시간

| 단계                   | 소요 시간         | 처리량          |
| ---------------------- | ----------------- | --------------- |
| Extract (API 다운로드) | 1시간 30분        | ~55 docs/min    |
| Extract (XML → MD)     | 15분              | ~333 docs/min   |
| Transform (전체)       | 25분              | ~200 docs/min   |
| Load (Parquet 변환)    | 2분               | ~2,500 docs/min |
| **총합**               | **약 2시간 12분** | -               |

### 데이터 크기 변화

| 단계             | 데이터 크기 | 압축률   |
| ---------------- | ----------- | -------- |
| XML (원본)       | 500 MB      | -        |
| Markdown         | 350 MB      | 30% 감소 |
| JSONL (final)    | 200 MB      | 60% 감소 |
| Parquet (snappy) | 50 MB       | 90% 감소 |

## 🛠️ 의존성 설치

```bash
# 필수 패키지
pip install -r requirements.txt

# 주요 패키지:
# - requests: DART API 호출
# - lxml: XML 파싱
# - pandas: 데이터 처리
# - pyarrow: Parquet 변환
# - python-dotenv: 환경변수 관리
```

## ⚙️ 환경 설정

`.env` 파일 생성:

```env
# DART API 키 (필수)
DART_API_KEY=your_dart_api_key_here

# PostgreSQL 연결 (Load 단계)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
```

## 📝 모듈화 설계

### 공통 유틸리티 (utils.py)

- `get_project_paths()`: 프로젝트 경로 관리
- `get_transform_paths()`: Transform 경로 관리
- `read_jsonl()`: JSONL 파일 읽기
- `write_jsonl()`: JSONL 파일 쓰기

### 경로 관리 예시

```python
from utils import get_transform_paths

# 모든 경로를 한 번에 가져오기
paths = get_transform_paths(__file__)
markdown_dir = paths['markdown_dir']
parser_dir = paths['parser_dir']
normalized_dir = paths['normalized_dir']
final_dir = paths['final_dir']
```

## 🎯 다음 단계

ETL 파이프라인 완료 후:

1. **임베딩 생성**: `service/embedding/` (TODO)
2. **벡터 DB 로드**: `service/etl/loader/loader.py` 완성
3. **RAG 시스템 구축**: `service/rag/`
4. **웹 인터페이스**: Streamlit 앱 연동

## 📚 참고 문서

- [Extract Module README](extractor/README.md) - TODO
- [Transform Module README](transform/README.md) - TODO
- [Loader Module README](loader/README.md) ✅
- [API Documentation](../docs/api.md) - TODO

## 🤝 기여

버그 리포트 및 개선 제안은 이슈로 등록해주세요.

## 📄 라이선스

MIT License - SKN18-3rd-5Team

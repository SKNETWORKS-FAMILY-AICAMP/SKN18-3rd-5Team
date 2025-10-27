# SKN18-3rd-5Team

## 프로젝트 구조

```text
├── app.py                 # Streamlit 진입점
├── pages/                 # Streamlit 멀티 페이지 모듈
│   ├── app_bootstrap.py   # 공통 페이지 설정 및 사이드바 메뉴 정의
│   ├── page1.py           # 채팅 Q&A 페이지
│   ├── data_tool.py       # 데이터 도구 페이지
│   └── views/             # 채팅 등 공통 뷰 컴포넌트 (page에서 이용)
│       ├── chat.py        # 채팅 UI 레이아웃
│       ├── {view}.py      #
│       ├── {veiw}.py      #
│       └── ...
├── service/               # LLM등 로직/기능
│   ├── chat_service.py    # SQLite 기반 채팅 세션 관리
│   └── ...
├── data/                  # 분석·시각화에 사용하는 원천 데이터
│   └── app_database.db    # SQLite 데이터베이스
├── assets/                # 이미지, 아이콘 등 정적 리소스
├── config/                # 환경 설정 파일 (예: YAML, JSON)
├── graph/                 # lang-graph
│   ├── state.py                  # 상태 스키마(QAState)
│   ├── app_graph.py              # 그래프 구성/compile/팩토리 함수
│   ├── nodes/                    # LangGraph 노드들
│   └── utils/
├── requirements.txt       # Python 의존성 목록
└── README.md
```

## 실행 방법

- HuggingFace에서 모델 미리 다운로드 실행

```bash
uv pip install -r requirements.txt
playwright install
python service/llm/setup_download.py
streamlit run app.py
```

##

데이터베이스 변경사항

### SQLite 마이그레이션

기존 JSON 파일 기반의 채팅 세션 관리를 SQLite 데이터베이스로 변경했습니다.

**변경된 부분:**

- 채팅 세션 및 메시지 데이터가 SQLite에 저장됨
- 기존 JSON 파일은 자동으로 `data/chat_sessions_backup/`으로 백업됨
- UI는 동일하게 유지되며 백엔드 로직만 변경됨

**마이그레이션 실행:**

```bash
python migrate_to_sqlite.py
```

**데이터베이스 구조:**

- `chat_sessions`: 채팅 세션 정보 (ID, 제목, 생성일시, 수정일시)
- `chat_messages`: 채팅 메시지 (세션ID, 역할, 내용, 타임스탬프)

## ETL 및 임베딩 생성 CLI

JSONL 파일 로딩 및 임베딩 생성을 위한 CLI 도구입니다.

### 환경 설정

```bash
# 가상환경 활성화
source .venv/bin/activate

# Docker Compose로 PostgreSQL 실행
docker-compose up -d
```

### CLI 사용법

#### 1. 모델 다운로드 (처음 실행 시 필수)

```bash
# 모든 임베딩 모델 다운로드
python service/etl/loader/loader_cli.py download

# E5 모델만 다운로드
python service/etl/loader/loader_cli.py download --model e5

# KakaoBank 모델만 다운로드
python service/etl/loader/loader_cli.py download --model kakaobank

# FinE5 모델 설정 (API 전용)
python service/etl/loader/loader_cli.py download --model fine5
```

#### 2. 데이터베이스 관리

```bash
# 데이터베이스 연결 테스트
python service/etl/loader/loader_cli.py db test

# 테이블 목록 조회
python service/etl/loader/loader_cli.py db list

# 스키마 생성 (테이블 생성)
python service/etl/loader/loader_cli.py db create
```

#### 2. 데이터 삭제 (TRUNCATE)

```bash
# 문서 테이블 데이터 삭제
python service/etl/loader/loader_cli.py truncate doc

# 임베딩 테이블 데이터 삭제
python service/etl/loader/loader_cli.py truncate vector

# 모든 테이블 데이터 삭제
python service/etl/loader/loader_cli.py truncate all
```

#### 3. 데이터 로드

```bash
# JSONL 파일을 chunks 테이블에 로드
python service/etl/loader/loader_cli.py load doc

# E5 모델로 임베딩 생성 및 저장
python service/etl/loader/loader_cli.py load vector --model e5

# KakaoBank 모델로 임베딩 생성 (1000개 제한)
python service/etl/loader/loader_cli.py load vector --model kakaobank --limit 1000

# 배치 크기 조정
python service/etl/loader/loader_cli.py load vector --model e5 --batch-size 50
```

### 전체 워크플로우

```bash
# 1. 모델 다운로드 (처음 실행 시 필수)
python service/etl/loader/loader_cli.py download

# 2. 데이터베이스 설정
python service/etl/loader/loader_cli.py db create

# 3. 문서 로드
python service/etl/loader/loader_cli.py load doc

# 4. 임베딩 생성 (E5 모델)
python service/etl/loader/loader_cli.py load vector --model e5

# 5. 임베딩 생성 (KakaoBank 모델)
python service/etl/loader/loader_cli.py load vector --model kakaobank

# 6. 상태 확인
python service/etl/loader/loader_cli.py db list
```

### 지원하는 임베딩 모델

- **E5**: `intfloat/multilingual-e5-small` (384차원)
- **KakaoBank**: `kakaobank/kf-deberta-base` (768차원)
- **FinE5**: `FinanceMTEB/FinE5` (API 기반, 1024차원)

### 실시간 진행률 로그

임베딩 생성 시 실시간으로 진행 상황을 확인할 수 있습니다:

```
🔄 진행률: 15.2% (176,543/1,161,721) | 새 임베딩: 176,543개 | 속도: 12.3개/초 | 예상 완료: 2.1시간
```

### 데이터베이스 구조

#### chunks 테이블

- `id`: 청크 ID (Primary Key)
- `chunk_id`: 청크 식별자
- `natural_text`: 청크 텍스트 내용
- `corp_name`: 기업명
- `document_name`: 문서명
- `doc_type`: 문서 타입

#### 임베딩 테이블

- `embeddings_multilingual_e5_small`: E5 모델 임베딩
- `embeddings_kakaobank_kf_deberta_base`: KakaoBank 모델 임베딩
- `embeddings_fine5_finance`: FinE5 모델 임베딩

각 임베딩 테이블은 `chunk_id`를 외래키로 참조합니다.

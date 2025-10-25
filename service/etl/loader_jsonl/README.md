# JSONL Loader - JSONL 파일 로딩 및 임베딩 생성 시스템

JSONL 파일을 PostgreSQL 데이터베이스에 로딩하고 임베딩을 생성하는 통합 시스템입니다.

## 📁 디렉토리 구조

```
service/etl/loader_jsonl/
├── __init__.py
├── generate_embeddings.py      # 임베딩 생성 모듈
├── jsonl_to_postgres.py        # JSONL 로딩 + CLI 유틸리티
├── loader_cli.py               # 통합 CLI
├── README.md                   # 이 파일
├── schema_jsonl.sql            # 데이터베이스 스키마
└── system_manager.py           # Docker + 스키마 관리
```

## 🚀 빠른 시작

### 1. 전체 파이프라인 실행

```bash
cd service/etl/loader_jsonl
python loader_cli.py run --jsonl-dir ../../../data/transform/final
```

### 2. 단계별 실행

```bash
# 1. 시스템 상태 확인
python loader_cli.py system health

# 2. 스키마 생성
python loader_cli.py schema create

# 3. JSONL 파일 로딩
python loader_cli.py load data --jsonl-dir ../../../data/transform/final

# 4. 임베딩 생성
python loader_cli.py embed --model intfloat/multilingual-e5-small
```

## 📋 CLI 명령어

### 전체 파이프라인

```bash
python loader_cli.py run --jsonl-dir <JSONL_DIR> [--batch-size 1000] [--embedding-model MODEL] [--skip-embeddings]
```

### 시스템 관리

```bash
python loader_cli.py system health    # 시스템 상태 확인
python loader_cli.py system reset     # 시스템 초기화
```

### Docker 관리

```bash
python loader_cli.py docker check     # Docker 상태 확인
python loader_cli.py docker start     # Docker 시작
python loader_cli.py docker stop      # Docker 중지
```

### 스키마 관리

```bash
python loader_cli.py schema create    # 스키마 생성
python loader_cli.py schema check     # 스키마 상태 확인
python loader_cli.py schema drop      # 스키마 삭제
```

### 데이터 로딩

```bash
python loader_cli.py load data --jsonl-dir <JSONL_DIR> [--batch-size 1000]
python loader_cli.py load stats       # 로딩 통계 조회
python loader_cli.py load clear       # 데이터 삭제
```

### 임베딩 생성

```bash
python loader_cli.py embed --model <MODEL_NAME>
```

## 🤖 지원하는 임베딩 모델

| 모델명                           | 테이블명                           | 차원 | 설명              |
| -------------------------------- | ---------------------------------- | ---- | ----------------- |
| `intfloat/multilingual-e5-small` | `embeddings_multilingual_e5_small` | 384  | 다국어 E5-Small   |
| `kakaobank/kf-deberta-base`      | `embeddings_kakaobank`             | 768  | KakaoBank DeBERTa |
| `FinanceMTEB/FinE5`              | `embeddings_fine5`                 | 4096 | 금융 특화 FinE5   |

### 모델 사용 예시

```bash
# 다국어 E5-Small (기본)
python loader_cli.py embed --model intfloat/multilingual-e5-small

# KakaoBank DeBERTa
python loader_cli.py embed --model kakaobank/kf-deberta-base

# 금융 특화 FinE5
python loader_cli.py embed --model FinanceMTEB/FinE5
```

## 🗄️ 데이터베이스 스키마

### 청크 테이블 (`chunks`)

```sql
CREATE TABLE chunks (
    chunk_id VARCHAR(500) PRIMARY KEY,
    doc_id VARCHAR(255),
    chunk_type VARCHAR(50),
    section_path TEXT,
    natural_text TEXT,
    structured_data JSONB,
    metadata JSONB,
    token_count INTEGER,
    merged_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 임베딩 테이블들

- `embeddings_multilingual_e5_small` (384차원)
- `embeddings_kakaobank` (768차원)
- `embeddings_fine5` (4096차원)

## 🔧 설정

### 환경 변수

```bash
export PG_HOST=localhost
export PG_PORT=5432
export PG_DB=skn_project
export PG_USER=postgres
export PG_PASSWORD=post1234
```

### Docker 설정

```yaml
# docker-compose.yml
version: "3.8"
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: skn_project
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: post1234
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## 📊 모니터링

### 로딩 통계 확인

```bash
python loader_cli.py load stats
```

### 시스템 상태 확인

```bash
python loader_cli.py system health
```

## 🚨 문제 해결

### 일반적인 문제들

1. **Docker 연결 실패**

   ```bash
   python loader_cli.py docker check
   python loader_cli.py docker start
   ```

2. **스키마 오류**

   ```bash
   python loader_cli.py schema drop
   python loader_cli.py schema create
   ```

3. **중복 데이터 오류**

   ```bash
   python loader_cli.py load clear
   python loader_cli.py load data --jsonl-dir <JSONL_DIR>
   ```

4. **임베딩 생성 실패**
   - 모델명 확인
   - 인터넷 연결 확인
   - 메모리 사용량 확인

### 로그 확인

```bash
# 상세 로그 출력
python loader_cli.py --verbose <command>
```

## 🔗 관련 시스템

- **RAG 시스템**: `service/rag_jsonl/` - 검색 및 생성
- **원본 로더**: `service/etl/loader/` - Parquet 기반 로더

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

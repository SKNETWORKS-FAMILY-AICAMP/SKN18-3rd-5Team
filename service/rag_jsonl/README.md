# RAG JSONL System

JSONL 파일을 직접 사용하는 RAG (Retrieval-Augmented Generation) 시스템입니다.

## 📋 주요 특징

- **JSONL 직접 지원**: Parquet 변환 없이 JSONL 파일에서 직접 임베딩 생성
- **PostgreSQL + pgvector**: 벡터 검색을 위한 PostgreSQL 확장 사용
- **다국어 지원**: 한국어/영어 혼합 문서 처리
- **실시간 검색**: 빠른 벡터 유사도 검색
- **기업별 필터링**: 특정 기업의 문서만 검색 가능

## 🚀 빠른 시작

### 1. 데이터 로드

```bash
# JSONL 파일을 PostgreSQL에 로드
cd service/etl/loader_jsonl
python loader_cli.py run --jsonl-dir ../../../data/transform/final
```

### 2. 검색 테스트

```bash
# RAG 시스템으로 검색
cd service/rag_jsonl/cli
python rag_jsonl_cli.py search --query "삼성전자 매출" --top-k 5
```

### 3. 통계 확인

```bash
# 시스템 통계 조회
python rag_jsonl_cli.py stats
```

## 📊 시스템 아키텍처

```
JSONL Files (data/transform/final/)
    ↓
JSONL Loader (service/etl/loader_jsonl/)
    ↓
PostgreSQL + pgvector
    ↓
RAG System (service/rag_jsonl/)
    ↓
Search Results
```

## 🤖 지원하는 임베딩 모델

| 모델명                           | 차원 | 설명              | 권장 사용                      |
| -------------------------------- | ---- | ----------------- | ------------------------------ |
| `intfloat/multilingual-e5-small` | 384  | 다국어 E5-Small   | 빠른 프로토타이핑, 다국어 지원 |
| `kakaobank/kf-deberta-base`      | 768  | KakaoBank DeBERTa | 한국어 금융 도메인 특화        |
| `FinanceMTEB/FinE5`              | 4096 | 금융 특화 FinE5   | 금융 도메인 최고 성능          |

## 🔧 사용법

### 검색 명령어

```bash
# 기본 검색
python rag_jsonl_cli.py search --query "AI 기술 개발"

# 상위 10개 결과
python rag_jsonl_cli.py search --query "매출 증가" --top-k 10

# 특정 기업만 검색
python rag_jsonl_cli.py search --query "연구개발비" --corp-filter "삼성전자"

# 최소 유사도 설정
python rag_jsonl_cli.py search --query "디지털 전환" --min-similarity 0.7

# 다른 임베딩 모델 사용
python rag_jsonl_cli.py search --query "ESG 경영" --model sentence-transformers/all-MiniLM-L6-v2

# 검색 결과 저장
python rag_jsonl_cli.py search --query "지속가능경영" --save-results
```

### 통계 명령어

```bash
# 전체 통계
python rag_jsonl_cli.py stats

# 특정 모델 통계
python rag_jsonl_cli.py stats --model sentence-transformers/all-MiniLM-L6-v2
```

## 📈 성능 비교

### 처리 시간

| 단계        | Parquet 방식        | JSONL 방식     |
| ----------- | ------------------- | -------------- |
| 데이터 변환 | 30분-1시간          | 0분 (생략)     |
| 데이터 로드 | 10-20분             | 30분-1시간     |
| **총 시간** | **40분-1시간 20분** | **30분-1시간** |

### 메모리 사용량

| 단계            | Parquet 방식 | JSONL 방식 |
| --------------- | ------------ | ---------- |
| 변환 시         | 16GB+        | 0GB        |
| 로드 시         | 8GB          | 8GB        |
| **최대 사용량** | **16GB+**    | **8GB**    |

## 🛠️ 환경 설정

### 1. 의존성 설치

```bash
pip install psycopg2-binary sentence-transformers pandas numpy tqdm
```

### 2. PostgreSQL + pgvector 설정

```bash
# PostgreSQL 설치 (Ubuntu/Debian)
sudo apt-get install postgresql postgresql-contrib

# pgvector 확장 설치
sudo -u postgres psql -c "CREATE EXTENSION vector;"
```

### 3. 환경변수 설정

```bash
# .env 파일 생성
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=rag_db
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your_password
```

## 📊 데이터베이스 스키마

### chunks 테이블

```sql
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
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

### embeddings\_{model_name} 테이블

```sql
CREATE TABLE embeddings_model_name (
    chunk_id VARCHAR(255) PRIMARY KEY,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
);
```

## 🔍 검색 예시

### 1. 기본 검색

```bash
python rag_jsonl_cli.py search --query "삼성전자 매출"
```

**결과:**

```
🔍 검색 결과: '삼성전자 매출'
📊 총 5개 결과
================================================================================

1. 청크 ID: 20241028_00382199_text_사업보고서_001
   기업: 삼성전자
   유사도: 0.8234
   텍스트: 삼성전자의 2024년 3분기 매출은 전년 대비 12% 증가한 67조원을 기록했습니다...
   토큰 수: 156
```

### 2. 기업별 필터링

```bash
python rag_jsonl_cli.py search --query "연구개발비" --corp-filter "SK하이닉스"
```

### 3. 고유사도 검색

```bash
python rag_jsonl_cli.py search --query "AI 반도체" --min-similarity 0.8
```

## 🎯 사용 사례

### 1. 기업 정보 검색

```bash
# 특정 기업의 재무 정보
python rag_jsonl_cli.py search --query "매출 증가율" --corp-filter "LG전자"

# ESG 관련 정보
python rag_jsonl_cli.py search --query "환경 경영" --corp-filter "현대자동차"
```

### 2. 산업 분석

```bash
# 반도체 산업 동향
python rag_jsonl_cli.py search --query "반도체 시장 전망"

# 자동차 산업 동향
python rag_jsonl_cli.py search --query "전기차 시장"
```

### 3. 기술 동향 분석

```bash
# AI 기술 개발
python rag_jsonl_cli.py search --query "인공지능 기술"

# 디지털 전환
python rag_jsonl_cli.py search --query "디지털 혁신"
```

## 🔧 고급 설정

### 1. 임베딩 모델 변경

```bash
# 한국어 특화 모델
python rag_jsonl_cli.py search --query "한국어 쿼리" --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# 영어 특화 모델
python rag_jsonl_cli.py search --query "English query" --model sentence-transformers/all-MiniLM-L6-v2

# 고성능 모델 (느리지만 정확)
python rag_jsonl_cli.py search --query "복잡한 쿼리" --model sentence-transformers/all-mpnet-base-v2
```

### 2. 배치 크기 조정

```bash
# 메모리 부족 시
python loader_cli.py load data --jsonl-dir ../../../data/transform/final --batch-size 500

# 메모리 충분 시
python loader_cli.py load data --jsonl-dir ../../../data/transform/final --batch-size 2000
```

## 📝 주의사항

1. **데이터 로드**: JSONL 파일을 먼저 PostgreSQL에 로드해야 검색 가능
2. **모델 일치**: 검색 시 사용하는 모델과 로드 시 사용한 모델이 일치해야 함
3. **메모리 관리**: 대용량 데이터 처리 시 배치 크기 조정 필요
4. **인덱스**: 첫 검색 전에 벡터 인덱스가 생성되는데 시간이 걸릴 수 있음

## 🚀 다음 단계

1. **성능 최적화**: 인덱스 튜닝, 쿼리 최적화
2. **모델 비교**: 여러 임베딩 모델의 성능 평가
3. **UI 개발**: 웹 인터페이스 구축
4. **API 서버**: REST API 서버 구축

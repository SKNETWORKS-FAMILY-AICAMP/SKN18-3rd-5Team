# JSONL Loader CLI

JSONL 파일을 PostgreSQL 데이터베이스에 로딩하고 임베딩을 생성하는 통합 시스템입니다.

## 🚀 빠른 시작

### 기본 워크플로우

```bash
# 1. Docker 시작
docker-compose up -d

# 2. 디렉토리 이동 및 데이터베이스 연결 테스트
cd service/etl/loader
python loader_cli.py db test

# 3. 스키마 생성
python loader_cli.py db create

# 4. 테이블 목록 확인
python loader_cli.py db list

# 5. 모델 다운로드 (처음 실행 시 필수)
python loader_cli.py download --model all

# 6. 문서 로드
python loader_cli.py load doc

# 7. 벡터 로드 (임베딩 생성)
python loader_cli.py load vector --model e5
```

## 📋 명령어 상세

### 데이터베이스 관리

```bash
python loader_cli.py db test          # 데이터베이스 연결 테스트
python loader_cli.py db create        # 스키마 생성
python loader_cli.py db list          # 테이블 목록 및 통계 조회
```

### 모델 다운로드

```bash
python loader_cli.py download --model all         # 모든 모델 다운로드
python loader_cli.py download --model e5          # E5 모델 다운로드
python loader_cli.py download --model kakaobank   # KakaoBank 모델 다운로드
python loader_cli.py download --model fine5       # FinE5 모델 설정 (API 전용)
```

### 데이터 삭제

```bash
python loader_cli.py truncate doc     # 문서 테이블 데이터 삭제
python loader_cli.py truncate vector  # 임베딩 테이블 데이터 삭제
python loader_cli.py truncate all     # 모든 테이블 데이터 삭제
```

### 데이터 로드

```bash
python loader_cli.py load doc                                    # 문서 로드
python loader_cli.py load vector --model e5                      # E5 모델로 임베딩 생성
python loader_cli.py load vector --model kakaobank               # KakaoBank 모델로 임베딩 생성
python loader_cli.py load vector --model fine5                   # FinE5 모델로 임베딩 생성
python loader_cli.py load vector --model e5 --limit 1000         # 1000개 청크만 처리
python loader_cli.py load vector --model e5 --batch-size 50      # 배치 크기 50으로 설정
```

## 🤖 지원하는 임베딩 모델

| 모델명                           | CLI 옵션    | 차원 | 설명                  |
| -------------------------------- | ----------- | ---- | --------------------- |
| `intfloat/multilingual-e5-small` | `e5`        | 384  | 다국어 E5-Small       |
| `kakaobank/kf-deberta-base`      | `kakaobank` | 768  | KakaoBank DeBERTa     |
| `FinanceMTEB/FinE5`              | `fine5`     | 4096 | 금융 특화 FinE5 (API) |

## 🔧 설정

### 환경 변수

```bash
# .env 파일에 설정
FIN_E5_API_KEY=your_api_key_here  # FinE5 API 사용 시 필요
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

### 실시간 진행률

임베딩 생성 시 60초마다 진행률이 표시됩니다:

```
🔄 진행률: 0.4% (4,100/1,160,239) | 새 임베딩: 4,100개 | 속도: 11.2개/초 | 예상 완료: 28.6시간
```

### 통계 확인

```bash
python loader_cli.py db list  # 테이블별 행 수 확인
```

## 🚨 문제 해결

### 일반적인 문제들

1. **Docker 연결 실패**

   ```bash
   docker-compose up -d
   python loader_cli.py db test
   ```

2. **모델 다운로드 실패**

   ```bash
   python loader_cli.py download --model e5
   ```

3. **FinE5 API 오류**

   - `.env` 파일에 `FIN_E5_API_KEY` 설정 확인
   - `python loader_cli.py download --model fine5`로 API 연결 테스트

4. **데이터 초기화**
   ```bash
   python loader_cli.py truncate all
   python loader_cli.py db create
   ```

## 📁 파일 구조

```
service/etl/loader/
├── loader_cli.py           # 메인 CLI
├── embeddings.py           # 임베딩 생성 모듈
├── jsonl_to_postgres.py    # JSONL 로딩 모듈
├── schema_jsonl.sql        # 데이터베이스 스키마
└── README.md              # 이 파일
```

## 🔗 관련 시스템

- **RAG 시스템**: `service/rag/` - 검색 및 생성
- **설정**: `config/vector_database.py` - 데이터베이스 설정

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

### 4. RAG 평가 실행

```bash
# 통합 RAG 평가 도구 실행
python -m service.rag_jsonl.cli.rag_evaluation_tool --top-k 3

# 기업별 필터링 평가
python -m service.rag_jsonl.cli.rag_evaluation_tool --top-k 5 --corp-filter "삼성전자"

# 다른 임베딩 모델로 평가
python -m service.rag_jsonl.cli.rag_evaluation_tool --model kakaobank --top-k 3
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
python rag_jsonl_cli.py search --query "ESG 경영" --model kakaobank/kf-deberta-base

# 검색 결과 저장
python rag_jsonl_cli.py search --query "지속가능경영" --save-results
```

### 통계 명령어

```bash
# 전체 통계
python rag_jsonl_cli.py stats

# 특정 모델 통계
python rag_jsonl_cli.py stats --model kakaobank/kf-deberta-base
```

### RAG 평가 도구

통합된 RAG 평가 도구는 검색 성능을 자동으로 평가하고 메트릭을 계산합니다.

#### 기본 사용법

```bash
# 기본 평가 (5개 쿼리, Top-K=3)
python -m service.rag_jsonl.cli.rag_evaluation_tool --top-k 3

# 더 많은 결과 검색
python -m service.rag_jsonl.cli.rag_evaluation_tool --top-k 5

# 최소 유사도 설정
python -m service.rag_jsonl.cli.rag_evaluation_tool --top-k 3 --min-similarity 0.7

# 특정 기업만 평가
python -m service.rag_jsonl.cli.rag_evaluation_tool --top-k 3 --corp-filter "삼성전자"
```

#### 모델별 평가

```bash
# 다국어 모델 (기본값)
python -m service.rag_jsonl.cli.rag_evaluation_tool --model multilingual-e5-small --top-k 3

# 한국어 금융 특화 모델
python -m service.rag_jsonl.cli.rag_evaluation_tool --model kakaobank --top-k 3

# 금융 도메인 최고 성능 모델
python -m service.rag_jsonl.cli.rag_evaluation_tool --model fine5 --top-k 3
```

#### 평가 결과

평가 실행 시 다음 파일들이 생성됩니다:

1. **`rag_evaluation_YYYYMMDD_HHMMSS.json`** - 메인 결과 파일

   - 검색 결과 + 메트릭 계산 포함
   - 각 쿼리별 상세 정보

2. **`detailed_results_TIMESTAMP.json`** - 상세 결과 파일

   - `overall_score`, `response_time_ms` 등 포함
   - `complete_evaluation` 형식

3. **`summary_report_TIMESTAMP.txt`** - 요약 리포트 파일
   - 텍스트 형식의 요약 리포트
   - 평균 성능 지표 + 개별 쿼리 결과

#### 평가 메트릭

- **Recall@K**: 예상 키워드가 검색된 문서에 포함된 비율
- **Precision@K**: 상위 K개 문서 중 관련 문서 비율
- **MRR**: 첫 번째 관련 문서의 순위 역수
- **NDCG@K**: 정규화된 할인 누적 이득
- **Keyword Coverage**: 키워드 커버리지
- **평균 유사도**: 검색 결과의 평균 유사도

#### 평가 결과 예시

```
📊 평가 결과 요약:
   - 총 쿼리 수: 5
   - 모델: multilingual-e5-small
   - 성공한 쿼리: 5/5

📈 평균 성능 지표:
   - Recall@K: 0.7667
   - Precision@K: 0.9333
   - MRR: 1.0000
   - NDCG@K: 1.0000
   - Keyword Coverage: 0.7667
   - 평균 유사도: 0.8902
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

### 1. RAG 시스템 성능 평가

```bash
# 전체 시스템 성능 평가
python -m service.rag_jsonl.cli.rag_evaluation_tool --top-k 5

# 특정 기업의 검색 성능 평가
python -m service.rag_jsonl.cli.rag_evaluation_tool --corp-filter "삼성전자" --top-k 3

# 모델별 성능 비교
python -m service.rag_jsonl.cli.rag_evaluation_tool --model multilingual-e5-small --top-k 3
python -m service.rag_jsonl.cli.rag_evaluation_tool --model kakaobank --top-k 3
python -m service.rag_jsonl.cli.rag_evaluation_tool --model fine5 --top-k 3
```

### 2. 기업 정보 검색

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
# 다국어 모델 (기본값)
python rag_jsonl_cli.py search --query "한국어 쿼리" --model intfloat/multilingual-e5-small

# 한국어 금융 특화 모델
python rag_jsonl_cli.py search --query "금융 쿼리" --model kakaobank/kf-deberta-base

# 금융 도메인 최고 성능 모델
python rag_jsonl_cli.py search --query "복잡한 금융 쿼리" --model FinanceMTEB/FinE5
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
2. **모델 비교**: `rag_evaluation_tool`을 사용한 여러 임베딩 모델의 성능 평가
3. **평가 확장**: 더 많은 평가 쿼리 추가 및 다양한 메트릭 구현
4. **UI 개발**: 웹 인터페이스 구축
5. **API 서버**: REST API 서버 구축
6. **자동화**: CI/CD 파이프라인에 평가 도구 통합

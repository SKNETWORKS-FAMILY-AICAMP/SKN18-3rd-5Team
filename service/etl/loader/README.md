# ETL Loader Module

Transform 단계의 최종 출력물(JSONL)을 Parquet 파일로 변환하는 모듈입니다.

## 📋 주요 기능

- **JSONL → Parquet 변환**: 여러 JSONL 파일을 하나의 Parquet 파일로 통합
- **데이터 타입 최적화**: 메모리 및 저장 공간 효율화
- **파티셔닝**: 기업별, 문서타입별로 데이터 분할 저장
- **압축**: Snappy, GZIP, ZSTD 등 다양한 압축 방식 지원

## 🚀 사용법

### 1. 기본 사용 (단일 Parquet 파일)

```bash
cd service/etl/loader
python jsonl_to_parquet.py
```

**입력**: `data/transform/final/*_chunks.jsonl`  
**출력**: `data/parquet/chunks.parquet`

### 2. 파티셔닝 (기업별 분리)

```bash
python jsonl_to_parquet.py --partition corp_name
```

**출력**: `data/parquet/by_corp_name/corp_name={회사명}/`

### 3. 압축 방식 변경

```bash
python jsonl_to_parquet.py --compression zstd
```

**압축 옵션**:

- `snappy`: 빠른 압축/해제 (기본값, 추천)
- `gzip`: 높은 압축률
- `zstd`: 균형잡힌 압축률과 속도
- `brotli`: 최고 압축률 (느림)

### 4. Python 코드에서 사용

```python
from service.etl.loader import ParquetConverter
from pathlib import Path

# 변환기 생성
converter = ParquetConverter(compression='snappy')

# JSONL 파일 읽기
input_dir = Path("data/transform/final")
jsonl_files = list(input_dir.glob("*_chunks.jsonl"))
df = converter.jsonl_to_dataframe(jsonl_files)

# Parquet 저장
output_path = Path("data/parquet/chunks.parquet")
converter.save_parquet(df, output_path)

# 통계 정보 확인
stats = converter.get_parquet_stats(output_path)
print(f"총 {stats['num_rows']:,}개 행")
```

## 📊 Parquet 파일 구조

### 스키마

| 컬럼명          | 타입     | 설명                        |
| --------------- | -------- | --------------------------- |
| chunk_id        | string   | 청크 고유 ID                |
| doc_id          | string   | 문서 ID                     |
| chunk_type      | category | 청크 타입 (text, table_row) |
| section_path    | string   | 섹션 경로                   |
| structured_data | object   | 구조화된 데이터 (딕셔너리)  |
| natural_text    | string   | 자연어 텍스트               |
| metadata        | object   | 메타데이터 (딕셔너리)       |
| token_count     | int16    | 토큰 수                     |

### 메타데이터 필드

| 필드명        | 타입     | 설명                |
| ------------- | -------- | ------------------- |
| corp_name     | string   | 기업명              |
| document_name | string   | 문서명              |
| rcept_dt      | string   | 접수일자 (YYYYMMDD) |
| doc_type      | category | 문서 타입           |
| data_category | category | 데이터 카테고리     |
| fiscal_year   | int16    | 사업연도            |
| keywords      | list     | 키워드 목록         |
| prev_context  | string   | 이전 문맥           |
| next_context  | string   | 다음 문맥           |

## 🔧 파티셔닝 전략

### 1. 기업별 파티셔닝 (추천)

```bash
python jsonl_to_parquet.py --partition corp_name
```

**장점**:

- 특정 기업 데이터만 빠르게 조회
- 기업별 분석에 최적화

**출력 구조**:

```
data/parquet/by_corp_name/
├── corp_name=삼성전자/
│   └── *.parquet
├── corp_name=SK하이닉스/
│   └── *.parquet
└── ...
```

### 2. 문서 타입별 파티셔닝

```bash
python jsonl_to_parquet.py --partition doc_type
```

**장점**:

- 문서 타입별 분석 (사업보고서, 분기보고서 등)
- 타입별 통계 및 필터링 효율화

### 3. 데이터 카테고리별 파티셔닝

```bash
python jsonl_to_parquet.py --partition data_category
```

**장점**:

- 재무, 일반 데이터 분리
- 카테고리별 임베딩 전략 차별화

## 📈 성능 비교

### 파일 크기 비교 (예시: 10,000개 청크)

| 포맷             | 크기  | 압축률   |
| ---------------- | ----- | -------- |
| JSONL (원본)     | 50 MB | -        |
| Parquet (snappy) | 15 MB | 70% 감소 |
| Parquet (gzip)   | 12 MB | 76% 감소 |
| Parquet (zstd)   | 11 MB | 78% 감소 |

### 읽기 속도 비교

| 포맷    | 읽기 시간 | 필터링 속도      |
| ------- | --------- | ---------------- |
| JSONL   | 2.5초     | 느림             |
| Parquet | 0.3초     | 빠름 (컬럼 기반) |

## 🎯 다음 단계

Parquet 파일 생성 후:

1. **임베딩 생성**

   ```bash
   python embedding.py --input data/parquet/chunks.parquet
   ```

2. **pgvector 로드**

   ```bash
   python loader.py --input data/parquet/chunks.parquet
   ```

3. **Pandas로 분석**
   ```python
   import pandas as pd
   df = pd.read_parquet("data/parquet/chunks.parquet")
   df.groupby('corp_name')['token_count'].sum()
   ```

## 🛠️ 의존성

```bash
pip install pandas pyarrow
```

## 📝 참고사항

- Parquet는 컬럼 기반 압축 포맷으로 데이터 분석에 최적화되어 있습니다
- 파티셔닝은 데이터 크기가 큰 경우(100MB 이상) 권장됩니다
- Snappy 압축은 속도와 압축률의 균형이 좋아 기본값으로 설정되어 있습니다

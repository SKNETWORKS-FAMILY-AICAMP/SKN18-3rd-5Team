# RAG System - 모듈 가이드

금융 문서 검색 및 생성을 위한 **Retrieval-Augmented Generation (RAG)** 시스템

---

## 📁 모듈 구조

```
service/rag/
├── retrieval/          # 검색 및 재정렬
│   ├── retriever.py    # 하이브리드 검색 (벡터+키워드)
│   └── eranker.py  # 하이브리드 리랭커
│
├── augmentation/       # 컨텍스트 증강
│   ├── augmenter.py    # 문서 정리 및 증강
│   └── formatters.py   # 출력 포맷터 (Prompt/Markdown/JSON)
│
├── generation/         # LLM 답변 생성 (내부 테스트용)
│   └── generator.py    # Ollama 생성기
│
├── evaluation/         # 성능 평가
│   ├── ragas_evaluator.py         # RAGAs 메트릭
│   ├── compare_retrieval_methods.py  # 검색 방법 비교
│   ├── evaluator.py    # 평가 도구
│   └── metrics.py      # 평가 메트릭
│
├── models/             # 임베딩 모델
│   ├── encoder.py      # 임베딩 인코더
│   ├── config.py       # 모델 설정
│   └── comparator.py   # 모델 비교
│
├── query/              # 쿼리 처리
│   └── temporal_parser.py  # 시간 표현 파싱
│
├── vectorstore/        # 벡터 DB
│   └── pgvector_store.py   # PostgreSQL + pgvector
│
├── cli/                # CLI 도구
│   ├── rag_cli.py                  # 기본 검색 CLI
│   ├── rag_evaluation_tool.py      # 평가 도구
│   └── compare_embedding_models.py # 모델 비교
│
└── rag_system.py       # 통합 RAG 시스템
```

---

## 🔍 1. Retrieval (검색)

### `retrieval/retriever.py` - 통합 검색 리트리버

**벡터 검색 + 키워드 검색 + 하이브리드 검색을 하나의 클래스로 통합**

#### 검색 방법 3가지

| 방법        | 설명                        | 장점             | 사용 시점  |
| ----------- | --------------------------- | ---------------- | ---------- |
| **vector**  | pgvector 임베딩 유사도      | 의미적 유사성    | 기본 검색  |
| **keyword** | PostgreSQL Full-Text Search | 정확한 용어 매칭 | 재시도 1차 |
| **hybrid**  | 벡터 + 키워드 결합 (RRF)    | 최고 성능        | 재시도 2차 |

```python
from service.rag.retrieval.retriever import Retriever
from service.rag.models.config import EmbeddingModelType

# 초기화
retriever = Retriever(
    model_type=EmbeddingModelType.MULTILINGUAL_E5_SMALL,
    enable_temporal_filter=True,
    enable_hybrid=True  # 하이브리드 검색 활성화
)

# 1. 벡터 검색 (기본)
results = retriever.search(
    query="배터리 기술",
    top_k=10,
    search_method="vector"
)

# 2. 키워드 검색
results = retriever.search(
    query="배터리 기술",
    top_k=10,
    search_method="keyword"
)

# 3. 하이브리드 검색 (벡터 + 키워드 RRF)
results = retriever.search(
    query="배터리 기술",
    top_k=10,
    search_method="hybrid"
)
```

**Rank Fusion 알고리즘**:

- **RRF (Reciprocal Rank Fusion)**: `score = Σ(1/(k+rank))`
- **Weighted Sum**: `score = w1 × vector + w2 × keyword`
- **Max Score**: 각 문서의 최고 점수

---

## 🎯 2. Reranker (재정렬)

### `retrieval/reranker.py` - 하이브리드 리랭커

**2단계 리랭킹: Rule-based → Cross-Encoder**

```
100개 후보 → Rule-based (20개) → Cross-Encoder (5개)
            ⚡ 빠른 필터링       🎯 정밀 평가
```

#### 내부 구조

```
HybridReranker
├── Stage 1: CombinedReranker (자동 포함)
│   ├── KeywordReranker (키워드 매칭)
│   ├── LengthReranker (문서 길이)
│   └── PositionReranker (문서 위치)
│
└── Stage 2: CrossEncoderReranker
    └── BAAI/bge-reranker-v2-m3
```

#### 사용법

```python
from service.rag.retrieval.reranker import HybridReranker

# 기본 사용 (모든 기능 포함)
reranker = HybridReranker(
    use_cross_encoder=True,        # Cross-Encoder 사용
    stage1_top_k_multiplier=3.0    # Stage 1에서 top_k × 3배 선택
)

final_results = reranker.rerank(
    query="2차전지 전망",
    candidates=search_results,
    top_k=5
)
```

#### 옵션

```python
# Cross-Encoder 끄기 (빠른 처리)
reranker = HybridReranker(use_cross_encoder=False)
# → Stage 1만 사용 (Rule-based만)

# 커스텀 모델 사용
reranker = HybridReranker(
    cross_encoder_model="BAAI/bge-reranker-v2-m3",
    device="cuda"  # GPU 사용
)
```

**성능**:

- ✅ **정확도**: Cross-Encoder 수준 (85-95%)
- ✅ **속도**: Stage 1 필터링으로 최적화 (40-50ms)
- ✅ **유연성**: Cross-Encoder 끄기 가능

---

## 📝 3. Augmentation (증강)

### `augmentation/augmenter.py` - 컨텍스트 증강

**검색 결과를 LLM 입력 형식으로 변환**

```python
from service.rag.augmentation.augmenter import DocumentAugmenter
from service.rag.augmentation.formatters import PromptFormatter

augmenter = DocumentAugmenter(
    max_context_length=4000,
    max_documents=5
)

formatter = PromptFormatter()

augmented = augmenter.augment(
    query="질문",
    search_results=검색결과,
    formatter=formatter
)

print(f"컨텍스트: {augmented.context_text}")
print(f"토큰 수: {augmented.token_count}")
print(f"인용: {augmented.citations}")
```

**처리 과정**:

1. 중복 제거
2. 토큰 제한
3. 포맷팅
4. 인용 정보 추출

---

### `augmentation/formatters.py` - 포맷터

**3가지 출력 형식**

```python
# 1. Prompt 형식 (LLM 입력용)
from service.rag.augmentation.formatters import PromptFormatter
formatter = PromptFormatter()

# 2. Markdown 형식 (사람 가독성)
from service.rag.augmentation.formatters import MarkdownFormatter
formatter = MarkdownFormatter()

# 3. JSON 형식 (API 응답)
from service.rag.augmentation.formatters import JSONFormatter
formatter = JSONFormatter()
```

---

## 🤖 4. Generation (생성)

### 파인튜닝된 Llama 3.2 3B 모델 사용

**LangGraph에서는 `service/llm/llm_client.py`의 파인튜닝 모델을 사용합니다**

⚠️ **주의**: `service/rag/generation/generator.py`는 **내부 테스트용**이므로 실제 서비스에서 사용하지 않습니다.

#### 실제 사용: 파인튜닝 모델

```python
from service.llm.llm_client import chat
from service.llm.prompt_templates import build_system_prompt, build_user_prompt

# 1. 사용자 레벨에 맞는 프롬프트 생성
user_level = "intermediate"  # "beginner", "intermediate", "advanced"
system_prompt = build_system_prompt(user_level)
user_prompt = build_user_prompt(
    question="2차전지 산업 전망은?",
    context="검색된 컨텍스트...",
    level=user_level
)

# 2. 파인튜닝된 Llama 3.2 3B LoRA 모델로 답변 생성
answer = chat(
    system=system_prompt,
    user=user_prompt,
    max_tokens=512
)

print(f"답변: {answer}")
```

#### 사용자 레벨별 프롬프트 템플릿

| 레벨             | 특징                            | 용도        |
| ---------------- | ------------------------------- | ----------- |
| **beginner**     | 쉬운 용어, 예시 중심            | 투자 초보자 |
| **intermediate** | 균형잡힌 설명, 주요 수치 포함   | 일반 투자자 |
| **advanced**     | 상세 수치, 추세 분석, 문서 근거 | 전문 투자자 |

#### 파인튜닝 모델 정보

- **베이스 모델**: Llama 3.2 3B
- **어댑터**: LoRA (Low-Rank Adaptation)
- **학습 데이터**: 금융 리포트 QA 데이터셋
- **특화 분야**: 한국 금융 문서 이해 및 답변 생성

#### 테스트용 Generator (내부 테스트만 사용)

```python
# ⚠️ 이 코드는 테스트용입니다. LangGraph에서 사용하지 마세요.
from service.rag.generation.generator import OllamaGenerator

generator = OllamaGenerator(
    base_url="http://localhost:11434",
    default_model="gemma2:2b"
)

# Ollama를 사용한 로컬 테스트
answer = generator.generate(
    query="테스트 질문",
    context="테스트 컨텍스트..."
)
```

---

## 📊 5. Evaluation (평가)

### `evaluation/ragas_evaluator.py` - RAG 품질 평가

**RAGAs 메트릭으로 RAG 시스템 평가**

#### 4가지 메트릭

| 메트릭                | 평가 대상 | 질문                         |
| --------------------- | --------- | ---------------------------- |
| **Faithfulness**      | 답변      | 컨텍스트에 근거하는가?       |
| **Answer Relevancy**  | 답변      | 질문과 관련있는가?           |
| **Context Precision** | 검색      | 정확한 문서를 찾았는가?      |
| **Context Recall**    | 검색      | 필요한 정보를 모두 찾았는가? |

```python
from service.rag.evaluation.ragas_evaluator import RAGASEvaluator

evaluator = RAGASEvaluator()

evaluation = evaluator.evaluate(
    query="2차전지 산업 전망은?",
    answer="2차전지 산업은 전기차 수요로 성장 전망입니다.",
    contexts=["컨텍스트 1", "컨텍스트 2"],
    ground_truth="예상 답변 (선택사항)"
)

print(f"Faithfulness:       {evaluation.faithfulness:.3f}")
print(f"Answer Relevancy:   {evaluation.answer_relevancy:.3f}")
print(f"Context Precision:  {evaluation.context_precision:.3f}")
print(f"Average Score:      {evaluation.average_score:.3f}")
```

---

### `evaluation/compare_retrieval_methods.py` - 검색 방법 비교

**Vector vs Keyword vs Hybrid 성능 비교**

```python
from service.rag.evaluation.compare_retrieval_methods import (
    RetrievalComparisonExperiment
)

experiment = RetrievalComparisonExperiment(db_config)
await experiment.initialize()

# 비교 실험 수행
results_df = await experiment.compare_methods(
    test_queries=[
        {'query': '질문1', 'embedding': [...]},
        {'query': '질문2', 'embedding': [...]}
    ],
    top_k=5,
    use_reranker=True
)

# 결과 분석
analysis = experiment.analyze_results(results_df)
experiment.save_results(results_df, analysis)
```

**출력**: CSV + JSON 파일 (속도, 정확도 비교)

---

## 🎨 6. Models (임베딩 모델)

### `models/encoder.py` - 임베딩 모델

**3가지 임베딩 모델 지원**

| 모델                  | 차원 | 특징         |
| --------------------- | ---- | ------------ |
| **E5-Small**          | 384  | 빠름, 다국어 |
| **KakaoBank DeBERTa** | 768  | 한국어 금융  |
| **FinE5**             | 4096 | 금융 특화    |

```python
from service.rag.models.encoder import get_encoder
from service.rag.models.config import EmbeddingModelType

# 인코더 로드
encoder = get_encoder(EmbeddingModelType.MULTILINGUAL_E5_SMALL)

# 쿼리 임베딩
query_emb = encoder.encode_query("2차전지 산업")

# 문서 임베딩 (배치)
doc_embs = encoder.encode_documents(["문서1", "문서2"])

# 벡터 차원
print(f"차원: {encoder.get_dimension()}")
```

---

## 🧩 7. Query (쿼리 분석)

### `query/temporal_parser.py` - 시간 표현 파싱

**쿼리에서 시간 정보 추출**

```python
from service.rag.query.temporal_parser import parse_temporal_query

# 예시 1
result = parse_temporal_query("2024년 2분기 실적은?")
# → {'year': 2024, 'quarter': 2}

# 예시 2
result = parse_temporal_query("작년 12월 매출은?")
# → {'year': 2024, 'month': 12}
```

**용도**: 시계열 데이터 필터링

---

## 🗄️ 8. VectorStore (벡터 DB)

### `vectorstore/pgvector_store.py` - PostgreSQL + pgvector

**벡터 데이터 저장 및 검색**

```python
from service.rag.vectorstore.pgvector_store import PGVectorStore

store = PGVectorStore(db_config)

# 임베딩 삽입
await store.insert_embeddings([
    {'chunk_id': 'id1', 'embedding': [...], 'metadata': {...}},
    {'chunk_id': 'id2', 'embedding': [...], 'metadata': {...}}
])

# 유사 검색
results = await store.search_similar(
    embedding=query_embedding,
    top_k=10,
    min_similarity=0.7
)

# HNSW 인덱스 생성
await store.create_index()
```

---

## 🔄 Self-RAG / Corrective RAG

### 개념: 답변 품질 검증 후 재검색

**문제**: 첫 검색으로 충분한 정보를 못 찾으면?
**해결**: 답변을 검증하고, 품질이 낮으면 다른 방법으로 재검색

### 작동 방식

```
1. Retrieve (Vector) → 문서 검색
2. Rerank → 재정렬
3. Generate → 답변 생성
4. GroundingCheck → [ref:] 있는지 검증
   ├─ ✅ 성공 → Guardrail로 진행
   └─ ❌ 실패 → Retrieve로 재시도 (최대 1회)
                (Vector 실패 → Keyword로 재검색)
```

### 구현 위치

- **LangGraph**: `graph/nodes/grounding_check.py`
- **로직**: 답변에 `[ref:]` 없으면 재시도
- **참고**: `graph/readme.md` (retry 엣지)

### 확장 예시

```python
# graph/nodes/grounding_check.py 확장 아이디어
def run(state: QAState) -> QAState:
    grounded = "[ref:" in state.get("draft_answer", "")

    if not grounded and state.get("retry_count", 0) < 1:
        # 재검색 전략 변경
        if state["search_method"] == "vector":
            state["search_method"] = "keyword"  # Vector → Keyword
        elif state["search_method"] == "keyword":
            state["search_method"] = "hybrid"   # Keyword → Hybrid

        state["retry_count"] = state.get("retry_count", 0) + 1
        # Retrieve 노드로 돌아감

    state["grounded"] = grounded
    return state
```

---

## 📖 전체 RAG 파이프라인 예시

```python
from service.rag.rag_system import RAGSystem
from service.rag.models.config import EmbeddingModelType

# RAG 시스템 초기화
rag = RAGSystem(
    model_type=EmbeddingModelType.MULTILINGUAL_E5_SMALL,
    enable_generation=True
)

# 검색 + 증강
response = rag.retrieve_and_augment(
    query="2차전지 산업 전망은?",
    top_k=5,
    use_reranker=True
)

print(f"검색된 문서: {len(response.retrieved_documents)}개")
print(f"컨텍스트: {response.augmented_context.token_count} 토큰")

# 전체 RAG (검색 + 생성)
full_response = rag.generate_answer(
    query="2차전지 산업 전망은?",
    top_k=5
)

print(f"답변: {full_response.generated_answer.answer}")
```

---

## 🛠️ CLI 사용법

### 기본 검색

```bash
cd service/rag/cli

# 기본 검색
python rag_cli.py search --query "삼성전자 매출"

# 상위 10개 결과
python rag_cli.py search --query "매출 증가" --top-k 10

# 특정 기업만 검색
python rag_cli.py search --query "연구개발비" --corp-filter "삼성전자"

# 최소 유사도 설정
python rag_cli.py search --query "AI 기술" --min-similarity 0.7
```

### RAG 평가

```bash
# 기본 평가
python rag_evaluation_tool.py --top-k 3

# 특정 기업 평가
python rag_evaluation_tool.py --corp-filter "삼성전자" --top-k 5

# 다른 모델로 평가
python rag_evaluation_tool.py --model kakaobank --top-k 3
```

### 통계 확인

```bash
python rag_cli.py stats
```

---

## 🎯 핵심 함수 요약표

| 모듈               | 클래스/함수                | 기능                                        |
| ------------------ | -------------------------- | ------------------------------------------- |
| **Retriever**      | `Retriever.search(method)` | 통합 검색 (vector/keyword/hybrid 선택 가능) |
| **Reranker**       | `HybridReranker.rerank()`  | 2단계 하이브리드 리랭킹                     |
| **Augmenter**      | `augment(q, res, fmt)`     | 컨텍스트 증강                               |
| **Generator**      | `generate(q, ctx)`         | 답변 생성                                   |
| **RAGASEvaluator** | `evaluate(q, a, ctx)`      | RAG 평가                                    |
| **Encoder**        | `encode_query(text)`       | 임베딩 생성                                 |
| **TemporalParser** | `parse_temporal_query(q)`  | 시간 파싱                                   |

---

## 📚 추가 자료

- **LangGraph 통합**: `/graph/readme.md`
- **모듈 상세 문서**: `/service/rag/MODULES_README.md`

---

**작성일**: 2025-01-26
**버전**: 3.0

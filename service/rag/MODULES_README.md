# RAG 모듈 상세 가이드

**각 모듈별 핵심 기능과 함수 설명**

---

## 📁 모듈 구조

```
service/rag/
├── retrieval/          # 검색 + 리랭킹
├── augmentation/       # 컨텍스트 증강
├── generation/         # LLM 답변 생성
├── evaluation/         # 성능 평가
├── models/             # 임베딩 모델
├── query/              # 쿼리 분석
├── vectorstore/        # 벡터 DB
└── cli/                # CLI 도구
```

---

## 🔍 1. Retrieval (검색)

### `retrieval/retriever.py`

**통합 검색 리트리버 (벡터/키워드/하이브리드)**

하나의 `Retriever` 클래스로 모든 검색 방법 지원

#### 검색 방법

| 함수               | 방법           | 설명               |
| ------------------ | -------------- | ------------------ |
| `vector_search()`  | pgvector       | 코사인 유사도 검색 |
| `keyword_search()` | PostgreSQL FTS | 정확한 키워드 매칭 |
| `hybrid_search()`  | 결합           | 두 방법 Fusion     |

#### 핵심 함수

```python
async def hybrid_search(
    query: str,                    # 쿼리 텍스트
    query_embedding: List[float],  # 쿼리 임베딩
    config: SearchConfig,          # 검색 설정
    top_k: int = 10                # 반환 문서 수
) -> List[Dict]
```

#### Fusion 알고리즘

- **RRF (Reciprocal Rank Fusion)**: `score = Σ(1/(k+rank))`
- **Weighted Sum**: `score = w1*vec + w2*kw`
- **Max Score**: 각 문서의 최고 점수

---

## 🎯 2. Reranker (재정렬)

### `retrieval/hybrid_reranker_combined.py`

**2단계 하이브리드 리랭킹**

#### 작동 원리

```
Stage 1: Rule-based (빠른 필터링)
  ↓ top_k * 2~3배 선택
Stage 2: Cross-Encoder (정밀 평가)
  ↓ 최종 top_k 선택
```

#### 핵심 클래스

```python
class HybridReranker:
    def __init__(
        stage1_reranker=None,        # 1단계 (기본: CombinedReranker)
        use_cross_encoder=True,       # Cross-Encoder 사용 여부
        stage1_top_k_multiplier=3.0,  # 1단계 배수
        cross_encoder_model="BAAI/bge-reranker-v2-m3"
    )

    def rerank(query, candidates, top_k) -> List[Dict]
```

---

## 📝 3. Augmentation (증강)

### `augmentation/augmenter.py`

**검색 결과를 LLM 입력으로 변환**

#### 핵심 함수

```python
def augment(
    query: str,                 # 원본 쿼리
    search_results: List[Dict], # 검색 결과
    formatter: BaseFormatter    # 포맷터
) -> AugmentedContext
```

#### 처리 과정

1. 중복 문서 제거 (deduplication)
2. 토큰 수 계산 및 제한
3. Formatter로 형식 변환
4. 메타데이터 추출 (citations)

#### 출력

```python
class AugmentedContext:
    context_text: str           # 포맷팅된 컨텍스트
    token_count: int            # 토큰 수
    documents_used: List[Dict]  # 사용된 문서
    citations: List[Dict]       # 인용 정보
```

---

### `augmentation/formatters.py`

**다양한 출력 형식**

#### Formatter 종류

```python
class PromptFormatter:
    def format(query, documents) -> str
    # 출력: "참고 문서:\n1. ...\n2. ..."

class MarkdownFormatter:
    def format(query, documents) -> str
    # 출력: "### 문서 1\n**제목**: ...\n**내용**: ..."

class JSONFormatter:
    def format(query, documents) -> str
    # 출력: JSON 문자열
```

---

## 🤖 4. Generation (생성)

### `generation/generator.py`

**LLM 답변 생성**

#### 핵심 클래스

```python
class OllamaGenerator(LLMGenerator):
    def generate(
        query: str,                      # 질문
        context: str,                    # 검색된 컨텍스트
        config: GenerationConfig = None  # 생성 설정
    ) -> GeneratedAnswer
```

#### 생성 설정

```python
@dataclass
class GenerationConfig:
    model: str = "gemma3:4b"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
```

#### 출력

```python
@dataclass
class GeneratedAnswer:
    answer: str              # 생성된 답변
    model: str               # 사용된 모델
    tokens_used: int         # 사용된 토큰
    generation_time_ms: float  # 생성 시간
```

---

## 📊 5. Evaluation (평가)

### `evaluation/ragas_evaluator.py`

**RAG 품질 평가 (RAGAs 메트릭)**

#### 평가 메트릭

```python
class RAGASEvaluator:
    def evaluate_faithfulness(answer, contexts) -> float
    # 답변이 컨텍스트에 근거하는가? (0.0~1.0)

    def evaluate_answer_relevancy(query, answer) -> float
    # 답변이 질문과 관련있는가? (0.0~1.0)

    def evaluate_context_precision(query, contexts) -> float
    # 검색된 컨텍스트가 정확한가? (0.0~1.0)

    def evaluate_context_recall(contexts, ground_truth) -> float
    # 필요한 정보를 모두 검색했는가? (0.0~1.0)
```

#### 통합 평가

```python
def evaluate(
    query: str,
    answer: str,
    contexts: List[str],
    ground_truth: str = None
) -> RAGEvaluation
```

#### 출력

```python
@dataclass
class RAGEvaluation:
    faithfulness: float          # 충실성
    answer_relevancy: float      # 답변 관련성
    context_precision: float     # 컨텍스트 정밀도
    context_recall: float        # 컨텍스트 재현율
    average_score: float         # 평균 점수
```

---

### `evaluation/compare_retrieval_methods.py`

**검색 방법 비교 실험**

#### 핵심 클래스

```python
class RetrievalComparisonExperiment:
    async def compare_methods(
        test_queries: List[Dict],  # 테스트 쿼리
        top_k: int = 5,
        use_reranker: bool = True
    ) -> pd.DataFrame
```

#### 비교 대상

- Vector Search (pgvector)
- Keyword Search (PostgreSQL FTS)
- Hybrid Search (RRF/Weighted)

#### 출력 메트릭

- `search_time_ms`: 검색 시간
- `rerank_time_ms`: 리랭킹 시간
- `faithfulness`: 충실성 점수
- `context_precision`: 정밀도
- `average_score`: 종합 점수

---

## 🧩 6. Query (쿼리 분석)

### `query/temporal_parser.py`

**시간 표현 파싱**

#### 핵심 함수

```python
def parse_temporal_query(query: str) -> Dict
```

#### 입출력 예시

```python
# 입력: "2024년 2분기 실적은?"
# 출력: {'year': 2024, 'quarter': 2}

# 입력: "작년 12월 매출은?"
# 출력: {'year': 2024, 'month': 12}
```

---

## 🎨 7. Models (임베딩 모델)

### `models/encoder.py`

**임베딩 모델 통합 관리**

#### 핵심 함수

```python
def get_encoder(
    model_type: EmbeddingModelType,
    device: str = None
) -> Encoder

class Encoder:
    def encode_query(text: str) -> List[float]
    # 쿼리 임베딩 (단일)

    def encode_documents(texts: List[str]) -> List[List[float]]
    # 문서 임베딩 (배치)

    def get_dimension() -> int
    # 벡터 차원 반환
```

#### 지원 모델

```python
class EmbeddingModelType(Enum):
    MULTILINGUAL_E5_SMALL = "intfloat/multilingual-e5-small"  # 384차원
    KAKAOBANK_DEBERTA = "kakaobank/kf-deberta-base"           # 768차원
    FINE5_FINANCE = "fine5-finance"                           # 4096차원
```

---

## 🗄️ 8. VectorStore (벡터 DB)

### `vectorstore/pgvector_store.py`

**PostgreSQL + pgvector 인터페이스**

#### 핵심 함수

```python
async def insert_embeddings(
    documents: List[Dict]  # [{'chunk_id', 'embedding', 'metadata'}, ...]
) -> None

async def search_similar(
    embedding: List[float],
    top_k: int = 10,
    min_similarity: float = 0.0
) -> List[Dict]

async def create_index() -> None
# HNSW 인덱스 생성
```

---

## 🔄 Self-RAG / Corrective RAG

### **개념**

답변 품질을 검증하고, 품질이 낮으면 다른 방법으로 재검색

### **구현 위치**

- **LangGraph**: `graph/nodes/grounding_check.py`
- **로직**: 답변에 `[ref:]` 없으면 재시도

### **작동 흐름**

```
1. Retrieve (Vector) → 문서 검색
2. Rerank → 재정렬
3. Generate → 답변 생성
4. GroundingCheck → [ref:] 검증
   ├─ 성공 → Guardrail로 진행
   └─ 실패 → Retrieve로 재시도 (최대 1회)
            (다른 검색 방법 사용 가능)
```

### **확장 아이디어**

```python
# graph/nodes/grounding_check.py
def run(state: QAState) -> QAState:
    ans = state.get("draft_answer", "")
    grounded = ("[ref:" in ans)

    if not grounded and state.get("retry_count", 0) < 1:
        # 재검색 전략 설정
        if state["search_method"] == "vector":
            state["search_method"] = "keyword"  # Vector 실패 → Keyword
        elif state["search_method"] == "keyword":
            state["search_method"] = "hybrid"   # Keyword 실패 → Hybrid

        state["retry_count"] = state.get("retry_count", 0) + 1
        # Retrieve 노드로 돌아감 (LangGraph 엣지 설정)

    state["grounded"] = grounded
    return state
```

### **이름**

- **Self-RAG**: 자기 검증 RAG
- **Corrective RAG (CRAG)**: 교정 RAG
- **Adaptive RAG**: 적응형 RAG

---

## 📖 전체 파이프라인 예시

```python
from service.rag.retrieval.retriever import Retriever
from service.rag.retrieval.reranker import HybridReranker
from service.rag.augmentation.augmenter import DocumentAugmenter
from service.rag.augmentation.formatters import PromptFormatter
from service.rag.generation.generator import OllamaGenerator
from service.rag.evaluation.ragas_evaluator import RAGASEvaluator
from service.rag.models.config import EmbeddingModelType

# 1. 통합 검색 (벡터/키워드/하이브리드)
retriever = Retriever(
    model_type=EmbeddingModelType.MULTILINGUAL_E5_SMALL,
    enable_temporal_filter=True,
    enable_hybrid=True  # 하이브리드 검색 활성화
)

# 벡터 검색 (기본)
results = retriever.search(
    query="2차전지 산업 전망",
    top_k=20,
    search_method="vector"
)

# 하이브리드 검색 (재시도 시)
results = retriever.search(
    query="2차전지 산업 전망",
    top_k=20,
    search_method="hybrid"
)

# 2. 하이브리드 리랭킹
reranker = HybridReranker(
    use_cross_encoder=True,
    stage1_top_k_multiplier=3.0
)
reranked = reranker.rerank(
    query="2차전지 산업 전망",
    candidates=results,
    top_k=5
)

# 3. 컨텍스트 증강
augmenter = DocumentAugmenter(max_context_length=4000)
formatter = PromptFormatter()
augmented = augmenter.augment(
    query="2차전지 산업 전망",
    search_results=reranked,
    formatter=formatter
)

# 4. 답변 생성
generator = OllamaGenerator()
answer = generator.generate(
    query="2차전지 산업 전망",
    context=augmented.context_text
)

# 5. RAG 평가
evaluator = RAGASEvaluator()
evaluation = evaluator.evaluate(
    query="2차전지 산업 전망",
    answer=answer.answer,
    contexts=[doc['chunk_text'] for doc in reranked]
)

print(f"답변: {answer.answer}")
print(f"Faithfulness: {evaluation.faithfulness:.3f}")
print(f"Overall Score: {evaluation.average_score:.3f}")
```

---

## 🎯 핵심 함수 치트시트

| 모듈                | 함수                             | 입력                   | 출력                 |
| ------------------- | -------------------------------- | ---------------------- | -------------------- |
| **Retriever**       | `search(query, top_k)`           | 쿼리 문자열            | 유사 문서 리스트     |
| **HybridRetriever** | `hybrid_search(query, emb, cfg)` | 쿼리 + 임베딩          | 하이브리드 검색 결과 |
| **Reranker**        | `rerank(query, cands, k)`        | 쿼리 + 후보            | 재정렬된 문서        |
| **HybridReranker**  | `rerank(query, cands, k)`        | 쿼리 + 후보            | 2단계 리랭킹 결과    |
| **Augmenter**       | `augment(q, results, fmt)`       | 쿼리 + 결과            | AugmentedContext     |
| **Generator**       | `generate(q, ctx)`               | 쿼리 + 컨텍스트        | GeneratedAnswer      |
| **RAGASEvaluator**  | `evaluate(q, a, ctx)`            | 쿼리 + 답변 + 컨텍스트 | RAGEvaluation        |
| **Encoder**         | `encode_query(text)`             | 텍스트                 | 임베딩 벡터          |
| **TemporalParser**  | `parse_temporal_query(q)`        | 쿼리                   | 시간 정보 Dict       |

---

**작성일**: 2025-01-26
**버전**: 1.0

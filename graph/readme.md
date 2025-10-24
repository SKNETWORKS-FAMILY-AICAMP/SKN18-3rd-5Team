## [허깅페이스]
- .env에 API KEY 추가
- https://huggingface.co/meta-llama/Llama-2-7b-hf 에서 라이센스 사용 등록
- https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/config.json 메세지 확인
  - 대기 : Your request to access model meta-llama/Llama-2-7b-hf is awaiting a review from the repo authors.
  - 승인 메세지 확인 : 

## [랭그래프]

  - 전체 연동 개념도
    ```text
    [ Streamlit UI ]
        ↓ (user_level session_state)
    [ LangGraph App ]
        ↓
    [ Router Node ]
    └── 레벨별 파라미터 (top_k, context_len)
    [ Generate Node ]
    └── PROMPT_TEMPLATES[level] 기반 시스템/유저 프롬프트 구성
        ↓
    [ FT 모델 + pgvector 검색 ]
        ↓
    [ 결과 + ref 반환 ]
    ```
  - 노드 워크플로우
    ```dot
    digraph LangGraphQA {
      rankdir=LR;
      node [shape=rect, style=filled, fillcolor="#f8fafc", color="#94a3b8", fontname="Pretendard"];

      START [shape=oval, label="START", fillcolor="#e2e8f0"];
      END   [shape=oval, label="END", fillcolor="#e2e8f0"];

      Router [label="Router\n- set user_level\n- meta(top_k/rerank_n/max_ctx_tokens)"];
      QueryRewrite [label="QueryRewrite\n- keyword/time/ticker enrich"];
      Retrieve [label="Retrieve (pgvector)\n- top_k by level\n- optional date freshness"];
      Rerank [label="Rerank (optional)\n- cross-encoder/bge reranker\n- pick n by level"];
      ContextTrim [label="ContextTrim\n- dedup + token cut\n- collect citations"];
      Generate [label="Generate (FT-LLM)\n- System: common + PROMPT_TEMPLATES[level]\n- User: question+context+structure\n- append [ref: report_id, date]"];
      GroundingCheck [label="GroundingCheck\n- ref present?\n- numbers/dates consistent?\n- retry if insufficient"];
      Guardrail [label="Guardrail\n- investment disclaimer\n- sensitive filter"];
      Answer [label="Answer\n- normalize citations\n- return answer+meta"];

      START -> Router -> QueryRewrite -> Retrieve -> Rerank -> ContextTrim -> Generate -> GroundingCheck -> Guardrail -> Answer -> END;
      GroundingCheck -> Retrieve [style=dashed, label="retry (≤1x)"];
    }
    ```
| 노드                      | 설명                                      |
| ----------------------- | ------------------------------------------ |
| **START**               | 사용자의 질문과 투자 레벨 정보를 받아 파이프라인 시작       |
| **Router**              | 사용자 레벨에 따라 검색 개수, 컨텍스트 길이, 답변 깊이 설정 |
| **QueryRewrite**        | 질문을 분석해 시점·티커·키워드를 보강해 검색 효율 증가      |
| **Retrieve (pgvector)** | 리포트 데이터베이스에서 의미상 유사한 문단을 top-k로 검색   |
| **Rerank**              | 검색된 문단 중 질문과 가장 밀접한 내용을 상위로 재정렬      |
| **ContextTrim**         | 중복 문장을 제거하고, 최대 토큰 길이 내로 컨텍스트를 정리    |
| **Generate (FT-LLM)**   | 파인튜닝된 모델이 레벨별 프롬프트에 맞춰 답변을 생성        |
| **GroundingCheck**      | 답변이 실제 문서 근거와 일치하는지, ref가 포함됐는지 검증   |
| **Guardrail**           | 투자 권유나 민감 표현을 필터링하고 안내 문구를 자동 추가     |
| **Answer**              | 중복 인용을 정리하고 근거 문단과 함께 최종 답변을 반환      |
| **END**                 | 사용자에게 레벨별 맞춤형 근거 기반 답변이 전달          |

## 테스트 코드
  - service/pgvector_client.py는 실제 PG 접속 대신 로컬 CSV(cleaned_shinhan_example.csv) 존재 시 그 데이터를, 없으면 샘플 더미 데이터를 검색 결과로 돌려줍니다.
  - service/llm_client.py는 데모용 요약 응답을 생성하고, 반드시 [ref: ...]를 붙여 GroundingCheck를 통과하도록 했습니다.

## 추후 실제 인프라 연결 시:
  - service/llm_client.py → 진짜 파인튜닝 모델 API/로컬 호출로 교체
  - service/embeddings.py → sentence-transformers 임베딩으로 교체
  - service/pgvector_client.py → asyncpg + pgvector 실쿼리로 교체

## 코드 포인트
  - 레벨 주입: pages/views/chat.py →
  - LangGraph 그래프: graph/app_graph.py
    - Router → QueryRewrite → Retrieve → Rerank → ContextTrim → Generate → GroundingCheck → Guardrail → Answer
  - 프롬프트: service/prompt_templates.py
    - System: 공통 규칙 + 레벨 템플릿
    - User: 레벨별 답변 구조 요구(초/중/고)

## TODO (실연결 체크리스트)

- pgvector 연결 시:
  - service/pgvector_client.py의 fetch_similar를 asyncpg 쿼리로 교체
  - requirements.txt에 이미 asyncpg, psycopg, pgvector 명시됨
- 임베딩:
  - service/embeddings.py를 sentence-transformers(bge 등)로 교체
- FT 모델:
  - service/llm_client.py의 chat()을 실제 LLM 호출로 교체
- Reranker:
  - service/reranker.py에 Cross-Encoder 도입
  

## 참고

- 코루틴 처리
  -  `import asyncio`
  -  LLM 호출처럼 I/O 지연이 큰 작업을 조금이라도 효율적으로 처리하려면 비동기 함수로 두는 편이 자연스럽다
  -  Streamlit이 현재 동기만 허용한다는 제약 때문에 run_sync 이용
  -  장기적으로는 asyncio.run이 가능한 별도 백엔드(예: FastAPI, Celery worker)에서 LangGraph를 돌리고 Streamlit은 REST/gRPC로 결과만 가져오는 구조 고려
- Grounding 체크
  - 답변에 반드시 실제 보고서 등 근거(reference)가 명시적으로 포함되어야 통과되는 절차
- top-k : 가장 관련 있는 상위 k개의 결과
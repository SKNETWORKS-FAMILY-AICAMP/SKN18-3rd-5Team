from graph.state import QAState
from service.rag.retrieval.retriever import Retriever
from service.rag.models.config import EmbeddingModelType

# 전역 초기화 (성능 최적화)
retriever = Retriever(
    model_type=EmbeddingModelType.MULTILINGUAL_E5_SMALL, enable_temporal_filter=True
)

def run(state: QAState) -> QAState:
    """
    검색 노드 실행 함수

    역할:
      - 질의(재작성된 질문)와 top_k(검색 개수)를 입력받아, 외부 RAG Retriever로부터 관련 문서들을 top_k 만큼 조회합니다.
      - 검색 결과를 LangGraph pipeline의 컨벤션에 맞게 정제하여 state["retrieved"]에 저장합니다.
      - 만약 오류가 발생하면 state["retrieved"]는 빈 리스트로 반환합니다.

    Args:
        state (QAState): 현 단계까지 축적된 세션 상태(딕셔너리).
          - state["rewritten_query"]: 검색에 사용할 쿼리(전처리/재작성된 질문).
          - state["meta"]["top_k"]: 검색 결과 최대 개수.

    Returns:
        QAState: "retrieved" 키에 검색 결과가 추가된 상태.
    """
    try:
        k = state["meta"]["top_k"]
        query = state["rewritten_query"]

        print(f"[Retrieve] start (top_k={k}, query={query[:100]}...)")

        # RAG Retriever 사용
        results = retriever.search(
            query=query,
            top_k=k,
            min_similarity=0.0,
            include_metadata=True,
            use_reranker=False,  # 다음 단계에서 처리
            include_context=True,
        )

        print(f"[Retrieve] fetched={len(results)}")

        # 결과를 LangGraph 형식으로 변환
        formatted_results = []
        # TODO: result["metadata"]에서 report_id/date/title/url을 꺼내는 구조로 통일하기
        for result in results:
            metadata = result.get("metadata") or {}
            report_id = result.get("report_id") or metadata.get("report_id", "")
            date = result.get("date") or metadata.get("date", "")
            url = result.get("url") or metadata.get("url", "")
            title = result.get("title") or metadata.get("title", "")
            chunk_id = result.get("chunk_id") or metadata.get("chunk_id", "")
            formatted_results.append(
                {
                    "chunk_text": result.get("content", ""),
                    "report_id": report_id or "",
                    "date": date or "",
                    "url": url or "",
                    "title": title or "",
                    "chunk_id": chunk_id or "",
                    "similarity": result.get("similarity", 0.0),
                    "metadata": metadata,
                }
            )

        state["retrieved"] = formatted_results
        print(f"[Retrieve] complete (stored={len(formatted_results)})")
        return state

    except Exception as e:
        print(f"[Retrieve] error={e}")
        state["retrieved"] = []
        return state

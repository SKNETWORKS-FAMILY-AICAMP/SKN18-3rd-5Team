# from graph.state import QAState
# from graph.utils.async_tools import run_sync

# from service.pgv_temp.pgvector_client import fetch_similar
# from service.pgv_temp.embeddings import embed_query


# def run(state: QAState) -> QAState:
#     """
#     주어진 세션 상태(state)에서 다음을 수행합니다:
#       1. 현재 질문의 임베딩 벡터(쿼리 벡터)를 생성합니다.
#       2. PGVector에서 top_k개 유사 문서를 검색합니다.
#       3. 검색 결과를 state["retrieved"]에 저장하여, 이후 답변 생성에 활용하게 합니다.

#     - "meta" dict에서 top_k를 읽어 검색 갯수를 결정합니다.
#     - embed_query와 fetch_similar는 비동기로 동작하므로, 동기 함수 내에서 안전하게 실행합니다.
#     """
#     k = state["meta"]["top_k"]

#     # rewritten_query를 임베딩 벡터로 변환
#     qv = run_sync(embed_query(state["rewritten_query"]))
#     # 임베딩 벡터를 기준으로 top_k개 유사 문서 검색
#     rows = run_sync(fetch_similar(qv, k=k))

#     # 검색 결과를 세션 상태에 저장
#     state["retrieved"] = rows
#     return state


from graph.state import QAState
from service.rag.retrieval.retriever import Retriever
from service.rag.models.config import EmbeddingModelType
import logging

logger = logging.getLogger(__name__)

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

        logger.info(f"Retrieving documents for query: {query[:100]}...")

        # RAG Retriever 사용
        results = retriever.search(
            query=query,
            top_k=k,
            min_similarity=0.0,
            include_metadata=True,
            use_reranker=False,  # 다음 단계에서 처리
            include_context=True,
        )

        logger.info(f"Retrieved {len(results)} documents")

        # 결과를 LangGraph 형식으로 변환
        formatted_results = []
        # TODO: result["metadata"]에서 report_id/date/title/url을 꺼내는 구조로 통일하기
        for result in results:
            formatted_results.append(
                {
                    "chunk_text": result.get("content", ""),
                    "report_id": result.get("report_id", ""),
                    "date": result.get("date", ""),
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "chunk_id": result.get("chunk_id", ""),
                    "similarity": result.get("similarity", 0.0),
                    "metadata": result.get("metadata", {}),
                }
            )

        state["retrieved"] = formatted_results
        return state

    except Exception as e:
        logger.error(f"Error in retrieve_pg: {e}")
        state["retrieved"] = []
        return state

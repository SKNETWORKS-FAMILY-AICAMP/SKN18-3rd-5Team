from graph.state import QAState
from graph.utils.async_tools import run_sync
from service.pgv_temp.pgvector_client import fetch_similar
from service.pgv_temp.embeddings import embed_query


def run(state: QAState) -> QAState:
    """
    주어진 세션 상태(state)에서 다음을 수행합니다:
      1. 현재 질문의 임베딩 벡터(쿼리 벡터)를 생성합니다.
      2. PGVector에서 top_k개 유사 문서를 검색합니다.
      3. 검색 결과를 state["retrieved"]에 저장하여, 이후 답변 생성에 활용하게 합니다.

    - "meta" dict에서 top_k를 읽어 검색 갯수를 결정합니다.
    - embed_query와 fetch_similar는 비동기로 동작하므로, 동기 함수 내에서 안전하게 실행합니다.
    """
    k = state["meta"]["top_k"]

    # rewritten_query를 임베딩 벡터로 변환
    qv = run_sync(embed_query(state["rewritten_query"]))
    # 임베딩 벡터를 기준으로 top_k개 유사 문서 검색
    rows = run_sync(fetch_similar(qv, k=k))

    # 검색 결과를 세션 상태에 저장
    state["retrieved"] = rows
    return state

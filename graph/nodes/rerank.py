from graph.state import QAState
from graph.utils.async_tools import run_sync
from service.pgv_temp.reranker import rerank as rr


def run(state: QAState) -> QAState:
    """
    세션 상태(state)에서 다음을 수행합니다:
      1. 이전 단계에서 검색된 문서 리스트(state['retrieved'])를 가져옵니다.
      2. 질문(state['question'])과 후보 문서 리스트(cand})를 reranker 모델에 넣어, 상위 n개(top_n=n)를 재정렬하여 추출합니다.
      3. 재정렬 결과를 state["reranked"]에 저장합니다.

    - meta["rerank_n"]에서 재정렬 결과 개수를 설정합니다.
    - service.pgv_temp.reranker.rerank는 비동기 함수이므로 동기 컨텍스트에서 안전하게 실행합니다.
    """
    cand = state.get("retrieved", [])
    n = state["meta"]["rerank_n"]
    state["reranked"] = run_sync(rr(state["question"], cand, top_n=n))
    return state

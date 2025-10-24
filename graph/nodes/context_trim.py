from graph.state import QAState

def run(state: QAState) -> QAState:
    """
    세션 상태(state)에서 context를 생성하고 관련 citation 정보를 추출합니다.

    1. 최우선적으로 reranked 문서 리스트(state['reranked'])를, 없으면 retrieved 리스트(state['retrieved'])를 가져옵니다.
    2. 각 문서의 'chunk_text'들을 "\n\n---\n\n" 으로 구분하여 이어 붙여 context string을 만듭니다.
    3. context는 최대 토큰 길이(meta["max_ctx_tokens"] * 4 chars)로 자릅니다.
    4. 각 문서의 citation 정보(report_id, date, url, title, chunk_id)를 추출하여 state["citations"]에 담습니다.

    Args:
        state (QAState): 질의 세션 상태

    Returns:
        QAState: context와 citations이 추가된 상태
    """
    # 1. rerank 결과가 있으면 그것을, 없으면 retrieve 결과를 사용
    items = state.get("reranked") or state.get("retrieved", [])

    # 2. context용 텍스트 추출 및 구분자 삽입
    chunks = [c["chunk_text"] for c in items]
    context = "\n\n---\n\n".join(chunks)

    # 3. context 최대 길이 제한 (토큰 추정: 대략적으로 1토큰~4자 기준으로 문자열 자름)
    state["context"] = context[: state["meta"]["max_ctx_tokens"] * 4]

    # 4. 문서별 citation(참고문헌) 정보 추출
    state["citations"] = [{
        "report_id": c["report_id"],
        "date": c["date"],
        "url": c["url"],
        "title": c["title"],
        "chunk_id": c["chunk_id"]
    } for c in items]

    return state
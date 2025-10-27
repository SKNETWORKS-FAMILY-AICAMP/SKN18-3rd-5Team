from graph.state import QAState
from graph.utils.citations import normalize


def run(state: QAState) -> QAState:
    """
    Answer 단계 함수.
    - 이전 단계에서 수집된 'citations'(참조 문헌/자료) 리스트를 정규화(normalize)합니다.
    - citations 필드는 list여야 하며, citation 포맷을 통일하여 downstream 평가와 UI 표시에 적합하게 만듭니다.
    - 정규화된 citations를 state["citations"]에 저장합니다.
    - state를 반환합니다.

    Args:
        state (QAState): 세션 상태 객체

    Returns:
        QAState: citations가 정규화된 상태 객체
    """
    citations = normalize(state.get("citations", []))
    state["citations"] = citations
    print(f"[Answer] complete (citations={len(citations)})")
    return state

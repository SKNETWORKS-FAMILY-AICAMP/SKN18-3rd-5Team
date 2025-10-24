from graph.state import QAState

def run(state: QAState) -> QAState:
    """
    GroundingCheck 단계:
    답변이 실제 근거(보고서 등)에 기반하고 있음을 확인합니다.
    
    1. draft_answer(초안 답변) 내에 "[ref:" 문자열이 포함되어 있는지 검사합니다.
       - "[ref:"가 있으면 답변이 근거(reference)를 명시적으로 포함하고 있다고 간주함
    2. 검사 결과를 state["grounded"]에 True/False로 저장합니다.

    Args:
        state (QAState): 질의 세션 상태

    Returns:
        QAState: grounded (근거 명시 여부) 플래그가 추가된 상태
    """
    ans = state.get("draft_answer", "")
    state["grounded"] = ("[ref:" in ans)
    
    return state
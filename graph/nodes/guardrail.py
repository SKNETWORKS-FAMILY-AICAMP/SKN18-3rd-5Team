from graph.state import QAState
from graph.utils.async_tools import run_sync

DISCLAIMER = "\n\n※ 본 답변은 정보 제공 목적이며 투자 권유가 아닙니다."


def _ensure_disclaimer(answer: str) -> str:
    """필수 투자 권유 아님 문구를 보장."""
    if "투자 권유가 아니다" in answer or DISCLAIMER in answer:
        return answer
    return answer + DISCLAIMER

def _check_policy(answer: str):
    """
    비동기 정책 검증(예: 민감 정보 포함 여부, IR/법률 위반 여부 등)을 향후 연동할 수 있도록 한 자리 표시자 함수입니다.
    현재는 아무 정책 검증도 수행하지 않으며 항상 None을 반환합니다.

    Args:
        answer (str): 평가 대상 답변 문자열

    Returns:
        None: 추후 정책 위반 시 flag/메시지 등으로 확장 예정
    """
    return None

def run(state: QAState) -> QAState:
    """
    Guardrail 단계 함수.
    - draft_answer에 투자 권유 아님을 명시하는 문구(DISCLAIMER)가 포함되어 있는지 검사합니다.
    - 만약 이미 '투자 권유가 아니다' 또는 DISCLAIMER 문구가 draft_answer에 포함되어 있지 않다면,
      답변 끝에 DISCLAIMER를 추가합니다.
    - policy_flag는 추가 정책 체크가 없는 경우 None으로 처리합니다.
    - 최종적으로 수정된 state를 반환합니다.

    Args:
        state (QAState): 질의 세션 상태 객체

    Returns:
        QAState: DISCLAIMER가 반영된 상태 객체
    """
    ans = state.get("draft_answer", "")
    print(f"[Guardrail] start (len={len(ans)})")
    state["draft_answer"] = run_sync(_ensure_disclaimer(ans))
    # 정책 위반 플래그: 향후 비동기 정책 엔진 연계를 고려해 run_sync 사용
    state["policy_flag"] = run_sync(_check_policy(state["draft_answer"]))
    print(f"[Guardrail] complete (disclaimer_appended={state['draft_answer'].endswith(DISCLAIMER)})")
    return state

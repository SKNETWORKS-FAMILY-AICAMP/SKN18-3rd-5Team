from graph.state import QAState
from graph.utils.async_tools import run_sync
from service.llm.llm_client import chat
from service.llm.prompt_templates import build_system_prompt, build_user_prompt

def run(state: QAState) -> QAState:
    """
    Generate 단계: LLM을 이용해 draft_answer(초안 답변)를 만듭니다.

    1. 사용자의 투자 수준(user_level)에 맞는 system prompt를 생성합니다.
    2. 질문 및 context와 수준 정보를 바탕으로 user prompt를 생성합니다.
    3. chat() 함수를 호출해 LLM 기반 draft answer를 생성합니다.
    4. 그 결과를 state["draft_answer"]에 저장하여 반환합니다.

    Args:
        state (QAState): 질의 처리 세션 상태

    Returns:
        QAState: draft_answer가 추가된 상태
    """
    lvl = state["user_level"]  # 사용자 수준 ("beginner" | "intermediate" | "advanced")
    sys_p = build_system_prompt(lvl)  # LLM에 넣을 시스템 프롬프트 생성
    usr_p = build_user_prompt(state["question"], state["context"], lvl)  # 사용자 프롬프트 생성
    
    ans = run_sync(chat(system=sys_p, user=usr_p, max_tokens=512))  # LLM 호출하여 답변 생성
    state["draft_answer"] = ans  # 초안 답변을 상태에 저장
    
    return state

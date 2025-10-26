from __future__ import annotations

from graph.state import QAState
from service.llm.llm_client import chat
from service.llm.prompt_templates import build_system_prompt, build_user_prompt


def run(state: QAState) -> QAState:
    """
    역할:
      - QAState 내의 user_level, question, context 등 필드를 이용해 시스템/유저 프롬프트를 생성
      - LLM(Chat)으로 답변 초안 생성 → state["draft_answer"]에 저장

    동작 흐름:
      1. state에서 유저 레벨, 질문, 컨텍스트 추출 (각각의 값이 없으면 디폴트 사용)
      2. 시스템 프롬프트(Based on user_level)와 유저 프롬프트(질문+문맥)를 생성
      3. chat() 함수 호출로 답변 텍스트 생성 (최대 512토큰)
      4. 생성된 답변을 state["draft_answer"]에 저장
      5. 예외 발생시 에러로그 남기고 안내 문구 반환

    Args:
        state (QAState): LangGraph에서 전달받은 워크플로 상태 딕셔너리

    Returns:
        QAState: draft_answer가 추가된 상태
    """
    try:
        # 1. 입력값 추출
        user_level = state.get("user_level", "intermediate")  # 유저 전문성 수준
        question = state.get("question", "")                  # 질문 텍스트
        context = state.get("context", "")                    # RAG 검색 컨텍스트

        print(f"[Generate] start (level={user_level}, ctx_len={len(context)})")

        # 2. 프롬프트 생성
        system_prompt = build_system_prompt(user_level)
        user_prompt = build_user_prompt(question, context, user_level)

        # 3. LLM 답변 생성
        answer = chat(
            system=system_prompt,
            user=user_prompt,
            max_tokens=512,
        )
        state["draft_answer"] = answer

        print(f"[Generate] complete (answer_chars={len(answer)})")
    except Exception as exc:
        # 예외 발생 시 로깅 및 안내 문구 반환
        print(f"[Generate] error={exc}")
        state["draft_answer"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."

    return state

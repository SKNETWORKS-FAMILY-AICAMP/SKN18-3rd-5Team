from graph.state import QAState
from graph.utils.level import defaults

def run(state: QAState) -> QAState:
    """
    사용자 레벨(user_level)에 따라 세션 상태(state)를 표준화하고,
    그에 맞는 주요 파라미터 셋(meta: top_k, rerank_n, max_ctx_tokens 등)을 할당한다.

    - user_level이 None이거나 잘못 지정된 경우 중간값('beginner')을 사용
    - meta에는 level별 기본값 dict를 저장함
      (참조: graph/utils/level.py의 default)
    """
    
    lvl = (state.get("user_level") or "beginner").lower()
    if lvl not in ("beginner","intermediate","advanced"):
        lvl = "beginner"
    state["user_level"] = lvl
    state["meta"] = defaults(lvl)
    return state

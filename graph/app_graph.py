from langgraph.graph import StateGraph, START, END
from graph.state import QAState
from graph.nodes import router, query_rewrite, retrieve_pg, rerank, context_trim, generate, grounding_check, guardrail, answer

def build_app():
    """
    역할:
      - Router: 입력 쿼리 라우팅/분기
      - QueryRewrite: 쿼리 전처리(예: 재작성)
      - Retrieve: 유사도 검색(벡터DB 등에서 관련 문서 조회)
      - Rerank: 검색결과 재정렬/필터링
      - ContextTrim: LLM 입력 토큰 제한에 맞게 컨텍스트 자르기/요약
      - Generate: LLM을 활용한 답안 초안 생성
      - GroundingCheck: 답변에 근거 명시([ref:] 등) 여부 검증
      - Guardrail: 정책 문구(DISCALIMER 등) 포함 및 정책 위반 체크
      - Answer: 참조문헌 등 정규화 및 최종 답변 반환

    그래프는 위 순서대로 각 노드를 직렬 연결해서 구성됩니다.
    마지막에는 compile()로 LangGraph 워크플로를 빌드하여 반환합니다.

    Returns:
        LangGraph: 완성된 질의응답 그래프 오브젝트
    """
    g = StateGraph(QAState)
    g.add_node("Router", router.run)
    g.add_node("QueryRewrite", query_rewrite.run)
    g.add_node("Retrieve", retrieve_pg.run)
    g.add_node("Rerank", rerank.run)
    g.add_node("ContextTrim", context_trim.run)
    g.add_node("Generate", generate.run)
    g.add_node("GroundingCheck", grounding_check.run)
    g.add_node("Guardrail", guardrail.run)
    g.add_node("Answer", answer.run)

    # 노드 간 엣지(흐름) 정의: 순차적으로 연결
    g.add_edge(START, "Router")
    g.add_edge("Router", "QueryRewrite")
    g.add_edge("QueryRewrite", "Retrieve")
    g.add_edge("Retrieve", "Rerank")
    g.add_edge("Rerank", "ContextTrim")
    g.add_edge("ContextTrim", "Generate")
    g.add_edge("Generate", "GroundingCheck")
    g.add_edge("GroundingCheck", "Guardrail")
    g.add_edge("Guardrail", "Answer")
    g.add_edge("Answer", END)
    return g.compile()
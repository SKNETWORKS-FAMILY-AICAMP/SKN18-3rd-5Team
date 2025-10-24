from typing import TypedDict, List, Dict, Optional

class QAState(TypedDict, total=False):
    question: str                              # 사용자가 입력한 원질문
    user_level: str                            # "beginner" | "intermediate" | "advanced" - 투자자 수준 설정
    
    # 질문 재작성
    rewritten_query: str                       # LLM 등으로 재작성된 질의문, retrieval에 사용
    
    retrieved: List[Dict]                      # vectordb retrieval 결과: [{chunk_text, report_id, date, title, url, score, chunk_id}]
    reranked: List[Dict]                       # rerank 모델을 거친 top-k 문서 리스트
    context: str                               # LLM에 주입할 context string
    
    draft_answer: str                          # LLM으로부터 받은 초안 답변
    citations: List[Dict]                      # 최종 답변에 포함된 레퍼런스 정보 [{report_id, date, url, title, chunk_id}]
    grounded: bool                             # 답변이 원문 근거 기반으로 작성됐는지 (Grounding)
    policy_flag: Optional[str]                 # 정책 위반 감지시 flag (ex: 금지 발언, OOS 등)
    
    meta: Dict                                 # {"top_k":int, "rerank_n":int, "max_ctx_tokens":int, ...} 등 질의 처리 주요 파라미터
    
    # vectordb의 결과 평가 (질문-컨텍스트 매칭 품질 등)
    # evaluation_result: str                     # 예/아니오 값 (ex: yes / no)
    # evaluation_score: float                    # 평가 점수 
    # evaluation_detail: str                     # 평가 사유/판단 근거 등 상세 설명

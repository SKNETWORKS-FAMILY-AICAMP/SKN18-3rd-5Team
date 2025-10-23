def defaults(level: str):
    """
    투자자 수준별(defaults) 주요 파라미터 세트 반환

    - beginner: 초보 투자자. 컨텍스트 길이와 retrieved 문서 수를 줄여 쉬운 요약을 우선
        - top_k: 벡터DB에서 retrieval할 문서 개수
        - rerank_n: 재정렬 평가할 문서 개수
        - max_ctx_tokens: LLM에 주입하는 최대 컨텍스트 토큰수
    - intermediate: 기본값. 일반적 사용자를 위한 중간 수준 파라미터
    - advanced: 상급자. 더 긴 문맥, 더 많은 후보 문서 노출로 숫자, 비교, 세부 정보 강조
    
    참고
    
    - 실제 질의/정답 세트를 가지고 LLM 성능을 측정하며 수동으로 파라미터 조정.
    - max_ctx_tokens는 모델 한계 토큰 수와 균형을 맞춰야 하고
    - top_k와 rerank_n은 너무 크면 노이즈·비용 증가, 너무 작으면 정보 누락 발생

    Args:
        level (str): "beginner" | "intermediate" | "advanced" 중 하나

    Returns:
        dict: 레벨별 파라미터 값 (top_k, rerank_n, max_ctx_tokens)
    """
    lvl = (level or "intermediate").lower()
    if lvl == "beginner":
        # 초급: 간결하고 쉬운 답변, 작은 context window
        return {"top_k": 3, "rerank_n": 3, "max_ctx_tokens": 2000}
    if lvl == "advanced":
        # 고급: 많은 자료, 더 긴 context, 수치·비교 중심
        return {"top_k": 5, "rerank_n": 6, "max_ctx_tokens": 3500}
    # 중급(기본): 균형 잡힌 설정
    return {"top_k": 4, "rerank_n": 4, "max_ctx_tokens": 2600}
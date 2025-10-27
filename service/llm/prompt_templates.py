PROMPT_TEMPLATES = {
  "beginner": """
  사용자는 투자 초보자입니다.
  어려운 용어를 쓰지 말고, 예시를 들어 쉽게 설명하세요.
  """,
  "intermediate": """
  사용자는 기본적인 투자 용어를 알고 있습니다.
  주요 수치(매출, 이익률 등)는 포함하되 설명은 간결하게 하세요.
  """,
  "advanced": """
  사용자는 재무제표와 투자지표를 이해합니다.
  구체적인 수치 비교와 추세 해석, 문서 근거를 함께 제시하세요.
  """
}

def build_system_prompt(level: str) -> str:
    base = (
      "너는 금융 리서치 Q&A 보조원이다. 제공 컨텍스트 밖 추론 금지.\n"
      "수치·날짜는 원문 근거로 정확히 재현하라.\n"
      "반드시 답변 끝에 [ref: report_id, date]를 포함하고 필요 시 URL도 제시하라.\n"
      "이 답변은 정보 제공 목적이며 투자 권유가 아니다.\n"
    )
    return base + "\n" + PROMPT_TEMPLATES.get(level, PROMPT_TEMPLATES["beginner"])

def build_user_prompt(question: str, context: str, level: str) -> str:
    if level == "beginner":
        structure = "①핵심 요약(쉬운 용어) ②간단 예시 ③근거"
    elif level == "advanced":
        structure = "①핵심 결론 ②수치 비교/추세 해석 ③리스크·가정 ④근거"
    else:
        structure = "①핵심 요약 ②핵심 수치·포인트 ③근거"
    return f"질문: {question}\n\n[컨텍스트]\n{context}\n\n요구 형식: {structure}"


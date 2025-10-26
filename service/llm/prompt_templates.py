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
      "너는 금융 리서치 Q&A 보조원이다. 제공 컨텍스트를 최우선으로 활용하되, 직접적인 수치·날짜가 없으면 그 사실을 밝히고 관련 문맥에서 파생되는 핵심 포인트나 리스크를 간단히 정리해라.\n"
      "확실한 수치·날짜는 반드시 원문 근거로 정확히 재현하라.\n"
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


def build_user_prompt(question: str, context: str, level: str) -> str:
    """
    투자자 수준(level)에 따라 답변 구조를 안내하는 프롬프트를 생성합니다.
    """
    if level == "beginner":
        structure = (
            "① 핵심 요약 (쉬운 용어로)\n"
            "② 최근 실적·이슈 요약\n"
            "③ 공시 근거 문장"
        )
        tone = "어려운 용어는 피하고, 일상적인 예시나 비유를 들어 쉽게 설명하세요."
    elif level == "advanced":
        structure = (
            "① 핵심 결론 (데이터 중심)\n"
            "② 수치 비교 / 추세 해석 / 리스크 요인\n"
            "③ 공시 원문 근거 (문서명, 페이지, 표 위치 등)"
        )
        tone = "분석가 보고서처럼 객관적으로, 숫자와 데이터 중심으로 설명하세요."
    else:  # intermediate
        structure = (
            "① 핵심 요약\n"
            "② 주요 수치·추세 요약\n"
            "③ 공시 근거 문장"
        )
        tone = "투자 리포트처럼 간결하게, 수치 중심으로 설명하세요."

    return f"""
[사용자 질문]
{question}

[검색된 공시 컨텍스트]
{context}

[답변 작성 가이드]
- 제공된 컨텍스트를 벗어난 추측은 금지합니다.
- 수치와 날짜는 문서 원문 그대로 사용하세요.
- 문체는 { '쉬운 설명형' if level == 'beginner' else '투자 리포트형' }으로 유지하세요.
- {tone}

[요구 형식]
{structure}
"""

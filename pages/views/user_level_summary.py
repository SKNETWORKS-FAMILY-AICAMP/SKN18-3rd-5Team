import streamlit as st

DEFAULT_LEVEL_SUMMARY = {
    "beginner": {"level": "파도 관찰자(초급)", "icon": "🔰", "color": "#4CAF50"},
    "intermediate": {"level": "파도 타는 서퍼(중급)", "icon": "⚡", "color": "#FF9800"},
    "advanced": {"level": "시장 항해자(고급)", "icon": "🚀", "color": "#F44336"},
}


def render_user_level_summary():
    stored_level = st.session_state.get("user_level")
    stored_info = st.session_state.get("user_level_info")

    info = stored_info if isinstance(stored_info, dict) else None
    if info is None and isinstance(stored_level, str):
        info = DEFAULT_LEVEL_SUMMARY.get(stored_level, {})

    if info:
        level_label = info.get("level") or DEFAULT_LEVEL_SUMMARY.get(stored_level, {}).get("level", "알 수 없음")
        icon = info.get("icon", "ℹ️")
        correct = info.get("correct_answers")
        total = info.get("total_questions")

        message = f"나의 투자 레벨: {icon} {level_label}"
        if isinstance(correct, int) and isinstance(total, int) and total > 0:
            message += f" · 정답 {correct}/{total}"

        st.info(message)
    else:
        st.info("ℹ️ 먼저 투자 레벨을 진단 받고 더욱 최적화된 답변을 받아보세요!")

import streamlit as st

# =========================
# User level summary
# =========================
def render_user_level_summary():
    user_level_info = st.session_state.get("user_level")

    if user_level_info:
        level = user_level_info.get("level", "알 수 없음")
        icon = user_level_info.get("icon", "ℹ️")
        color = user_level_info.get("color", "#2563eb")
        correct = user_level_info.get("correct_answers", 0)
        total = user_level_info.get("total_questions", 0)
        
        st.info(f"나의 투자 레벨 {icon} {level}")
    else:
        st.info("[초급] 설정에 맞춰 답변해 드립니다. 투자 레벨 진단 받고 최적화된 답변을 받아보세요!")
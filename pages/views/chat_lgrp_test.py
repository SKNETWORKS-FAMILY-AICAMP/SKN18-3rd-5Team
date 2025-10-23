import streamlit as st
from graph.app_graph import build_app

# test
def render_chat_lgrp_test():
    st.markdown("---")
    st.title("LangGraph 테스트챗")

    # ex: 2차 전지 산업 어떄?
    q = st.text_input("질문을 입력하세요")
    level = st.selectbox("사용자 레벨", ["beginner","intermediate","advanced"], index=1)

    if "lg_app" not in st.session_state:
        st.session_state["lg_app"] = build_app()

    if st.button("질문하기") and q:
        app = st.session_state["lg_app"]
        state = app.invoke({"question": q, "user_level": level})  # sync 버전
        st.markdown(state["draft_answer"])
        with st.expander("출처 보기"):
            for c in state.get("citations", []):
                st.write(f"- {c['title']} ({c['date']}), {c['url']} [ref: {c['report_id']}]")
                
    st.markdown("---")
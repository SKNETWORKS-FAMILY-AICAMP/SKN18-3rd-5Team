from __future__ import annotations
from datetime import datetime
import streamlit as st
from pages.app_bootstrap import render_sidebar, render_page_title  # 필수

# ---------------------------
# 기본 설정
# ---------------------------
st.set_page_config(
    page_title="Investment Q&A System",
    page_icon="🤖",
    layout="wide",
)
render_sidebar()

# ---------------------------
# UTILS
# ---------------------------
def _format_timestamp(timestamp: datetime) -> str:
    return timestamp.strftime("%Y-%m-%d %H:%M")

# ---------------------------
# VIEW
# ---------------------------
def render_top():
    # 시스템 이름 등 구현
    pass
    
# 대시보드 구현
def render_status_overview() -> None:
    ## 예시 입니다 -> 수정
    
    """Display current RAG / LLM training status metrics."""
    st.subheader("RAG/LLM 학습 현황")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("임베딩 문서 수", value="1,248", delta="+32")
    with col2:
        st.metric("마지막 학습", value=_format_timestamp(datetime.now()))
    with col3:
        st.metric("평균 응답 정확도", value="92%", delta="+3%")
    st.progress(0.6, text="재학습 파이프라인 진행률 (예시)")
    st.caption("실제 값은 백엔드 연동 후 갱신하세요.")

# 나의 투자 수준 선택하기
def render_user_level():
    st.subheader("🎯 나의 투자 수준 선택하기")
    
    # MD 파일의 문제은행 (적응형 트리 구조)
    all_questions = [
        # Q1 (초급·개념)
        {
            "id": "Q1",
            "q": "PER은 무엇을 의미하나요?",
            "options": ["A. 주가/주당순이익(Price/Earnings)", "B. 주가/매출액", "C. 영업이익/자산", "D. 배당성향"],
            "answer": "A. 주가/주당순이익(Price/Earnings)",
            "difficulty": "초급",
            "type": "개념"
        },
        # Q2 (초급·개념)
        {
            "id": "Q2",
            "q": "'매출'과 '영업이익'의 관계로 옳은 것은?",
            "options": ["A. 매출 = 영업이익 + 판관비", "B. 영업이익 = 매출 – 매출원가 – 판관비", "C. 영업이익 = 순이익 + 이자비용", "D. 매출 = 영업이익 – 원가"],
            "answer": "B. 영업이익 = 매출 – 매출원가 – 판관비",
            "difficulty": "초급",
            "type": "개념"
        },
        # Q3 (초급·해석)
        {
            "id": "Q3",
            "q": "매출 10조, 영업이익 1조일 때 영업이익률은?",
            "options": ["A. 1%", "B. 5%", "C. 10%", "D. 20%"],
            "answer": "C. 10%",
            "difficulty": "초급",
            "type": "해석"
        },
        # Q4 (초급·개념)
        {
            "id": "Q4",
            "q": "'2Q25' 표기는 무엇을 뜻하나요?",
            "options": ["A. 2025년 2월", "B. 2025년 2분기", "C. 2025년 4월", "D. 2025년 상반기 전체"],
            "answer": "B. 2025년 2분기",
            "difficulty": "초급",
            "type": "개념"
        },
        # Q5 (초급·개념)
        {
            "id": "Q5",
            "q": "ROE는 무엇의 비율인가요?",
            "options": ["A. 순이익/총자산", "B. 영업이익/자본", "C. 순이익/자기자본", "D. 매출/자본"],
            "answer": "C. 순이익/자기자본",
            "difficulty": "초급",
            "type": "개념"
        },
        # Q6 (중급·해석)
        {
            "id": "Q6",
            "q": "자산 100, 부채 60일 때 부채비율(부채/자본)은?",
            "options": ["A. 60%", "B. 150%", "C. 166%", "D. 40%"],
            "answer": "B. 150%",
            "difficulty": "중급",
            "type": "해석"
        },
        # Q7 (중급·개념)
        {
            "id": "Q7",
            "q": "전분기 대비 성장을 나타내는 표현은?",
            "options": ["A. YoY", "B. QoQ", "C. CAGR", "D. MoM"],
            "answer": "B. QoQ",
            "difficulty": "중급",
            "type": "개념"
        },
        # Q8 (중급·개념)
        {
            "id": "Q8",
            "q": "Free Cash Flow(FCF)에 대한 설명으로 가장 적절한 것은?",
            "options": ["A. 순이익에서 법인세만 뺀 값", "B. 영업현금흐름 – CAPEX", "C. 영업이익 – 감가상각", "D. 매출 – 원가 – 판관비"],
            "answer": "B. 영업현금흐름 – CAPEX",
            "difficulty": "중급",
            "type": "개념"
        },
        # Q9 (중급·해석)
        {
            "id": "Q9",
            "q": "A사 매출 1조 중 배터리 6천억, 나머지 4천억. 배터리 매출 비중은?",
            "options": ["A. 40%", "B. 50%", "C. 60%", "D. 70%"],
            "answer": "C. 60%",
            "difficulty": "중급",
            "type": "해석"
        },
        # Q10 (고급·개념)
        {
            "id": "Q10",
            "q": "EV/EBITDA 멀티플을 선호하는 상황으로 가장 적절한 것은?",
            "options": ["A. 자산가치 중심 산업", "B. 감가상각이 큰 자본집약 산업", "C. 적자 상태의 스타트업", "D. 현금성 자산이 매우 큰 인터넷 플랫폼"],
            "answer": "B. 감가상각이 큰 자본집약 산업",
            "difficulty": "고급",
            "type": "개념"
        },
        # Q11 (고급·개념)
        {
            "id": "Q11",
            "q": "은행의 NIM(Net Interest Margin)에 직접적 영향을 주는 요인은?",
            "options": ["A. 환율", "B. 기준금리와 예대금리 스프레드", "C. 법인세율", "D. R&D비용"],
            "answer": "B. 기준금리와 예대금리 스프레드",
            "difficulty": "고급",
            "type": "개념"
        },
        # Q12 (고급·해석)
        {
            "id": "Q12",
            "q": "WACC 상승 시 일반적으로 옳은 해석은?",
            "options": ["A. 기업가치 상승 압력", "B. 투자 프로젝트의 현재가치가 커짐", "C. 투자 의사결정 문턱이 높아짐", "D. 배당성향 자동 증가"],
            "answer": "C. 투자 의사결정 문턱이 높아짐",
            "difficulty": "고급",
            "type": "해석"
        },
        # Q13 (고급·개념)
        {
            "id": "Q13",
            "q": "EBITDA와 영업이익(OP)의 차이를 가장 잘 설명한 것은?",
            "options": ["A. EBITDA는 이자·세금·감가상각·무형상각을 차감한 이익", "B. EBITDA는 감가상각·무형상각을 더한 영업이익", "C. 영업이익은 현금흐름을 반영", "D. 두 지표는 동일"],
            "answer": "B. EBITDA는 감가상각·무형상각을 더한 영업이익",
            "difficulty": "고급",
            "type": "개념"
        },
        # Q14 (고급·개념)
        {
            "id": "Q14",
            "q": "'Guidance'와 'Consensus'의 차이는?",
            "options": ["A. Guidance=애널리스트 전망, Consensus=회사 제시", "B. Guidance=회사 전망, Consensus=애널리스트 평균 전망", "C. 둘 다 회사 발표", "D. 둘 다 애널리스트 추정"],
            "answer": "B. Guidance=회사 전망, Consensus=애널리스트 평균 전망",
            "difficulty": "고급",
            "type": "개념"
        },
        # Q15 (중급·해석)
        {
            "id": "Q15",
            "q": "매출 총이익률(Gross Margin)이 하락했다. 가장 합리적인 1차 원인 가설은?",
            "options": ["A. 판관비 증가", "B. 매출원가율 상승", "C. 법인세 증가", "D. 감가상각비 감소"],
            "answer": "B. 매출원가율 상승",
            "difficulty": "중급",
            "type": "해석"
        },
        # Q16 (중급·해석)
        {
            "id": "Q16",
            "q": "두 분기 연속 매출 정체·영업이익률 하락. 가장 가능성 높은 설명은?",
            "options": ["A. 제품 믹스 개선", "B. 판관비 증가 또는 할인 확대", "C. CAPEX 감소", "D. 법인세 환급 증가"],
            "answer": "B. 판관비 증가 또는 할인 확대",
            "difficulty": "중급",
            "type": "해석"
        }
    ]
    
    # 적응형 트리 구조를 위한 상태 초기화
    if 'current_question_id' not in st.session_state:
        st.session_state.current_question_id = "Q1"  # Q1부터 시작
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
    if 'correct_answers' not in st.session_state:
        st.session_state.correct_answers = 0
    if 'test_completed' not in st.session_state:
        st.session_state.test_completed = False
    if 'question_path' not in st.session_state:
        st.session_state.question_path = []  # 문제 경로 추적
    if 'used_questions' not in st.session_state:
        st.session_state.used_questions = []  # 이미 사용된 문제들 추적
    
    # 적응형 난이도 시스템 (중복 방지 포함)
    def get_next_question(current_id, is_correct):
        # 난이도별 문제 그룹
        beginner_questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        intermediate_questions = ["Q6", "Q7", "Q8", "Q9", "Q15", "Q16"]
        advanced_questions = ["Q10", "Q11", "Q12", "Q13", "Q14"]
        
        # 현재 문제의 난이도 파악
        current_difficulty = None
        if current_id in beginner_questions:
            current_difficulty = "beginner"
        elif current_id in intermediate_questions:
            current_difficulty = "intermediate"
        elif current_id in advanced_questions:
            current_difficulty = "advanced"
        
        # 정답이면 상위 난이도, 오답이면 하위 난이도
        if is_correct:
            if current_difficulty == "beginner":
                target_questions = intermediate_questions
            elif current_difficulty == "intermediate":
                target_questions = advanced_questions
            else:  # advanced
                target_questions = advanced_questions  # 최고 난이도 유지
        else:
            if current_difficulty == "advanced":
                target_questions = intermediate_questions
            elif current_difficulty == "intermediate":
                target_questions = beginner_questions
            else:  # beginner
                target_questions = beginner_questions  # 최저 난이도 유지
        
        # 사용되지 않은 문제 중에서 선택
        available_questions = [q for q in target_questions if q not in st.session_state.used_questions]
        
        if available_questions:
            # 랜덤하게 선택 (다양성을 위해)
            import random
            return random.choice(available_questions)
        else:
            # 해당 난이도의 모든 문제를 다 풀었으면 다른 난이도에서 선택
            all_available = [q for q in (beginner_questions + intermediate_questions + advanced_questions) 
                           if q not in st.session_state.used_questions]
            if all_available:
                import random
                return random.choice(all_available)
            else:
                return "END"
    
    # 테스트가 완료되지 않은 경우에만 문제 표시
    if not st.session_state.test_completed:
        # 현재 문제 ID로 문제 찾기
        current_question = None
        for q in all_questions:
            if q['id'] == st.session_state.current_question_id:
                current_question = q
                break
        
        # 난이도별 색상과 아이콘
        difficulty_colors = {
            "초급": ("🔰", "#4CAF50"),
            "중급": ("⚡", "#FF9800"), 
            "고급": ("🔥", "#F44336")
        }
        
        icon, color = difficulty_colors.get(current_question["difficulty"], ("❓", "#9E9E9E"))
        
        # 진행 상황 표시
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {color}20, {color}40); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="margin: 0; color: {color};">
                {icon} {current_question["difficulty"]} 문제 | 진행도: {st.session_state.question_count}/5
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 문제 표시
        st.write(f"**문제 {st.session_state.question_count + 1}:** {current_question['q']}")
        
        # 문제 폼
        with st.form(f"question_form_{st.session_state.question_count}"):
            answer = st.radio("선택하세요", current_question["options"], index=None)
            submitted = st.form_submit_button("✅ 답안 제출")
        
        # 답안 처리
        if submitted and answer is not None:
            is_correct = (answer == current_question["answer"])
            st.session_state.question_count += 1
            
            # 현재 문제를 경로와 사용된 문제 목록에 추가
            st.session_state.question_path.append(current_question)
            st.session_state.used_questions.append(st.session_state.current_question_id)
            
            # 답안 결과 즉시 표시
            if is_correct:
                st.success("🎉 정답입니다!")
                # 정답도 초록색으로 표시
                st.markdown(f"<div style='background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; border: 1px solid #c3e6cb;'>"
                           f"<strong>✅ 정답:</strong> {current_question['answer']}</div>", unsafe_allow_html=True)
                st.session_state.correct_answers += 1
                
                # 2초 후 다음 문제로 자동 이동
                import time
                time.sleep(2)
                
                # 다음 문제 결정
                next_question_id = get_next_question(st.session_state.current_question_id, is_correct)
                if next_question_id == "END" or st.session_state.question_count >= 5:
                    st.session_state.test_completed = True
                else:
                    st.session_state.current_question_id = next_question_id
                st.rerun()
            else:
                st.error("❌ 오답입니다!")
                # 오답이면 정답을 초록색으로 표시
                st.markdown(f"<div style='background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; border: 1px solid #c3e6cb;'>"
                           f"<strong>💡 정답:</strong> {current_question['answer']}</div>", unsafe_allow_html=True)
                
                # 2초 후 다음 문제로 자동 이동
                import time
                time.sleep(2)
                
                # 다음 문제 결정 (적응형 트리 구조)
                next_question_id = get_next_question(st.session_state.current_question_id, is_correct)
                
                if next_question_id == "END" or st.session_state.question_count >= 5:
                    # 테스트 완료
                    st.session_state.test_completed = True
                    st.rerun()
                else:
                    # 다음 문제로 이동
                    st.session_state.current_question_id = next_question_id
                    st.rerun()
    
    # 최종 결과 표시
    if st.session_state.test_completed:
        st.markdown("---")
        st.markdown("### 🎯 최종 결과")
        
        # 정답 수에 따른 레벨 결정 (5문제 기준)
        # 풀린 문제들을 분석하여 최고 난이도 확인
        max_difficulty = "초급"
        for question in st.session_state.question_path:
            if question['difficulty'] == "고급":
                max_difficulty = "고급"
            elif question['difficulty'] == "중급" and max_difficulty != "고급":
                max_difficulty = "중급"
        
        # 최고 난이도와 정답률을 종합하여 레벨 결정
        if max_difficulty == "고급" and st.session_state.correct_answers >= 4:
            final_level = "고급"
            level_icon = "🚀"
            level_color = "#F44336"
        elif max_difficulty == "중급" and st.session_state.correct_answers >= 3:
            final_level = "중급"
            level_icon = "⚡"
            level_color = "#FF9800"
        else:
            final_level = "초급"
            level_icon = "🔰"
            level_color = "#4CAF50"
        
        st.success(f"{level_icon} **{final_level}자**로 분류되었습니다!")
        
        # 결과 상세 정보
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("맞춘 문제", f"{st.session_state.correct_answers}/5")
        with col2:
            st.metric("최종 레벨", final_level)
        with col3:
            st.metric("정답률", f"{st.session_state.correct_answers/5*100:.0f}%")
        
        # 레벨별 맞춤 가이드
        if final_level == "초급":
            st.info("💡 **초급자 맞춤 가이드**: 기본 개념부터 차근차근 학습하세요. 안전한 투자부터 시작하는 것을 권장합니다.")
        elif final_level == "중급":
            st.info("⚡ **중급자 맞춤 가이드**: 기술적 분석과 기본적 분석을 함께 활용하여 더 정확한 투자 판단을 내려보세요.")
        else:
            st.info("🚀 **고급자 맞춤 가이드**: 복합적 분석과 리스크 관리를 통해 고도화된 투자 전략을 수립하세요.")
        
        # 재시작 버튼
        if st.button("🔄 테스트 다시 시작"):
            for key in ['current_question_id', 'question_count', 'correct_answers', 'test_completed', 'question_path', 'used_questions']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


# 실행
def _render():
    render_top()
    render_status_overview()
    render_user_level()

    st.write("---")
    st.caption("© 2025 · SK Networks Family AI Camp 18기 - 3rd - 5Team")

if __name__== "__main__":
    _render()
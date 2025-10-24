
import time
import streamlit as st

# ------------------------------
# Question Bank (적응형 트리 구조)
# ------------------------------

ALL_QUESTIONS = [
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

TOTAL_QUESTIONS = 5     # 총 5문제로 테스트 진행


#########################
# 스타일 적용
#########################
def _inject_styles():
    st.markdown(
        """
        <style>
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }
        div[data-testid="stForm"] {
            max-width: 600px;
            width: 100%;
            min-height: 300px;
            margin: 0 auto 24px auto;
            padding: 24px 26px 20px 26px;
            background: #ffffff;
            border-radius: 18px;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            animation: fadeInUp 0.45s ease;
        }
        div[data-testid="stForm"] > div:first-child p {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 5px;
        }
        div[data-testid="stForm"] p.question-body {
            font-size: 0.98rem;
            margin-bottom: 16px;
            color: #374151;
        }
        div[data-testid="stForm"] div[data-testid="stRadio"] {
            padding: 0;
            margin-bottom: 18px;
        }
        div[data-testid="stForm"] div[data-testid="stRadio"] label {
            font-size: 0.95rem;
        }
        div[data-testid="stForm"] .stButton > button {
            width: 100%;
            height: 46px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: #ffffff;
        }
        div[data-testid="stForm"] .stButton > button:hover {
            filter: brightness(1.05);
        }
        .quiz-progress {
            max-width: 600px;
            margin: 0 auto 20px auto;
            padding: 14px 20px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            gap: 14px;
            background: rgba(15, 23, 42, 0.06);
            animation: fadeInUp 0.35s ease;
        }
        .quiz-progress__icon {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            font-size: 1.35rem;
        }
        .quiz-progress__title {
            font-size: 1.05rem;
            font-weight: 600;
            color: #1f2937;
            margin: 0;
        }
        .quiz-progress__subtitle {
            font-size: 0.92rem;
            color: #4b5563;
            margin: 2px 0 0 0;
        }
        .feedback-inline {
            font-size: 0.92rem;
            font-weight: 600;
            color: #2563eb;
            margin: 0;
            padding-left: 8px;
        }
        .feedback-inline.error {
            color: #dc2626;
        }
        [data-testid="final-level-card"] {
            max-width: 600px;
            margin: 0 auto 28px auto;
            padding: 32px 30px;
            border-radius: 22px;
            background: linear-gradient(145deg, rgba(37, 99, 235, 0.12), rgba(124, 58, 237, 0.12));
            box-shadow: 0 22px 45px rgba(15, 23, 42, 0.18);
            text-align: center;
            animation: fadeInUp 0.45s ease;
        }
        [data-testid="final-level-card"] .level-title {
            font-size: 2.3rem;
            font-weight: 700;
            margin-bottom: 12px;
            letter-spacing: -0.02em;
        }
        [data-testid="final-level-card"] .rate-text {
            font-weight: bold;
            color: #000;
            margin-bottom: 10px;
        }
        [data-testid="final-level-card"] .guide-text {
            font-size: 0.98rem;
            line-height: 1.55;
            color: #4b5563;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


#########################
# 문제풀이 실행 로직
########################

def _provide_feedback_and_advance(current_question, is_correct):
    """
    피드백을 표시한 뒤 잠시 대기하고 다음 문제로 이동합니다.
    """
    if is_correct:
        st.session_state.correct_answers += 1
    time.sleep(0.5)
    next_question_id = _get_next_question_alt(st.session_state.current_question_id, is_correct)
    if next_question_id == "END" or st.session_state.question_count >= TOTAL_QUESTIONS:
        st.session_state.test_completed = True
    else:
        st.session_state.current_question_id = next_question_id
    st.rerun()


def _compute_final_level(question_results, correct_answers):
    difficulty_rank = {"초급": 0, "중급": 1, "고급": 2}
    highest_correct_rank = max(
        (difficulty_rank[result["difficulty"]] for result in question_results if result["is_correct"]),
        default=-1
    )
    if highest_correct_rank == 2 and correct_answers >= 3:
        return "시장 항해자(초급)", "🚀", "#F44336"
    if highest_correct_rank >= 1 and correct_answers >= 2:
        return "파도 타는 서퍼(중급)", "⚡", "#FF9800"
    return "파도 관찰자(고급)", "🔰", "#4CAF50"


def _reset_user_level_state():
    for key in ['current_question_id', 'question_count', 'correct_answers', 'test_completed', 'question_path', 'question_results', 'user_level']:
        if key in st.session_state:
            del st.session_state[key]

def _get_next_question_alt(current_id, is_correct):
    """
    문제 트리맵 기반 적응형 이동.
    st.session_state.question_count는 방금 푼 문제의 누적 개수를 나타냅니다.
    """
    question_count = st.session_state.get("question_count", 0)
    
    # 5문제 완료 시 종료
    if question_count >= TOTAL_QUESTIONS:
        return "END"
    
    transitions_by_step = {
        1: {  # Step 1 -> Step 2
            "Q1": {"correct": "Q6", "incorrect": "Q2"},
        },
        2: {  # Step 2 -> Step 3
            "Q6": {"correct": "Q10", "incorrect": "Q7"},
            "Q2": {"correct": "Q6", "incorrect": "Q3"},
        },
        3: {  # Step 3 -> Step 4
            "Q10": {"correct": "Q12", "incorrect": "Q8"},
            "Q7": {"correct": "Q10", "incorrect": "Q4"},
            "Q6": {"correct": "Q10", "incorrect": "Q7"},
            "Q3": {"correct": "Q6", "incorrect": "Q5"},
        },
        4: {  # Step 4 -> Step 5
            "Q12": {"correct": "Q14", "incorrect": "Q9"},
            "Q8": {"correct": "Q10", "incorrect": "Q4"},
            "Q4": {"correct": "Q7", "incorrect": "Q5"},
            "Q6": {"correct": "Q10", "incorrect": "Q7"},
            "Q7": {"correct": "Q10", "incorrect": "Q15"},
        },
    }
    
    step_transitions = transitions_by_step.get(question_count)
    if not step_transitions:
        return "END"
    
    transition = step_transitions.get(current_id)
    if not transition:
        return "END"
    
    return transition["correct" if is_correct else "incorrect"]

def render_user_level():
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
    if 'question_results' not in st.session_state:
        st.session_state.question_results = []  # 문제별 정오답 기록
    
    _inject_styles()
    
    st.markdown("---")
    st.markdown(
    """
    <h3 style="text-align: center; margin-top: 1rem; margin-bottom: 0rem;">
        📋 나의 투자 지식 레벨은?
    </h3>
    """,
    unsafe_allow_html=True
)
    st.markdown("---")

    
    # 테스트가 완료되지 않은 경우에만 문제 표시
    if not st.session_state.test_completed:
        # 현재 문제 ID로 문제 찾기
        current_question = next(
            (q for q in ALL_QUESTIONS if q['id'] == st.session_state.current_question_id),
            None
        )
        
        if current_question is None:
            st.warning("예상치 못한 문제가 발생했습니다. 테스트를 다시 시작합니다.")
            _reset_user_level_state()
            st.rerun()
            return
        
        # 난이도별 색상과 아이콘
        difficulty_colors = {
            "🌊파도 관찰자(초급)": ("🌊", "#4CAF50"),
            "🚤파도 타는 서퍼(중급)": ("🚤", "#FF9800"), 
            "🛳️시장 항해자(고급)": ("🛳️", "#F44336")
        }
        
        icon, color = difficulty_colors.get(current_question["difficulty"], ("❓", "#9E9E9E"))
        
        remaining_questions = TOTAL_QUESTIONS - st.session_state.question_count
        st.markdown(
            f"""
            <div class="quiz-progress" data-transition-key="{st.session_state.question_count}">
                <div class="quiz-progress__icon" style="background-color: {color}26; color: {color};">{icon}</div>
                <div>
                    <p class="quiz-progress__title">{current_question['difficulty']} 문제 : {remaining_questions}문제 남았습니다</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # 문제 폼
        with st.form(f"question_form_{st.session_state.question_count}"):
            st.markdown(
                f"<p>문제 {st.session_state.question_count + 1}</p>"
                f"<p class='question-body'>{current_question['q']}</p>",
                unsafe_allow_html=True,
            )
            answer = st.radio(
                "선택하세요",
                current_question["options"],
                index=None,
                label_visibility="collapsed",
            )
            btn_col, feedback_col = st.columns([0.58, 0.42])
            with btn_col:
                submitted = st.form_submit_button("✅ 답안 제출")
            with feedback_col:
                feedback_placeholder = st.empty()
        
        # 답안 처리
        if submitted and answer is not None:
            is_correct = (answer == current_question["answer"])
            st.session_state.question_count += 1
            st.session_state.question_path.append(current_question)
            st.session_state.question_results.append(
                {
                    "id": current_question["id"],
                    "difficulty": current_question["difficulty"],
                    "is_correct": is_correct,
                }
            )
            
            feedback_text = "✅ 정답입니다!" if is_correct else f"❌ 오답입니다 · 정답: {current_question['answer']}"
            feedback_class = "feedback-inline" + (" error" if not is_correct else "")
            feedback_placeholder.markdown(
                f"<p class='{feedback_class}'>{feedback_text}</p>",
                unsafe_allow_html=True,
            )
            
            _provide_feedback_and_advance(current_question, is_correct)
    
    
    # 최종 결과 표시
    if st.session_state.test_completed:
        final_level, level_icon, level_color = _compute_final_level(
            st.session_state.question_results,
            st.session_state.correct_answers,
        )
        st.session_state.user_level = {
            "level": final_level,
            "icon": level_icon,
            "color": level_color,
            "correct_answers": st.session_state.correct_answers,
            "total_questions": TOTAL_QUESTIONS,
        }
        
        level_guides = {
            "파도 관찰자(초급)": "💬 기본 개념을 반복 학습하며 안전한 투자 방법부터 차근히 익혀 보세요.",
            "파도 타는 서퍼(중급)": "💬 기본적 분석과 기술적 분석을 병행하여 자신만의 투자 전략을 고도화해 보세요.",
            "시장 항해자(고급)": "💬 리스크 관리와 포트폴리오 다각화를 통해 고도화된 전략을 실행해 보세요."
        }
        guide_text = level_guides.get(final_level, "")
        st.markdown(
            f"""
            <div data-testid="final-level-card">
                <div class="level-title" style="color: {level_color};">{level_icon} {final_level}</div>
                <div class="rate-text">정답률 : {st.session_state.correct_answers/TOTAL_QUESTIONS*100:.0f}% ({st.session_state.correct_answers}/{TOTAL_QUESTIONS})</div>
                <div class="guide-text">{guide_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # 재시작 버튼
        if st.button("🔄 테스트 다시 시작"):
            _reset_user_level_state()
            st.rerun()
    

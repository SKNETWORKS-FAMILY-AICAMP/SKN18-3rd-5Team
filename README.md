# SKN18-3rd-5Team

## 프로젝트 구조

```text
├── app.py                 # Streamlit 진입점
├── pages/                 # Streamlit 멀티 페이지 모듈
│   ├── app_bootstrap.py   # 공통 페이지 설정 및 사이드바 메뉴 정의
│   ├── page1.py           # 채팅 Q&A 페이지
│   ├── data_tool.py       # 데이터 도구 페이지
│   └── views/             # 채팅 등 공통 뷰 컴포넌트 (page에서 이용)
│       ├── chat.py        # 채팅 UI 레이아웃
│       ├── {view}.py      # 
│       ├── {veiw}.py      # 
│       └── ...
├── service/               # LLM등 로직/기능
│   ├── chat_service.py    # SQLite 기반 채팅 세션 관리
│   └── ...
├── data/                  # 분석·시각화에 사용하는 원천 데이터
│   └── app_database.db    # SQLite 데이터베이스
├── assets/                # 이미지, 아이콘 등 정적 리소스
├── config/                # 환경 설정 파일 (예: YAML, JSON)
├── graph/                 # lang-graph
│   ├── state.py                  # 상태 스키마(QAState)
│   ├── app_graph.py              # 그래프 구성/compile/팩토리 함수
│   ├── nodes/                    # LangGraph 노드들
│   └── utils/
├── requirements.txt       # Python 의존성 목록
└── README.md
```

## 실행 방법
- HuggingFace에서 모델 미리 다운로드 실행
```bash
uv pip install -r requirements.txt
python service/llm/setup_download.py
streamlit run app.py
```
##
 데이터베이스 변경사항

### SQLite 마이그레이션

기존 JSON 파일 기반의 채팅 세션 관리를 SQLite 데이터베이스로 변경했습니다.

**변경된 부분:**
- 채팅 세션 및 메시지 데이터가 SQLite에 저장됨
- 기존 JSON 파일은 자동으로 `data/chat_sessions_backup/`으로 백업됨
- UI는 동일하게 유지되며 백엔드 로직만 변경됨

**마이그레이션 실행:**
```bash
python migrate_to_sqlite.py
```

**데이터베이스 구조:**
- `chat_sessions`: 채팅 세션 정보 (ID, 제목, 생성일시, 수정일시)
- `chat_messages`: 채팅 메시지 (세션ID, 역할, 내용, 타임스탬프)
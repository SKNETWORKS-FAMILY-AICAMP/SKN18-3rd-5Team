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
├── data/                  # 분석·시각화에 사용하는 원천 데이터
├── assets/                # 이미지, 아이콘 등 정적 리소스
├── config/                # 환경 설정 파일 (예: YAML, JSON)
├── requirements.txt       # Python 의존성 목록
└── README.md
```

## 실행 방법

```bash
uv pip install -r requirements.txt
streamlit run app.py
```
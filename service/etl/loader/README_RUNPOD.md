# RunPod 환경에서 Parquet 변환 실행 가이드

> **이 가이드는 처음 RunPod를 사용하는 사용자를 위한 단계별 안내서입니다.**

## 📋 목차

1. [사전 준비](#사전-준비)
2. [RunPod 연결 및 환경 설정](#runpod-연결-및-환경-설정)
3. [파일 업로드](#파일-업로드)
4. [Parquet 변환 실행](#parquet-변환-실행)
5. [Git LFS로 결과물 저장](#git-lfs로-결과물-저장)

---

## 사전 준비

### 1. RunPod 인스턴스 생성

RunPod 대시보드에서 새 Pod를 생성할 때 다음 사양을 선택하세요:

- **메모리**: **32GB 권장** (16GB는 메모리 부족 위험)
- **디스크**: **10GB 이상** 권장
- **컴퓨트 타입**: **CPU** (Parquet 변환은 GPU 불필요)

> ⚠️ **중요**: 디스크 확장 시 Pod가 재시작되어 데이터가 손실될 수 있으므로 처음부터 충분한 용량을 설정하세요.

### 2. SSH 키 설정

**로컬 터미널에서 실행:**

```bash
# SSH 키 생성 (이미 생성된 경우 생략)
ssh-keygen -t ed25519 -C "runpod"

# 생성된 공개키 확인
cat ~/.ssh/id_ed25519.pub
```

**RunPod 대시보드에서:**

1. **Settings** → **SSH Public Keys** 메뉴로 이동
2. 위에서 확인한 공개키를 복사하여 등록

---

## RunPod 연결 및 환경 설정

### 1. SSH 접속 정보 확인

**RunPod 대시보드에서:**

1. 생성한 Pod 선택
2. **Connect** 탭 클릭
3. **SSH over exposed TCP** 섹션에서 접속 정보 확인:
   - IP 주소: `205.196.19.26` (예시)
   - 포트번호: `11236` (예시)

### 2. SSH 접속

**로컬 터미널에서 실행:**

```bash
# SSH 접속 (포트번호와 IP는 RunPod 대시보드에서 확인한 값으로 변경)
ssh -p 11236 -i ~/.ssh/id_ed25519 root@205.196.19.26
```

> ✅ 성공 시 RunPod 터미널 프롬프트가 표시됩니다: `root@xxxxxxxx:/workspace#`

### 3. 시스템 패키지 업데이트

**RunPod 터미널에서 실행:**

```bash
sudo apt update
sudo apt install python3-venv
```

### 4. Python 가상환경 생성 및 패키지 설치

**RunPod 터미널에서 실행:**

```bash
cd /workspace
python -m venv .venv
source .venv/bin/activate

# Parquet 변환에 필요한 패키지만 설치
pip install pandas pyarrow
```

> 💡 **팁**: requirements.txt 대신 필요한 패키지만 설치하여 메모리를 절약합니다.

### 5. Git LFS 설치 및 설정

**RunPod 터미널에서 실행:**

```bash
# Git LFS 설치
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# 설치 확인
git lfs version

# Git LFS 초기화
git lfs install
```

### 6. LFS 레포지토리 클론 및 설정

**RunPod 터미널에서 실행:**

```bash
cd /workspace
git clone https://github.com/JINA1003/LFS.git
cd LFS

# Git 사용자 정보 설정 (본인의 정보로 변경)
git config --global user.email "your.email@example.com"
git config --global user.name "Your Name"

# Parquet 파일을 LFS로 추적
git lfs track "*.parquet"
git add .gitattributes
git commit -m "Track parquet files with LFS"
```

---

## 파일 업로드

### SCP로 필요한 파일들 업로드

**로컬 터미널에서 실행 (새 터미널 창 열기):**

> ⚠️ **주의**: RunPod SSH 터미널이 아닌 **로컬 터미널**에서 실행하세요!

```bash
# 1. JSONL 파일들 업로드 (약 1.5GB, 5-10분 소요)
scp -P 11236 -i ~/.ssh/id_ed25519 -r /Users/jina/Documents/GitHub/SKN18-3rd-5Team/data/transform/final root@205.196.19.26:/workspace/

# 2. 변환 스크립트 업로드
scp -P 11236 -i ~/.ssh/id_ed25519 /Users/jina/Documents/GitHub/SKN18-3rd-5Team/service/etl/loader/jsonl_to_parquet.py root@205.196.19.26:/workspace/
```

> 📝 **참고**:
>
> - 포트번호(`11236`)와 IP 주소(`205.196.19.26`)는 본인의 RunPod 정보로 변경하세요.
> - 파일 경로(`/Users/jina/...`)는 본인의 로컬 경로로 변경하세요.

### 업로드 확인

**RunPod 터미널에서 실행:**

```bash
# 파일 개수 확인
ls -la /workspace/final/ | wc -l

# 전체 용량 확인
du -sh /workspace/final/
```

---

## Parquet 변환 실행

### 1. 변환 스크립트 실행

**RunPod 터미널에서 실행:**

```bash
cd /workspace
source .venv/bin/activate

# 32GB 메모리 환경에서 권장 배치 사이즈: 10000
python jsonl_to_parquet.py --streaming --batch-size 10000
```

> ⏱️ **예상 소요 시간**: 30분 ~ 1시간

### 2. 진행 상황 모니터링 (선택사항)

**별도의 SSH 터미널에서 실행:**

```bash
# 새 터미널에서 RunPod 재접속
ssh -p 11236 -i ~/.ssh/id_ed25519 root@205.196.19.26

# 메모리 사용량 모니터링
htop

# 또는 간단한 모니터링
watch -n 5 'free -h'
```

> 📊 **정상 상태**:
>
> - CPU: 100% 사용 (정상)
> - 메모리: 5-20% 사용 (정상)
> - 디스크: 30% 이하 (정상)

### 3. 변환 결과 확인

**RunPod 터미널에서 실행:**

```bash
# Parquet 파일 생성 확인
ls -la /workspace/LFS/chunks.parquet

# 파일 크기 확인
du -h /workspace/LFS/chunks.parquet
```

---

## Git LFS로 결과물 저장

### 1. Parquet 파일을 LFS로 추가

**RunPod 터미널에서 실행:**

```bash
cd /workspace/LFS

# Parquet 파일 추가
git add chunks.parquet
git commit -m "Add parquet file"

# GitHub에 push
git push
```

> 🔐 **인증 필요**: GitHub 계정 정보 입력 또는 Personal Access Token 사용

### 2. 로컬에서 결과물 다운로드 (선택사항)

**로컬 터미널에서 실행:**

```bash
# SCP로 다운로드
scp -P 11236 -i ~/.ssh/id_ed25519 root@205.196.19.26:/workspace/LFS/chunks.parquet /Users/jina/Documents/GitHub/SKN18-3rd-5Team/

# 또는 Git pull로 다운로드
cd /Users/jina/Documents/GitHub/SKN18-3rd-5Team
git pull
```

## 경로 구조

```
/workspace/
├── final/                    # 입력: JSONL 파일들 (5000개)
│   ├── 20241028000193_chunks.jsonl
│   ├── 20241028000251_chunks.jsonl
│   └── ...
├── LFS/                      # 출력: LFS 레포지토리
│   ├── .git/
│   ├── .gitattributes
│   └── chunks.parquet        # 최종 결과물
├── jsonl_to_parquet.py       # 변환 스크립트
├── requirements.txt          # 의존성
└── .venv/                    # Python 가상환경
```

## 성능 최적화 팁

### 1. 메모리 모니터링

```bash
# 별도 터미널에서 메모리 사용량 확인
htop
# 또는
watch -n 5 'free -h'
```

### 2. 배치 사이즈 조정

- **32GB 메모리**: 5000-10000
- **16GB 메모리**: 2000-5000
- **메모리 부족 시**: 1000-2000

### 3. 압축 방식 선택

```bash
# 기본 (빠른 압축/해제)
python jsonl_to_parquet.py --compression snappy

# 높은 압축률 (느림)
python jsonl_to_parquet.py --compression gzip
```

## 문제 해결

### 1. 메모리 부족 오류

- 배치 사이즈를 줄이기: `--batch-size 1000`
- 메모리 모니터링으로 확인

### 2. 디스크 공간 부족

- RunPod 대시보드에서 Container Disk 크기 증가
- 불필요한 파일 정리: `sudo apt clean`

### 3. SSH 연결 실패

- RunPod 대시보드에서 Pod 상태 확인
- 포트번호와 IP 주소 재확인
- Pod 재시작

## 예상 처리 시간

- **5000개 JSONL 파일**: 30분-1시간
- **배치 사이즈 5000**: 빠른 처리
- **배치 사이즈 2000**: 안전한 처리
- **배치 사이즈 1000**: 메모리 부족 시 사용

## 비용 최적화

- **32GB CPU Pod**: 시간당 $0.24
- **처리 시간**: 1시간 이내
- **총 비용**: $0.24 이하
- **실패 시 재시작 비용 고려**

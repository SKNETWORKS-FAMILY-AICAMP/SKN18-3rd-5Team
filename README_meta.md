# 발표 자료 메모 README

> 추후 제거

---

## 프로젝트명

**공시 문서 및 애널리스트 분석 리포트 기반의 투자 Q&A 시스템**

- 사용자의 투자 지식 수준 인식 로직으로 적응형 시스템 (개인화)

---

## [🧩 ETL의 정의]

**ETL**은 **Extract** → **Transform** → **Load의** 줄임말이에요.
즉, 데이터를 가져오고(Extract), 가공하고(Transform), 저장하는(Load) 전체 과정을 말해요.

✅ 한 문장으로 요약하면
**데이터를 깨끗하게 정리해서 쓸 수 있는 형태로 옮기는 파이프라인**이에요.

⚙️ 2. ETL의 프로세스 요약
| 단계 | 의미 | 예시 |
| ----------------- | ------------------------------------ | ---------------------------- |
| **E (Extract)** | 여러 곳(웹, DB, API 등)에서 원본 데이터를 끌어오는 단계 | API로 XML 데이터 다운로드 |
| **T (Transform)** | 불필요한 데이터를 제거하고, 사람이 읽을 수 있는 형태로 가공 | XML → 자연어 문서화 (MD + YAML 변환) |
| **L (Load)** | 가공된 데이터를 데이터베이스나 벡터DB에 저장 | Embedding 후 VectorDB에 저장 |

```scss
API (XML)
 ↓
> [Extract]
 ↓
문서화 (MD + YAML)
 ↓
> [Transform]
 ↓
청크 임베딩 → 벡터DB
 ↓
> [Load]
 ↓
질의 → 검색 → 답변 생성
```

### 🧠 왜 ETL이 중요한가?

| 이유                     | 설명                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------- |
| **1️⃣ 정확성 확보**       | 원본 데이터를 그대로 쓰면 오류나 중복이 많아요. ETL은 데이터를 정제(clean)해서 일관성을 보장해요. |
| **2️⃣ 효율성 향상**       | 사람이 일일이 변환하는 대신 자동화된 파이프라인으로 처리 속도가 수십 배 빨라집니다.               |
| **3️⃣ 분석/AI 활용 기반** | LLM, 머신러닝, 대시보드, 리포트 등 모든 데이터 분석은 ETL을 거친 후 가능합니다.                   |
| **4️⃣ 재사용성 확보**     | 동일한 ETL 파이프라인을 유지하면, 나중에 새로운 데이터를 추가로 연결하기 쉽습니다.                |

### 구현

> RAG 위주, FineTuning 간단 구현

---

## [문서화]

> XML → 의미있는 자연어 레이어(문서화) → 청크/임베딩

### 1. 왜 문서화가 유리한가

- 의미 단위 보존: XML은 필드/태그 나열(맥락 부족). 자연어 문장으로 “증감/비교/기간/사유”를 함께 서술하면 검색 적중률↑, 답변 품질↑.
- 중복·노이즈 제거: 태그명/속성/불필요 공백 등 불용 토큰을 줄여 임베딩 효율↑.
- 출처 인용이 쉬움: “DART, 접수번호, 페이지/섹션”을 **문장과** 함께 같이 저장하니 근거 제시 정확.
- 질문 적합성: 사용자가 자연어로 묻기 때문에, 컨텍스트도 자연어가 맞음.

**따라서 문서화 없이 원본 XML 그대로 임베딩하면**:
태그가 섞인 텍스트가 늘어져 검색 품질 저하, 숫자만 많은 청크는 문맥 매칭 약화, 응답에 근거 문구가 어색해질 확률이 높습니다.

### 2. ETL - RAG

- E : Dart -> API -> XML -> 약 5000개 -> Google Drive / -> 1개의 .jsonl (5000라인) 로컬에 저장
- T : JSONL & XML -> 정규화 -> 자연어 문서화 (yaml / md) -> 로컬 저장 (docs/{corp}/{year}/{reprt_code}/{rcept_no}.md) -> 청크 (1차/2차 ) + 메타 동봉 -> .parquet 파일 생성
- L : Parquet & MD -> text 임베딩 -> pgvector -> (키워드 인덱스 색인) -> 문서 CDN 업로드 보류 -> 서빙 (질의 및 응답)

### 3. ETL - FineTuning

-

---

## [Fine-Tuning]

### 1. 목적

- 증권사 애널리스트 리포트 요약/코멘트를 이용해서 리스크/밸류에이션/가이던스 해석 포함한 "공시 해석 지능" 학습

### 2. Llama Factory

- 참고 : [SK Tech 블로그](https://devocean.sk.com/blog/techBoardDetail.do?ID=166098)
- 정의 :  
  LLaMA Factory는 Meta의 LLaMA 및 다양한 오픈소스 LLM(대형언어모델) 계열 모델을 쉽게 파인튜닝, 프롬프트 미세조정(Instruction Tuning), 채팅/웹UI 배포까지 지원하는 올인원 오픈소스 라이브러리입니다.  
  기존 HuggingFace Transformers 기반이며, 다양한 미세조정 방법(LoRA, QLoRA, P-Tuning 등), 데이터셋 포맷 지원, Web UI(그라디오 기반)로 GUI 환경에서 손쉽게 커스텀 학습/테스트/배포를 할 수 있습니다.  
  즉, **LLM 모델 파인튜닝과 운영**을 코드 몇 줄, 혹은 클릭만으로 진행할 수 있게 돕는 학습·서빙용 **프레임워크**입니다.

#### [설치]

- 다시 정리 예정

### 파인튜닝 옵션

| 항목                            | 값                        | 설명                                                   |
| ------------------------------- | ------------------------- | ------------------------------------------------------ |
| **Stage**                       | `sft`                     | Supervised Fine-Tuning 단계                            |
| **Base Model**                  | `meta-llama/Llama-3.2-3B` | 파인튜닝할 기본 모델                                   |
| **Template**                    | `alpaca`                  | 프롬프트 형식 템플릿 (instruction, input, output 구조) |
| **Finetuning Type**             | `lora`                    | PEFT(파라미터 효율적 튜닝) 방식                        |
| **Rank (r)**                    | `8`                       | 저랭크 차원 수                                         |
| **Alpha**                       | `16`                      | 스케일링 계수                                          |
| **Dropout**                     | `0.05`                    | LoRA 드롭아웃 비율                                     |
| **Quantization Bit**            | `4`                       | 4bit 양자화로 GPU 메모리 절약                          |
| **방식**                        | QLoRA                     | 양자화된 모델에 LoRA 튜닝 적용                         |
| **Batch Size**                  | `1`                       | GPU 메모리 제약 고려                                   |
| **Gradient Accumulation Steps** | `8`                       | 실질적 배치 크기 확장                                  |
| **Learning Rate**               | `1e-4`                    | 학습률                                                 |
| **Epochs**                      | `3`                       | 학습 반복 횟수                                         |
| **Scheduler**                   | `cosine`                  | 학습률 스케줄러                                        |
| **Warmup Ratio**                | `0.1`                     | 전체 스텝의 10%는 워밍업 구간                          |
| **Eval Strategy**               | `steps`                   | 일정 step마다 평가                                     |
| **Eval Steps / Save Steps**     | `500`                     | 500 step마다 평가 및 저장                              |
| **Save Total Limit**            | `3`                       | 체크포인트 최대 3개 유지                               |
| **Load Best Model**             | `true`                    | eval_loss 기준으로 최고 성능 모델 불러오기             |
| **Metric for Best Model**       | `eval_loss`               | 낮을수록 좋음 (`greater_is_better=false`)              |
| **Precision**                   | `fp16`                    | Half-Precision으로 메모리 절약                         |
| **Gradient Checkpointing**      | `true`                    | 그래디언트 캐시로 VRAM 절감                            |
| **Flash Attention**             | `fa2`                     | 고속 Attention 연산                                    |
| **Packing**                     | `true`                    | 시퀀스 효율 향상                                       |
| **Num Workers**                 | `4`                       | DataLoader 병렬 처리                                   |
| **Pin Memory**                  | `true`                    | 데이터 로딩 속도 향상                                  |
| **Optimizer**                   | `adamw_torch`             | AdamW 최적화기 사용                                    |
| **Seed**                        | `42`                      | 재현성 확보                                            |
| **Report**                      | `tensorboard`             | 학습 로그 시각화                                       |

### 파인튜닝 결과

```text
***** Running training *****
>>   Num examples = 955
>>   Num Epochs = 3
>>   Instantaneous batch size per device = 1
>>   Total train batch size (w. parallel, distributed & accumulation) = 8
>>   Gradient Accumulation steps = 8
>>   Total optimization steps = 360
>>   Number of trainable parameters = 12,156,928

***** train metrics *****
  epoch                    =         3.0
  total_flos               = 185635548GF
  train_loss               =      0.8458
  train_runtime            =  2:30:35.06
  train_samples_per_second =       0.317
  train_steps_per_second   =        0.04

***** eval metrics *****
  epoch                   =        3.0
  eval_loss               =     0.5694
  eval_runtime            = 0:01:40.81
  eval_samples_per_second =      1.061
  eval_steps_per_second   =      1.061
```

        output_dir: output/llama-3-8b-Instruct-bnb-4bit/qlora
        #logging_steps: 10
        #save_steps: 500
        plot_loss: true
        overwrite_output_dir: true

        per_device_train_batch_size: 1
        gradient_accumulation_steps: 4
        learning_rate: 1.0e-4
        num_train_epochs: 3.0
        lr_scheduler_type: cosine
        warmup_ratio: 0.1
        bf16: true
        #report_to: none

        seed: 42
        val_size: 0.1
        per_device_eval_batch_size: 1
        eval_strategy: steps
        eval_steps: 100


        do_eval: true
        #eval_strategy: steps
        #eval_steps: 100
        save_strategy: steps
        save_steps: 100
        logging_steps: 20

        load_best_model_at_end: true
        metric_for_best_model: "eval_loss"
        greater_is_better: false

        report_to: ["tensorboard"]
        resize_vocab: false
        upcast_layernorm: true
        ```

7. 학습 시작

   - CLI

     ```bash
     # 내 경로 확인 (/workspace/LLaMA-Factory/)
     pwd

     # (선택) 캐시 경로 고정해두면 재사용에 좋아요
     export HF_HOME=/workspace/.cache/huggingface

     # 학습 시작
     llamafactory-cli train config/llama3-8b-instruct-bnb-4bit-unsloth.yaml
     ```

   - Config 추천 설정

     ```yaml
     #.yaml 파일
     # 평가/저장/로깅 전략
     do_eval: true
     eval_strategy: steps # 또는 "epoch"
     eval_steps: 100 # 데이터/스텝 규모에 맞게
     save_strategy: steps
     save_steps: 100
     logging_steps: 20

     # 베스트 모델 저장
     load_best_model_at_end: true
     metric_for_best_model: "eval_loss"
     greater_is_better: false

     # 로깅 백엔드(선택)
     report_to: ["tensorboard"] # 또는 ["wandb"] 사용 시 WANDB_API_KEY 필요

     # 토크나이저 경고 대응
     resize_vocab: true

     #(권장) 4bit 학습 안정화
     upcast_layernorm: true
     ```

   - Output 확인
     ```python
     # 커맨드
     ls -al {config에 지정한 ouput_dir}
     # 예시
     ls -al /output/llama-3-8b-Instruct-bnb-4bit/qlora
     ```

8. 추론

   ```bash
   # 인자 설정
   llamafactory-cli chat \
   --model_name_or_path="unsloth/llama-3-8b-Instruct-bnb-4bit" \
   --adapter_name_or_path="output/llama-3-8b-Instruct-bnb-4bit/qlora" \
   --template="llama3" \
   --finetuning_type="lora" \
   --quantization_bit=4 \
   --temperature=0

   # 테스트 후 히스토리 제거
   clear

   # 챗 종료
   exit
   ```

9. 모델 저장

   - 학습된 Lora adapter를 base 모델과 합쳐 저장
   - 저장 파라미터
     - model_name_or_path : base 모델을 지정, 학습 효율화를 위해 사용했던 양자화 모델이 아닌 원래 모델을 지정
     - adapter_name_or_path: Lora adapter가 저장된 위치
     - template : 모델의 형식
     - finetuning_type: 파인튜닝 방법
     - export_dir: 병합된 모델이 저장될 위치
     - export_size: 모델의 큰 경우 분할될 크기 (GB)
     - export_device: 모델 병합을 처리할 디바이스 지정 (cpu and cuda)
     - export_hub_model_id : huggingface에 업로드 할 경우 아이디
   - CLI

     ```bash
     # llama3 모델을 다운로드 해야하기 때문에 라이센스 동의
     huggingface-cli login

     # Lora Adapter를 병합
     llamafactory-cli export \
     --model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct" \
     --adapter_name_or_path="output/llama-3-8b-Instruct-bnb-4bit/qlora" \
     --template="llama3" \
     --finetuning_type="lora" \
     --export_dir="/output/Meta-Llama-3-8B-Instruct" \
     --export_size=2 \
     --export_device="cpu"

     # 분할 저장 확인
     ls -al output/Meta-Llama-3-8B-Instruct/
     ```

10. HuggingFace 모델 업로드
    - 강의 자료 참고

---

## [랭그래프]

- 전체 연동 개념도
  ```text
  [ Streamlit UI ]
      ↓ (user_level session_state)
  [ LangGraph App ]
      ↓
  [ Router Node ]
  └── 레벨별 파라미터 (top_k, context_len)
  [ Generate Node ]
  └── PROMPT_TEMPLATES[level] 기반 시스템/유저 프롬프트 구성
      ↓
  [ FT 모델 + pgvector 검색 ]
      ↓
  [ 결과 + ref 반환 ]
  ```
- 참고 : ![[graph/readme.md]]

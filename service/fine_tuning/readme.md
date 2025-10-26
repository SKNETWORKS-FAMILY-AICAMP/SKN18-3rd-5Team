## LLaMA Factory 실행

```bash
# 2-1) 필수 공백 제거 (instruction/output이 비지 않은 샘플만)
jq '[ .[]
  | select(.instruction!=null and (.instruction|tostring|gsub("\\s+";"")|length)>0)
  | select(.output     !=null and (.output     |tostring|gsub("\\s+";"")|length)>0)
]' data/train.json > data/train.nonempty.json

jq '[ .[]
  | select(.instruction!=null and (.instruction|tostring|gsub("\\s+";"")|length)>0)
  | select(.output     !=null and (.output     |tostring|gsub("\\s+";"")|length)>0)
]' data/test.json > data/test.nonempty.json

# 2-2) 제로폭/불가시 공백 제거
for f in data/train.nonempty.json data/test.nonempty.json; do
  jq '[ .[]
    | .instruction = (.instruction|tostring|gsub("\\u200b|\\u200c|\\u200d|\\ufeff|\\u00a0";""))
    | .input       = ((.input // "")|tostring|gsub("\\u200b|\\u200c|\\u200d|\\ufeff|\\u00a0";""))
    | .output      = (.output|tostring|gsub("\\u200b|\\u200c|\\u200d|\\ufeff|\\u00a0";""))
  ]' "$f" > "${f%.json}.clean.json"
done
```

```bash
jq '[ .[] | .input = ((.input // "")|tostring|.[0:8000]) ]' \
  data/train.nonempty.clean.json > data/train.slice.json

jq '[ .[] | .input = ((.input // "")|tostring|.[0:8000]) ]' \
  data/test.nonempty.clean.json > data/test.slice.json
```


```bash
# 1) 메모리 단편화 완화
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 2) 학습 재시작 (eval/save 일치)
llamafactory-cli train \
  --stage sft \
  --model_name_or_path meta-llama/Llama-3.2-3B \
  --dataset_dir data \
  --dataset ko_report_summary_train \
  --eval_dataset ko_report_summary_eval \
  --template alpaca \
  --quantization_bit 4 \
  --finetuning_type lora \
  --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 \
  --cutoff_len 4096 \
  --output_dir output/llama-3-2-3b-4bit/qlora \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --num_train_epochs 3 \
  --lr_scheduler_type cosine --warmup_ratio 0.1 \
  --bf16 true \
  --do_train true --do_eval true --val_size 0.0 \
  --eval_strategy steps --eval_steps 200 \
  --save_strategy steps --save_steps 200 --save_total_limit 3 \
  --load_best_model_at_end true --metric_for_best_model eval_loss --greater_is_better false \
  --report_to tensorboard --seed 42 \
  --gradient_checkpointing true --flash_attn fa2
```

### 파인튜닝 결과 
```text
***** eval metrics *****
  eval_loss                   =     1.4073
  eval_model_preparation_time =     0.0019
  eval_runtime                = 0:01:28.56
  eval_samples_per_second     =      6.775
  eval_steps_per_second       =      6.775
```



---


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
llamafactory-cli train --stage sft --model_name_or_path meta-llama/Llama-3.2-3B \
--dataset_dir data --dataset ko_report_summary_train --eval_dataset ko_report_summary_eval \
--template alpaca --quantization_bit 4 --finetuning_type lora \
--lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
--lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 \
--cutoff_len 4096 --output_dir output/llama-3-2-3b-4bit/qlora \
--per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 --learning_rate 1e-4 --num_train_epochs 3 \
--lr_scheduler_type cosine --warmup_ratio 0.1 \
--fp16 true --bf16 false \
--do_train true --do_eval true --val_size 0.0 \
--eval_strategy steps --eval_steps 500 \
--save_strategy steps --save_steps 500 --save_total_limit 3 \
--load_best_model_at_end true --metric_for_best_model eval_loss --greater_is_better false \
--report_to tensorboard --seed 42 \
--gradient_checkpointing true \
--flash_attn fa2 --packing true \
--dataloader_num_workers 4 --dataloader_pin_memory true \
--save_only_model true --optim adamw_torch


결과?



---

### 추론


  


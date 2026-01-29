# PADA-coder

# data collect

# evaluate model
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/attentioncode/merge_model/qwen2.5-coder-7B_lora_att64 \
    --served-model-name qwen-lora \
    --port 8000 \
    --max-model-len 18000 \
    --trust-remote-code

# data collect
cd /root/autodl-tmp/attentioncode/data_collect/programming
python main.py  --dataset_path /root/autodl-tmp/attentioncode/data_collect/input_data/APPS/dataset/probs.jsonl --testfile /root/autodl-tmp/attentioncode/data_collect/input_data/APPS/dataset/probs.jsonl --strategy lpw --model Qwen3 --max_iters 8 --port 8000

# evaluate
cd /root/autodl-tmp/attentioncode/evaluate/code_evaluation/programming
cd /root/autodl-tmp/attentioncode/evaluate/code_plan_Evaluation
cd /root/autodl-tmp/attentioncode/evaluate/data_collect/programming
cd /root/autodl-tmp/attentioncode/evaluate/base_test/programming

python main.py  --dataset_path /root/autodl-tmp/attentioncode/data/test/final_dataset_with_tests.jsonl --testfile /root/autodl-tmp/attentioncode/data/test/final_dataset_with_tests.jsonl --strategy lpw --model Qwen3 --max_iters 8 --port 8000

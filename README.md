# **<center> PADA-Coder: Improving Plan-Following Code Generation via Perturbation-Verified Attention Distillation and Dynamic Alignment</center>**

This repository contains the code, data, and models for "PADA-Coder: Improving Plan-Following Code Generation via Perturbation-Verified Attention Distillation and Dynamic Alignment."

🔗 **GitHub**: [https://anonymous.4open.science/r/PADA-Coder](https://anonymous.4open.science/r/PADA-Coder)

📜 **Paper**: [Added later]() | 📊 **Benchmark**: [APPS, HumanEval, MBPP+, LiveCodeBench]() | 🤖 **Models**: [Added later]()

**📢 Notice: Ongoing Maintenance**:

This repository is currently under active development. Core code, execution scripts, and related materials are being updated regularly.

## **Key Features**

### **1\. PADA Framework**

We propose **PADA (Perturbation-Verified Attention Distillation and Dynamic Alignment)**, a unified framework that addresses "Attention Allocation Imbalance" (Drift & Dispersion) in LLMs through:

* **Maximum-DID Matrix**: Constructs an optimal attention target matrix using **Distillation Information Density (DID)** and **Perturbation Analysis** to filter noise and retain only high-value logic tokens.  
* **Dynamic Attention Alignment**: A training strategy incorporating a **Progress-Aware Sliding Window** and a **Difficulty-Aware Gating Mechanism** to dynamically calibrate the model's focus on relevant plan steps.

### **2\. Comprehensive Benchmarks**

We evaluate PADA-Coder on a diverse set of datasets, covering simple syntax implementation to complex algorithmic reasoning and mathematical problem solving.

### **3\. Performance Improvements**

PADA-Coder achieves:

✅ **16.7% improvement** in Pass@1 on average across complex benchmarks.

✅ **Superiority over SOTA methods** (SPA, LeaF, LPW) on APPS and LiveCodeBench.

✅ **Comparable performance** to ultra-large models (e.g., Gemini-2.5-Pro) using significantly smaller parameters (3B-7B).

## **📊 Datasets**

We conduct extensive experiments on the following datasets mentioned in the paper.

| Benchmark | Description | Difficulty |
| :---- | :---- | :---- |
| **APPS** | Contains 10,000 Python problems categorized into **Introductory**, **Interview**, and **Competition** levels. | ⭐⭐⭐ |
| **HumanEval** | A set of 164 handwritten Python programming problems assessing functional correctness. | ⭐ |
| **MBPP(+)** | **Mostly Basic Programming Problems**. We use the sanitized **MBPP+** version (399 samples). | ⭐⭐ |
| **LiveCodeBench** | A collection of 164 contest problems published between Sept 2024 and Feb 2025\. | ⭐⭐⭐ |
| **GSM8K** | Grade school math word problems requiring multi-step Chain-of-Thought (CoT) reasoning. | Math |
| **MATH-500** | A challenging subset of the MATH dataset covering algebra, calculus, and geometry. | Math |

---


Our entire work flow can be summarized as follows:

<div align="center">
<img src="pics\method.jpg" width="800px">
</div>

**Overview of PADA:** Our framework comprises three steps:(1) **Attention Extraction:** we extract the attention matrices from both teacher and student models corresponding to correct and erroneous outputs, followed by Code-wise Aggregation to derive importance vectors. (2) **Construction of Maximum DID Matrix via Perturbation Analysis:** Based on DID, we get $k$, $\eta$ and $\tau$. the top-$k$ key tokens are selected and categorized into consensus, divergence, and Distractors based on the ranked attention scores. The consensus and divergence undergo perturbation analysis to yield the key tokens set $K_{pos}$, while the Distractors are processed via neg-only sampling to obtain the $K_{neg}$. Then we construct the Maximum-DID Attention Target Matrix with these sets. (3) **Dynamic Attention Alignment training:** The student model updates its attention matrix via a sliding window and gating mechanism to align with the target matrix, thereby enhancing generation accuracy.

## **Getting Started**

<span id='all\_catalogue'/>

### **Table of Contents:**

* <a href='\#Environment Preparation'>1. Environment Preparation </a>  
* <a href='\#Data Collection'>2. Data Collection (Plan & Code) </a>  
  * <a href='\#Launch vLLM Server'>2.1 Launch vLLM Server  
  * <a href='\#Run Data Generation'>2.2 Run Data Generation  
* <a href='\#Attention Extraction'>3. Attention Extraction </a>  
* <a href='\#Dual-Gating Metrics'>4. Dual-Gating System Metrics </a>  
* <a href='\#Perturbation Analysis'>5. Perturbation Analysis </a>  
* <a href='\#Training'>6. Dynamic Attention Alignment Training </a>

<span id='Environment Preparation'/>

### **1\. Environment Preparation <a href='\#all\_catalogue'>\[Back to Top\]</a>**

Please first clone the repo and install the required environment, which can be done by running the following commands:
```shell
conda create -n PADA python=3.10

conda activate PADA

# Install requirements  
python -m pip install -r requirements.txt
```
<span id='Data Collection'>

### **2\. Data Collection (Plan & Code) <a href='\#all\_catalogue'>\[Back to Top\]</a>**

For open-source models, we use the OpenAI compatible server based on vLLM. Please refer [OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server) for detailed instructions to setup the local servers.

<span id='Launch vLLM Server'>

#### **2.1 Launch vLLM Server <a href='\#all\_catalogue'>\[Back to Top\]</a>**
```shell
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen3-32B \
    --served-model-name Qwen3-32B \
    --port 8000 \
    --max-model-len 18000 \
    --trust-remote-code
```
<span id='Run Data Generation'/>

#### **2.2 Run Data Generation <a href='\#all\_catalogue'>\[Back to Top\]</a>**
```shell
cd ./data_collect/programming

python generate_data.py \
    --root_dir ../output_data/APPS/Qwen3-32B/ \
    --name initial_experiment \
    --dataset_path ../input_data/APPS/dataset/train.jsonl \
    --testfile ../input_data/APPS/dataset/train.jsonl \
    --strategy "plan_then_code" \
    --model Qwen3-32B \
    --max_iters 5 \
    --port 8000
```
**Available options for** \--dataset\_path: HumanEval, MBPP, HumanEval-ET, MBPP-ET, LiveCode, APPS, CodeContests.

<span id='Attention Extraction'/>

### **3\. Attention Extraction <a href='\#all\_catalogue'>\[Back to Top\]</a>**

Extract attention matrices from Teacher and Student models to identify raw key tokens.
```shell
cd data_prepare

python attkeyword.py \
    --model_path "$MODEL_DIR" \
    --input_file "$INPUT_DATA" \
    --output_file "$OUTPUT_FILE" \
    --top_k_ratio $TOP_K \
    --max_score_threshold $THRESHOLD \
    --window_size $WINDOW \
    --target_layer_start $START_LAYER
```
<span id='Dual-Gating Metrics'/>

### **4\. Dual-Gating System Metrics <a href='\#all\_catalogue'>\[Back to Top\]</a>**

Calculate metrics required for the **Difficulty-Aware Gating Mechanism**.

#### **Calculate Model Perplexity (Uncertainty)**
```shell
cd data_prepare

python calculate_uncertainty.py \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_DATA" \
    --output_file "$OUTPUT_DATA" \
    --device_map "auto"

```
#### **Calculate Structural Complexity (AST)**
```shell
python AST_difficulty.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE"

```
<span id='Perturbation Analysis'/>

### **5\. Perturbation Analysis <a href='\#all\_catalogue'>\[Back to Top\]</a>**

Construct the final **Maximum-DID Matrix** by filtering tokens based on ![][image1]PPL.
```shell
python Perturbation.py \
    --teacher_file "$TEACHER_JSONL" \
    --student_file "$STUDENT_JSONL" \
    --output_file "$OUTPUT_FILE" \
    --eval_model_path "$EVAL_MODEL" \
    --k_ratio $K_RATIO \
    --rho_min $RHO_MIN \
    --rho_max $RHO_MAX

```
<span id='Training'/>

### **6\. Dynamic Attention Alignment Training <a href='\#all\_catalogue'>\[Back to Top\]</a>**

Train the student model using the PADA objective.

#### **For Llama-3.2-3B**
```shell
python train_llama.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr $LR \
    --epochs $EPOCHS \
    --target_layers_start $TARGET_LAYER_START \
    --target_layers_end $TARGET_LAYER_END \
    --target_heads "$TARGET_HEADS" \
    --attn_loss_start $ATTN_LOSS_START \
    --attn_loss_end $ATTN_LOSS_END \
    --alpha_mu $ALPHA_MU \
    --alpha_tau $ALPHA_TAU \
    --complexity_scale $COMPLEXITY_SCALE

```
#### **For Qwen3 / Qwen2.5**
```shell
python train_llama.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr $LR \
    --epochs $EPOCHS \
    --target_layers_start $TARGET_LAYER_START \
    --target_layers_end $TARGET_LAYER_END \
    --target_heads "$TARGET_HEADS" \
    --attn_loss_start $ATTN_LOSS_START \
    --attn_loss_end $ATTN_LOSS_END \
    --alpha_mu $ALPHA_MU \
    --alpha_tau $ALPHA_TAU \
    --complexity_scale $COMPLEXITY_SCALE

```

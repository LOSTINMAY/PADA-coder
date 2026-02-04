# **\<center\> PADA-Coder: Improving Plan-Following Code Generation via Perturbation-Verified Attention Distillation and Dynamic Alignment\</center\>**

This repository contains the code, data, and models for "PADA-Coder: Improving Plan-Following Code Generation via Perturbation-Verified Attention Distillation and Dynamic Alignment."

🔗 **GitHub**: [https://anonymous.4open.science/r/PADA-Coder](https://anonymous.4open.science/r/PADA-Coder)

📜 **Paper**: [Coming Soon](https://www.google.com/search?q) | 📊 **Benchmark**: [APPS, HumanEval, MBPP+, LiveCodeBench](https://www.google.com/search?q) | 🤖 **Models**: [PADA-Qwen2.5-7B, PADA-Llama3.2-3B](https://www.google.com/search?q)

**📢 Notice: Ongoing Maintenance**:

This repository is currently under active development. Core code, execution scripts, and related materials are being updated regularly.

## **Key Features**

### **1\. PADA Framework**

We propose **PADA (Perturbation-Verified Attention Distillation and Dynamic Alignment)**, a unified framework that addresses "Attention Allocation Imbalance" (Drift & Dispersion) in LLMs through:

* **Maximum-DID Matrix**: Constructs an optimal attention target matrix using **Distillation Information Density (DID)** and **Perturbation Analysis** (![][image1]PPL) to filter noise and retain only high-value logic nodes.  
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

## **Getting Started**

\<span id='all\_catalogue'/\>

### **Table of Contents:**

* \<a href='\#Environment Preparation'\>1. Environment Preparation \</a\>  
* \<a href='\#Data Collection'\>2. Data Collection (Plan & Code) \</a\>  
  * \<a href='\#Launch vLLM Server'\>2.1 Launch vLLM Server  
  * \<a href='\#Run Data Generation'\>2.2 Run Data Generation  
* \<a href='\#Attention Extraction'\>3. Attention Extraction \</a\>  
* \<a href='\#Dual-Gating Metrics'\>4. Dual-Gating System Metrics \</a\>  
* \<a href='\#Perturbation Analysis'\>5. Perturbation Analysis \</a\>  
* \<a href='\#Training'\>6. Dynamic Attention Alignment Training \</a\>

\<span id='Environment Preparation'/\>

### **1\. Environment Preparation \<a href='\#all\_catalogue'\>\[Back to Top\]\</a\>**

Please first clone the repo and install the required environment, which can be done by running the following commands:
```shell
conda create \-n PADA python=3.10

conda activate PADA

\# Install requirements  
python \-m pip install \-r requirements.txt
```
\<span id='Data Collection'/\>

### **2\. Data Collection (Plan & Code) \<a href='\#all\_catalogue'\>\[Back to Top\]\</a\>**

For open-source models, we use the OpenAI compatible server based on vLLM. Please refer [OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server) for detailed instructions to setup the local servers.

\<span id='Launch vLLM Server'/\>

#### **2.1 Launch vLLM Server \<a href='\#all\_catalogue'\>\[Back to Top\]\</a\>**
```shell
CUDA\_VISIBLE\_DEVICES=0 python \-m vllm.entrypoints.openai.api\_server \\  
    \--model /path/to/Qwen3-32B \\  
    \--served-model-name Qwen3-32B \\  
    \--port 8000 \\  
    \--max-model-len 18000 \\  
    \--trust-remote-code
```
\<span id='Run Data Generation'/\>

#### **2.2 Run Data Generation \<a href='\#all\_catalogue'\>\[Back to Top\]\</a\>**
```shell
cd ./data\_collect/programming

python generate\_data.py \\  
    \--root\_dir ../output\_data/APPS/Qwen3-32B/ \\  
    \--name initial\_experiment \\  
    \--dataset\_path ../input\_data/APPS/dataset/train.jsonl \\  
    \--testfile ../input\_data/APPS/dataset/train.jsonl \\  
    \--strategy "plan\_then\_code" \\  
    \--model Qwen3-32B \\  
    \--max\_iters 5 \\  
    \--port 8000
```
**Available options for** \--dataset\_path: HumanEval, MBPP, HumanEval-ET, MBPP-ET, LiveCode, APPS, CodeContests.

\<span id='Attention Extraction'/\>

### **3\. Attention Extraction \<a href='\#all\_catalogue'\>\[Back to Top\]\</a\>**

Extract attention matrices from Teacher and Student models to identify raw key tokens.
```shell
cd data\_prepare

python attkeyword.py \\  
    \--model\_path "$MODEL\_DIR" \\  
    \--input\_file "$INPUT\_DATA" \\  
    \--output\_file "$OUTPUT\_FILE" \\  
    \--top\_k\_ratio $TOP\_K \\  
    \--max\_score\_threshold $THRESHOLD \\  
    \--window\_size $WINDOW \\  
    \--target\_layer\_start $START\_LAYER
```
\<span id='Dual-Gating Metrics'/\>

### **4\. Dual-Gating System Metrics \<a href='\#all\_catalogue'\>\[Back to Top\]\</a\>**

Calculate metrics required for the **Difficulty-Aware Gating Mechanism**.

#### **Calculate Model Perplexity (Uncertainty)**
```shell
cd data\_prepare

python calculate\_uncertainty.py \\  
    \--model\_path "$MODEL\_PATH" \\  
    \--input\_file "$INPUT\_DATA" \\  
    \--output\_file "$OUTPUT\_DATA" \\  
    \--device\_map "auto"
```
#### **Calculate Structural Complexity (AST)**
```shell
python AST\_difficulty.py \\  
    \--input\_file "$INPUT\_FILE" \\  
    \--output\_file "$OUTPUT\_FILE"
```
\<span id='Perturbation Analysis'/\>

### **5\. Perturbation Analysis \<a href='\#all\_catalogue'\>\[Back to Top\]\</a\>**

Construct the final **Maximum-DID Matrix** by filtering tokens based on ![][image1]PPL.
```shell
python Perturbation.py \\  
    \--teacher\_file "$TEACHER\_JSONL" \\  
    \--student\_file "$STUDENT\_JSONL" \\  
    \--output\_file "$OUTPUT\_FILE" \\  
    \--eval\_model\_path "$EVAL\_MODEL" \\  
    \--k\_ratio $K\_RATIO \\  
    \--rho\_min $RHO\_MIN \\  
    \--rho\_max $RHO\_MAX
```
\<span id='Training'/\>

### **6\. Dynamic Attention Alignment Training \<a href='\#all\_catalogue'\>\[Back to Top\]\</a\>**

Train the student model using the PADA objective.

#### **For Llama-3.2-3B**
```shell
python train\_llama.py \\  
    \--model\_path "$MODEL\_PATH" \\  
    \--data\_path "$DATA\_PATH" \\  
    \--output\_dir "$OUTPUT\_DIR" \\  
    \--batch\_size $BATCH\_SIZE \\  
    \--grad\_accum $GRAD\_ACCUM \\  
    \--lr $LR \\  
    \--epochs $EPOCHS \\  
    \--target\_layers\_start $TARGET\_LAYER\_START \\  
    \--target\_layers\_end $TARGET\_LAYER\_END \\  
    \--target\_heads "$TARGET\_HEADS" \\  
    \--attn\_loss\_start $ATTN\_LOSS\_START \\  
    \--attn\_loss\_end $ATTN\_LOSS\_END \\  
    \--alpha\_mu $ALPHA\_MU \\  
    \--alpha\_tau $ALPHA\_TAU \\  
    \--complexity\_scale $COMPLEXITY\_SCALE
```
#### **For Qwen3 / Qwen2.5**
```shell
python train\_qwen.py \\  
    \--model\_path "$MODEL\_PATH" \\  
    \--data\_path "$DATA\_PATH" \\  
    \--output\_dir "$OUTPUT\_DIR" \\  
    \--batch\_size $BATCH\_SIZE \\  
    \--grad\_accum $GRAD\_ACCUM \\  
    \--lr $LR \\  
    \--epochs $EPOCHS \\  
    \--target\_layers\_start $TARGET\_LAYER\_START \\  
    \--target\_layers\_end $TARGET\_LAYER\_END \\  
    \--target\_heads "$TARGET\_HEADS" \\  
    \--attn\_loss\_start $ATTN\_LOSS\_START \\  
    \--attn\_loss\_end $ATTN\_LOSS\_END \\  
    \--alpha\_mu $ALPHA\_MU \\  
    \--alpha\_tau $ALPHA\_TAU \\  
    \--complexity\_scale $COMPLEXITY\_SCALE
```

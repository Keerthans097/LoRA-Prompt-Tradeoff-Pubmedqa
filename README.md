# LoRA vs Prompt Engineering — PubMedQA

This repository contains the code, experiments, and results for comparing **LoRA fine-tuning** vs **prompt engineering** on the **PubMedQA** dataset.  
The project investigates which approach is more effective and efficient for biomedical question answering tasks with **yes/no/maybe** answers.

---

## Research Goal

- **Question:** Is **LoRA fine-tuning** or **prompt engineering** more effective in small, specialized domains?  
- **Domain:** Biomedical QA (PubMed abstracts)  
- **Dataset:** [PubMedQA](https://github.com/pubmedqa/pubmedqa)  
- **Comparison:**
  - LoRA fine-tuning (different ranks, baselines)  
  - Prompting: zero-shot, domain-specific, chain-of-thought (CoT)  
- **Metrics:** Accuracy, Macro F1, GPU memory usage


```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Keerthans097/LoRA-Prompt-Tradeoff-Pubmedqa.git
cd LoRA-Prompt-Tradeoff-Pubmedqa
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Hugging Face authentication  

```bash
huggingface-cli login
```


---

## Running Experiments

### 1. LoRA fine-tuning
Set in config:
```python
CONFIG["USE_QLORA"] = True
```
Run training cells in the notebook. Outputs:
- Fine-tuned LoRA adapter 
- Test metrics (accuracy, F1)
- Confusion matrix plots

### 2. Prompt baselines
Set:
```python
CONFIG["USE_QLORA"] = False
```
Evaluates:
- Zero-shot
- Domain-specific
- Chain-of-thought (CoT)

Each style logs metrics and confusion matrices.

---

## Logging & Results

- **experiment_log.jsonl** → detailed per-run logs  
- **experiment_leaderboard.csv** → leaderboard summary (best runs per mode)  
- **artifacts/** → confusion matrices, plots  

Results browser cell lets you view:
- Best run per mode (LoRA, zero-shot, domain, CoT)
- Accuracy/F1 bar plots
- Links to logs

---

## Using LoRA Adapter

Adapter checkpoint is available on [Hugging Face Hub](https://huggingface.co/Keerthan097/LoRA-Prompt-Tradeoff-PubMedQA).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    device_map="auto"
)
tok = AutoTokenizer.from_pretrained("Keerthan097/LoRA-Prompt-Tradeoff-PubMedQA")

model = PeftModel.from_pretrained(base, "Keerthan097/LoRA-Prompt-Tradeoff-PubMedQA")
```
# Results Summary


## LoRA (QLoRA)

## LoRA Results (PubMedQA)

| model | mode | Train Split | rank | epochs | lr | test_acc | test_f1 | train_loss | train_runtime | train_samples_per_sec | gpu_mem_allocated_mb | gpu_mem_reserved_mb | gpu_max_mem_allocated_mb |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| meta-llama/Meta-Llama-3.1-8B | qlora | 1000 | 16 | 5 | 3.00E-05 | 0.9133 | 0.8542 | 0.6295 | 4065.93 | 1.045 | 7942.92 | 16886 | 16829.31 |
| meta-llama/Meta-Llama-3.1-8B | qlora | 1000 | 8 | 5 | 3.00E-05 | 0.9133 | 0.8542 | 0.6297 | 4001.03 | 1.062 | 7702.92 | 16426 | 16345.21 |
| meta-llama/Meta-Llama-3.1-8B | qlora | 1000 | 4 | 5 | 3.00E-05 | 0.9133 | 0.8542 | 0.6298 | 3908.34 | 1.087 | 7582.92 | 16156 | 16104.09 |
| meta-llama/Meta-Llama-3.1-8B | qlora | 512 | 16 | 7 | 2.00E-05 | 0.9133 | 0.8475 | 0.6564 | 3356.89 | 1.068 | 7942.92 | 16886 | 16829.31 |
| meta-llama/Meta-Llama-3.1-8B | qlora | 512 | 8 | 7 | 2.00E-05 | 0.9133 | 0.8475 | 0.6561 | 3308.16 | 1.083 | 7702.92 | 16426 | 16345.21 |
| meta-llama/Meta-Llama-3.1-8B | qlora | 512 | 4 | 7 | 2.00E-05 | 0.9133 | 0.8475 | 0.6563 | 3297.87 | 1.087 | 7582.92 | 16156 | 16104.09 |
| meta-llama/Meta-Llama-3.1-8B | qlora | 128 | 16 | 15 | 1.00E-05 | 0.8400 | 0.8184 | 0.7960 | 1800.04 | 1.067 | 7942.92 | 16886 | 16829.31 |
| meta-llama/Meta-Llama-3.1-8B | qlora | 128 | 8 | 15 | 1.00E-05 | 0.8333 | 0.8033 | 0.7943 | 1775.37 | 1.081 | 7702.92 | 16426 | 16345.21 |
| meta-llama/Meta-Llama-3.1-8B | qlora | 128 | 8 | 10 | 1.00E-05 | 0.8133 | 0.7620 | 0.8409 | 1182.45 | 1.082 | 7702.92 | 16426 | 16345.21 |
| meta-llama/Meta-Llama-3.1-8B | qlora | 128 | 4 | 15 | 1.00E-05 | 0.8400 | 0.8184 | 0.7952 | 1772.56 | 1.083 | 7582.92 | 16166 | 16104.06 |

---
## Prompt-Based Baseline Results


| Mode          | Train Split | Test Accuracy | Test Macro F1 | GPU Mem Allocated (MB) | GPU Mem Reserved (MB) | Runtime (s) |
|---------------|-------------|---------------|---------------|-------------------------|-----------------------|-------------|
| prompt/zero   | 1000        | 0.6867        | 0.4714        | 6834.69                 | 9656.00               | 204.68      |
| prompt/domain | 1000        | 0.6933        | 0.4829        | 6834.69                 | 9656.00               | 204.68      |
| prompt/cot    | 1000        | 0.6933        | 0.4850        | 6834.69                 | 9656.00               | 204.68      |
| prompt/zero   | 512         | 0.6867        | 0.4714        | 6834.69                 | 9656.00               | 216.80      |
| prompt/domain | 512         | 0.6933        | 0.4829        | 6834.69                 | 9656.00               | 216.80      |
| prompt/cot    | 512         | 0.6933        | 0.4850        | 6834.69                 | 9656.00               | 216.80      |
| prompt/zero   | 128         | 0.6867        | 0.4714        | 6834.69                 | 9656.00               | 176.15      |
| prompt/domain | 128         | 0.6933        | 0.4829        | 6834.69                 | 9656.00               | 176.15      |
| prompt/cot    | 128         | 0.6933        | 0.4850        | 6834.69                 | 9656.00               | 176.15      |



---
## Scoring Method:
 
Logits-Based Scoring 
- Instead of generating, the model is scored directly on fixed verbalizers:
  - **yes → `" yes"`**
  - **no → `" no"`**
  - **maybe → `" maybe"`**
- For each candidate label, we compute the **log-likelihood** of producing that token sequence given the prompt.
- The label with the highest probability is selected as the prediction.
---

##  Notes
- LoRA adapters are trained with low-rank settings (R=4/8).  
- Prompt baselines tested with zero/few-shot, domain, and chain-of-thought.  
- GPU memory and runtime are logged for efficiency comparisons.  

---


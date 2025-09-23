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

| run_id | timestamp | model | rank | epochs | lr | test_acc | test_f1 | mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1758565325.000000 | 2025-09-22 18:24:21 | meta-llama/Meta-Llama-3.1-8B | 16.000000 | 5.000000 | 0.000030 | 0.913333 | 0.854231 | qlora |
| 1758500237.000000 | 2025-09-22 0:19:33 | meta-llama/Meta-Llama-3.1-8B | 8.000000 | 5.000000 | 0.000030 | 0.913333 | 0.854231 | qlora |
| 1758636905.000000 | 2025-09-23 14:17:18 | meta-llama/Meta-Llama-3.1-8B | 4.000000 | 5.000000 | 0.000030 | 0.913333 | 0.854231 | qlora |
| 1758623855.000000 | 2025-09-23 10:39:47 | meta-llama/Meta-Llama-3.1-8B | 16.000000 | 7.000000 | 0.000020 | 0.913333 | 0.847478 | qlora |
| 1758619710.000000 | 2025-09-23 9:30:43 | meta-llama/Meta-Llama-3.1-8B | 8.000000 | 7.000000 | 0.000020 | 0.913333 | 0.847478 | qlora |
| 1758640741.000000 | 2025-09-23 15:21:14 | meta-llama/Meta-Llama-3.1-8B | 4.000000 | 7.000000 | 0.000020 | 0.913333 | 0.847478 | qlora |
| 1758628052.000000 | 2025-09-23 11:49:44 | meta-llama/Meta-Llama-3.1-8B | 16.000000 | 15.000000 | 0.000010 | 0.840000 | 0.818353 | qlora |
| 1758632685.000000 | 2025-09-23 13:06:57 | meta-llama/Meta-Llama-3.1-8B | 4.000000 | 15.000000 | 0.000010 | 0.840000 | 0.818353 | qlora |
| 1758476432.000000 | 2025-09-21 18:40:38 | meta-llama/Meta-Llama-3.1-8B | 8.000000 | 5.000000 | 0.000030 | 0.833333 | 0.820111 | qlora(old metric) |
| 1758630064.000000 | 2025-09-23 12:23:17 | meta-llama/Meta-Llama-3.1-8B | 8.000000 | 15.000000 | 0.000010 | 0.833333 | 0.803337 | qlora |
| 1758625856.000000 | 2025-09-23 11:13:08 | meta-llama/Meta-Llama-3.1-8B | 8.000000 | 10.000000 | 0.000010 | 0.813333 | 0.761998 | qlora |
| 1758468939.000000 | 2025-09-21 15:36:11 | meta-llama/Meta-Llama-3.1-8B | 4.000000 | 5.000000 | 0.000020 | 0.460000 | 0.403954 | qlora (old metric) |


---

##  Notes
- LoRA adapters are trained with low-rank settings (R=4/8).  
- Prompt baselines tested with zero/few-shot, domain, and chain-of-thought.  
- GPU memory and runtime are logged for efficiency comparisons.  

---


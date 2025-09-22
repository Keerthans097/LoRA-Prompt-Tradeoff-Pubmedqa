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


---

##  Notes
- LoRA adapters are trained with low-rank settings (R=4/8).  
- Prompt baselines tested with zero/few-shot, domain, and chain-of-thought.  
- GPU memory and runtime are logged for efficiency comparisons.  

---

## Citation
```
@misc{keerthan2025lora,
  title={LoRA vs Prompt Engineering Trade-off on PubMedQA},
  author={Keerthan S.},
  year={2025},
  howpublished={GitHub},
  url={https://github.com/Keerthans097/LoRA-Prompt-Tradeoff-Pubmedqa}
}
```

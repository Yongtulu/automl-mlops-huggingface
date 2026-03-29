# Medical Symptom Classification — End-to-End MLOps Pipeline

[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange?logo=amazonaws)](https://aws.amazon.com/sagemaker/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)

An end-to-end MLOps pipeline that classifies medical symptom descriptions into
22 disease categories using AWS SageMaker and HuggingFace.

> **AIG130 — Applied Intelligent Systems, Project 3**

---

## Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| LightGBM (SageMaker Autopilot) | 100% — training set | Overfitting on 853 samples |
| **Bio_ClinicalBERT (HuggingFace)** | **76.4% — test set** | **Deployed to production** |

---

## Architecture

```
HuggingFace Hub                     Amazon S3 (central storage)
  ├── Dataset ──────────────────────────────────────┐
  └── Bio_ClinicalBERT ──────────────────┐          │
                                          │          │
                              ┌───────────┴──────────┴─────────────┐
                              │     SageMaker Pipelines (CI/CD)    │
                              │  ┌──────────────┐                  │
                              │  │ Step 1       │                  │
                              │  │ Processing   │ ml.m5.xlarge     │
                              │  │ Job          │                  │
                              │  └──────┬───────┘                  │
                              │         │                          │
                              │  ┌──────▼───────┐                  │
                              │  │ Step 2       │                  │
                              │  │ Training     │ ml.g4dn.xlarge   │
                              │  │ Job          │ (GPU)            │
                              │  └──────────────┘                  │
                              └─────────────────────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   MLflow Tracking   │ (experiment comparison)
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   Model Registry    │ (version + approval)
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │ SageMaker Endpoint  │ ml.m5.xlarge
                              └──────────┬──────────┘
                                    ┌────┘
                         ┌──────────▼──────────┐   ┌──────────────────┐
                         │   Model Monitor     │──▶│  CloudWatch +    │
                         │  (hourly drift check)│   │  EventBridge     │
                         └─────────────────────┘   │  (auto-retrain)  │
                                                    └──────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │    Gradio Demo      │ (web interface)
                              └─────────────────────┘
```

---

## Pipeline Steps

| # | Step | Tool | Instance |
|---|------|------|----------|
| 1 | Load dataset from HuggingFace Hub | HuggingFace Datasets API | — |
| 2 | Store raw data | Amazon S3 | — |
| 3 | Feature engineering | SageMaker Processing Job | ml.m5.xlarge |
| 4 | AutoML baseline | SageMaker Autopilot | managed |
| 5 | Fine-tune Bio_ClinicalBERT | SageMaker Training Job | ml.g4dn.xlarge |
| 6 | Experiment tracking | AWS MLflow (managed) | — |
| 7 | Automate pipeline | SageMaker Pipelines (DAG) | — |
| 8 | Version & approve model | SageMaker Model Registry | — |
| 9 | Real-time serving | SageMaker Endpoint | ml.m5.xlarge |
| 10 | Monitor for data drift | SageMaker Model Monitor | ml.m5.xlarge |
| 11 | Web demo | Gradio | — |

---

## Tech Stack

- **Dataset:** `gretelai/symptom_to_diagnosis` (HuggingFace Hub)
- **Model:** `emilyalsentzer/Bio_ClinicalBERT` (HuggingFace Hub)
- **AutoML:** AWS SageMaker Autopilot
- **Training:** HuggingFace Transformers + SageMaker Training Job
- **Tracking:** AWS MLflow (managed tracking server)
- **CI/CD:** AWS SageMaker Pipelines
- **Registry:** AWS SageMaker Model Registry
- **Serving:** AWS SageMaker Endpoint
- **Monitoring:** AWS SageMaker Model Monitor + CloudWatch
- **Demo:** Gradio (`share=True`)
- **AWS Region:** us-east-2

---

## Repository Structure

```
symptom-classifier-mlops/
├── notebooks/
│   ├── 01_data_loading.ipynb         # Load from HuggingFace Hub → S3
│   ├── 02_processing_job.ipynb       # SageMaker Processing Job
│   ├── 03_autopilot.ipynb            # AutoML baseline
│   ├── 04_biobert_training.ipynb     # Fine-tune Bio_ClinicalBERT
│   ├── 05_mlflow_tracking.ipynb      # Log & compare experiments
│   ├── 06_sagemaker_pipelines.ipynb  # CI/CD DAG automation
│   ├── 07_model_registry.ipynb       # Version & approve model
│   ├── 08_endpoint_deployment.ipynb  # Deploy + test endpoint
│   ├── 09_model_monitor.ipynb        # Data drift monitoring
│   └── 10_gradio_demo.ipynb          # Web demo interface
│
├── scripts/
│   ├── preprocessing.py              # Processing Job entry point
│   └── train.py                      # Training Job entry point
│
├── screenshots/                       # AWS Console & demo screenshots
│
└── docs/
    └── report.pdf                     # Full project report
```

---

## Key Findings

### Why LightGBM 100% ≠ Good Model
Autopilot found LightGBM with 100% training accuracy. However, with only 853
training samples, the model simply memorised the data (overfitting). LightGBM
uses TF-IDF word-frequency features — it cannot understand the meaning or
context of words.

### Why Bio_ClinicalBERT Works Better
Pre-trained on MIMIC-III clinical notes, Bio_ClinicalBERT already understands
medical language before fine-tuning begins. Transfer learning allows it to
achieve 76.4% test accuracy with only 853 samples — a genuine measurement on
held-out data.

### Per-class Highlights
| Category | F1 Score |
|----------|----------|
| Arthritis | 1.00 ✅ |
| Cervical Spondylosis | 1.00 ✅ |
| Fungal Infection | 1.00 ✅ |
| Dengue | 0.00 ⚠️ |
| Drug Reaction | 0.00 ⚠️ |

Dengue and Drug Reaction score 0.00 because their symptoms heavily overlap
with other diseases, and only ~39 training samples exist per class.

---

## Setup

### Prerequisites
- AWS account with SageMaker access (us-east-2)
- IAM role with `AmazonSageMakerFullAccess` + `AmazonS3FullAccess`
- S3 bucket in us-east-2

### Configuration
Replace the following in each notebook before running:

```python
BUCKET = "YOUR-S3-BUCKET-NAME"       # Your S3 bucket
REGION = "us-east-2"                 # AWS region
ROLE   = "YOUR-SAGEMAKER-ROLE-ARN"   # IAM role ARN
```

### Run Order
Run notebooks **01 → 10** in sequence. Each notebook outputs to S3, which
the next notebook reads from.

> **GPU Quota:** Step 04/06 require `ml.g4dn.xlarge`.
> Request via AWS Console → Service Quotas → SageMaker →
> "ml.g4dn.xlarge for training job usage" → increase to 1.

### Cost Estimate
| Service | Estimated Cost |
|---------|---------------|
| Autopilot job (~1 hour) | ~$5–15 |
| Training Job (ml.g4dn.xlarge, ~45 min) | ~$0.56 |
| MLflow Server (Small, ~4 hours) | ~$0.60 |
| Endpoint (ml.m5.xlarge, ~2 hours) | ~$0.23 |
| S3 storage (<1 GB) | ~$0.01 |
| **Total** | **~$7–16** |

⚠️ **Delete the endpoint after use** — it charges ~$0.115/hour continuously.

---

## Ethical Considerations

1. **Accuracy limitations** — 76.4% means ~1 in 4 predictions is wrong.
   In a medical setting this is too high for clinical use without validation.

2. **Not a substitute for doctors** — this system is for educational purposes
   only. Always consult a qualified medical professional.

3. **Model bias** — dengue and drug reaction have F1=0.00, meaning patients
   with these conditions receive no useful prediction.

4. **Data privacy** — medical data is highly sensitive. Real deployment would
   require HIPAA/PIPEDA compliance and additional AWS security configuration.

5. **Vendor lock-in** — the pipeline depends entirely on AWS services.
   Core logic (HuggingFace Transformers) is cloud-agnostic to reduce risk.

---

## References

- Alsentzer et al. (2019). *Publicly Available Clinical BERT Embeddings*. NAACL Clinical NLP Workshop.
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL-HLT.
- Gretel AI. [symptom_to_diagnosis dataset](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)

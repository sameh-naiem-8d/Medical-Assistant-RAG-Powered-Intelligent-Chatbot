# Artifact Setup

The complete MedBridge AI runtime artifacts are stored in this repository using Git LFS.

No Hugging Face repository or external artifact service is required.

## Requirements

- Git
- Git LFS
- Python 3.11

## Clone the complete repository

```powershell
git lfs install
git clone https://github.com/Mohamed-515/Medbridge-Ai.git
cd Medbridge-Ai
git lfs pull
```

## Verify the artifacts

```powershell
git lfs ls-files
python scripts/verify_artifacts.py
```

Expected runtime artifacts:

- `disease_classifier.pkl`
- `disease_label_encoder.pkl`
- `symptom_columns.pkl`
- `medical_knowledge.pkl`
- `faiss.index`
- `knowledge_base.pkl`
- `maqa_clean_data.pkl`
- `maqa_embeddings.pkl`

The large files must have their real sizes. Very small text files indicate unresolved Git LFS pointers.

Do not commit `.env`, credentials, virtual environments, caches, or logs.

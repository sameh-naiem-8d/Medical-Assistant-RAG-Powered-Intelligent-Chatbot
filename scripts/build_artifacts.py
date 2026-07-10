from __future__ import annotations

import argparse
import gc
import json
import pickle
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_maqa(data_root: Path, include_test: bool) -> pd.DataFrame:
    paths = [data_root / "First Dataset" / "MAQA_Train.xlsx"]
    if include_test:
        paths.append(data_root / "First Dataset" / "MAQA_Test.xlsx")
    frames = [pd.read_excel(path, engine="openpyxl") for path in paths]
    maqa = pd.concat(frames, ignore_index=True)
    maqa = maqa[["q_body", "a_body", "category"]].copy()
    for column in ["q_body", "a_body", "category"]:
        maqa[column] = maqa[column].map(clean_text)
    maqa = maqa[(maqa["q_body"] != "") & (maqa["a_body"] != "")]
    maqa = maqa.drop_duplicates(subset=["q_body", "a_body"]).reset_index(drop=True)
    return maqa


def build_rag_artifacts(args: argparse.Namespace) -> dict[str, object]:
    from sentence_transformers import SentenceTransformer
    import faiss

    maqa = load_maqa(args.data_root, args.include_maqa_test)
    if args.max_rag_rows and len(maqa) > args.max_rag_rows:
        maqa = maqa.sample(n=args.max_rag_rows, random_state=args.random_state).reset_index(drop=True)

    knowledge_base = maqa[["q_body", "a_body", "category"]].to_dict("records")
    corpus = [
        f"السؤال: {item['q_body']}\nالإجابة: {item['a_body']}\nالتخصص: {item['category']}"
        for item in knowledge_base
    ]

    with open(args.artifacts_dir / "maqa_clean_data.pkl", "wb") as file:
        pickle.dump(maqa, file)
    with open(args.artifacts_dir / "knowledge_base.pkl", "wb") as file:
        pickle.dump(knowledge_base, file)

    model = SentenceTransformer(args.embedding_model)
    embedding_dim = int(model.get_sentence_embedding_dimension())
    temp_embeddings_path = args.artifacts_dir / "maqa_embeddings.tmp.dat"
    progress_path = args.artifacts_dir / "maqa_embeddings.progress.json"
    chunk_size = args.encode_chunk_size

    rows_completed = 0
    if temp_embeddings_path.exists() and progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as file:
            progress = json.load(file)
        same_build = (
            progress.get("row_count") == len(corpus)
            and progress.get("embedding_dim") == embedding_dim
            and progress.get("embedding_model") == args.embedding_model
        )
        if same_build:
            rows_completed = int(progress.get("rows_completed", 0))
        else:
            temp_embeddings_path.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)

    embeddings = np.memmap(
        temp_embeddings_path,
        dtype="float32",
        mode="r+" if temp_embeddings_path.exists() else "w+",
        shape=(len(corpus), embedding_dim),
    )

    print(
        f"Encoding {len(corpus)} MAQA rows with {embedding_dim}-dim embeddings "
        f"in chunks of {chunk_size}. Resuming at row {rows_completed}.",
        flush=True,
    )

    for start in range(rows_completed, len(corpus), chunk_size):
        end = min(start + chunk_size, len(corpus))
        chunk_embeddings = model.encode(
            corpus[start:end],
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        embeddings[start:end] = chunk_embeddings
        embeddings.flush()

        with open(progress_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "rows_completed": end,
                    "row_count": len(corpus),
                    "embedding_dim": embedding_dim,
                    "embedding_model": args.embedding_model,
                },
                file,
                indent=2,
            )
        print(f"Encoded rows {end}/{len(corpus)}", flush=True)

    index = faiss.IndexFlatIP(embedding_dim)
    for start in range(0, len(corpus), chunk_size):
        end = min(start + chunk_size, len(corpus))
        index.add(np.asarray(embeddings[start:end], dtype="float32"))

    with open(args.artifacts_dir / "maqa_embeddings.pkl", "wb") as file:
        pickle.dump(np.asarray(embeddings, dtype="float32"), file, protocol=pickle.HIGHEST_PROTOCOL)
    faiss.write_index(index, str(args.artifacts_dir / "faiss.index"))

    embeddings.flush()
    del embeddings
    gc.collect()
    temp_embeddings_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)

    return {"rag_rows": len(maqa), "embedding_dim": embedding_dim}


def build_classifier_artifacts(args: argparse.Namespace) -> dict[str, object]:
    train = pd.read_csv(args.data_root / "Training.csv")
    test = pd.read_csv(args.data_root / "Testing.csv")

    symptom_columns = [column for column in train.columns if column != "prognosis"]
    X_train = train[symptom_columns]
    X_test = test[symptom_columns]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train["prognosis"].astype(str).str.strip())
    y_test = encoder.transform(test["prognosis"].astype(str).str.strip())

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "labels": encoder.classes_.tolist(),
    }

    joblib.dump(model, args.artifacts_dir / "disease_classifier.pkl")
    joblib.dump(encoder, args.artifacts_dir / "disease_label_encoder.pkl")
    joblib.dump(symptom_columns, args.artifacts_dir / "symptom_columns.pkl")
    return metrics


def build_medical_knowledge_artifact(args: argparse.Namespace) -> dict[str, int]:
    descriptions_df = pd.read_csv(
        args.data_root / "symptom_Description.csv",
        header=None,
        names=["disease", "description"],
    )
    precautions_df = pd.read_csv(
        args.data_root / "symptom_precaution.csv",
        header=None,
        names=["disease", "precaution_1", "precaution_2", "precaution_3", "precaution_4"],
    )
    severity_df = pd.read_csv(
        args.data_root / "Symptom_severity.csv",
        header=None,
        names=["symptom", "severity"],
    )

    knowledge = {
        "descriptions": dict(zip(descriptions_df["disease"].str.strip(), descriptions_df["description"].fillna("").str.strip())),
        "precautions": {
            row["disease"].strip(): [
                clean_text(row[column])
                for column in ["precaution_1", "precaution_2", "precaution_3", "precaution_4"]
                if clean_text(row[column])
            ]
            for _, row in precautions_df.fillna("").iterrows()
        },
        "severity": dict(zip(severity_df["symptom"].str.strip(), severity_df["severity"].astype(int))),
    }
    joblib.dump(knowledge, args.artifacts_dir / "medical_knowledge.pkl")
    return {
        "description_rows": len(descriptions_df),
        "precaution_rows": len(precautions_df),
        "severity_rows": len(severity_df),
    }


def parse_args() -> argparse.Namespace:
    service_root = Path(__file__).resolve().parents[1]
    workspace_root = service_root.parent
    parser = argparse.ArgumentParser(description="Build MedBridge AI model and RAG artifacts.")
    parser.add_argument("--data-root", type=Path, default=workspace_root)
    parser.add_argument("--artifacts-dir", type=Path, default=service_root / "artifacts")
    parser.add_argument("--max-rag-rows", type=int, default=25000, help="Use 0 for the full MAQA corpus.")
    parser.add_argument("--include-maqa-test", action="store_true")
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-chunk-size", type=int, default=1024)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.max_rag_rows = None if args.max_rag_rows == 0 else args.max_rag_rows

    summary: dict[str, object] = {
        "classifier": build_classifier_artifacts(args),
        "medical_knowledge": build_medical_knowledge_artifact(args),
    }
    if not args.skip_rag:
        summary["rag"] = build_rag_artifacts(args)

    with open(args.artifacts_dir / "artifact_metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

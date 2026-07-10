from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import numpy as np

from .config import Settings
from .safety import normalize_text

logger = logging.getLogger(__name__)


UNSAFE_RAG_TERMS = {
    "دواء",
    "ادوية",
    "أدوية",
    "مضاد",
    "مضاد حيوي",
    "جرعة",
    "حبوب",
    "حقنة",
    "بخاخ",
    "وصفة",
    "روشتة",
    "عملية",
    "جراحة",
    "تناول",
    "خذ",
    "استعمل",
    "antibiotic",
    "medicine",
    "dose",
    "tablet",
    "pill",
    "injection",
    "spray",
    "inhaler",
    "surgery",
    "operation",
    "prescription",
}

EMERGENCY_CONTRADICTION_TERMS = {
    "انتظر",
    "راقب فقط",
    "غير طارئ",
    "ليس طارئ",
    "لا تحتاج طوارئ",
    "لا داعي للطوارئ",
    "wait",
    "monitor only",
    "not emergency",
}

GENERIC_SHORT_RAG_ANSWERS = {
    "راجع الطبيب",
    "استشر الطبيب",
    "اذهب للطبيب",
    "لا اعلم",
    "لا أعرف",
    "نعم",
    "لا",
}


def _arabic_letter_count(text: str) -> int:
    return sum(1 for char in text if "\u0600" <= char <= "\u06FF")


def _latin_letter_count(text: str) -> int:
    return sum(1 for char in text if ("a" <= char.lower() <= "z"))


def sanitize_rag_answer(answer: str) -> str:
    cleaned = " ".join(str(answer or "").split())
    if not cleaned:
        return ""

    normalized = normalize_text(cleaned)
    if not normalized:
        return ""
    if normalized in {normalize_text(item) for item in GENERIC_SHORT_RAG_ANSWERS}:
        return ""
    if len(normalized.split()) < 5:
        return ""
    if any(normalize_text(term) in normalized for term in UNSAFE_RAG_TERMS):
        return ""

    arabic_count = _arabic_letter_count(cleaned)
    latin_count = _latin_letter_count(cleaned)
    if latin_count > arabic_count and arabic_count < 12:
        return ""

    return cleaned[:900]


def is_rag_case_safe(case: dict[str, Any], urgency_level: str | None = None) -> bool:
    answer = sanitize_rag_answer(str(case.get("a_body", "")))
    question = " ".join(str(case.get("q_body", "")).split())
    if not answer or len(question) < 8:
        return False

    combined = normalize_text(f"{question} {answer}")
    if urgency_level == "High" and any(normalize_text(term) in combined for term in EMERGENCY_CONTRADICTION_TERMS):
        return False
    return True


def filter_rag_cases_for_prompt(
    cases: list[dict[str, Any]],
    urgency_level: str | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for case in cases:
        if not is_rag_case_safe(case, urgency_level=urgency_level):
            continue
        item = dict(case)
        item["a_body"] = sanitize_rag_answer(str(item.get("a_body", "")))
        item["q_body"] = " ".join(str(item.get("q_body", "")).split())[:700]
        filtered.append(item)
        if len(filtered) >= limit:
            break
    return filtered


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.index = None
        self.knowledge_base: list[dict[str, Any]] = []
        self.embedder = None
        self.reranker = None
        self._embedder_unavailable = False
        self._reranker_unavailable = False
        self._load_artifacts()

    def _load_pickle(self, filename: str) -> Any:
        with open(self.settings.artifacts_dir / filename, "rb") as file:
            return pickle.load(file)

    def _load_artifacts(self) -> None:
        try:
            kb_path = self.settings.artifacts_dir / "knowledge_base.pkl"
            if kb_path.exists():
                loaded = self._load_pickle("knowledge_base.pkl")
                self.knowledge_base = self._coerce_records(loaded)

            index_path = self.settings.artifacts_dir / "faiss.index"
            if index_path.exists():
                import faiss

                self.index = faiss.read_index(str(index_path))
        except Exception as exc:  # pragma: no cover - defensive startup guard
            logger.warning("Could not load RAG artifacts: %s", exc)

    def _coerce_records(self, loaded: Any) -> list[dict[str, Any]]:
        if hasattr(loaded, "to_dict"):
            return loaded.to_dict("records")
        if isinstance(loaded, list):
            return [dict(item) for item in loaded]
        return []

    @property
    def ready(self) -> bool:
        return self.index is not None and bool(self.knowledge_base)

    def artifact_status(self) -> dict[str, bool]:
        artifacts_dir = self.settings.artifacts_dir
        return {
            "faiss.index": (artifacts_dir / "faiss.index").exists(),
            "maqa_clean_data.pkl": (artifacts_dir / "maqa_clean_data.pkl").exists(),
            "maqa_embeddings.pkl": (artifacts_dir / "maqa_embeddings.pkl").exists(),
            "knowledge_base.pkl": (artifacts_dir / "knowledge_base.pkl").exists(),
        }

    def _get_embedder(self):
        if self._embedder_unavailable:
            raise RuntimeError("RAG embedder unavailable")
        if self.embedder is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer

            try:
                self.embedder = SentenceTransformer(self.settings.embedding_model, local_files_only=True)
            except TypeError:
                self.embedder = SentenceTransformer(self.settings.embedding_model)
            except Exception:
                self._embedder_unavailable = True
                raise
        return self.embedder

    def _get_reranker(self):
        if not self.settings.enable_reranker:
            return None
        if self._reranker_unavailable:
            return None
        if self.reranker is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import CrossEncoder

            try:
                self.reranker = CrossEncoder(self.settings.reranker_model, local_files_only=True)
            except TypeError:
                self.reranker = CrossEncoder(self.settings.reranker_model)
            except Exception:
                self._reranker_unavailable = True
                return None
        return self.reranker

    def retrieve(self, query: str, top_k: int | None = None, rerank_top_k: int | None = None) -> list[dict[str, Any]]:
        if not self.ready or not query.strip():
            return []

        top_k = top_k or self.settings.rag_top_k
        rerank_top_k = rerank_top_k or self.settings.rerank_top_k

        try:
            query_vector = self._get_embedder().encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
            scores, indices = self.index.search(query_vector, min(top_k, len(self.knowledge_base)))
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return []

        candidates: list[dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index < 0 or index >= len(self.knowledge_base):
                continue
            item = dict(self.knowledge_base[int(index)])
            item["score"] = float(score)
            candidates.append(item)

        return self._rerank(query, candidates, rerank_top_k)

    def filter_evidence(
        self,
        cases: list[dict[str, Any]],
        urgency_level: str | None = None,
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        threshold = self.settings.min_rag_score_for_evidence
        for case in cases:
            try:
                score = float(case.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            if score < threshold:
                continue
            item = dict(case)
            item["score"] = score
            if not is_rag_case_safe(item, urgency_level=urgency_level):
                continue
            item["a_body"] = sanitize_rag_answer(str(item.get("a_body", "")))
            item["q_body"] = " ".join(str(item.get("q_body", "")).split())[:700]
            filtered.append(item)
        return filtered

    def _rerank(self, query: str, candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        if not candidates:
            return []

        try:
            reranker = self._get_reranker()
            if reranker is None:
                return candidates[:top_n]
            pairs = [
                (
                    query,
                    f"السؤال: {case.get('q_body', '')}\nالإجابة: {case.get('a_body', '')}\nالتخصص: {case.get('category', '')}",
                )
                for case in candidates
            ]
            rerank_scores = reranker.predict(pairs)
            for case, score in zip(candidates, np.asarray(rerank_scores).tolist(), strict=False):
                case["score"] = float(score)
            candidates.sort(key=lambda item: item["score"], reverse=True)
        except Exception as exc:
            logger.warning("RAG reranking failed, returning FAISS order: %s", exc)

        return candidates[:top_n]

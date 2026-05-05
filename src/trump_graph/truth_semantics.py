from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Protocol

import numpy as np
import pandas as pd

TRUTH_TOPIC_LABELS: tuple[str, ...] = (
    "elections_campaigns",
    "immigration_border",
    "economy_trade_taxes",
    "foreign_policy_diplomacy",
    "defense_military_veterans",
    "law_justice_crime",
    "congress_governance",
    "media_communications",
    "health_public_health",
    "energy_environment",
    "endorsements_appointments",
    "social_culture_education",
)

DEFAULT_ENTITY_TYPES: tuple[str, ...] = ("PER", "ORG", "LOC")
DEFAULT_EMBEDDING_DIM = 384

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_TITLE_ENTITY_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,})){0,3}\b")

_DETERMINISTIC_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "elections_campaigns": ("election", "vote", "campaign", "endorse", "poll", "ballot", "primary"),
    "immigration_border": ("border", "immigration", "illegal", "alien", "deport", "ice", "migrant"),
    "economy_trade_taxes": ("economy", "tax", "tariff", "trade", "market", "jobs", "inflation"),
    "foreign_policy_diplomacy": ("israel", "hamas", "china", "russia", "ukraine", "peace", "foreign"),
    "defense_military_veterans": ("military", "navy", "army", "veteran", "war", "defense", "troop"),
    "law_justice_crime": ("crime", "court", "judge", "justice", "attorney", "law", "criminal"),
    "congress_governance": ("congress", "senate", "house", "government", "shutdown", "speaker"),
    "media_communications": ("media", "news", "fox", "cnn", "abc", "nbc", "interview"),
    "health_public_health": ("health", "vaccine", "hospital", "doctor", "medical", "drug"),
    "energy_environment": ("energy", "oil", "gas", "epa", "environment", "electric"),
    "endorsements_appointments": ("endorsement", "appoint", "nominate", "congratulations", "representative"),
    "social_culture_education": ("school", "education", "culture", "sports", "children", "college"),
}


@dataclass(frozen=True)
class TruthSemanticConfig:
    backend: str
    topic_model_id: str
    ner_model_id: str
    sentiment_model_id: str
    embedding_model_id: str
    device: str
    batch_size: int
    topic_threshold: float
    max_topic_labels: int
    entity_score_threshold: float
    max_chunk_chars: int
    embedding_dim: int = DEFAULT_EMBEDDING_DIM


class TruthSemanticEnricher(Protocol):
    def enrich(self, posts_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        ...


def split_text_for_models(text: str, max_chars: int) -> list[str]:
    if max_chars < 100:
        raise ValueError("max_chars must be >= 100")
    clean_text = str(text).strip()
    if not clean_text:
        return [""]

    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT_PATTERN.split(clean_text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(sentence[index : index + max_chars] for index in range(0, len(sentence), max_chars))
            continue
        next_value = sentence if not current else f"{current} {sentence}"
        if len(next_value) <= max_chars:
            current = next_value
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks or [clean_text[:max_chars]]


def select_topic_labels(
    label_scores: Mapping[str, float],
    *,
    threshold: float,
    max_labels: int,
) -> list[dict[str, Any]]:
    if max_labels < 1:
        raise ValueError("max_labels must be >= 1")
    ranked = sorted(
        ((label, float(score)) for label, score in label_scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    selected = [(label, score) for label, score in ranked if score >= threshold][:max_labels]
    if not selected and ranked:
        selected = [ranked[0]]
    return [{"label": label, "score": round(float(score), 6)} for label, score in selected]


def normalize_entity_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).replace("##", "")).strip(" ,.;:!?\"'")
    return cleaned


def filter_entities(
    entities: Iterable[Mapping[str, Any]],
    *,
    score_threshold: float,
    allowed_types: Iterable[str] = DEFAULT_ENTITY_TYPES,
) -> list[dict[str, Any]]:
    allowed = {entity_type.upper() for entity_type in allowed_types}
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_entity in entities:
        raw_type = str(raw_entity.get("entity_group") or raw_entity.get("entity") or "").upper()
        entity_type = raw_type.split("-")[-1]
        if entity_type not in allowed:
            continue
        score = float(raw_entity.get("score", 0.0))
        if score < score_threshold:
            continue
        text = normalize_entity_text(str(raw_entity.get("word") or raw_entity.get("text") or ""))
        if not text:
            continue
        key = (entity_type.lower(), text.lower())
        current = deduped.get(key)
        if current is None or score > float(current["score"]):
            deduped[key] = {"text": text, "type": entity_type, "score": round(score, 6)}
    return sorted(deduped.values(), key=lambda entity: (entity["type"], entity["text"].lower()))


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def serialize_enriched_truth_posts(enriched_df: pd.DataFrame) -> pd.DataFrame:
    out = enriched_df.copy()
    for column in [
        "topic_labels",
        "topic_scores",
        "entities",
        "semantic_nodes",
        "mentions",
        "hashtags",
        "urls",
    ]:
        if column in out.columns:
            out[column] = out[column].map(json_dumps_compact)
    if "created_at_utc" in out.columns:
        out["created_at_utc"] = out["created_at_utc"].astype(str)
    if "week_start" in out.columns:
        out["week_start"] = out["week_start"].astype(str)
    if "week_end" in out.columns:
        out["week_end"] = out["week_end"].astype(str)
    return out


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, list) else []


def parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, dict) else {}


class DeterministicTruthSemanticEnricher:
    def __init__(self, config: TruthSemanticConfig):
        self.config = config

    def enrich(self, posts_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        enriched_rows: list[dict[str, Any]] = []
        embeddings: list[np.ndarray] = []
        for row in posts_df.to_dict(orient="records"):
            text = str(row["text"])
            topic_scores = self._topic_scores(text)
            selected_topics = select_topic_labels(
                topic_scores,
                threshold=self.config.topic_threshold,
                max_labels=self.config.max_topic_labels,
            )
            entities = self._entities(text)
            sentiment_label, sentiment_score = self._sentiment(text)
            semantic_nodes = semantic_nodes_from_features(
                selected_topics,
                entities,
                row.get("hashtags", []),
                row.get("mentions", []),
                include_auxiliary=True,
            )
            enriched_rows.append(
                {
                    **row,
                    "topic_labels": selected_topics,
                    "topic_scores": topic_scores,
                    "top_topic": selected_topics[0]["label"] if selected_topics else "",
                    "top_topic_score": selected_topics[0]["score"] if selected_topics else 0.0,
                    "entities": entities,
                    "sentiment_label": sentiment_label,
                    "sentiment_score": sentiment_score,
                    "semantic_nodes": semantic_nodes,
                }
            )
            embeddings.append(self._embedding(text))
        if not embeddings:
            return pd.DataFrame(enriched_rows), np.empty((0, self.config.embedding_dim), dtype="float32")
        return pd.DataFrame(enriched_rows), np.vstack(embeddings).astype("float32")

    def _topic_scores(self, text: str) -> dict[str, float]:
        lowered = text.lower()
        scores: dict[str, float] = {}
        for label, keywords in _DETERMINISTIC_TOPIC_KEYWORDS.items():
            hits = sum(1 for keyword in keywords if keyword in lowered)
            scores[label] = min(0.99, 0.08 + (hits * 0.22))
        return scores

    def _entities(self, text: str) -> list[dict[str, Any]]:
        raw_entities = []
        for match in _TITLE_ENTITY_PATTERN.finditer(text):
            value = match.group(0)
            if value.lower() in {"the", "this", "president", "america", "american"}:
                continue
            entity_type = "ORG" if value.isupper() and len(value) > 2 else "PER"
            raw_entities.append({"word": value, "entity_group": entity_type, "score": 0.72})
        return filter_entities(raw_entities, score_threshold=self.config.entity_score_threshold)

    def _sentiment(self, text: str) -> tuple[str, float]:
        lowered = text.lower()
        negative_hits = sum(keyword in lowered for keyword in ("bad", "crime", "disaster", "corrupt", "fake"))
        positive_hits = sum(keyword in lowered for keyword in ("great", "win", "peace", "congratulations", "strong"))
        if positive_hits > negative_hits:
            return "positive", 0.66
        if negative_hits > positive_hits:
            return "negative", 0.66
        return "neutral", 0.62

    def _embedding(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeats = math.ceil(self.config.embedding_dim / len(digest))
        raw = (digest * repeats)[: self.config.embedding_dim]
        vector = (np.frombuffer(raw, dtype=np.uint8).astype("float32") / 127.5) - 1.0
        norm = float(np.linalg.norm(vector))
        return vector if norm <= 1e-12 else vector / norm


class HuggingFaceTruthSemanticEnricher:
    def __init__(self, config: TruthSemanticConfig):
        self.config = config
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeError(
                "Truth semantic build needs Hugging Face dependencies. Install requirements-ml.txt "
                "or run with --semantic-backend deterministic for smoke tests."
            ) from error

        device = self._resolve_device(config.device)
        self.topic_pipeline = pipeline(
            "zero-shot-classification",
            model=config.topic_model_id,
            device=device,
        )
        self.ner_pipeline = pipeline(
            "token-classification",
            model=config.ner_model_id,
            aggregation_strategy="simple",
            device=device,
        )
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model=config.sentiment_model_id,
            device=device,
        )
        self.embedding_model = SentenceTransformer(config.embedding_model_id, device="cuda" if device == 0 else "cpu")

    @staticmethod
    def _resolve_device(device: str) -> int:
        normalized = device.strip().lower()
        if normalized == "cpu":
            return -1
        if normalized.startswith("cuda"):
            return 0
        if normalized != "auto":
            raise ValueError("Truth semantic device must be one of: auto, cuda, cpu")
        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except ImportError:
            return -1

    def enrich(self, posts_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        rows: list[dict[str, Any]] = []
        texts = [str(text) for text in posts_df["text"].tolist()]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=max(1, int(self.config.batch_size)),
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        for row in posts_df.to_dict(orient="records"):
            text = str(row["text"])
            chunks = split_text_for_models(text, self.config.max_chunk_chars)
            topic_scores = self._topic_scores(chunks)
            selected_topics = select_topic_labels(
                topic_scores,
                threshold=self.config.topic_threshold,
                max_labels=self.config.max_topic_labels,
            )
            entities = self._entities(chunks)
            sentiment_label, sentiment_score = self._sentiment(chunks)
            semantic_nodes = semantic_nodes_from_features(
                selected_topics,
                entities,
                row.get("hashtags", []),
                row.get("mentions", []),
                include_auxiliary=True,
            )
            rows.append(
                {
                    **row,
                    "topic_labels": selected_topics,
                    "topic_scores": topic_scores,
                    "top_topic": selected_topics[0]["label"] if selected_topics else "",
                    "top_topic_score": selected_topics[0]["score"] if selected_topics else 0.0,
                    "entities": entities,
                    "sentiment_label": sentiment_label,
                    "sentiment_score": sentiment_score,
                    "semantic_nodes": semantic_nodes,
                }
            )

        return pd.DataFrame(rows), embeddings

    def _topic_scores(self, chunks: list[str]) -> dict[str, float]:
        scores = {label: 0.0 for label in TRUTH_TOPIC_LABELS}
        for chunk in chunks:
            prepared_chunk = self._prepare_chunk_for_tokenizer(chunk, self.topic_pipeline.tokenizer)
            result = self.topic_pipeline(
                prepared_chunk,
                candidate_labels=list(TRUTH_TOPIC_LABELS),
                multi_label=True,
                hypothesis_template="This post is about {}.",
            )
            for label, score in zip(result.get("labels", []), result.get("scores", []), strict=False):
                scores[str(label)] = max(scores.get(str(label), 0.0), float(score))
        return {label: round(score, 6) for label, score in scores.items()}

    def _entities(self, chunks: list[str]) -> list[dict[str, Any]]:
        all_entities: list[Mapping[str, Any]] = []
        for chunk in chunks:
            prepared_chunk = self._prepare_chunk_for_tokenizer(chunk, self.ner_pipeline.tokenizer)
            all_entities.extend(self.ner_pipeline(prepared_chunk))
        return filter_entities(
            all_entities,
            score_threshold=self.config.entity_score_threshold,
            allowed_types=DEFAULT_ENTITY_TYPES,
        )

    def _sentiment(self, chunks: list[str]) -> tuple[str, float]:
        scores: dict[str, list[float]] = {"negative": [], "neutral": [], "positive": []}
        for chunk in chunks:
            prepared_chunk = self._prepare_chunk_for_tokenizer(chunk, self.sentiment_pipeline.tokenizer)
            result = self.sentiment_pipeline(prepared_chunk)
            if result:
                label = str(result[0].get("label", "neutral")).lower()
                if "neg" in label:
                    scores["negative"].append(float(result[0].get("score", 0.0)))
                elif "pos" in label:
                    scores["positive"].append(float(result[0].get("score", 0.0)))
                else:
                    scores["neutral"].append(float(result[0].get("score", 0.0)))
        averaged = {label: (sum(values) / len(values) if values else 0.0) for label, values in scores.items()}
        best_label, best_score = max(averaged.items(), key=lambda item: (item[1], item[0]))
        return best_label, round(float(best_score), 6)

    @staticmethod
    def _prepare_chunk_for_tokenizer(chunk: str, tokenizer: Any) -> str:
        max_length = int(getattr(tokenizer, "model_max_length", 512))
        if max_length <= 0 or max_length > 8192:
            max_length = 512
        input_ids = tokenizer.encode(
            str(chunk),
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        return tokenizer.decode(input_ids, skip_special_tokens=True)


def semantic_nodes_from_features(
    topic_labels: Iterable[Mapping[str, Any]],
    entities: Iterable[Mapping[str, Any]],
    hashtags: Iterable[str],
    mentions: Iterable[str],
    *,
    include_auxiliary: bool,
) -> list[str]:
    nodes: list[str] = []
    for topic in topic_labels:
        label = str(topic.get("label", "")).strip().lower()
        if label:
            nodes.append(f"topic::{label}")
    for entity in entities:
        entity_type = str(entity.get("type", "")).strip().lower()
        text = normalize_entity_text(str(entity.get("text", ""))).lower()
        if entity_type and text:
            nodes.append(f"{entity_type}::{text}")
    if include_auxiliary:
        nodes.extend(f"hashtag::{str(tag).strip().lower()}" for tag in hashtags if str(tag).strip())
        nodes.extend(f"mention::{str(mention).strip().lower()}" for mention in mentions if str(mention).strip())
    return list(dict.fromkeys(nodes))


def make_truth_semantic_enricher(config: TruthSemanticConfig) -> TruthSemanticEnricher:
    backend = config.backend.strip().lower()
    if backend == "hf":
        return HuggingFaceTruthSemanticEnricher(config)
    if backend == "deterministic":
        return DeterministicTruthSemanticEnricher(config)
    raise ValueError("semantic backend must be one of: hf, deterministic")

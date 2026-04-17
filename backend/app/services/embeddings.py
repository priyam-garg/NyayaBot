from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.config import get_settings


@lru_cache
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = get_embedder().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def vector_size() -> int:
    return get_embedder().get_sentence_embedding_dimension()

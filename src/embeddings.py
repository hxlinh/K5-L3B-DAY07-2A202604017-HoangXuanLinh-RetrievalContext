"""Real embedding provider: google/embeddinggemma-2 run locally with sentence-transformers."""
from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class GemmaEmbedder:
    """google/embeddinggemma-2 (Apache-2.0), text encoder only, 768 dimensions.

    Following the model card:
    - documents use the "Document" prompt -> "title: none | text: {chunk}" (same for every strategy);
    - questions use the "QuestionAnswering" prompt -> "task: question answering | query: {question}";
    - vectors are L2-normalized, so cosine distance in pgvector is the right metric;
    - float32 on CPU (bfloat16 only on GPUs that support it; float16 is unsafe for this model).
    Any loading or inference error is raised: there is no fallback to mock vectors.
    """

    DOCUMENT_PROMPT = "Document"
    QUERY_PROMPT = "QuestionAnswering"

    def __init__(self, model_name: str = "google/embeddinggemma-2", batch_size: int = 16) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self.model = SentenceTransformer(
            model_name,
            config_kwargs={"vision_config": None, "audio_config": None},   # text-only: 270M params
            model_kwargs={"dtype": dtype},
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = str(self.model.device)
        self.dtype = str(dtype).replace("torch.", "")
        self.dimensions = self.model.get_embedding_dimension()
        if self.dimensions != 768:
            raise RuntimeError(f"Unexpected embedding size {self.dimensions} for {model_name}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, prompt_name=self.DOCUMENT_PROMPT, batch_size=self.batch_size,
                                    normalize_embeddings=True, show_progress_bar=len(texts) > 50)
        return vectors.tolist()

    def embed_query(self, question: str) -> list[float]:
        return self.model.encode(question, prompt_name=self.QUERY_PROMPT, normalize_embeddings=True).tolist()

    def describe(self) -> dict:
        return {"provider": "local sentence-transformers", "model": self.model_name, "dimensions": self.dimensions,
                "dtype": self.dtype, "device": self.device, "document_prompt": self.DOCUMENT_PROMPT,
                "query_prompt": self.QUERY_PROMPT, "normalized": True, "cost_usd": 0}

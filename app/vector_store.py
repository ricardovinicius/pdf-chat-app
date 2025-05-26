import faiss
import numpy as np
from typing import List

class InMemoryVectorStore:
    def __init__(self, embedding_dim: int):
        """
        Inicializa um índice FAISS em memória.

        :param embedding_dim: Dimensão dos vetores de embedding.
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.texts: List[str] = []

    def add(self, embeddings: np.ndarray, texts: List[str]) -> None:
        """
        Adiciona embeddings e seus textos associados ao índice.

        :param embeddings: Vetores numpy de shape (n_chunks, embedding_dim)
        :param texts: Lista de textos (chunks) na mesma ordem dos vetores
        """
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> \
    List[str]:
        """
        Realiza busca vetorial no índice FAISS.

        :param query_embedding: Vetor de embedding da consulta (shape: 1 x embedding_dim)
        :param top_k: Número de resultados a retornar
        :return: Lista de textos mais similares
        """
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.texts[i] for i in indices[0] if
                i < len(self.texts)]

    def is_empty(self) -> bool:
        """
        Verifica se o índice está vazio.
        """
        return self.index.ntotal == 0



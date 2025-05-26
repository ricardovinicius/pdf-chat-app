# embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inicializa o modelo de embeddings.

        :param model_name: nome do modelo pré-treinado do sentence-transformers
        """
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """
        Gera embeddings para uma lista de textos.

        :param texts: lista de strings a serem convertidas em embeddings
        :return: array numpy de embeddings
        """
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Gera embedding para um texto único (consulta).

        :param query: string da consulta
        :return: vetor numpy do embedding da consulta
        """
        embedding = self.model.encode([query], show_progress_bar=True, normalize_embeddings=True)
        return embedding

def main():
    embedder = Embedder()

    texts = ["Test 1", "Test 2"]
    embeddings = embedder.encode_texts(texts)

    print(embeddings)

if __name__ == "__main__":
    main()

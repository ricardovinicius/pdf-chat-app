# chat_agent.py

from typing import List
import numpy as np

from embedder import Embedder
from llm_client import OllamaClient
from pdf_processor import extract_text_from_pdf, \
    split_text_into_chunks
from vector_store import InMemoryVectorStore


class ChatAgent:
    def __init__(self, embedder, vector_store, llm_client, top_k: int = 5):
        """
        Inicializa o agente de chat com os componentes necessários.

        :param embedder: Instância do componente de embeddings.
        :param vector_store: Instância do armazenamento vetorial (FAISS).
        :param llm_client: Cliente para comunicação com o modelo LLM.
        :param top_k: Número de chunks a serem recuperados para o contexto.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.top_k = top_k

    def embed_question(self, question: str) -> np.ndarray:
        """
        Gera o embedding da pergunta do usuário.

        :param question: Pergunta em linguagem natural.
        :return: Vetor de embedding da pergunta.
        """
        return self.embedder.encode_query(question)

    def retrieve_context(self, question_embedding: np.ndarray) -> List[str]:
        """
        Recupera os trechos mais relevantes com base no embedding da pergunta.

        :param question_embedding: Vetor da pergunta.
        :return: Lista de chunks de texto relevantes.
        """
        return self.vector_store.search(question_embedding, self.top_k)

    def build_prompt(self, context_chunks: List[str], question: str) -> str:
        """
        Constrói o prompt a ser enviado para a LLM.

        :param context_chunks: Trechos relevantes recuperados.
        :param question: Pergunta original do usuário.
        :return: Prompt completo para o modelo.
        """
        context = "\n\n".join(context_chunks)
        prompt = f"""
Contexto:
{context}

Pergunta:
{question}

Responda de forma clara e objetiva com base no contexto acima.
"""
        return prompt.strip()

    def generate_answer(self, prompt: str) -> str:
        """
        Envia o prompt para a LLM e retorna a resposta.

        :param prompt: Prompt formatado com contexto e pergunta.
        :return: Resposta gerada pela LLM.
        """
        return self.llm_client.generate(prompt)

    def answer_question(self, question: str) -> str:
        """
        Orquestra o processo completo de geração de resposta com recuperação de contexto.

        :param question: Pergunta do usuário.
        :return: Resposta final gerada pela LLM.
        """
        question_embedding = self.embed_question(question)
        context_chunks = self.retrieve_context(question_embedding)
        prompt = self.build_prompt(context_chunks, question)
        return self.generate_answer(prompt)

def main():
    pdf_path = "historico.pdf"

    with open(pdf_path, "rb") as pdf_file:
        pdf_text = extract_text_from_pdf(pdf_file)

    chunks = split_text_into_chunks(pdf_text)

    embedder = Embedder()

    embeddings = embedder.encode_texts(chunks)
    vector_store = InMemoryVectorStore(384)
    vector_store.add(embeddings, chunks)

    llm_client = OllamaClient()

    chat_agent = ChatAgent(embedder, vector_store, llm_client)

    print(chat_agent.answer_question("Quais discplinas foram cursadas pelo estudante?"))

if __name__ == "__main__":
    main()

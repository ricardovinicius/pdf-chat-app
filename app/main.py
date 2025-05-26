# main.py

import streamlit as st
from io import BytesIO

from llm_client import OllamaClient
from pdf_processor import extract_text_from_pdf, split_text_into_chunks
from embedder import Embedder
from vector_store import InMemoryVectorStore
from ai_agent import ChatAgent

embedder = Embedder()
llm_client = OllamaClient()
vector_store = InMemoryVectorStore(384)
chat_agent = ChatAgent(embedder, vector_store, llm_client)

# Título da aplicação
st.title("PDF Q&A com RAG + LLM (via Ollama)")

# Upload do PDF
uploaded_file = st.file_uploader("Faça upload de um PDF", type=["pdf"])

if uploaded_file is not None:
    # Extrai texto do PDF
    text = extract_text_from_pdf(BytesIO(uploaded_file.read()))

    # Divide o texto em chunks
    chunks = split_text_into_chunks(text)

    # Gera embeddings para os chunks
    embeddings = embedder.encode_texts(chunks)
    vector_store.add(embeddings, chunks)

    st.success("PDF processado com sucesso!")

    # Campo de pergunta
    question = st.text_input("Digite sua pergunta sobre o PDF")

    if question:
        # Gera resposta com base nos chunks mais relevantes
        response = chat_agent.answer_question(question)

        # Exibe a resposta
        st.markdown("**Resposta:**")
        st.write(response)

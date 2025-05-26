# PDF RAG Chatbot

Este projeto implementa um sistema de **Perguntas e Respostas com PDFs** utilizando o paradigma de **RAG (Retrieval-Augmented Generation)** com **LLMs locais via Ollama** e uma interface interativa via **Streamlit**.

## Visão Geral

O sistema permite que o usuário envie arquivos PDF e, a partir deles, faça perguntas em linguagem natural. A aplicação utiliza técnicas de NLP para extrair informações relevantes e gerar respostas contextualizadas com base no conteúdo dos documentos.

## Arquitetura

app/
├── main.py # Interface principal (Streamlit)
├── pdf_processor.py # Extração e divisão de texto do PDF
├── embedder.py # Geração de embeddings via Sentence-Transformers
├── vector_store.py # Armazenamento e recuperação com FAISS
├── chat_agent.py # Lógica de recuperação e interação com LLM
└── utils.py # Funções auxiliares


## Componentes Principais

- **Extração de texto:** Utiliza `PyPDF2` para extrair texto de arquivos PDF recebidos via Streamlit.
- **Chunking:** Segmenta o texto em pedaços semânticos para posterior indexação.
- **Embeddings:** Utiliza `sentence-transformers` (modelo MiniLM) para transformar texto em vetores numéricos.
- **Armazenamento vetorial:** FAISS é usado para armazenar e recuperar vetores semanticamente similares.
- **LLM (Local):** Integração com modelos como `mistral` ou `llama3` rodando via `Ollama` para geração de respostas.
- **Interface:** Desenvolvida com Streamlit, permite upload de PDF e chat em tempo real.

## Como Executar

### Requisitos

- Python 3.10+
- Ollama instalado e com modelo carregado (`mistral`, `llama3`, etc.)
- Dependências listadas em `requirements.txt`

### Instalação

```bash
pip install -r requirements.txt
```

### Execução

```
streamlit run app/main.py
```

### Exemplos de Uso

    Faça upload de um arquivo PDF.

    Digite uma pergunta sobre o conteúdo do arquivo.

    A resposta será gerada com base nos trechos mais relevantes do documento.

### Tecnologias Utilizadas

    Streamlit

    sentence-transformers

    FAISS

    Ollama

    PyPDF2

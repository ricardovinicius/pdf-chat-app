"""
Este módulo fornece funcionalidades para extração e segmentação de texto a partir de documentos em formato PDF.
É utilizado como etapa inicial no pipeline de sistemas baseados em Recuperação Aumentada por Geração (RAG).
"""
from typing import IO

import pymupdf
import pymupdf4llm
from nltk.tokenize import sent_tokenize


def extract_text_from_pdf(file: IO[bytes]) -> str:
    """
    Extrai e retorna todo o texto contido no PDF.
    """
    document = pymupdf.open(stream=file)
    md_text = pymupdf4llm.to_markdown(document)

    return md_text

def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[str]:
    """
    Divide o texto em pedaços com sobreposição para contexto.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Se adicionar a sentença ultrapassar o limite
        if len(current_chunk) + len(sentence) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[
                            -chunk_overlap:] + " "  # reaproveita fim
        current_chunk += sentence + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def main():
    pdf_path = "historico.pdf"

    with open(pdf_path, "rb") as pdf_file:
        pdf_text = extract_text_from_pdf(pdf_file)

    # print(pdf_text)
    for chunk in split_text_into_chunks(pdf_text):
        print(chunk)

if __name__ == "__main__":
    main()
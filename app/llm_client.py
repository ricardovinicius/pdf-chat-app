# llm_client.py

import ollama

class OllamaClient:
    def __init__(self, model: str = "mistral"):
        """
        Cliente para interagir com o modelo LLM executado via Ollama.

        :param model: Nome do modelo carregado no Ollama.
        """
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Gera uma resposta para o prompt fornecido.

        :param prompt: Prompt completo com contexto e pergunta.
        :return: Resposta gerada pela LLM.
        """
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content']

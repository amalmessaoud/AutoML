import ollama


def call_llm(prompt: str, model: str = "llama3.1:8b") -> str:
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

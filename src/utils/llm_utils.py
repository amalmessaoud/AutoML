import ollama


def call_ollama(prompt: str, model: str = 'llama3.1:8b') -> str:
    response = ollama.chat(
        model=model, messages=[{'role': 'user', 'content': prompt}], format='json'
    )
    return response['message']['content']

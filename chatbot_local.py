import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# Cargar los vectores y textos
index = faiss.read_index("index.faiss")
with open("textos.pkl", "rb") as f:
    textos = pickle.load(f)

# Cargar el modelo de embeddings
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Función para encontrar el contexto más cercano
def buscar_contexto(pregunta, k=3):
    vector = modelo.encode([pregunta])
    D, I = index.search(np.array(vector), k)
    contexto = "\n\n".join([textos[i] for i in I[0]])
    return contexto

# Función para enviar pregunta al modelo DeepSeek (Ollama)
def preguntar_ollama(contexto, pregunta):
    prompt = f"""
Eres un asistente académico experto. Usa el siguiente contexto para responder de forma clara y completa:

Contexto:
{contexto}

Pregunta: {pregunta}
"""
    respuesta = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "deepseek-coder:instruct", "prompt": prompt, "stream": False}
    )
    return respuesta.json()["response"]

# Chat interactivo
print("🤖 Chatbot activo. Escribe 'salir' para terminar.\n")

while True:
    pregunta = input("Tú: ")
    if pregunta.lower() == "salir":
        break
    contexto = buscar_contexto(pregunta)
    respuesta = preguntar_ollama(contexto, pregunta)
    print("\n🤖 Chatbot:", respuesta, "\n")
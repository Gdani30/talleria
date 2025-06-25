from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Leer el PDF
reader = PdfReader("documento.pdf")
textos = [page.extract_text() for page in reader.pages if page.extract_text()]

# Crear los vectores (embeddings)
modelo = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = modelo.encode(textos)

# Guardar el índice vectorial
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
faiss.write_index(index, "index.faiss")

# Guardar los textos alineados
with open("textos.pkl", "wb") as f:
    pickle.dump(textos, f)

print("PDF vectorizado con éxito.")
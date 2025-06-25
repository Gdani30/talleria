# talleria
# Chatbot Personalizado Basado en Documentos

## 📄 Descripción del proyecto

Este proyecto es un chatbot académico que responde preguntas sobre un archivo PDF usando inteligencia artificial local (Ollama + DeepSeek) y búsqueda semántica.

---

## 🔧 Herramientas utilizadas

- Python 3.x
- Ollama + modelo `deepseek-coder:instruct`
- sentence-transformers
- faiss-cpu
- PyPDF2
- GitHub Desktop

---

📚 Lo que aprendí en este taller

Durante el desarrollo de este proyecto aprendí a construir un chatbot personalizado capaz de responder preguntas sobre el contenido de un documento PDF, usando inteligencia artificial local.

Algunos de los aprendizajes que me llevé:

Comprendí cómo funciona un modelo de lenguaje (LLM) ejecutado localmente a través de herramientas como Ollama y DeepSeek.

Aprendí a transformar el contenido de un PDF en vectores semánticos utilizando embeddings con la librería sentence-transformers.

Descubrí cómo aplicar FAISS para realizar búsquedas por similitud y conectar preguntas con fragmentos relevantes del documento.

Desarrollé un flujo completo de chatbot, desde la lectura del PDF hasta la interacción con el usuario.

Me familiaricé con herramientas de desarrollo como GitHub Desktop y Cursor, y entendí la importancia de organizar un proyecto en repositorio.

A pesar de no tener experiencia previa en programación, logré entender cómo se conectan las distintas piezas para que un chatbot funcione de forma autónoma y local.

Este proyecto fue un gran primer paso para adentrarme en el mundo de la inteligencia artificial aplicada y el desarrollo de soluciones personalizadas con Python.

---

## ⚙️ Cómo ejecutar el sistema

1. Clonar el repositorio o descargarlo como ZIP.
2. Crear un entorno virtual:

```bash
python -m venv env
.\env\Scripts\activate



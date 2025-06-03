# 📚 Research Paper Buddy

A smart AI assistant to help you understand and explore research papers with ease. Upload any academic PDF and ask questions — powered by **LangChain**, **FAISS**, and **Groq LLM** with **Ollama embeddings**.

---

## 🚀 Features

- 📄 Upload any research paper PDF
- 💬 Ask questions and get instant contextual answers
- 🔍 Semantic search using FAISS vector store
- 🧠 Local embeddings via Ollama
- ⚡ Ultra-fast responses using Groq’s `mixtral-8x7b` model
- 🖼️ Grad-CAM-style document similarity display (if enabled)

---

## 🛠️ Tech Stack

- `LangChain` (v0.3.25)
- `FAISS` for vector similarity
- `Ollama` for local embeddings
- `Groq LLM` for fast inference
- `Streamlit` for interactive UI
- `Arxiv`, `Wikipedia` tools for enhanced RAG
- `Python 3.12`, `venv`

---

## 🧪 Setup Instructions

```bash
# Clone this repository
git clone https://github.com/soniakashyap001/research-paper-buddy.git
cd research-paper-buddy

# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the app
streamlit run app.py

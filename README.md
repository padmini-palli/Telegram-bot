# Gemini 2.5 Flash Telegram Bot (Text-based RAG)

This project builds a **Telegram bot** powered by **Google Gemini 2.5 Flash** for text-based question answering and summarization.  
It uses **local FAISS vector search** for document retrieval and **Vertex AI embeddings** for semantic similarity.

---

##  Features

| Feature | Description |
|----------|-------------|
| **Text Q&A with Context** | `/ask` command retrieves best matching document snippet via FAISS and generates a Gemini answer. |
| **Summarization** | `/summarize` summarizes the last few user-bot interactions. |
| **Contextual Memory** | Keeps last 3 interactions per user for coherent replies. |
| **FAISS Vector Store** | Stores embeddings locally for fast retrieval. |
| **Lightweight Setup** | Runs entirely on your machine (except Gemini API). |

---

##  Tech Stack

- **Telegram Bot Framework:** `python-telegram-bot` (async)
- **Generative Model:** `gemini-2.5-flash`
- **Embedding Model:** `text-embedding-005`
- **Vector Store:** `FAISS`
- **Language Embeddings:** `Vertex AI`
- **Cloud Platform:** Google Cloud Vertex AI
- **Optional Local DB:** SQLite (future extension)

---
##  Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/gemini-telegram-bot.git
cd gemini-telegram-bot

pip install -r requirements.txt
python telegram_bot.py

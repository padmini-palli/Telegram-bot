# Telegram Bot (Text + Image Captioning)

This project builds a **Telegram bot** powered by **Google Gemini 2.5 Flash** for text-based question answering and summarization,  
and **Salesforce BLIP (Image Captioning Model)** for describing uploaded images.

It combines **text reasoning** and **vision understanding** in one bot — powered by **Google Cloud Vertex AI** and **Hugging Face Transformers**.

---

##  Features

| Feature | Description |
|----------|-------------|
| **Text Q&A with RAG** | `/ask` retrieves relevant context using FAISS + Vertex AI embeddings and queries Gemini 2.5 Flash. |
| **Summarization** | `/summarize` command summarizes the last few user-bot messages. |
| **Contextual Memory** | Retains the last 3 user interactions for coherence. |
| **Image Captioning (BLIP)** | Upload any image and the bot automatically generates a short description. |
| **FAISS Vector Store** | Local semantic search for RAG-based question answering. |
| **Gemini + Hugging Face Combo** | Gemini handles text intelligence; BLIP handles image understanding. |
| **Lightweight Deployment** | Runs locally or in GCP with minimal setup. |

---

##  Tech Stack

| Component | Technology |
|------------|-------------|
| **Text Model** | Google **Gemini 2.5 Flash** (via Vertex AI) |
| **Image Model** | Salesforce **BLIP (blip-image-captioning-large)** |
| **Vector Store** | **FAISS** |
| **Embeddings** | `text-embedding-005` (Vertex AI) |
| **Framework** | `python-telegram-bot` (async) |
| **Cloud** | Google Cloud Vertex AI |
| **Dependencies** | Hugging Face Transformers, Torch, Pillow |

---

##  Installation

###  Clone or Install the BLIP related packages
```bash
!git clone https://github.com/salesforce/LAVIS.git
!cd LAVIS
!pip install -e .

### Clone the Repository
```bash
git clone https://github.com/padmini-palli/Telegram-bot.git
cd telegram-bot

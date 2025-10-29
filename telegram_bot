import os
import asyncio
import logging
import json
import numpy as np
import faiss
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)
from google.cloud import aiplatform
import google.generativeai as genai
import nest_asyncio
import asyncio
nest_asyncio.apply()
from vertexai.generative_models import GenerationConfig, GenerativeModel, SafetySetting
# ============== CONFIG =====================

PROJECT_ID = "gpeg-camps-platform"
LOCATION = "us-central1"
EMBED_MODEL = "text-embedding-005"
GEN_MODEL = "gemini-2.5-flash"
logging.basicConfig(level=logging.INFO)

# ============== SAMPLE DOCS =====================
DOCS = {
    "policy.txt": "Employees must submit leave requests 2 days in advance. Sick leaves one per month and casal leave is 2 per month. Remote work is allowed up to 3 days a week.",
    "recipe.txt": "To make pasta: boil water, add salt, cook pasta for 8-10 minutes, then drain and mix with sauce. To make briyani: boil water, add masala, add chicken, cook for 15 mins,  add rice, cook for 10 mins, wait for 15 mins before serve",
    "faq.txt": "To reset your password, go to settings, click 'Reset Password', and follow the email instructions.",
}

os.makedirs("data", exist_ok=True)
DOC_PATH = "data/docs.json"
if not os.path.exists(DOC_PATH):
    with open(DOC_PATH, "w") as f:
        json.dump(DOCS, f)

# ============== MEMORY + CACHE =====================
USER_HISTORY = {}   # {user_id: [ {"q": query, "a": answer}, ... ] }
EMBED_CACHE = {}    # {text: np.ndarray}

# ============== EMBEDDINGS =====================
def get_text_embedding(text: str):
    """Generate and cache embeddings"""
    if text in EMBED_CACHE:
        return EMBED_CACHE[text]
    from vertexai.preview.language_models import TextEmbeddingModel
    model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
    emb = np.array(model.get_embeddings([text])[0].values, dtype="float32")
    EMBED_CACHE[text] = emb
    return emb

def build_faiss_index():
    """Build FAISS index for docs"""
    with open(DOC_PATH) as f:
        docs = json.load(f)
    keys, vectors = [], []
    for k, v in docs.items():
        keys.append(k)
        vectors.append(get_text_embedding(v))
    vectors = np.vstack(vectors)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, "data/index.faiss")
    with open("data/keys.json", "w") as f:
        json.dump(keys, f)
    logging.info("✅ FAISS index built successfully.")

def retrieve_similar(query, top_k=1):
    """Return best matching doc snippet"""
    index = faiss.read_index("data/index.faiss")
    with open("data/keys.json") as f:
        keys = json.load(f)
    qvec = get_text_embedding(query).reshape(1, -1)
    D, I = index.search(qvec, top_k)
    best_key = keys[I[0][0]]
    with open(DOC_PATH) as f:
        docs = json.load(f)
    return docs[best_key], best_key

if not os.path.exists("data/index.faiss"):
    build_faiss_index()

# ============== GEMINI =====================
async def generate_answer(context_text, user_query, user_history=None):
    """Generate response using Gemini with context and last 3 turns"""
    hist_text = ""
    if user_history:
        for h in user_history[-3:]:
            hist_text += f"User: {h['q']}\nBot: {h['a']}\n"

    prompt = f"""
You are a helpful assistant. Use the context below to answer.

Context:
{context_text}

Recent conversation:
{hist_text}

User query:
{user_query}

Answer clearly and concisely.
"""
    model = GenerativeModel(GEN_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()

# ============== COMMAND HANDLERS =====================
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        " *Commands:*\n"
        "/ask <query> → Ask about docs\n"
        "/summarize → Summarize last few interactions",
        parse_mode="Markdown"
    )

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Please provide a query. Example: /ask What is the leave policy?")
        return

    context_text, source = retrieve_similar(query)
    history = USER_HISTORY.get(user_id, [])

    try:
        answer = await generate_answer(context_text, query, history)
    except Exception as e:
        await update.message.reply_text(f" Error generating answer: {e}")
        return

    # Update memory
    USER_HISTORY.setdefault(user_id, []).append({"q": query, "a": answer})
    USER_HISTORY[user_id] = USER_HISTORY[user_id][-3:]  # keep last 3

    await update.message.reply_text(
        f" *Answer:*\n{answer}\n\n📄 *Source:* `{source}`",
        parse_mode="Markdown"
    )

async def summarize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Summarize the user's recent chat"""
    user_id = update.message.from_user.id
    history = USER_HISTORY.get(user_id, [])
    if not history:
        await update.message.reply_text("No recent chat to summarize.")
        return

    chat_text = "\n".join([f"User: {h['q']}\nBot: {h['a']}" for h in history])
    prompt = f"Summarize the following conversation briefly:\n\n{chat_text}"

    model = GenerativeModel(GEN_MODEL)
    response = model.generate_content(prompt)
    summary = response.text.strip()
    await update.message.reply_text(f" *Summary:*\n{summary}", parse_mode="Markdown")

# ============== MAIN =====================
async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("summarize", summarize))
    logging.info("Bot started…")
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())

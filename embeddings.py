# import os
# import asyncio
# import logging
# import json
# import numpy as np
# import faiss
# from telegram import Update
# from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
# import google.generativeai as genai

# TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# GEN_MODEL = "gemini-2.0-flash-exp"
# EMBED_MODEL = "models/text-embedding-004"

# genai.configure(api_key=GOOGLE_API_KEY)

# logging.basicConfig(level=logging.INFO)

# DOCS = {
#     "policy.txt": "Employees must submit leave requests 2 days in advance. Remote work is allowed up to 3 days a week.",
#     "recipe.txt": "To make pasta: boil water, add salt, cook pasta for 8-10 minutes, then drain and mix with sauce.",
#     "faq.txt": "To reset your password, go to settings, click 'Reset Password', and follow the email instructions.",
#     "personal.txt": "My name is Susu and age is 105. Eats chicken and favourite colour is Red.",
# }

# os.makedirs("data", exist_ok=True)
# DOC_PATH = "data/docs.json"
# if not os.path.exists(DOC_PATH):
#     with open(DOC_PATH, "w") as f:
#         json.dump(DOCS, f)

# USER_HISTORY = {}
# EMBED_CACHE = {}

# def get_text_embedding(text: str):
#     """Generate and cache embeddings using Google Generative AI"""
#     if text in EMBED_CACHE:
#         return EMBED_CACHE[text]
#     result = genai.embed_content(
#         model=EMBED_MODEL,
#         content=text,
#         task_type="retrieval_document"
#     )
#     emb = np.array(result['embedding'], dtype="float32")
#     EMBED_CACHE[text] = emb
#     return emb

# def build_faiss_index():
#     """Build FAISS index for docs"""
#     with open(DOC_PATH) as f:
#         docs = json.load(f)
#     keys, vectors = [], []
#     for k, v in docs.items():
#         keys.append(k)
#         vectors.append(get_text_embedding(v))
#     vectors = np.vstack(vectors)
#     index = faiss.IndexFlatL2(vectors.shape[1])
#     index.add(vectors)
#     faiss.write_index(index, "data/index.faiss")
#     with open("data/keys.json", "w") as f:
#         json.dump(keys, f)
#     logging.info("✅ FAISS index built successfully.")

# def retrieve_similar(query, top_k=1):
#     """Return best matching doc snippet"""
#     index = faiss.read_index("data/index.faiss")
#     with open("data/keys.json") as f:
#         keys = json.load(f)
    
#     result = genai.embed_content(
#         model=EMBED_MODEL,
#         content=query,
#         task_type="retrieval_query"
#     )
#     qvec = np.array(result['embedding'], dtype="float32").reshape(1, -1)
    
#     D, I = index.search(qvec, top_k)
#     best_key = keys[I[0][0]]
#     with open(DOC_PATH) as f:
#         docs = json.load(f)
#     return docs[best_key], best_key

# if not os.path.exists("data/index.faiss"):
#     build_faiss_index()

# async def generate_answer(context_text, user_query, user_history=None):
#     """Generate response using Gemini with context and last 3 turns"""
#     hist_text = ""
#     if user_history:
#         for h in user_history[-3:]:
#             hist_text += f"User: {h['q']}\nBot: {h['a']}\n"

#     prompt = f"""
# You are a helpful assistant. Use the context below to answer.

# Context:
# {context_text}

# Recent conversation:
# {hist_text}

# User query:
# {user_query}

# Answer clearly and concisely.
# """
#     model = genai.GenerativeModel(GEN_MODEL)
#     response = model.generate_content(prompt)
#     return response.text.strip()

# async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     await update.message.reply_text(
#         "💡 *Commands:*\n"
#         "/help → Show this help message\n"
#         "/ask <query> → Ask about docs\n"
#         "/summarize → Summarize last few interactions",
#         parse_mode="Markdown"
#     )

# async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     user_id = update.message.from_user.id
#     query = " ".join(context.args)
#     if not query:
#         await update.message.reply_text("Please provide a query. Example: /ask What is the leave policy?")
#         return

#     context_text, source = retrieve_similar(query)
#     history = USER_HISTORY.get(user_id, [])

#     try:
#         answer = await generate_answer(context_text, query, history)
#     except Exception as e:
#         await update.message.reply_text(f"⚠️ Error generating answer: {e}")
#         return

#     USER_HISTORY.setdefault(user_id, []).append({"q": query, "a": answer})
#     USER_HISTORY[user_id] = USER_HISTORY[user_id][-3:]

#     await update.message.reply_text(
#         f"🧠 *Answer:*\n{answer}\n\n📄 *Source:* `{source}`",
#         parse_mode="Markdown"
#     )

# async def summarize(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Summarize the user's recent chat"""
#     user_id = update.message.from_user.id
#     history = USER_HISTORY.get(user_id, [])
#     if not history:
#         await update.message.reply_text("No recent chat to summarize.")
#         return

#     chat_text = "\n".join([f"User: {h['q']}\nBot: {h['a']}" for h in history])
#     prompt = f"Summarize the following conversation briefly:\n\n{chat_text}"

#     model = genai.GenerativeModel(GEN_MODEL)
#     response = model.generate_content(prompt)
#     summary = response.text.strip()
#     await update.message.reply_text(f"📝 *Summary:*\n{summary}", parse_mode="Markdown")

# async def main():
#     app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
#     app.add_handler(CommandHandler("help", help_command))
#     app.add_handler(CommandHandler("ask", ask))
#     app.add_handler(CommandHandler("summarize", summarize))
#     logging.info("🚀 Bot started…")
#     await app.run_polling()

# if __name__ == "__main__":
#     asyncio.run(main())

import os
import json
import asyncio
import logging
import numpy as np
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# =====================================
# 🔧 Setup
# =====================================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
logging.basicConfig(level=logging.INFO)

# Initialize SentenceTransformer model (lightweight for Replit)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================
# 📄 Sample Docs
# =====================================
DOCS = {
    "policy.txt": "Employees must submit leave requests 2 days in advance. Remote work is allowed up to 3 days a week.",
    "recipe.txt": "To make pasta: boil water, add salt, cook pasta for 8-10 minutes, then drain and mix with sauce.",
    "faq.txt": "To reset your password, go to settings, click 'Reset Password', and follow the email instructions.",
    "personal.txt": "My name is Susu and age is 105. Eats chicken and favourite colour is Red.",
}

os.makedirs("data", exist_ok=True)
DOC_PATH = "data/docs.json"
if not os.path.exists(DOC_PATH):
    with open(DOC_PATH, "w") as f:
        json.dump(DOCS, f)

# =====================================
# 🧠 Memory
# =====================================
USER_HISTORY = {}  # {user_id: [{"q": query, "a": answer}]}
EMBED_CACHE = {}   # {text: np.ndarray}

# =====================================
# 🔍 Embeddings with SentenceTransformer
# =====================================
def get_text_embedding(text: str):
    """Generate embeddings and cache results using SentenceTransformer"""
    if text in EMBED_CACHE:
        return EMBED_CACHE[text]
    emb = embed_model.encode([text])[0].astype("float32")
    EMBED_CACHE[text] = emb
    return emb

def build_local_index():
    """Build simple in-memory embedding index"""
    with open(DOC_PATH) as f:
        docs = json.load(f)
    keys, vectors = [], []
    for k, v in docs.items():
        keys.append(k)
        vectors.append(get_text_embedding(v))
    np.save("data/keys.npy", np.array(keys))
    np.save("data/vectors.npy", np.vstack(vectors))
    logging.info("✅ Local index built successfully.")

def retrieve_similar(query, top_k=1):
    """Return most similar document snippet"""
    if not os.path.exists("data/vectors.npy"):
        build_local_index()

    keys = np.load("data/keys.npy", allow_pickle=True)
    vectors = np.load("data/vectors.npy")

    qvec = get_text_embedding(query)
    sims = np.dot(vectors, qvec) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(qvec))
    best_idx = np.argmax(sims)
    best_key = keys[best_idx]
    with open(DOC_PATH) as f:
        docs = json.load(f)
    return docs[best_key], best_key

# =====================================
# 🤖 Gemini Generator
# =====================================
async def generate_answer(context_text, user_query, user_history=None):
    """Generate a response using Gemini"""
    hist_text = ""
    if user_history:
        for h in user_history[-3:]:
            hist_text += f"User: {h['q']}\nBot: {h['a']}\n"

    prompt = f"""
You are a helpful assistant. Use the context below to answer.

Context:
{context_text}

Conversation history:
{hist_text}

User query:
{user_query}

Answer clearly and concisely.
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# =====================================
# 🧾 Telegram Commands
# =====================================
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "💡 *Commands:*\n"
        "/ask <query> → Ask about docs\n"
        "/summarize → Summarize recent chat",
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
        logging.error(f"Gemini error: {e}")
        await update.message.reply_text(f"⚠️ Error generating answer: {e}")
        return

    USER_HISTORY.setdefault(user_id, []).append({"q": query, "a": answer})
    USER_HISTORY[user_id] = USER_HISTORY[user_id][-3:]

    await update.message.reply_text(
        f"🧠 *Answer:*\n{answer}\n\n📄 *Source:* `{source}`",
        parse_mode="Markdown"
    )

async def summarize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    history = USER_HISTORY.get(user_id, [])
    if not history:
        await update.message.reply_text("No recent chat to summarize.")
        return

    chat_text = "\n".join([f"User: {h['q']}\nBot: {h['a']}" for h in history])
    prompt = f"Summarize briefly:\n\n{chat_text}"

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    summary = response.text.strip()

    await update.message.reply_text(f"📝 *Summary:*\n{summary}", parse_mode="Markdown")

# =====================================
# 🚀 Main Entrypoint
# =====================================
async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("summarize", summarize))
    logging.info("🚀 Bot started…")
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())

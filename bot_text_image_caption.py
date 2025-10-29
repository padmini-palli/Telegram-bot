import os
import json
import asyncio
import logging
import numpy as np
import faiss
from dotenv import load_dotenv
from PIL import Image, ExifTags
from io import BytesIO
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from google.cloud import aiplatform
import google.generativeai as genai
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# =====================================
#  Load environment variables
# =====================================
import nest_asyncio
import asyncio
nest_asyncio.apply()
from vertexai.generative_models import GenerationConfig, GenerativeModel, SafetySetting

# ============== CONFIG =====================

PROJECT_ID = "XXX-XXX-XXXX"
LOCATION = "us-XXXXX"
EMBED_MODEL = "text-embedding-005"
GEN_MODEL = "gemini-2.5-flash"

import vertexai
#from vertexai.language_models import GenerativeModel, GenerationConfig, SafetySetting

# Initialize Vertex AI
vertexai.init(project="gpeg-camps-platform", location="us-central1")
logging.basicConfig(level=logging.INFO)

# =====================================
#  Initialize sample documents
# =====================================
DOCS = {
    "policy.txt": "Employees must submit leave requests 2 days in advance. Remote work is allowed up to 3 days a week.",
    "recipe.txt": "To make pasta: boil water, add salt, cook pasta for 8-10 minutes, then drain and mix with sauce.To make briyani: boil water, add masala, add chicken, cook for 15 mins,  add rice, cook for 10 mins, wait for 15 mins before serve",
    "faq.txt": "To reset your password, go to settings, click 'Reset Password', and follow the email instructions.",
}

os.makedirs("data", exist_ok=True)
DOC_PATH = "data/docs.json"
if not os.path.exists(DOC_PATH):
    with open(DOC_PATH, "w") as f:
        json.dump(DOCS, f)

USER_HISTORY = {}
EMBED_CACHE = {}

# =====================================
#  Embeddings + FAISS
# =====================================
def get_text_embedding(text: str):
    if text in EMBED_CACHE:
        return EMBED_CACHE[text]
    from vertexai.preview.language_models import TextEmbeddingModel
    model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
    emb = np.array(model.get_embeddings([text])[0].values, dtype="float32")
    EMBED_CACHE[text] = emb
    return emb

def build_faiss_index():
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

# =====================================
#  Gemini Text Generation
# =====================================
async def generate_answer(context_text, user_query, user_history=None):
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
    model = GenerativeModel(GEN_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()

# =====================================
#  Telegram Commands
# =====================================
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        " *Commands:*\n"
        "/ask <query> → Ask about docs\n"
        "/summarize → Summarize recent chat\n"
        "Send an image → Get local analysis (no API call)",
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

    USER_HISTORY.setdefault(user_id, []).append({"q": query, "a": answer})
    USER_HISTORY[user_id] = USER_HISTORY[user_id][-3:]

    await update.message.reply_text(
        f" *Answer:*\n{answer}\n\n📄 *Source:* `{source}`",
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
    model = GenerativeModel(GEN_MODEL)
    response = model.generate_content(prompt)
    summary = response.text.strip()
    await update.message.reply_text(f" *Summary:*\n{summary}", parse_mode="Markdown")

# =====================================
#  Local Image Analyzer (PIL)
# =====================================
# async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     photo = update.message.photo[-1]  # highest resolution
#     file = await photo.get_file()
#     img_bytes = await file.download_as_bytearray()

#     image = Image.open(BytesIO(img_bytes))
#     width, height = image.size
#     mode = image.mode
#     exif_data = {}

#     try:
#         exif = image._getexif()
#         if exif:
#             for tag, value in exif.items():
#                 decoded = ExifTags.TAGS.get(tag, tag)
#                 exif_data[decoded] = str(value)
#     except Exception:
#         pass

#     analysis = (
#         f"🖼️ *Image Analysis (Local)*\n"
#         f"- Resolution: {width}x{height}\n"
#         f"- Mode: {mode}\n"
#         f"- Format: {image.format}\n"
#     )

#     if exif_data:
#         analysis += f"- EXIF Data: {len(exif_data)} tags found\n"
#     else:
#         analysis += "- No EXIF metadata found\n"

#     await update.message.reply_text(analysis, parse_mode="Markdown")

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

async def handle_image(update, context):
    file = await update.message.photo[-1].get_file()
    file_path = "user_image.jpg"
    await file.download_to_drive(file_path)
    await update.message.reply_text("🧠 Analyzing your image...")

    image = Image.open(file_path).convert("RGB")
    with torch.no_grad():
        inputs = processor(image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(output[0], skip_special_tokens=True)

    await update.message.reply_text(f"📸 *Description:* {caption}", parse_mode="Markdown")


# =====================================
# 🚀 Main Entrypoint
# =====================================
async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("summarize", summarize))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    logging.info("🚀 Bot started…")
    await app.run_polling()

# if __name__ == "__main__":
#     asyncio.run(main())

if __name__ == "__main__":
    try:
        asyncio.run(main())
        #await main()
    except RuntimeError:
        # Fix for asyncio.run() inside interactive environment
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())

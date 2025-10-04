
# bot.py
import os
from dotenv import load_dotenv
load_dotenv()  # Carga .env en local

import asyncio
from functools import partial

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes,
    filters, AIORateLimiter
)

# ====== IMPORTA TU AGENTE ======
from agente import AgenteMultiAPI

# ====== CONFIG OBLIGATORIA ======
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Falta TELEGRAM_TOKEN en variables de entorno (.env o Render)")

# Opcionales según tus herramientas
if not os.environ.get("GOOGLE_API_KEY"):
    print("WARN: Falta GOOGLE_API_KEY; tu agente fallará si la requiere.")
if not os.environ.get("WEATHER_API_KEY"):
    print("WARN: Falta WEATHER_API_KEY si vas a usar ConsultarClima.")

# Detectar si estamos en Render (webhooks) o local (polling)
PORT = int(os.environ.get("PORT", "0"))  # Render siempre pasa PORT
BASE_URL = (
    os.environ.get("WEBHOOK_URL")  # si quieres forzar una URL
    or os.environ.get("RENDER_EXTERNAL_URL")  # Render la inyecta automáticamente
)

# ====== SESIONES POR CHAT ======
SESSIONS: dict[int, AgenteMultiAPI] = {}
FIRST_STEP_PENDING: dict[int, bool] = {}

def get_agent(chat_id: int) -> AgenteMultiAPI:
    if chat_id not in SESSIONS:
        SESSIONS[chat_id] = AgenteMultiAPI()
        FIRST_STEP_PENDING[chat_id] = True
    return SESSIONS[chat_id]

def set_first_step(chat_id: int, value: bool) -> None:
    FIRST_STEP_PENDING[chat_id] = value

def needs_first_step(chat_id: int) -> bool:
    return FIRST_STEP_PENDING.get(chat_id, True)

def split_telegram(text: str, max_len: int = 4000):
    out = []
    while len(text) > max_len:
        cut = text.rfind("\n", 0, max_len)
        if cut == -1:
            cut = max_len
        out.append(text[:cut])
        text = text[cut:]
    if text:
        out.append(text)
    return out

# ====== RESPUESTAS FIJAS ======
SALUDO_ESPECIAL = "Hola, soy tu asistente personal, para conocer mis capacidades digita 'help'"
INSTRUCCION_START = "Para comenzar digita start"
HELP_TEXT = (
    "Puedo ayudarte con:\n"
    "• Consultar clima (ConsultarClima)\n"
    "• Noticias del mercado (NoticiasWallStreet)\n"
    "• Buscar en la web (BuscarWeb)\n"
    "• Obtener fecha/hora (ObtenerFecha)\n"
    "• Hora por ciudad en tu mensaje (ObtenerFechaDesdePrompt)\n\n"
    "Tip: envía cualquier pregunta y te respondo. Escribe 'exit' para terminar."
)

# ====== HANDLERS ======
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    get_agent(chat_id)
    set_first_step(chat_id, False)
    await update.message.reply_text(SALUDO_ESPECIAL)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    get_agent(chat_id)
    await update.message.reply_text(HELP_TEXT)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in SESSIONS:
        del SESSIONS[chat_id]
    FIRST_STEP_PENDING[chat_id] = True
    await update.message.reply_text("✅ Memoria reiniciada para este chat.")

async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    low = user_text.lower()
    agent = get_agent(chat_id)

    # Salidas rápidas
    if low in {"exit", "salir", "/exit"}:
        await update.message.reply_text("Adiós, gracias por usar el asistente.")
        return
    if low in {"help", "/help"}:
        await update.message.reply_text(HELP_TEXT)
        return

    # Primer paso
    if needs_first_step(chat_id):
        if low in {"start", "star"}:
            set_first_step(chat_id, False)
            await update.message.reply_text(SALUDO_ESPECIAL)
            return
        await update.message.reply_text(INSTRUCCION_START)
        return

    # “Reinicio amable” de saludo
    if low in {"start", "star", "/start"}:
        set_first_step(chat_id, False)
        await update.message.reply_text(SALUDO_ESPECIAL)
        return

    # Llamada al agente (usa invoke con clave "input")
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    loop = asyncio.get_running_loop()
    try:
        call = partial(agent.agente.invoke, {"input": user_text})
        resp = await loop.run_in_executor(None, call)
        result = resp["output"] if isinstance(resp, dict) and "output" in resp else str(resp)
    except Exception as e:
        result = f"⚠️ Error al invocar el agente: {e}"

    for chunk in split_telegram(str(result)):
        await update.message.reply_text(chunk)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    print(f"Error en handler: {context.error}")
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("⚠️ Ocurrió un error. Intenta de nuevo.")
    except Exception:
        pass

# ====== ARRANQUE ======
def build_application() -> Application:
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .rate_limiter(AIORateLimiter())
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))
    app.add_error_handler(error_handler)
    return app

def main():
    application = build_application()

    # Modo Render (webhooks) si tenemos PORT y BASE_URL
    if PORT and BASE_URL:
        webhook_path = f"/webhook/{TELEGRAM_TOKEN}"
        webhook_url = f"{BASE_URL.rstrip('/')}{webhook_path}"
        print(f"[INFO] Usando WEBHOOK en {webhook_url} (listen 0.0.0.0:{PORT})")

        application.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=webhook_path,
            webhook_url=webhook_url,
            drop_pending_updates=True,
        )
    else:
        # Local (polling)
        print("[INFO] Modo local: long polling")
        application.run_polling(close_loop=False, drop_pending_updates=True)

if __name__ == "__main__":
    main()

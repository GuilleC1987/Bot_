
# bot.py
import os
import logging
from dotenv import load_dotenv

# En Render las env vars vienen del panel; load_dotenv() no estorba si no hay .env
load_dotenv()

import asyncio
from functools import partial

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    AIORateLimiter,
)

# ====== LOGGING ======
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("bot")

# ====== IMPORTA TU AGENTE ======
from agente import AgenteMultiAPI

# ====== CONFIG ======
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Falta TELEGRAM_TOKEN en variables de entorno.")

if not os.environ.get("GOOGLE_API_KEY"):
    log.warning("Falta GOOGLE_API_KEY; tu agente fallará si la requiere.")

# Un “pool” simple de sesiones por chat para mantener memoria por usuario
SESSIONS: dict[int, AgenteMultiAPI] = {}

# Estado por chat para el “primer paso” (True => aún no ha digitado start/star)
FIRST_STEP_PENDING: dict[int, bool] = {}


def get_agent(chat_id: int) -> AgenteMultiAPI:
    """Devuelve un agente por chat; crea uno nuevo si no existe."""
    if chat_id not in SESSIONS:
        log.info(f"Creando nueva sesión de agente para chat_id={chat_id}")
        SESSIONS[chat_id] = AgenteMultiAPI()
        FIRST_STEP_PENDING[chat_id] = True  # nuevo chat => requiere 'start'
    return SESSIONS[chat_id]


def set_first_step(chat_id: int, value: bool) -> None:
    FIRST_STEP_PENDING[chat_id] = value


def needs_first_step(chat_id: int) -> bool:
    return FIRST_STEP_PENDING.get(chat_id, True)


def split_telegram(text: str, max_len: int = 4000):
    """Telegram limita ~4096 chars. Partimos en trozos seguros."""
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
    """Comando /start: lo tratamos como si el usuario ya hubiera digitado start."""
    chat_id = update.effective_chat.id
    get_agent(chat_id)  # asegura sesión y marca primer paso True por defecto
    set_first_step(chat_id, False)
    await update.message.reply_text(SALUDO_ESPECIAL)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /help."""
    chat_id = update.effective_chat.id
    get_agent(chat_id)  # asegura sesión
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

    # Salidas
    if low in {"exit", "salir", "/exit"}:
        await update.message.reply_text("Adiós, gracias por usar el asistente.")
        return

    # Help (como texto)
    if low in {"help", "/help"}:
        await update.message.reply_text(HELP_TEXT)
        return

    # Primer mensaje → pedir 'start'
    if needs_first_step(chat_id):
        if low in {"start", "star"}:
            set_first_step(chat_id, False)
            await update.message.reply_text(SALUDO_ESPECIAL)
            return
        await update.message.reply_text(INSTRUCCION_START)
        return

    # Si escribe start/star luego, re-saludamos
    if low in {"start", "star", "/start"}:
        set_first_step(chat_id, False)
        await update.message.reply_text(SALUDO_ESPECIAL)
        return

    # ---- Llamada al LLM (invoke con clave "input") ----
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    loop = asyncio.get_running_loop()
    try:
        call = partial(agent.agente.invoke, {"input": user_text})
        resp = await loop.run_in_executor(None, call)

        result = resp["output"] if isinstance(resp, dict) and "output" in resp else str(resp)
    except Exception as e:
        log.exception("Fallo al invocar el agente")
        result = f"⚠️ Error al invocar el agente: {e}"

    for chunk in split_telegram(str(result)):
        await update.message.reply_text(chunk)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.error(f"Error en handler: {context.error}")
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("⚠️ Ocurrió un error. Intenta de nuevo.")
    except Exception:
        pass


# ====== APP ======
def main():
    log.info("Iniciando bot en modo long-polling (Render Worker)…")
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .rate_limiter(AIORateLimiter())
        .concurrent_updates(True)  # mejor throughput
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))
    app.add_error_handler(error_handler)

    # En Render no hay que abrir puertos para long-polling
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
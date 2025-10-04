# bot.py
import os
import warnings
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram.ext import Application, CommandHandler, MessageHandler, filters



if os.name == "nt":
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from dotenv import load_dotenv
load_dotenv()  # .env local

# Silenciar deprecations verbosas de LangChain
try:
    from langchain_core._api import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

import asyncio
from functools import partial
import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# AIORateLimiter
AIORateLimiter = None
try:
    from telegram.ext import AIORateLimiter as _AIORateLimiter
    AIORateLimiter = _AIORateLimiter
except Exception as e:
    logging.getLogger("bot").warning(
        f"AIORateLimiter no disponible ({e}). Continuando sin rate limiter…"
    )


from agente import AgenteMultiAPI

# ====== CONFIG ======
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Falta TELEGRAM_TOKEN en variables de entorno (.env o sistema)")

if not os.environ.get("GOOGLE_API_KEY"):
    print("WARN: Falta GOOGLE_API_KEY; tu agente fallará si la requiere.")

# Un “pool” simple de sesiones por chat para mantener memoria por usuario
SESSIONS: dict[int, AgenteMultiAPI] = {}

# Estado por chat para el “primer paso” (True => aún no ha digitado start/star)
FIRST_STEP_PENDING: dict[int, bool] = {}

def get_agent(chat_id: int) -> AgenteMultiAPI:
    """Devuelve un agente por chat; crea uno nuevo si no existe."""
    if chat_id not in SESSIONS:
        SESSIONS[chat_id] = AgenteMultiAPI()
        FIRST_STEP_PENDING[chat_id] = True  # nuevo chat inicia bloqueado
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
    "• Precio ciptomonedas más importantes (PrecioCRIPTOMonedas)\n"
    "• Buscar en la web (BuscarWeb)\n"
    "• Obtener fecha/hora (ObtenerFecha)\n"
    "• Hora por ciudad en tu mensaje (ObtenerFechaDesdePrompt)\n\n"
    "Tip: envía cualquier pregunta y te respondo. Escribe 'exit' para terminar."
)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # /, /health, /live → 200 OK
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
      
        return

def start_health_server():
    """Inicia un HTTP server simple para que Render detecte el puerto abierto."""
    port = int(os.environ.get("PORT", "10000"))  # Render inyecta PORT en web services
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    logging.getLogger("bot").info(f"Health server escuchando en 0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

# ====== HANDLERS TELEGRAM ======
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

    if low in {"exit", "salir", "/exit"}:
        await update.message.reply_text("Adiós, gracias por usar el asistente.")
        return

    if low in {"help", "/help"}:
        await update.message.reply_text(HELP_TEXT)
        return

    if needs_first_step(chat_id):
        if low in {"start", "star"}:
            set_first_step(chat_id, False)
            await update.message.reply_text(SALUDO_ESPECIAL)
            return
        await update.message.reply_text(INSTRUCCION_START)
        return

    if low in {"start", "star", "/start"}:
        set_first_step(chat_id, False)
        await update.message.reply_text(SALUDO_ESPECIAL)
        return

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
    logging.getLogger("bot").error(f"Error en handler: {context.error}")
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("⚠️ Ocurrió un error. Intenta de nuevo.")
    except Exception:
        pass

# ====== FACTORÍA DE APPLICATION ======
def build_application() -> Application:
    builder = Application.builder().token(TELEGRAM_TOKEN)
  
    if AIORateLimiter is not None:
        try:
            builder = builder.rate_limiter(AIORateLimiter())
        except Exception as e:
            logging.getLogger("bot").warning(f"No se aplicó AIORateLimiter: {e}")

    app = builder.build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))
    app.add_error_handler(error_handler)
    return app

# ====== MAIN ======
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    
    if os.environ.get("PORT"):
        th = threading.Thread(target=start_health_server, daemon=True)
        th.start()

    application = build_application()
    
    print("Bot corriendo con long polling.")
    application.run_polling(close_loop=False)

if __name__ == "__main__":
    main()

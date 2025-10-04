# bot.py
import os
import warnings
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
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
)

from agente import AgenteMultiAPI

# ---- Ajustes mínimos de entorno (evita ruido en Windows) ----
if os.name == "nt":
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# .env local (si está disponible)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Silenciar deprecations verbosas de LangChain
try:
    from langchain_core._api import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

log = logging.getLogger("bot")

# ====== CONFIG ======
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Falta TELEGRAM_TOKEN en variables de entorno (.env o sistema)")

if not os.environ.get("GOOGLE_API_KEY"):
    log.warning("WARN: Falta GOOGLE_API_KEY; tu agente fallará si la requiere.")

# Sesiones por chat (memoria por usuario)
SESSIONS: dict[int, AgenteMultiAPI] = {}
FIRST_STEP_PENDING: dict[int, bool] = {}  # True => aún no digitó 'start'

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
SALUDO_ESPECIAL = "Hola, soy tu asistente. Para ver mis capacidades, escribe 'help'."
INSTRUCCION_START = "Para comenzar digita start"
HELP_TEXT = (
    "Puedo ayudarte con:\n"
    "• Consultar clima (ConsultarClima)\n"
    "• Precio de criptomonedas (ObtenerPrecioCripto)\n"
    "• Top de criptomonedas (ObtenerTopCripto)\n"
    "• Buscar en la web (BuscarWeb)\n"
    "• Obtener fecha/hora (ObtenerFecha)\n"
    "• Hora por ciudad en tu mensaje (ObtenerFechaDesdePrompt)\n\n"
    "Tip: envía cualquier pregunta y te respondo. Escribe 'exit' para terminar."
)

# ====== MINI HEALTH SERVER PARA RENDER ======
class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"ok")
    def log_message(self, format, *args):
        return  # silencio en logs

def start_health_server():
    port = int(os.environ.get("PORT", "10000"))  # Render inyecta PORT
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    log.info(f"Health server escuchando en 0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

# ====== INDICADOR 'TYPING' REPETIDO ======
async def _typing_job(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    except Exception:
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
    # --- definir SIEMPRE chat_id/variables primero ---
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    low = user_text.lower()
    agent = get_agent(chat_id)

    # --- salidas rápidas (sin typing) ---
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

    # --- indicador typing (ya con chat_id definido) ---
    typing_job = None
    jq = getattr(context.application, "job_queue", None)
    if jq is not None:
        typing_job = jq.run_repeating(
            _typing_job,
            interval=4.0,
            first=0.0,
            data=chat_id,
            name=f"typing-{chat_id}",
        )
    else:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # --- llamada al agente ---
    try:
        loop = asyncio.get_running_loop()
        # ⬇️ ahora usamos el método de alto nivel con fallback
        result = await loop.run_in_executor(None, partial(agent.procesar_consulta, user_text))
    except Exception as e:
        result = f"⚠️ Error al invocar el agente: {e}"
    finally:
        if typing_job:
            typing_job.schedule_removal()

    # --- responde en trozos seguros ---
    for chunk in split_telegram(str(result)):
        await update.message.reply_text(chunk)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    err_text = str(context.error or "")
    log.error(f"Error en handler: {err_text}")
    # Manejo silencioso del 409 si hay otra instancia leyendo updates
    if "Conflict: terminated by other getUpdates request" in err_text:
        return
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("⚠️ Ocurrió un error. Intenta de nuevo.")
    except Exception:
        pass

# ====== FACTORÍA DE APPLICATION ======
def build_application() -> Application:
    builder = Application.builder().token(TELEGRAM_TOKEN)
    # Rate limiter opcional
    try:
        from telegram.ext import AIORateLimiter
        builder = builder.rate_limiter(AIORateLimiter())
    except Exception as e:
        log.warning(f"AIORateLimiter no disponible ({e}). Continuando sin rate limiter…")

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

    # Health server solo si Render define PORT
    if os.environ.get("PORT"):
        th = threading.Thread(target=start_health_server, daemon=True)
        th.start()

    application = build_application()

    # Evita 409 si hay dos instancias leyendo updates
    if os.environ.get("DISABLE_POLLING") == "1":
        log.info("DISABLE_POLLING=1 → no se inicia long polling (solo health server).")
        threading.Event().wait()  # mantener proceso vivo
        return

    log.info("Bot corriendo con long polling.")
    application.run_polling(
        close_loop=False,
        drop_pending_updates=True,
    )

if __name__ == "__main__":
    main()

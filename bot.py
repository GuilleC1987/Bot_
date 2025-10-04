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
    """Inicia un HTTP server simple para que Render detecte el puerto abierto."""
    port = int(os.environ.get("PORT", "10000"))  # Render inyecta PORT en web services
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
    ...
    # Inicia indicador 'typing' periódico mientras se procesa la respuesta
    typing_job = None
    jq = getattr(context.application, "job_queue", None)
    if jq is not None:
        typing_job = jq.run_repeating(
            _typing_job,
            interval=4.0,   # cada ~4 s
            first=0.0,      # enviar de inmediato
            data=chat_id,
            name=f"typing-{chat_id}",
        )
    else:
        # Fallback si no hay JobQueue disponible en esta versión/entorno
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        loop = asyncio.get_running_loop()
        call = partial(agent.agente.invoke, {"input": user_text})
        resp = await loop.run_in_executor(None, call)
        result = resp["output"] if isinstance(resp, dict) and "output" in resp else str(resp)
    except Exception as e:
        result = f"⚠️ Error al invocar el agente: {e}"
    finally:
        if typing_job:
            typing_job.schedule_removal()


    # Salidas rápidas
    if low in {"exit", "salir", "/exit"}:
        await update.message.reply_text("Adiós, gracias por usar el asistente.")
        return

    if low in {"help", "/help"}:
        await update.message.reply_text(HELP_TEXT)
        return

    # Primer mensaje de cualquier tipo -> instrucción de start
    if needs_first_step(chat_id):
        if low in {"start", "star"}:
            set_first_step(chat_id, False)
            await update.message.reply_text(SALUDO_ESPECIAL)
            return
        await update.message.reply_text(INSTRUCCION_START)
        return

    # Reinicio amable de saludo si escribe start/star otra vez
    if low in {"start", "star", "/start"}:
        set_first_step(chat_id, False)
        await update.message.reply_text(SALUDO_ESPECIAL)
        return

    # Inicia indicador 'typing' periódico mientras se procesa la respuesta
    typing_job = context.application.job_queue.run_repeating(
        _typing_job,
        interval=4.0,   # actualiza cada ~4 s
        first=0.0,      # envía inmediatamente
        data=chat_id,
        name=f"typing-{chat_id}",
    )

    try:
        loop = asyncio.get_running_loop()
        call = partial(agent.agente.invoke, {"input": user_text})
        resp = await loop.run_in_executor(None, call)
        result = resp["output"] if isinstance(resp, dict) and "output" in resp else str(resp)
    except Exception as e:
        result = f"⚠️ Error al invocar el agente: {e}"
    finally:
        # Detener el typing sí o sí
        typing_job.schedule_removal()

    for chunk in split_telegram(str(result)):
        await update.message.reply_text(chunk)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.error(f"Error en handler: {context.error}")
    # Manejo amable de conflicto 409 (dos instancias)
    err_text = str(context.error or "")
    if "Conflict: terminated by other getUpdates request" in err_text:
        # No spamear al usuario: solo loguear
        return
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("⚠️ Ocurrió un error. Intenta de nuevo.")
    except Exception:
        pass

# ====== FACTORÍA DE APPLICATION ======
def build_application() -> Application:
    builder = Application.builder().token(TELEGRAM_TOKEN)

    # Rate limiter opcional si está disponible
    try:
        from telegram.ext import AIORateLimiter as _AIORateLimiter
        builder = builder.rate_limiter(_AIORateLimiter())
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

    # SUGERENCIA para evitar 409 en despliegues paralelos:
    # - Pon DISABLE_POLLING=1 en las instancias que NO deban leer updates.
    if os.environ.get("DISABLE_POLLING") == "1":
        log.info("DISABLE_POLLING=1 → no se inicia long polling (solo health server).")
        th = threading.Event()
        th.wait()  # mantener proceso vivo
        return

    print("Bot corriendo con long polling.")
    application.run_polling(
        close_loop=False,
        drop_pending_updates=True,  # evita colas antiguas
    )

if __name__ == "__main__":
    main()

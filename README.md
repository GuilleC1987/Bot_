# Bot de Telegram — Asistente Multi‑API

Pequeño bot que integra clima, web search, cripto y acciones usando LangChain + Telegram.

## Requisitos
- Python 3.10+
- Cuenta de Telegram BotFather para obtener `TELEGRAM_TOKEN`

## Instalación rápida
```bash
git clone <tu-repo>
cd <tu-repo>
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Variables de entorno
Crea un archivo `.env` en la raíz con lo necesario:
```
TELEGRAM_TOKEN=xxxxxxxx:yyyyyyyyyyyyyyyyyyyyyyyyyyyy
GOOGLE_API_KEY=tu_clave_opcional   # si usas Gemini
WEATHER_API_KEY=tu_clave_opcional  # si usas clima
# Para Render (instancias secundarias sin polling):
# DISABLE_POLLING=1
```
> `GOOGLE_API_KEY` y `WEATHER_API_KEY` son opcionales según tus herramientas activas.

## Ejecutar (local)
```bash
python bot.py
```
Chatea con tu bot en Telegram. Comandos: `/start`, `/help`, `/reset`.

## Despliegue en Render (web service)
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python bot.py`
- Añade las variables de entorno (mínimo `TELEGRAM_TOKEN`). El bot abre un puerto (`PORT`) para health-check.

## Problemas comunes
- **409 Conflict (getUpdates):** solo una instancia debe hacer long‑polling. En las otras, pon `DISABLE_POLLING=1`.
- **ModuleNotFoundError: dotenv:** añade `python-dotenv` a `requirements.txt`.
- **Cripto rate limit:** vuelve a intentar luego; hay backoff/caché.

---
Minimal README. Ajusta según tu proyecto.

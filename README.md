# Bot Asistente Personal (Telegram) — Multi‑API

Asistente conversacional para **Telegram** con herramientas integradas de clima, finanzas/mercados, búsqueda web, fechas y **criptomonedas** (con tolerancia a rate limits). Preparado para ejecutarse **localmente** y desplegarse como **Web Service en Render** (incluye health check HTTP).

## ✨ Funcionalidades
- **Clima** (WeatherAPI): `ConsultarClima` — clima actual por ciudad.
- **Acciones** (Yahoo/alt): `PrecioAccion` — precio actual con fallbacks.
- **Criptomonedas** (CoinGecko + fallbacks): `ObtenerPrecioCripto`, `ObtenerTopCripto`.
- **Búsqueda web** (DuckDuckGo): `BuscarWeb`.
- **Fechas**: `ObtenerFecha`, `ObtenerFechaDesdePrompt` (detecta ciudad).
- Memoria por chat y agente con **LangChain + Google Generative AI (Gemini)**.

## 🗂️ Estructura (simplificada)
```
.
├─ bot.py                 # Arranque del bot, handlers, health server (Render)
├─ agente.py              # Orquestación: LLM + herramientas (LangChain)
├─ herramientas.py        # Implementación de las herramientas (clima, cripto, etc.)
├─ prompts.py             # Prompts del sistema/conversación
├─ requirements.txt
└─ .env.example           # Plantilla de variables de entorno
```

## ✅ Requisitos
- **Python 3.10+**
- **Token de Telegram** (BotFather): `TELEGRAM_TOKEN`
- **Google Generative AI** (Gemini): `GOOGLE_API_KEY`
- **WeatherAPI**: `WEATHER_API_KEY`
- (Opcional) `COINGECKO_BASE`, `PORT`, `DISABLE_POLLING`

## 🧰 Instalación local

```bash
# 1) Clonar
git clone <URL_DE_TU_REPO>.git
cd <CARPETA_DEL_REPO>

# 2) Crear y activar entorno
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3) Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

### `.env` (crear en la raíz)
Copia y edita según tus claves:
```env
TELEGRAM_TOKEN=123456:ABCDEF...      # Bot de Telegram
GOOGLE_API_KEY=AIza...               # Gemini
WEATHER_API_KEY=xxxxxxxxxxxxxxxx     # WeatherAPI
# COINGECKO_BASE=https://api.coingecko.com/api/v3   # (opcional; default)
# PORT=10000                          # Render lo inyecta automáticamente
# DISABLE_POLLING=0                  # 1 para desactivar long polling (útil en entornos con varias instancias)
```

## ▶️ Ejecutar localmente

```bash
python bot.py
```
- El bot usa **long polling** y mostrará “Bot corriendo con long polling.”
- Abre Telegram, busca tu bot y envía **/start**.

## 💬 Uso en Telegram (comandos y ejemplos)
- **/start** → saludo inicial.
- **/help** → lista de capacidades.
- **/reset** → limpia memoria de la sesión.

Ejemplos de mensajes:
- `Clima en Madrid`
- `Precio de AAPL`
- `bitcoin` · `eth vs: eur` · `top 5`
- `Búscame noticias de Tesla`
- `Qué hora es en Buenos Aires`

## ☁️ Despliegue en Render (Web Service)
1. Conecta tu repo de GitHub a Render y crea un **Web Service**.
2. **Build Command**:  
   ```bash
   pip install --upgrade pip && pip install -r requirements.txt
   ```
3. **Start Command**:  
   ```bash
   python bot.py
   ```
4. **Environment** → Variables:
   - `TELEGRAM_TOKEN`, `GOOGLE_API_KEY`, `WEATHER_API_KEY`
   - (Opcional) `DISABLE_POLLING=1` en **todas** las instancias que **no** deban leer updates.
5. El bot expone un **health server** en `PORT` (Render lo detecta).

> ⚠️ **Importante**: solo **una** instancia debe leer updates de Telegram por **long polling**.  
> Si despliegas varias instancias (o pruebas local y Render a la vez), pon `DISABLE_POLLING=1` en todas menos en la principal para evitar el error **409 Conflict**.

## 🧪 Recomendaciones
- Instalar el rate limiter opcional de PTB:  
  ```bash
  pip install "python-telegram-bot[rate-limiter]"
  ```
- `requirements.txt` debe incluir `python-dotenv` si quieres cargar `.env` localmente.
- Para logs más verbosos, exporta:
  ```bash
  export LOG_LEVEL=DEBUG
  ```

## 🛠️ Solución de problemas

**409 Conflict / getUpdates**  
- Causa: dos instancias hacen polling.  
- Solución: dejar **solo una** con `DISABLE_POLLING=0` y el resto `DISABLE_POLLING=1`. También evita ejecutar local y Render a la vez.

**CoinGecko rate limit**  
- El bot implementa backoff, caché y fallbacks (CryptoCompare/Binance).  
- Si ves: “CoinGecko está limitando…”, espera unos segundos y reintenta.

**Fallo de yfinance / Yahoo**  
- Se usan **fallbacks**: Yahoo Quote API y Stooq. Aun así, algunos símbolos/mercados pueden no estar disponibles temporalmente.

**No se ve el “typing”**  
- El bot programa un job periódico; si el `JobQueue` no está disponible en tu entorno, envía una sola acción de typing como fallback. El mensaje seguirá enviándose normalmente.

**dotenv no encontrado**  
- Asegúrate de tenerlo en `requirements.txt` e instalado: `pip install python-dotenv`.

## 👩‍💻 Desarrollo
- Memoria por chat en `bot.py` (diccionarios en memoria).
- Herramientas “string-only” registradas con LangChain en `agente.py`.
- `herramientas.py` contiene:
  - `HerramientaCripto` con **CoinGecko** + fallbacks y caché.
  - `HerramientaMercadosYahoo` con resolver de símbolos y múltiples fuentes de precio.
  - `HerramientaClima` (WeatherAPI).
  - `HerramientaBusquedaWeb` (DuckDuckGo).
  - `HerramientaFecha` (geocoding + timezone).

## 🤝 Contribuir
1. Crea un branch: `git checkout -b feat/mi-mejora`
2. Haz cambios y tests
3. Pull Request a `main`

## 📄 Licencia
MIT — úsalo libremente.

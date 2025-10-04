# herramientas.py
from __future__ import annotations

import os
import time
import re
import csv
import random
from io import StringIO
from datetime import datetime, timedelta, timezone

import requests
import certifi
import urllib3
import yfinance as yf

# ---------- TLS/CA: usar certifi (no desactivar verificación) ----------
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------- Zona horaria ----------
try:
    import zoneinfo as zinfo                     # Python 3.9+
except Exception:                                # pragma: no cover
    import backports.zoneinfo as zinfo           # pip install backports.zoneinfo

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except Exception:                                # pragma: no cover
    from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from timezonefinder import TimezoneFinder    # pip install timezonefinder
except Exception:                                # pragma: no cover
    TimezoneFinder = None

from geopy.geocoders import Nominatim

# ---------- Constantes ----------
WEATHER_BASE_URL = "http://api.weatherapi.com/v1"
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# ---------- Helpers ----------
def _fmt_epoch(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")

def _unix_from_any(x) -> int:
    try:
        return int(x)
    except Exception:
        return int(time.time())


# ============================================================
#                       FECHA / HORA
# ============================================================
class HerramientaFecha:
    """
    Obtiene fecha/hora; intenta detectar ciudad desde un prompt,
    geocodifica con Nominatim y deriva zona horaria con TimezoneFinder.
    """

    def __init__(self, user_agent: str = "herramienta-fecha/1.0", country_bias: str | None = None):
        self.geocoder = Nominatim(user_agent=user_agent, timeout=10)
        self.tzf = TimezoneFinder() if TimezoneFinder else None
        self.country_bias = country_bias
        self._nlp = None  # spaCy se carga perezosamente

        self.DEFAULT_TZ_KEY = "America/El_Salvador"
        self.DEFAULT_CITY_LABEL = "San Salvador, El Salvador"
        self.DEFAULT_FALLBACK_OFFSET = -6  # UTC-6 todo el año

    # ------------------------ Interno ------------------------
    def _ensure_nlp(self):
        if self._nlp is not None:
            return
        try:
            import spacy
            try:
                self._nlp = spacy.load("es_core_news_sm")
            except Exception:
                self._nlp = spacy.load("en_core_web_sm")
        except Exception:
            self._nlp = None

    def _extract_candidates_spacy(self, prompt: str) -> list[str]:
        self._ensure_nlp()
        if not self._nlp:
            return []
        doc = self._nlp(prompt)
        cands = [ent.text.strip() for ent in doc.ents if ent.label_ in {"GPE", "LOC"}]
        return sorted(set(cands), key=len, reverse=True)

    def _extract_candidates_regex(self, prompt: str) -> list[str]:
        pattern = r"(?:\b(en|de|para|a)\b)\s+(?:la\s+|el\s+)?([A-Za-zÁÉÍÓÚÜÑñ]+(?:\s+[A-Za-zÁÉÍÓÚÜÑñ]+){0,3})"
        hits = re.findall(pattern, prompt, flags=re.IGNORECASE)
        cands = [h[1].strip() for h in hits]
        STOP = {"hora", "tiempo", "hoy", "mañana", "ayer", "utc", "gmt"}
        cands = [c for c in cands if len(c) >= 3 and c.lower() not in STOP]
        return sorted(set(cands), key=len, reverse=True)

    def _geocode_first(self, query: str):
        try:
            kwargs = {"language": "es", "addressdetails": True}
            if self.country_bias:
                kwargs["country_codes"] = self.country_bias
            loc = self.geocoder.geocode(query, **kwargs)
            if not loc:
                return None
            if self.country_bias:
                allowed = {c.strip().lower() for c in self.country_bias.split(",") if c.strip()}
                cc = (loc.raw.get("address", {}).get("country_code") or "").lower()
                if allowed and cc not in allowed:
                    return None
            return (loc.latitude, loc.longitude, loc.address)
        except Exception:
            return None

    def _timezone_from_latlon(self, lat: float, lon: float) -> str | None:
        try:
            if not self.tzf:
                return None
            return self.tzf.timezone_at(lat=lat, lng=lon)
        except Exception:
            return None

    def _safe_zoneinfo(self, tzkey: str, fallback_offset_hours: int | None = None):
        try:
            return zinfo.ZoneInfo(tzkey)
        except Exception:
            try:
                import tzdata  # noqa: F401
            except Exception:
                pass
            try:
                return zinfo.ZoneInfo(tzkey)
            except Exception:
                if fallback_offset_hours is not None:
                    return timezone(timedelta(hours=fallback_offset_hours), name=tzkey)
                return timezone.utc

    # ------------------------ Pública ------------------------
    def obtener_fecha_actual(self, formato: str = "completo") -> str:
        ahora = datetime.now()
        formatos = {
            "completo": ahora.strftime("%A, %d de %B de %Y a las %H:%M:%S"),
            "fecha": ahora.strftime("%d/%m/%Y"),
            "hora": ahora.strftime("%H:%M:%S"),
            "iso": ahora.isoformat(),
            "simple": ahora.strftime("%Y-%m-%d %H:%M"),
        }
        return formatos.get(formato, formatos["completo"])

    def detectar_ciudad(self, prompt: str) -> dict | None:
        if not prompt or not prompt.strip():
            return None

        candidates = self._extract_candidates_spacy(prompt) or self._extract_candidates_regex(prompt)
        if not candidates:
            tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑñ]+", prompt)
            for n in (3, 2, 1):
                for i in range(len(tokens) - n + 1):
                    candidates.append(" ".join(tokens[i:i + n]))

        seen = set()
        for cand in candidates:
            if not cand or cand.lower() in seen:
                continue
            seen.add(cand.lower())
            geo = self._geocode_first(cand)
            if not geo:
                continue
            lat, lon, display_name = geo
            tz = self._timezone_from_latlon(lat, lon)
            if tz:
                return {"query": cand, "lat": lat, "lon": lon, "display_name": display_name, "timezone": tz}
        return None

    def obtener_fecha_desde_prompt(self, prompt: str, formato: str = "completo") -> str:
        info = self.detectar_ciudad(prompt) if prompt else None

        if info and info.get("timezone"):
            tzkey = info["timezone"]
            label = info.get("display_name") or info.get("query") or tzkey
            tz = self._safe_zoneinfo(tzkey)
        else:
            tzkey = self.DEFAULT_TZ_KEY
            label = self.DEFAULT_CITY_LABEL
            tz = self._safe_zoneinfo(tzkey, fallback_offset_hours=self.DEFAULT_FALLBACK_OFFSET)

        ahora = datetime.now(tz)
        formatos = {
            "completo": ahora.strftime("%A, %d de %B de %Y a las %H:%M:%S"),
            "fecha": ahora.strftime("%d/%m/%Y"),
            "hora": ahora.strftime("%H:%M:%S"),
            "iso": ahora.isoformat(),
            "simple": ahora.strftime("%Y-%m-%d %H:%M"),
        }
        base = formatos.get(formato, formatos["completo"])
        return f"{base} — {label} ({tzkey})"


# ============================================================
#                         CLIMA
# ============================================================
class HerramientaClima:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = WEATHER_BASE_URL

    def obtener_clima_actual(self, ciudad: str) -> str:
        try:
            url = f"{self.base_url}/current.json"
            params = {"key": self.api_key, "q": ciudad, "lang": "es"}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            location = data["location"]
            current = data["current"]

            clima_info = (
                f"Clima en {location['name']}, {location['country']}\n"
                f"Temperatura: {current['temp_c']}°C (se siente como {current['feelslike_c']}°C)\n"
                f"Condición: {current['condition']['text']}\n"
                f"Viento: {current['wind_kph']} km/h\n"
                f"Humedad: {current['humidity']}%\n"
                f"Visibilidad: {current['vis_km']} km\n"
                f"Última actualización: {current['last_updated']}"
            )
            return clima_info

        except Exception as e:
            return f"Error al obtener el clima: {str(e)}"


# ============================================================
#                   MERCADOS (Yahoo/Stooq)
# ============================================================
class HerramientaAccionesStooq:
    """
    Precio de acciones usando Stooq (CSV público, sin API key).
    - No depende de Yahoo ni yfinance.
    - Convierte tickers US a formato 'aapl.us', 'tsla.us', etc.
    - Incluye un pequeño diccionario de nombres comunes -> ticker.
    """

    # Nombres frecuentes -> ticker
    SEED_TICKERS = {
        "TESLA": "TSLA", "TESLA INC": "TSLA", "TESLA, INC.": "TSLA",
        "APPLE": "AAPL", "APPLE INC": "AAPL",
        "ALPHABET": "GOOGL", "GOOGLE": "GOOGL",
        "MICROSOFT": "MSFT", "AMAZON": "AMZN",
        "META": "META", "FACEBOOK": "META",
        "NVIDIA": "NVDA", "NETFLIX": "NFLX",
        "BERKSHIRE": "BRK-B", "BRK": "BRK-B", "BRK.B": "BRK-B",
        "SP500": "SPY", "S&P500": "SPY", "S&P 500": "SPY",
    }

    @staticmethod
    def _sanitize(q: str) -> str:
        import re
        return re.sub(r"[^A-Za-z0-9\-. ]+", "", (q or "").strip()).upper()

    def _name_to_ticker(self, q: str) -> str | None:
        u = self._sanitize(q)
        return self.SEED_TICKERS.get(u)

    def _to_stooq_symbol(self, ticker: str) -> str:
        """
        Stooq usa minúsculas y sufijos de mercado:
        - Acciones US:  aapl.us, tsla.us, googl.us, spy.us
        - Reemplaza '.' por '-' (ej. BRK.B -> brk-b.us)
        """
        t = (ticker or "").strip().lower().replace(".", "-")
        if not t:
            return ""
        # si no viene sufijo, asumimos US
        if not t.endswith(".us") and not t.endswith(".pl") and not t.endswith(".de") and not t.endswith(".uk"):
            t = f"{t}.us"
        return t

    def _fetch_csv_quote(self, stooq_symbol: str) -> dict | None:
        """
        Llama a: https://stooq.com/q/l/?s=<sym>&f=sd2t2ohlcv&h&e=csv
        Devuelve dict con c/o/h/l/pc y ts si hay datos, None si no hay.
        """
        url = f"https://stooq.com/q/l/?s={stooq_symbol}&f=sd2t2ohlcv&h&e=csv"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        rows = list(csv.DictReader(StringIO(r.text)))
        if not rows:
            return None
        row = rows[0]
        # Stooq pone 'N/A' si no encuentra
        if (row.get("Close") or "").upper() in {"N/A", "N/D", ""}:
            return None

        def f(x):
            try:
                return float(x)
            except Exception:
                return None

        # Nota: Stooq no trae "previous close" directo aquí → lo dejamos N/D
        return {
            "c": f(row.get("Close")),
            "o": f(row.get("Open")),
            "h": f(row.get("High")),
            "l": f(row.get("Low")),
            "pc": None,
            "t": int(datetime.now(timezone.utc).timestamp()),
        }

    def obtener_precio_accion(self, symbol_o_nombre: str) -> str:
        """
        Acepta nombre o ticker:
         - 'Tesla' -> TSLA -> tsla.us
         - 'TSLA'  -> tsla.us
         - 'AAPL'  -> aapl.us
        """
        q = self._sanitize(symbol_o_nombre)
        if not q:
            return "Debes indicar un símbolo o nombre (ej. AAPL, Apple)."

        # 1) si ya parece ticker (AAPL/TSLA/BRK-B), úsalo; si es nombre, intenta mapa
        ticker = q if (len(q) <= 6 or "-" in q or "." in q) else self._name_to_ticker(q)
        if not ticker:
            # también probamos algunos aliases comunes
            ticker = self._name_to_ticker(q) or q

        sym = self._to_stooq_symbol(ticker)
        if not sym:
            return f"No pude interpretar '{symbol_o_nombre}'."

        try:
            quote = self._fetch_csv_quote(sym)
        except Exception:
            quote = None

        if not quote or quote.get("c") is None:
            return f"No pude obtener el precio de {ticker} en este momento."

        ts = quote["t"]
        def fmt_epoch(ts_: int):
            return datetime.fromtimestamp(int(ts_), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        return (
            f"=== {ticker.upper()} ===\n"
            f"Precio actual: {quote['c']}\n"
            f"Apertura:     {quote['o'] if quote['o'] is not None else 'N/D'}\n"
            f"Máximo día:   {quote['h'] if quote['h'] is not None else 'N/D'}\n"
            f"Mínimo día:   {quote['l'] if quote['l'] is not None else 'N/D'}\n"
            f"Cierre prev.: {quote['pc'] if quote['pc'] is not None else 'N/D'}\n"
            f"Hora:         {fmt_epoch(ts)}"
        )
# ============================================================
#                       BÚSQUEDA WEB
# ============================================================
class HerramientaBusquedaWeb:
    def __init__(self):
        self.base_url = "https://api.duckduckgo.com/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }

    def buscar_web(self, consulta: str, cantidad: int = 3) -> str:
        try:
            params = {"q": consulta, "format": "json", "no_html": 1, "skip_disambig": 1}
            response = requests.get(self.base_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            resultados_texto = f"Resultados de búsqueda para '{consulta}':\n\n"
            if data.get("Answer"):
                resultados_texto += f"Respuesta directa: {data['Answer']}\n\n"

            if data.get("AbstractText"):
                resultados_texto += f"Resumen: {data['AbstractText']}\n\n"
                if data.get("AbstractSource"):
                    resultados_texto += f"Fuente: {data['AbstractSource']}\n\n"

            if data.get("RelatedTopics"):
                resultados_texto += "Temas relacionados:\n\n"
                for i, tema in enumerate(data["RelatedTopics"][:cantidad], 1):
                    if isinstance(tema, dict) and tema.get("Text"):
                        texto = tema["Text"][:200] + "..." if len(tema["Text"]) > 200 else tema["Text"]
                        resultados_texto += f"{i}. {texto}\n\n"
                resultados_texto += "\n"

            if data.get("Definition"):
                resultados_texto += f"Definición: {data['Definition']}\n\n"
                if data.get("DefinitionSource"):
                    resultados_texto += f"Fuente: {data['DefinitionSource']}\n\n"

            if not any([data.get("Answer"), data.get("AbstractText"), data.get("RelatedTopics"), data.get("Definition")]):
                return self._busqueda_alternativa(consulta)

            return resultados_texto.strip()
        except Exception as e:
            return f"Error al buscar en la web: {str(e)}"

    def _busqueda_alternativa(self, consulta: str) -> str:
        try:
            url = "https://duckduckgo.com/html/"
            params = {"q": consulta}
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                content = response.text
                titles = re.findall(r'<a[^>]*class="result__a"[^>]*>([^<]+)</a>', content)
                if titles:
                    resultado = f"Resultados de búsqueda para '{consulta}':\n\n"
                    for i, title in enumerate(titles[:3], 1):
                        resultado += f"{i}. {title}\n\n"
                    return resultado.strip()
            return f"Búsqueda realizada para '{consulta}'. No se encontraron resultados."
        except Exception as e:
            return f"Error al buscar en la web: {str(e)}"


# ============================================================
#                        CRIPTO (CoinGecko)
# ============================================================
class HerramientaCripto:
    """
    Precio de criptomonedas usando CoinGecko con tolerancia a fallos.
    - Backoff ante 429/5xx y “cortacircuito” temporal global.
    - Caché en memoria (TTL) de precios.
    - Fallbacks: CryptoCompare (sin API key) y Binance (si existe par USDT).
    """
    BASE = os.environ.get("COINGECKO_BASE", COINGECKO_BASE_URL)

    def __init__(self, price_ttl: int = 30):
        self.sess = requests.Session()
        self.sess.headers.update({"User-Agent": "MiAgente/1.0 (+https://example.local)"})
        self._ids_cache = {"ts": 0, "maps": None}   # refresca cada 6h
        self._price_cache: dict[tuple[str, str], dict] = {}  # (coin_key, vs) -> {ts, data}
        self.PRICE_TTL = max(5, int(price_ttl))
        self._gecko_block_until = 0.0

    # ----- Semillas mínimas para resolver símbolos comunes -----
    SEED_COINS = {
        "bitcoin": ("BTC", "Bitcoin"),
        "ethereum": ("ETH", "Ethereum"),
        "tether": ("USDT", "Tether"),
        "binancecoin": ("BNB", "BNB"),
        "solana": ("SOL", "Solana"),
        "ripple": ("XRP", "XRP"),
        "usd-coin": ("USDC", "USD Coin"),
        "cardano": ("ADA", "Cardano"),
        "dogecoin": ("DOGE", "Dogecoin"),
        "tron": ("TRX", "TRON"),
        "polkadot": ("DOT", "Polkadot"),
        "litecoin": ("LTC", "Litecoin"),
        "matic-network": ("MATIC", "Polygon"),
        "bitcoin-cash": ("BCH", "Bitcoin Cash"),
        "chainlink": ("LINK", "Chainlink"),
        "stellar": ("XLM", "Stellar"),
        "uniswap": ("UNI", "Uniswap"),
        "cosmos": ("ATOM", "Cosmos"),
        "monero": ("XMR", "Monero"),
        "algorand": ("ALGO", "Algorand"),
    }

    def _seed_maps(self):
        by_id = set(self.SEED_COINS.keys())
        by_symbol, by_name, id_to_symbol = {}, {}, {}
        for cid, (sym, name) in self.SEED_COINS.items():
            id_to_symbol[cid] = sym.upper()
            by_symbol.setdefault(sym.lower(), []).append(cid)
            by_name.setdefault(name.lower(), []).append(cid)
        return {"by_id": by_id, "by_symbol": by_symbol, "by_name": by_name, "id_to_symbol": id_to_symbol}

    # ----------------------- Backoff HTTP -----------------------
    def _sleep_backoff(self, attempt: int, retry_after: str | None) -> None:
        base = float(retry_after) if retry_after else min(2 ** attempt, 10)
        time.sleep(base + random.uniform(0, 0.5))

    def _get(self, path: str, params: dict | None = None, max_attempts: int = 4):
        url = f"{self.BASE}{path}"
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                r = self.sess.get(url, params=params or {}, timeout=12)
            except requests.RequestException as e:
                last_exc = e
                if attempt < max_attempts:
                    self._sleep_backoff(attempt, None)
                    continue
                break

            if r.status_code == 429 or r.status_code >= 500:
                retry_after = r.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 60.0
                self._gecko_block_until = time.time() + delay
                if attempt < max_attempts:
                    self._sleep_backoff(attempt, retry_after)
                    continue
                last_exc = requests.RequestException(f"HTTP {r.status_code} CoinGecko (rate/server)")
                break

            try:
                return r.json()
            except ValueError as e:
                last_exc = e
                if attempt < max_attempts:
                    self._sleep_backoff(attempt, None)
                    continue
                break

        raise requests.RequestException("Rate limit o error persistente de CoinGecko.") from last_exc

    # ------------------- Resolución de IDs -------------------
    def _load_ids(self, force: bool = False):
        now = time.time()
        if now < self._gecko_block_until and not force:
            if not self._ids_cache["maps"]:
                self._ids_cache = {"ts": now, "maps": self._seed_maps()}
            return self._ids_cache["maps"]

        if (not force) and self._ids_cache["maps"] and (now - self._ids_cache["ts"] < 6 * 3600):
            return self._ids_cache["maps"]

        try:
            data = self._get("/coins/list", {"include_platform": "false"})
            maps = {"by_id": set(), "by_symbol": {}, "by_name": {}, "id_to_symbol": {}}
            for it in data:
                cid = it["id"]
                sym = (it.get("symbol") or "").lower()
                name = (it.get("name") or "").lower()
                maps["by_id"].add(cid)
                if sym:
                    maps["by_symbol"].setdefault(sym, []).append(cid)
                if name:
                    maps["by_name"].setdefault(name, []).append(cid)
                if cid and sym:
                    maps["id_to_symbol"][cid] = sym.upper()
            self._ids_cache = {"ts": now, "maps": maps}
            return maps
        except requests.RequestException:
            self._gecko_block_until = now + 60.0
            maps = self._seed_maps()
            self._ids_cache = {"ts": now, "maps": maps}
            return maps

    def _resolve_id(self, q: str) -> tuple[str | None, str | None]:
        q = (q or "").strip().lower().lstrip("$")
        if not q:
            return None, None
        maps = self._load_ids(force=False)
        if q in maps["by_id"]:
            return q, maps["id_to_symbol"].get(q)
        if q in maps["by_symbol"]:
            c = maps["by_symbol"][q][0]
            return c, maps["id_to_symbol"].get(c)
        if q in maps["by_name"]:
            c = maps["by_name"][q][0]
            return c, maps["id_to_symbol"].get(c)
        for name, lst in maps["by_name"].items():
            if q in name:
                c = lst[0]
                return c, maps["id_to_symbol"].get(c)
        sym_guess = q.upper() if q.isalpha() and 2 <= len(q) <= 6 else None
        return None, sym_guess

    # ----------------------- Fallbacks -----------------------
    def _fallback_cc_price(self, symbol_upper: str, vs: str) -> dict | None:
        try:
            url = "https://min-api.cryptocompare.com/data/price"
            r = self.sess.get(url, params={"fsym": symbol_upper, "tsyms": vs.upper()}, timeout=10)
            r.raise_for_status()
            js = r.json()
            val = js.get(vs.upper())
            if isinstance(val, (int, float)):
                return {"price": float(val), "chg": None}
        except Exception:
            pass
        return None

    def _fallback_binance_price(self, symbol_upper: str, vs: str) -> dict | None:
        try:
            if vs.lower() not in {"usd", "usdt"}:
                return None
            pair = f"{symbol_upper}USDT"
            url = "https://api.binance.com/api/v3/ticker/price"
            r = self.sess.get(url, params={"symbol": pair}, timeout=8)
            r.raise_for_status()
            js = r.json()
            px = js.get("price")
            if px is not None:
                return {"price": float(px), "chg": None}
        except Exception:
            pass
        return None

    # ------------------------- Pública -------------------------
    def obtener_precio_cripto(self, consulta: str, vs: str = "usd") -> str:
        s = (consulta or "").strip()
        parts = s.split()
        m = re.search(r"\bvs\s*[:=]\s*([a-zA-Z]{2,5})\b", s, re.I)
        if m:
            vs = m.group(1).lower()
            coin = re.sub(r"\bvs\s*[:=]\s*[a-zA-Z]{2,5}\b", "", s, flags=re.I).strip()
        elif len(parts) >= 2 and parts[-1].isalpha() and len(parts[-1]) <= 5:
            vs = parts[-1].lower()
            coin = " ".join(parts[:-1]).strip()
        else:
            coin = s

        cid, sym_upper = self._resolve_id(coin)
        if not cid and not sym_upper:
            return f"No pude reconocer la cripto '{consulta}'. Prueba con 'bitcoin', 'btc', 'ethereum', 'eth'."

        key = ((cid or f"sym:{sym_upper}"), vs.lower())
        now = time.time()
        hit = self._price_cache.get(key)
        if hit and now - hit["ts"] < self.PRICE_TTL:
            p = hit["data"]
        else:
            p = None
            gecko_err = None

            if cid and now >= self._gecko_block_until:
                try:
                    js = self._get("/simple/price", {"ids": cid, "vs_currencies": vs, "include_24hr_change": "true"})
                    if isinstance(js, dict) and cid in js and vs in js[cid]:
                        p = {"price": js[cid][vs], "chg": js[cid].get(f"{vs}_24h_change")}
                except requests.RequestException as e:
                    gecko_err = e

            sym_try = sym_upper
            if not sym_try and cid:
                maps = self._ids_cache["maps"] or self._seed_maps()
                sym_try = (maps.get("id_to_symbol") or {}).get(cid)

            if p is None and sym_try:
                p = self._fallback_cc_price(sym_try, vs)
            if p is None and sym_try:
                p = self._fallback_binance_price(sym_try, vs)

            if p is None:
                if gecko_err or (now < self._gecko_block_until):
                    return "CoinGecko está limitando o con error ahora mismo y los proveedores de respaldo no respondieron. Intenta más tarde."
                return "No pude obtener el precio ahora mismo. Intenta más tarde."

            self._price_cache[key] = {"ts": now, "data": p}

        chg = p.get("chg")
        chg_txt = f"{chg:.2f}%" if isinstance(chg, (int, float)) else "N/D"
        display = (cid.capitalize() if cid else (sym_upper or "Cripto"))
        return f"{display} → {p['price']} {vs.upper()} (24h: {chg_txt})"

    def _tool_cripto_top(self, entrada: str) -> str:
        n = 10
        s = (entrada or "").strip()
        if s:
            m = re.search(r"(\d+)", s)
            if m:
                try:
                    n = max(1, min(int(m.group(1)), 50))
                except Exception:
                    n = 10
        # Llama al método correcto (con 's' y argumento 'n')
        return self.herramienta_cripto.obtener_top_criptos(n=n, vs="usd")


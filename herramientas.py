import os
import time
import random

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

def _ensure_ca_bundle_safe_path():
    try:
        import certifi, shutil
        cafile = certifi.where()
        # Si la ruta tiene acentos o espacios, copia a C:\certs\cacert.pem
        if any(ord(ch) > 127 for ch in cafile) or " " in cafile:
            target = r"C:\certs\cacert.pem"
            os.makedirs(r"C:\certs", exist_ok=True)
            try:
                shutil.copyfile(cafile, target)
            except PermissionError:
                if os.path.exists(target):
                    os.remove(target)
                shutil.copyfile(cafile, target)
            os.environ["SSL_CERT_FILE"] = target
            os.environ["REQUESTS_CA_BUNDLE"] = target
            os.environ["CURL_CA_BUNDLE"] = target
        else:
            os.environ.setdefault("SSL_CERT_FILE", cafile)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
            os.environ.setdefault("CURL_CA_BUNDLE", cafile)
    except Exception:
        pass

_ensure_ca_bundle_safe_path()



import requests
import os
from datetime import datetime, timedelta, timezone
import yfinance as yf
import time  # <- antes tenías "import time, requests" (requests ya está importado)
from bs4 import BeautifulSoup

import os
import ssl
import certifi
import urllib3

# ✅ SOLUCIÓN SSL - Configurar certificados correctamente
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Configurar SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Deshabilitar warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configurar yfinance para usar requests session con SSL verificado
import yfinance as yf
import requests

# Crear session con configuración SSL
session = requests.Session()
session.verify = certifi.where()  # Usar certificados de certifi


# Usa el módulo con alias para evitar confusiones
try:
    import zoneinfo as zinfo            # Python 3.9+
except Exception:
    import backports.zoneinfo as zinfo  # pip install backports.zoneinfo

# Opcional en Windows: provee base IANA
try:
    import tzdata                       # pip install tzdata
except Exception:
    pass

import re
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError            # Python 3.9+
except Exception:
    from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # pip install backports.zoneinfo

try:
    from timezonefinder import TimezoneFinder    # pip install timezonefinder
except Exception:
    TimezoneFinder = None

from geopy.geocoders import Nominatim

# ✅ AGREGADO: URLs base para APIs externas
WEATHER_BASE_URL = "http://api.weatherapi.com/v1"
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"   


# --- helpers opcionales compartidos ---
def _fmt_epoch(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")

def _unix_from_any(x) -> int:
    # yfinance devuelve providerPublishTime (int) o published_at
    try:
        return int(x)
    except Exception:
        return int(time.time())

class HerramientaFecha:
    """
    Herramienta para obtener fecha y hora actual, detectando ciudad del prompt:
      - spaCy (opcional) para NER (GPE/LOC)
      - Geopy (Nominatim) para geocodificar
      - TimezoneFinder para derivar la zona horaria desde lat/lon
    """


    def __init__(self, user_agent: str = "herramienta-fecha/1.0", country_bias: str | None = None):
        """
        Args:
            user_agent: cadena requerida por Nominatim
            country_bias: restringe búsqueda por país (códigos ISO2 separados por coma, ej: 'sv,mx,es').
                          Si None, no restringe.
        """
        self.geocoder = Nominatim(user_agent=user_agent, timeout=10)
        self.tzf = TimezoneFinder() if TimezoneFinder else None
        self.country_bias = country_bias
        self._nlp = None  # spaCy se carga perezosamente

        self.DEFAULT_TZ_KEY = "America/El_Salvador"
        self.DEFAULT_CITY_LABEL = "San Salvador, El Salvador"
        self.DEFAULT_FALLBACK_OFFSET = -6  # ES no usa DST, UTC-6 todo el año

    # ------------------------
    # Utilidades internas
    # ------------------------
    def _ensure_nlp(self):
        if self._nlp is not None:
            return
        try:
            import spacy
            try:
                self._nlp = spacy.load("es_core_news_sm")
            except Exception:
                # Fallback a inglés si el modelo ES no está instalado
                self._nlp = spacy.load("en_core_web_sm")
        except Exception:
            self._nlp = None  # Seguimos sin NER; usaremos regex/heurísticas

    def _extract_candidates_spacy(self, prompt: str) -> list[str]:
        """Extrae posibles lugares usando NER (GPE/LOC) si spaCy está disponible."""
        self._ensure_nlp()
        if not self._nlp:
            return []
        doc = self._nlp(prompt)
        cands = [ent.text.strip() for ent in doc.ents if ent.label_ in {"GPE", "LOC"}]
        # Orden de mayor a menor longitud para preferir frases más completas
        return sorted(set(cands), key=len, reverse=True)

    def _extract_candidates_regex(self, prompt: str) -> list[str]:
        """
        Heurística sin NER: solo acepta patrones tipo 'en X', 'de X', 'para X'
        donde X = 1 a 4 tokens alfabéticos. Evita palabras sueltas capitalizadas.
        """
        pattern = r"(?:\b(en|de|para|a)\b)\s+(?:la\s+|el\s+)?([A-Za-zÁÉÍÓÚÜÑñ]+(?:\s+[A-Za-zÁÉÍÓÚÜÑñ]+){0,3})"
        hits = re.findall(pattern, prompt, flags=re.IGNORECASE)
        # Nos quedamos solo con el grupo de lugar (índice 1)
        cands = [h[1].strip() for h in hits]

    # Filtrado básico para reducir ruido
        STOP = {"hora", "tiempo", "hoy", "mañana", "ayer", "utc", "gmt"}
        cands = [c for c in cands if len(c) >= 3 and c.lower() not in STOP]

    # Unificar y ordenar por longitud (mayor primero)
        cands = sorted(set(cands), key=len, reverse=True)
        return cands

    def _geocode_first(self, query: str):
        """Geocodifica un string y devuelve (lat, lon, display_name) o None."""
        try:
            kwargs = {"language": "es", "addressdetails": True}
            if self.country_bias:
                kwargs["country_codes"] = self.country_bias  # ej: 'sv' o 'sv,gt,hn'
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
    # ------------------------
    # API pública
    # ------------------------
    def obtener_fecha_actual(self, formato: str = "completo") -> str:
        """
        Igual que tu versión original: hora local del servidor.
        formato: 'completo', 'fecha', 'hora', 'iso', 'simple'
        """
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
        """
        Intenta detectar una ciudad/lugar en el prompt (sin mapeo).
        Retorna {'query': <texto_detectado>, 'lat': float, 'lon': float, 'display_name': str, 'timezone': str}
        o None si no logra detectarla.
        """
        if not prompt or not prompt.strip():
            return None

        # 1) spaCy (si está) → candidatos
        candidates = self._extract_candidates_spacy(prompt)

        # 2) Fallback regex/heurístico
        if not candidates:
            candidates = self._extract_candidates_regex(prompt)

        # 3) Último recurso: intentar geocodificar fragmentos 1–3 tokens desde todo el prompt
        if not candidates:
            tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑñ]+", prompt)
            # Generamos n-gramas grandes primero
            for n in (3, 2, 1):
                for i in range(len(tokens) - n + 1):
                    candidates.append(" ".join(tokens[i:i+n]))

        # Geocodificar en orden de preferencia
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
                return {
                    "query": cand,
                    "lat": lat,
                    "lon": lon,
                    "display_name": display_name,
                    "timezone": tz
                }
        return None

    def obtener_fecha_desde_prompt(self, prompt: str, formato: str = "completo") -> str:
        """
        Si detecta ciudad en el prompt, usa su zona horaria.
        Si NO detecta, DEFAULT: San Salvador, El Salvador.
        Siempre devuelve hora + ciudad (y tz).
        """
        info = self.detectar_ciudad(prompt) if prompt else None

        if info and info.get("timezone"):
            tzkey = info["timezone"]
            label = info.get("display_name") or info.get("query") or tzkey
            # Para zonas distintas a la default no usamos offset fijo (por si tienen DST)
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

    def _safe_zoneinfo(self, tzkey: str, fallback_offset_hours: int | None = None):
        """
        Devuelve zinfo.ZoneInfo(tzkey). Si no hay base IANA en el sistema,
        intenta (re)cargar tzdata y vuelve a intentar. Si aún falla, usa offset fijo o UTC.
        """
        try:
            return zinfo.ZoneInfo(tzkey)
        except Exception:
        # Intentar nuevamente tras asegurar tzdata (en Windows)
            try:
                import tzdata  # no pasa nada si ya estaba importado
            except Exception:
                pass
            try:
                return zinfo.ZoneInfo(tzkey)
            except Exception:
                if fallback_offset_hours is not None:
                    return timezone(timedelta(hours=fallback_offset_hours), name=tzkey)
                return timezone.utc

# ✅ AGREGADO: Clase completa para manejo de API de clima
class HerramientaClima:
    """
    ✅ NUEVA CLASE: Herramienta para obtener información meteorológica
    Integra con WeatherAPI para consultas de clima en tiempo real
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = WEATHER_BASE_URL
    
    # ✅ AGREGADO: Método principal para consultar clima actual
    def obtener_clima_actual(self, ciudad:str) -> str:
        """
        ✅ NUEVO MÉTODO: Obtiene información meteorológica actual de una ciudad
        Parámetros: ciudad (str) - Nombre de la ciudad a consultar
        Retorna: str - Información formateada del clima o mensaje de error
        """
        try:
            url = f"{self.base_url}/current.json"
            params = {
                "key": self.api_key,
                "q": ciudad,
                "lang": "es"  # ✅ AGREGADO: Respuestas en español
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            # ✅ AGREGADO: Extracción y formateo de datos meteorológicos
            location = data["location"]
            current = data["current"]
            clima_info = f"""
            Clima en {location['name']}, {location["country"]}
            Temperatura: {current["temp_c"]}°C (se siente como {current["feelslike_c"]}°C)
            Condicion: {current["condition"]["text"]}
            Viento: {current["wind_kph"]} km/h
            Humedad: {current["humidity"]}%
            Visibilidad: {current["vis_km"]} km
            Ultima actualizacion: {current["last_updated"]}
            """.strip()
            return clima_info
        except Exception as e:
            return f"Error al obtener el clima: {str(e)}"


# ✅ AGREGADO: Clase completa para manejo de API de noticias financieras
class HerramientaMercadosYahoo:
    def __init__(self):
        import os
        self._sess = requests.Session()
        self._sess.headers.update({"User-Agent": "Mozilla/5.0"})

        # Si configuraste un CA custom en el entorno, úsalo SOLO para nuestras requests manuales (resolver).
        ca = (os.environ.get("REQUESTS_CA_BUNDLE")
              or os.environ.get("SSL_CERT_FILE")
              or os.environ.get("CURL_CA_BUNDLE"))
        if ca and os.path.exists(ca):
            self._sess.verify = ca

    @staticmethod
    def _sanitize(q: str) -> str:
        q = (q or "").strip()
        if q.startswith("$"):  # "$TSLA" -> "TSLA"
            q = q[1:]
        q = q.replace("–", "-").replace("—", "-")
        if "." in q and "-" not in q:  # BRK.B -> BRK-B
            q = q.replace(".", "-")
        return re.sub(r"[^A-Za-z0-9\-. ]+", "", q).strip()

    @staticmethod
    def _fmt_epoch(ts: int) -> str:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def _resolver_yahoo_simbolo(self, query: str) -> str | None:
        """
        Resuelve texto libre a ticker (ej. 'Tesla'/'$TESLA' -> 'TSLA') sin mapeos manuales.
        1) query2.finance.yahoo.com/v1/finance/search
        2) autoc.finance.yahoo.com/autoc (fallback)
        3) si parece ticker, úsalo tal cual (saneado)
        """
        q_raw = (query or "").strip()
        if not q_raw:
            return None
        q = self._sanitize(q_raw)
        allowed_exchs = {"NASDAQ", "NASDAQGS", "NYSE", "NMS", "NYQ", "NCM", "NSQ"}

        # --- 1) Buscador principal ---
        try:
            resp = self._sess.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": q, "quotesCount": 20, "newsCount": 0},
                timeout=12,
            )
            resp.raise_for_status()
            js = resp.json()
            quotes = js.get("quotes") or []
            if quotes:
                equities    = [it for it in quotes if (it.get("quoteType", "").upper() == "EQUITY")]
                equities_us = [it for it in equities if (it.get("exchDisp", "").upper() in allowed_exchs)]
                pool = equities_us or equities or quotes

                def score(it):
                    sc   = 0
                    name = (it.get("shortname") or it.get("longname") or "").lower()
                    sym  = (it.get("symbol") or "").upper()
                    qt   = (it.get("quoteType") or "").upper()
                    exch = (it.get("exchDisp") or "").upper()
                    for tok in q_raw.lower().split():
                        if tok and tok in name:
                            sc += 6
                    if qt == "EQUITY": sc += 10
                    if exch in allowed_exchs: sc += 6
                    if len(sym) <= 6: sc += 1
                    if sym == q.upper(): sc += 2
                    if qt and qt != "EQUITY": sc -= 10
                    return sc

                best = max(pool, key=score)
                sym = (best.get("symbol") or "").upper().replace(".", "-")
                if sym:
                    return sym
        except Exception:
            pass

        # --- 2) Fallback: endpoint alterno 'autoc' ---
        try:
            resp = self._sess.get(
                "https://autoc.finance.yahoo.com/autoc",
                params={"query": q, "region": 1, "lang": "en"},
                headers={"Accept": "application/json"},
                timeout=12,
            )
            resp.raise_for_status()
            js = resp.json()
            results = (js.get("ResultSet") or {}).get("Result") or []
            if results:
                def score2(r):
                    sc  = 0
                    sym = (r.get("symbol") or "").upper()
                    nam = (r.get("name") or "").lower()
                    exch= (r.get("exchDisp") or r.get("exch") or "").upper()
                    typ = (r.get("type") or r.get("typeDisp") or "")
                    for tok in q_raw.lower().split():
                        if tok and tok in nam:
                            sc += 6
                    if typ in ("S", "Equity"): sc += 10
                    if exch in allowed_exchs: sc += 6
                    if len(sym) <= 6: sc += 1
                    if sym == q.upper(): sc += 2
                    if typ and typ not in ("S", "Equity"): sc -= 10
                    return sc

                best = max(results, key=score2)
                sym = (best.get("symbol") or "").upper().replace(".", "-")
                if sym:
                    return sym
        except Exception:
            pass

        # --- 3) Fallback final ---
        q_up = q.upper()
        if q_up and " " not in q_up and all(ch.isalnum() or ch in {"-", "."} for ch in q_up):
            return q_up.replace(".", "-")
        return None

    def obtener_precio_accion(self, symbol_o_nombre: str) -> str:
        """
        Acepta nombre o ticker. Resuelve con Yahoo search (sin mapeos) y usa yfinance.
        ¡OJO! No pasamos session a yfinance (usa curl_cffi interno).
        """
        q = self._sanitize(symbol_o_nombre)
        sym = self._resolver_yahoo_simbolo(q)
        if not sym:
            return f"No pude resolver el símbolo para '{symbol_o_nombre}'. Prueba con el ticker (p. ej., TSLA)."

        # 👇 sin session
        t = yf.Ticker(sym)

        # 1) fast_info
        c = o = h = l = pc = None
        try:
            fi = getattr(t, "fast_info", None)
            if fi:
                c  = getattr(fi, "last_price", None)
                o  = getattr(fi, "open", None)
                h  = getattr(fi, "day_high", None)
                l  = getattr(fi, "day_low", None)
                pc = getattr(fi, "previous_close", None)
        except Exception:
            pass

        # 2) fallback history
        if c is None or o is None or h is None or l is None:
            try:
                df = t.history(period="1d", interval="1m")
                if df is None or df.empty:
                    df = t.history(period="5d", interval="1d")
                if df is not None and not df.empty:
                    last = df.iloc[-1]
                    if "Close" in last and c is None: c = float(last["Close"])
                    if "Open"  in last and o is None: o = float(last["Open"])
                    if "High" in df and h is None:    h = float(df["High"].max())
                    if "Low"  in df and l is None:    l = float(df["Low"].min())
                    if pc is None and len(df) >= 2 and "Close" in df:
                        try: pc = float(df["Close"].iloc[-2])
                        except Exception: pass
            except Exception:
                pass

        if c is None:
            return f"No hay datos de precio para {sym}. (¿símbolo correcto?)"

        ts = int(time.time())
        return (
            f"=== {sym} ===\n"
            f"Precio actual: {c}\n"
            f"Apertura:     {o if o is not None else 'N/D'}\n"
            f"Máximo día:   {h if h is not None else 'N/D'}\n"
            f"Mínimo día:   {l if l is not None else 'N/D'}\n"
            f"Cierre prev.: {pc if pc is not None else 'N/D'}\n"
            f"Hora:         {self._fmt_epoch(ts)}"
        )

    def obtener_velas(self, symbol: str, resolution: str = "1m", dias: int = 1):
        """
        OHLCV; no pasamos session a yfinance.
        """
        q   = self._sanitize(symbol)
        sym = self._resolver_yahoo_simbolo(q) or q

        interval_map = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m","60m":"60m","D":"1d","1d":"1d"}
        interval = interval_map.get(resolution, "1m")
        period = f"{dias}d" if interval != "1m" or dias > 1 else "1d"

        # 👇 sin session
        df = yf.download(tickers=sym, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return f"No hay datos de velas para {sym}."
        o = df["Open"].tolist(); h = df["High"].tolist(); l = df["Low"].tolist()
        c = df["Close"].tolist(); v = df["Volume"].tolist()
        t = [int(pd_ts.timestamp()) for pd_ts in df.index.to_pydatetime()]
        return {"s":"ok","t":t,"o":o,"h":h,"l":l,"c":c,"v":v}




class HerramientaBusquedaWeb:
    
    def __init__(self):
        self.base_url = "https://api.duckduckgo.com/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }
    
    def buscar_web(self, consulta:str, cantidad:int = 3) -> str:
        try:
            params = {
                "q": consulta,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }

            response = requests.get(
                self.base_url, params=params, headers=self.headers, timeout=10
            )
            response.raise_for_status()
            data = response.json()

            resultados_texto = f"Resultados de busqueda para '{consulta}':\n\n"

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
                resultados_texto += f"Definicion: {data['Definition']}\n\n"
                if data.get("DefinitionSource"):
                    resultados_texto += f"Fuente: {data['DefinitionSource']}\n\n"
            
            if not any([data.get("Answer"), data.get("AbstractText"), data.get("RelatedTopics"), data.get("Definition")]):
                return self._busqueda_alternativa(consulta)

            return resultados_texto.strip()

        except Exception as e:
            return f"Error al buscar en la web: {str(e)}"
    
    def _busqueda_alternativa(self, consulta:str) -> str:
        try:
            url = "https://duckduckgo.com/html/"
            params = {
                "q": consulta
            }

            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                content = response.text

                import re
                titles = re.findall(r'<a[^>]*class="result__a"[^>]*>([^<]+)</a>', content)

                if titles:
                    resultado = f"Resultados de busqueda para '{consulta}':\n\n"
                    for i, title in enumerate(titles[:3], 1):
                        resultado += f"{i}. {title}\n\n"
                    
                    return resultado.strip()

            return f"Busqueda realizada para '{consulta}'. No se encontraron resultados."
                

        except Exception as e:
            return f"Error al buscar en la web: {str(e)}"


class HerramientaCripto:
    """
    Precio de criptomonedas usando CoinGecko (free).
    - Reintentos con backoff ante 429/5xx.
    - Caché en memoria (TTL) para reducir rate limit.
    """
    BASE = os.environ.get("COINGECKO_BASE", "https://api.coingecko.com/api/v3")

    def __init__(self, price_ttl: int = 30, top_ttl: int = 60):
        self.sess = requests.Session()
        self.sess.headers.update({"User-Agent": "MiAgente/1.0 (+https://example.local)"})
        self._ids_cache = {"ts": 0, "maps": None}   # refresca cada 6h
        self._price_cache = {}                      # (coin_id, vs) -> {ts, data}
        self._top_cache = {"ts": 0, "key": None, "data": None}  # cache para TOP
        self.PRICE_TTL = max(5, int(price_ttl))
        self.TOP_TTL = max(10, int(top_ttl))

    # ---------- núcleo HTTP con backoff ----------
    def _sleep_backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = float(retry_after) if retry_after else min(2 ** attempt, 10)
        time.sleep(delay)

    def _get(self, path: str, params: dict | None = None, max_attempts: int = 4):
        url = f"{self.BASE}{path}"
        for attempt in range(1, max_attempts + 1):
            r = self.sess.get(url, params=params or {}, timeout=12)
            # Manejo de rate limit / server errors
            if r.status_code == 429 or r.status_code >= 500:
                self._sleep_backoff(attempt, r.headers.get("Retry-After"))
                continue
            r.raise_for_status()
            return r.json()
        raise requests.RequestException("Rate limit o error persistente de CoinGecko.")

    # ---------- resolución de ID ----------
    def _load_ids(self, force: bool = False):
        now = time.time()
        if (not force) and self._ids_cache["maps"] and (now - self._ids_cache["ts"] < 6 * 3600):
            return self._ids_cache["maps"]

        data = self._get("/coins/list", {"include_platform": "false"})
        maps = {"by_id": set(), "by_symbol": {}, "by_name": {}}
        for it in data:
            cid = it["id"]
            sym = (it.get("symbol") or "").lower()
            name = (it.get("name") or "").lower()
            maps["by_id"].add(cid)
            if sym:
                maps["by_symbol"].setdefault(sym, []).append(cid)
            if name:
                maps["by_name"].setdefault(name, []).append(cid)
        self._ids_cache = {"ts": now, "maps": maps}
        return maps

    def _resolve_id(self, q: str) -> str | None:
        q = (q or "").strip().lower().lstrip("$")
        if not q:
            return None
        m = self._load_ids()

        if q in m["by_id"]:
            return q
        if q in m["by_symbol"]:
            return m["by_symbol"][q][0]
        if q in m["by_name"]:
            return m["by_name"][q][0]

        # Búsqueda parcial por nombre (p.ej., 'bitco' → 'bitcoin')
        for name, lst in m["by_name"].items():
            if q in name:
                return lst[0]
        return None

    # ---------- API pública ----------
    def obtener_precio(self, consulta: str, vs: str = "usd") -> str:
        """
        Entrada flexible:
          - "btc" / "bitcoin"
          - "eth usd" (cambia vs)
        """
        s = (consulta or "").strip()
        parts = s.split()
        if len(parts) >= 2 and parts[-1].isalpha() and len(parts[-1]) <= 4:
            vs = parts[-1].lower()
            coin = " ".join(parts[:-1])
        else:
            coin = s

        cid = self._resolve_id(coin)
        if not cid:
            return f"No pude reconocer la cripto '{consulta}'. Prueba con 'bitcoin', 'btc', 'ethereum', 'eth'."

        key = (cid, vs.lower())
        now = time.time()
        hit = self._price_cache.get(key)
        if hit and now - hit["ts"] < self.PRICE_TTL:
            p = hit["data"]
        else:
            js = self._get(
                "/simple/price",
                {"ids": cid, "vs_currencies": vs, "include_24hr_change": "true"},
            )
            if not isinstance(js, dict) or cid not in js or vs not in js[cid]:
                return f"No hay precio disponible para {cid} en {vs.upper()}."
            p = {
                "price": js[cid][vs],
                "chg": js[cid].get(f"{vs}_24h_change"),
            }
            self._price_cache[key] = {"ts": now, "data": p}

        chg = p["chg"]
        chg_txt = f"{chg:.2f}%" if isinstance(chg, (int, float)) else "N/D"
        return f"{cid.capitalize()} → {p['price']} {vs.upper()} (24h: {chg_txt})"

    def obtener_top_criptos(self, n: int = 10, vs: str = "usd") -> str:
        """
        Devuelve el Top N por market cap con cambio 24h. Usa caché corto para bajar el rate limit.
        """
        try:
            n = max(1, min(int(n), 50))
        except Exception:
            n = 10
        vs = (vs or "usd").lower()

        # caché simple por (n, vs)
        key = (n, vs)
        now = time.time()
        if self._top_cache["data"] and self._top_cache["key"] == key and (now - self._top_cache["ts"] < self.TOP_TTL):
            js = self._top_cache["data"]
        else:
            js = self._get(
                "/coins/markets",
                {
                    "vs_currency": vs,
                    "order": "market_cap_desc",
                    "per_page": n,
                    "page": 1,
                    "price_change_percentage": "24h",
                },
            )
            if not isinstance(js, list) or not js:
                return "No pude obtener el top de criptomonedas."
            self._top_cache.update({"ts": now, "key": key, "data": js})

        out = [f"Top {n} criptos por capitalización ({vs.upper()}):", ""]
        for i, it in enumerate(js[:n], 1):
            name = it.get("name", "N/D")
            sym = (it.get("symbol") or "").upper()
            px = it.get("current_price")
            ch = it.get("price_change_percentage_24h")
            ch_txt = f"{ch:.2f}%" if isinstance(ch, (int, float)) else "N/D"
            out.append(f"{i}. {name} ({sym}): {px} {vs.upper()}  |  24h: {ch_txt}")
        return "\n".join(out)

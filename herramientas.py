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
        Acepta nombre o ticker. Resuelve con Yahoo search (sin mapeos) y usa:
        1) yfinance.fast_info
        2) Yahoo Quote API (query1)
        3) Stooq CSV
        """
        sym = self._resolver_yahoo_simbolo(symbol_o_nombre)
        if not sym:
            return f"No pude resolver el símbolo para '{symbol_o_nombre}'. Prueba con el ticker (p. ej., TSLA)."

        # --- 1) yfinance.fast_info (sin session explícita) ---
        c = o = h = l = pc = None
        try:
            t = yf.Ticker(sym)  # ¡no pasar session!
            fi = getattr(t, "fast_info", None)
            if fi:
                c  = getattr(fi, "last_price", None)
                o  = getattr(fi, "open", None)
                h  = getattr(fi, "day_high", None)
                l  = getattr(fi, "day_low", None)
                pc = getattr(fi, "previous_close", None)
        except Exception:
            pass

        # --- 2) Yahoo Quote API directo ---
        if c is None:
            q = self._yahoo_quote_api(sym)
            if q:
                c, o, h, l, pc = q["c"], q["o"], q["h"], q["l"], q["pc"]
                ts = q["t"]
            else:
                ts = int(time.time())
        else:
            ts = int(time.time())

        # --- 3) Stooq fallback ---
        if c is None:
            q2 = self._stooq_quote(sym)
            if q2:
                c, o, h, l, pc, ts = q2["c"], q2["o"], q2["h"], q2["l"], q2["pc"], q2["t"]

        if c is None:
            return f"No hay datos de precio para {sym} ahora mismo."

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

    def _yahoo_quote_api(self, symbol: str) -> dict | None:
        """
        Fallback directo a la API pública de Yahoo Quote.
        Devuelve dict con c/o/h/l/pc si se puede; None si falla.
        """
        url = "https://query1.finance.yahoo.com/v7/finance/quote"
        params = {"symbols": symbol}
        try:
            r = requests.get(
                url, params=params, timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            r.raise_for_status()
            js = r.json()
            res = ((js or {}).get("quoteResponse") or {}).get("result") or []
            if not res:
                return None
            q = res[0]
            out = {
                "c": q.get("regularMarketPrice"),
                "o": q.get("regularMarketOpen"),
                "h": q.get("regularMarketDayHigh"),
                "l": q.get("regularMarketDayLow"),
                "pc": q.get("regularMarketPreviousClose"),
                "t": q.get("regularMarketTime"),
            }
            # válido si al menos tenemos precio actual
            if out["c"] is None:
                return None
            return out
        except Exception:
            return None

    def _stooq_quote(self, symbol: str) -> dict | None:
        """
        Último recurso: Stooq (CSV). Útil cuando Yahoo/yfinance falla en ciertos entornos.
        """
        try:
            # Stooq usa minúsculas; índices y algunos símbolos tienen formato especial
            s = symbol.lower()
            url = f"https://stooq.com/q/l/?s={s}&f=sd2t2ohlcv&h&e=csv"
            r = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
            r.raise_for_status()
            # parse CSV
            content = r.text.strip()
            rows = list(csv.DictReader(StringIO(content)))
            if not rows:
                return None
            row = rows[0]
            if row.get("Close") in ("N/D", "N/A", None, ""):
                return None
            # Construimos campos similares
            def f(x):
                try: return float(x)
                except Exception: return None
            out = {
                "c": f(row.get("Close")),
                "o": f(row.get("Open")),
                "h": f(row.get("High")),
                "l": f(row.get("Low")),
                "pc": None,            # Stooq “lite” no trae previous close en este endpoint
                "t": int(datetime.now(timezone.utc).timestamp()),
            }
            if out["c"] is None:
                return None
            return out
        except Exception:
            return None


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



    SEED_COINS = {
        "bitcoin":        ("BTC", "Bitcoin"),
        "ethereum":       ("ETH", "Ethereum"),
        "tether":         ("USDT", "Tether"),
        "binancecoin":    ("BNB", "BNB"),
        "solana":         ("SOL", "Solana"),
        "ripple":         ("XRP", "XRP"),
        "usd-coin":       ("USDC", "USD Coin"),
        "cardano":        ("ADA", "Cardano"),
        "dogecoin":       ("DOGE","Dogecoin"),
        "tron":           ("TRX", "TRON"),
        "polkadot":       ("DOT", "Polkadot"),
        "litecoin":       ("LTC", "Litecoin"),
        "matic-network":  ("MATIC","Polygon"),
        "bitcoin-cash":   ("BCH", "Bitcoin Cash"),
        "chainlink":      ("LINK","Chainlink"),
        "stellar":        ("XLM","Stellar"),
        "uniswap":        ("UNI","Uniswap"),
        "cosmos":         ("ATOM","Cosmos"),
        "monero":         ("XMR","Monero"),
        "algorand":       ("ALGO","Algorand"),
    }

    def _seed_maps(self):
        """Construye índices in-memory a partir de SEED_COINS."""
        by_id = set(self.SEED_COINS.keys())
        by_symbol = {}
        by_name = {}
        id_to_symbol = {}
        for cid, (sym, name) in self.SEED_COINS.items():
            id_to_symbol[cid] = sym.upper()
            by_symbol.setdefault(sym.lower(), []).append(cid)
            by_name.setdefault(name.lower(), []).append(cid)
        return {"by_id": by_id, "by_symbol": by_symbol, "by_name": by_name, "id_to_symbol": id_to_symbol}

    def _load_ids(self, force: bool = False):
        """Devuelve mapas de resolución. Si CoinGecko está bloqueado, usa mapas semilla sin lanzar."""
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
            # Marca bloqueo y usa semilla
            self._gecko_block_until = now + 60.0
            maps = self._seed_maps()
            self._ids_cache = {"ts": now, "maps": maps}
            return maps

    def _resolve_id(self, q: str) -> tuple[str | None, str | None]:
        """
        Devuelve (coin_id, symbol_upper) o (None, None).
        Nunca lanza; si no puede, intenta resolver contra mapas semilla.
        """
        q = (q or "").strip().lower().lstrip("$")
        if not q:
            return None, None

        # 1) Usa mapas (CG o semilla) sin lanzar
        maps = self._load_ids(force=False)

        # consultas directas
        if q in maps["by_id"]:
            return q, maps["id_to_symbol"].get(q)
        if q in maps["by_symbol"]:
            cid = maps["by_symbol"][q][0]
            return cid, maps["id_to_symbol"].get(cid)
        if q in maps["by_name"]:
            cid = maps["by_name"][q][0]
            return cid, maps["id_to_symbol"].get(cid)

        # 2) Búsqueda parcial por nombre
        for name, lst in maps["by_name"].items():
            if q in name:
                cid = lst[0]
                return cid, maps["id_to_symbol"].get(cid)

        sym_guess = q.upper() if q.isalpha() and 2 <= len(q) <= 6 else None
        return None, sym_guess  # esto permite al precio usar fallbacks por símbolo

    def obtener_precio_cripto(self, consulta: str, vs: str = "usd") -> str:
        s = (consulta or "").strip()
        parts = s.split()
        import re as _re
        m = _re.search(r"\bvs\s*[:=]\s*([a-zA-Z]{2,5})\b", s, _re.I)
        if m:
            vs = m.group(1).lower()
            coin = _re.sub(r"\bvs\s*[:=]\s*[a-zA-Z]{2,5}\b", "", s, flags=_re.I).strip()
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
                    js = self._get(
                        "/simple/price",
                        {"ids": cid, "vs_currencies": vs, "include_24hr_change": "true"},
                    )
                    if isinstance(js, dict) and cid in js and vs in js[cid]:
                        p = {"price": js[cid][vs], "chg": js[cid].get(f"{vs}_24h_change")}
                except requests.RequestException as e:
                    gecko_err = e

            sym_try = sym_upper
            if not sym_try and cid:
                # intenta derivar símbolo desde mapas
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

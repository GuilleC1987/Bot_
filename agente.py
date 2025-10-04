

import re
from typing import Any

import requests
from datetime import datetime, timezone

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.base import BaseCallbackHandler

from prompts import get_system_prompt, get_conversation_prompt
from herramientas import (
    HerramientaFecha,
    HerramientaClima,
    HerramientaMercadosYahoo,
    HerramientaBusquedaWeb,
    HerramientaCripto,
)
import os

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


class AgenteMultiAPI:
    """
    Asistente que integra múltiples APIs (clima, noticias/mercados, búsqueda web, fecha).
    """

    def __init__(self):
        self.nombre = "Asistente Personal"

        # Herramientas externas
        self.herramienta_clima = HerramientaClima(os.getenv("WEATHER_API_KEY"))
        self.herramienta_noticias_financieras = HerramientaMercadosYahoo()
        self.herramienta_busqueda = HerramientaBusquedaWeb()
        self.herramienta_fecha = HerramientaFecha()
        self.herramienta_cripto = HerramientaCripto()

        # Modelo LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
        )

        # Memoria conversacional
        self.memoria = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

        # Tools para LangChain
        self.herramientas = self._crear_herramientas_langchain()

        # Prompts
        base_system = get_system_prompt().format(nombre=self.nombre)

        # Agente conversacional
        self.agente = initialize_agent(
            tools=self.herramientas,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memoria,
            verbose=False,                   # ↓ menos ruido en consola
            handle_parsing_errors=True,
            max_iterations=4,
            agent_kwargs={
                "system_message": base_system,
                "prefix": "Eres un asistente útil. Responde en español o inglés y de forma directa.",
            },
        )


    def _tool_fecha_desde_prompt_str(self, entrada: str) -> str:
        """
        Tool 'string-only' para fecha por ciudad.
        Puedes incluir 'formato: X' en el texto (X ∈ completo|fecha|hora|iso|simple).
        """
        s = (entrada or "").strip()
        m = re.search(r"formato\s*[:=]\s*(completo|fecha|hora|iso|simple)", s, re.I)
        formato = (m.group(1).lower() if m else "completo")
        try:
            return self.herramienta_fecha.obtener_fecha_desde_prompt(s, formato=formato)
        except Exception as e:
            return f"⚠️ Error al obtener fecha desde prompt: {e}"
    
    def _tool_cripto_top(self, entrada: str) -> str:
        """
        Lista el top N cripto por market cap.
        Entrada opcional: número o texto tipo 'top 5' (default 10).
        """
        n = 10
        s = (entrada or "").strip()
        if s:
            import re
            m = re.search(r"(\d+)", s)
            if m:
                try:
                    n = max(1, min(int(m.group(1)), 50))
                except Exception:
                    n = 10
        return self.herramienta_cripto.obtener_top_cripto(cantidad=n, vs="usd")

    def _tool_cripto_precio(self, entrada: str) -> str:
        """
        Precio de una cripto específica. Ej: 'bitcoin', 'eth', 'sol'.
        También puedes indicar moneda fiat: 'eth vs: eur'
        """
        s = (entrada or "").strip()
        if not s:
            return "Indica la criptomoneda (ej. 'bitcoin', 'eth', 'sol')."
        import re
        m = re.search(r"vs\s*[:=]\s*([a-zA-Z]+)", s)
        vs = (m.group(1).lower() if m else "usd")
        return self.herramienta_cripto.obtener_precio_cripto(s, vs=vs)

    # ------------------------------------------------------------

    def _crear_herramientas_langchain(self) -> list:
        """
        Construye la lista de Tools para el agente.
        Forzamos string-only con infer_schema=False y agregamos alias por compatibilidad.
        """
        from langchain.tools import Tool

        herramientas = [
            Tool.from_function(
                name="ConsultarClima",
                func=self.herramienta_clima.obtener_clima_actual,
                description="Obtiene información del clima actual de una ciudad. Input: nombre de la ciudad.",
                infer_schema=False,  # <- fuerza un único 'tool_input' (str)
            ),

  

            Tool.from_function(
                name="PrecioAccion",
                func=self.herramienta_noticias_financieras.obtener_precio_accion,
                description=(
                    "Obtén el precio actual de una acción a partir del símbolo o NOMBRE. "
                    "Ej.: 'Apple' o 'AAPL'. Devuelve precio, apertura, máximo, mínimo y hora."
                ),
                infer_schema=False,
            ),
            Tool.from_function(
                name="BuscarWeb",
                func=self.herramienta_busqueda.buscar_web,
                description="Busca información general en la web. Input: término/pregunta.",
                infer_schema=False,
            ),
            Tool.from_function(
                name="ObtenerFecha",
                func=self.herramienta_fecha.obtener_fecha_actual,
                description="Obtiene fecha/hora local del servidor ('completo','fecha','hora','iso','simple').",
                infer_schema=False,
            ),
            Tool.from_function(
                name="ObtenerFechaDesdePrompt",
                func=lambda s: self._tool_fecha_desde_prompt_str(s),
                description=(
                    "Devuelve fecha/hora local detectando ciudad en el texto; "
                    "opcional 'formato: completo|fecha|hora|iso|simple' en el mismo texto."
                ),
                infer_schema=False,
            ),
            Tool.from_function(
                name="ObtenerTopCripto",
                func=lambda s: self._tool_cripto_top(s),
                description=(
                    "Obtiene el top N cripto por market cap. "
                    "Entrada opcional: número o texto tipo 'top 5' (default 10)."
                ),
                infer_schema=False,
            ),
            Tool.from_function(
                name="ObtenerPrecioCripto",
                func=lambda s: self._tool_cripto_precio(s),
                description=(
                    "Obtiene el precio de una cripto específica. "
                    "Ej: 'bitcoin', 'eth', 'sol'. "
                    "También puedes indicar moneda fiat: 'eth vs: eur'."
                ),
                infer_schema=False,
            ),
            Tool(
                name="PrecioCrypto",
                func=self.herramienta_cripto.precio,
                description=(
                    "Precio actual de una criptomoneda. Input: símbolo o nombre (ej. 'BTC', 'bitcoin', 'btc/usdt'). "
                    "Devuelve precio, cambio 24h y market cap."
                ),
            ),
            Tool(
                name="TopCriptos",
                func=self.herramienta_cripto.top,
                description="Top N criptomonedas por market cap. Input: número N (por defecto 10).",
            ),
            ]


        return herramientas
    
    def procesar_consulta(self, consulta: str) -> str:
        """Procesa la consulta con el agente y devuelve el texto resultante."""
        try:
            resp = self.agente.invoke({"input": consulta})
            if isinstance(resp, dict) and "output" in resp:
                return resp["output"]
            return str(resp)
        except Exception as e:
            return f"Error al procesar la consulta: {e}"

    def mostrar_capacidades_del_agente(self):
        print("Capacidades del agente:")
        print("-" * 20)
        for herramienta in self.herramientas:
            print(f"Nombre: {herramienta.name}")
            print(f"Descripción: {herramienta.description}")
            print("-" * 20)


# agente.py
from __future__ import annotations

import os
import re
from typing import Any

from datetime import datetime, timezone
import requests

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts import get_system_prompt, get_conversation_prompt
from herramientas import (
    HerramientaFecha,
    HerramientaClima,
    HerramientaAccionesStooq,
    HerramientaBusquedaWeb,
    HerramientaCripto,
)


os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _ensure_ca_bundle_safe_path():

    try:
        import certifi, shutil
        cafile = certifi.where()
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
    Asistente que integra múltiples APIs (clima, búsqueda web, fecha, precion top cripto).
    """

    def __init__(self):
        self.nombre = "Asistente Personal"

        # Herramientas externas (de herramientas.py)
        self.herramienta_clima = HerramientaClima(os.getenv("WEATHER_API_KEY"))
        self.herramienta_acciones  = HerramientaAccionesStooq()
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

        # Prompt del sistema
        base_system = get_system_prompt().format(nombre=self.nombre)

        # Agente conversacional
        
        self.agente = initialize_agent(
            tools=self.herramientas,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memoria,
            
            early_stopping_method="generate",

            max_iterations=int(os.getenv("AGENT_MAX_STEPS", "8")),
            max_execution_time=int(os.getenv("AGENT_MAX_SECS", "25")),
            handle_parsing_errors=True,
            verbose=False,
            agent_kwargs={
                "system_message": base_system,

                "prefix": (
                    "Eres un asistente útil. Responde en español o inglés y de forma directa. "
                    "Si tras pocos intentos con herramientas no estás seguro, responde directamente sin más herramientas."
                ),
            },
        )


    # ------------------- Wrappers string-only -------------------
    def _tool_fecha_desde_prompt_str(self, entrada: str) -> str:
        """
        Fecha/hora por ciudad detectada en el texto.
        Permite 'formato: completo|fecha|hora|iso|simple'.
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
            m = re.search(r"(\d+)", s)
            if m:
                try:
                    n = max(1, min(int(m.group(1)), 50))
                except Exception:
                    n = 10
        # ✅ herramientas.py expone obtener_top_criptos(n:int, vs:str='usd')
        return self.herramienta_cripto.obtener_top_criptos(n=n, vs="usd")

    def _tool_cripto_precio(self, entrada: str) -> str:
        """
        Precio de una cripto específica. Ej: 'bitcoin', 'eth', 'sol'.
        Puedes indicar moneda: 'eth vs: eur' o 'btc usd'.
        """
        s = (entrada or "").strip()
        if not s:
            return "Indica la criptomoneda (ej. 'bitcoin', 'eth', 'sol')."
        m = re.search(r"vs\s*[:=]\s*([a-zA-Z]{2,5})", s)
        vs = (m.group(1).lower() if m else "usd")
        # ✅ herramientas.py expone obtener_precio_cripto(consulta:str, vs:str='usd')
        return self.herramienta_cripto.obtener_precio_cripto(s, vs=vs)

    # ------------------- Construcción de Tools -------------------
    def _crear_herramientas_langchain(self) -> list:
        """
        Construye la lista de Tools para el agente.
        TODAS string-only (infer_schema=False) para evitar 'Missing some input keys'.
        """
        return [
            Tool.from_function(
                name="ConsultarClima",
                func=self.herramienta_clima.obtener_clima_actual,
                description="Clima actual de una ciudad. Input: nombre de la ciudad.",
                infer_schema=False,
            ),
            Tool.from_function(
                name="PrecioAccion",
                func=self.herramienta_acciones.obtener_precio_accion,
                description=(
                    "Precio de una acción por símbolo o NOMBRE. Ej.: 'Apple' o 'AAPL'. "
                    "Devuelve precio, apertura, máximo, mínimo y hora."
                ),
                infer_schema=False,
            ),
            Tool.from_function(
                name="BuscarWeb",
                func=self.herramienta_busqueda.buscar_web,
                description="Busca información general en la web. Input: término o pregunta.",
                infer_schema=False,
            ),
            Tool.from_function(
                name="ObtenerFecha",
                func=self.herramienta_fecha.obtener_fecha_actual,
                description="Fecha/hora del servidor ('completo','fecha','hora','iso','simple').",
                infer_schema=False,
            ),
            Tool.from_function(
                name="ObtenerFechaDesdePrompt",
                func=self._tool_fecha_desde_prompt_str,
                description=(
                    "Fecha/hora local detectando ciudad en el texto; "
                    "puedes incluir 'formato: completo|fecha|hora|iso|simple'."
                ),
                infer_schema=False,
            ),
            # --- CRIPTO ---
            Tool.from_function(
                name="ObtenerPrecioCripto",
                func=self._tool_cripto_precio,
                description="Precio cripto. Ej.: 'bitcoin', 'btc', 'eth usd', 'sol vs: eur'.",
                infer_schema=False,
            ),
            Tool.from_function(
                name="ObtenerTopCripto",
                func=self._tool_cripto_top,
                description="Top N criptos por market cap. Input opcional: número (default 10).",
                infer_schema=False,
            ),
        ]

    # ------------------- API pública -------------------
    def procesar_consulta(self, consulta: str) -> str:
        """Procesa la consulta con el agente y, si se queda sin pasos/tiempo, hace fallback al LLM directo."""
        try:
            resp = self.agente.invoke({"input": consulta})
            if isinstance(resp, dict) and "output" in resp:
                return resp["output"]
            return str(resp)
        except Exception as e:
            msg = str(e)
            if "Agent stopped due to iteration limit or time limit" in msg:
                # Fallback: responder sin herramientas
                try:
                    llm_resp = self.llm.invoke(consulta)
                    return getattr(llm_resp, "content", str(llm_resp)) or "Lo siento, no pude completar la consulta."
                except Exception as e2:
                    return f"No pude completar la consulta ahora mismo ({e2})."
            return f"Error al procesar la consulta: {e}"

    def mostrar_capacidades_del_agente(self):
        print("Capacidades del agente:")
        print("-" * 20)
        for herramienta in self.herramientas:
            print(f"Nombre: {herramienta.name}")
            print(f"Descripción: {herramienta.description}")
            print("-" * 20)

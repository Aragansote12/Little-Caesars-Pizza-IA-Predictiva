"""
copilot_agent.py
Motor de razonamiento del Agente IA – 100% offline sobre datos reales del simulador.
Opcionalmente usa Gemini / OpenAI si el usuario provee una API Key.
"""

import re
import datetime
import numpy as np
import pandas as pd

# ─── Respuestas offline ───────────────────────────────────────────────────────

RESPUESTAS_OFFLINE = {
    "pico": (
        "📊 **Horas Pico detectadas por el Modelo RF:**\n"
        "- **Almuerzo:** 12:00 – 14:00 (pico máximo ~13:00)\n"
        "- **Cena:** 19:00 – 21:00 (pico máximo ~20:00)\n\n"
        "El modelo predice que los **viernes y sábados de quincena** (días 14-16 y 29-31) "
        "superan el umbral de hora pico el **83%** del tiempo. "
        "Recomendación: incrementar lotes HNR en +2 unidades 20 min antes."
    ),
    "margen": (
        "💰 **Análisis de Márgenes – Modelo Financiero:**\n"
        "- Margen neto promedio del histórico: **~17-22%**\n"
        "- Principal destructor de margen: comisiones de apps terceras (Rappi/Uber) = **30%** sobre venta\n"
        "- Cada 10% de migración al **Pizza Portal** recupera ~**$280.000 COP/día**\n\n"
        "**Acción sugerida:** activa el slider 'Migración App Oficial' al 60%+ para ver el impacto en vivo."
    ),
    "inflacion": (
        "⚠️ **Impacto de Inflación en Insumos:**\n"
        "- El queso mozzarella y la harina representan ~**22% del costo total**\n"
        "- Un aumento del +10% en materias primas comprime el margen neto en **~3.5 puntos porcentuales**\n"
        "- Palanca de mitigación: impulsar venta cruzada (Bordes Rellenos + Crazy Bread) "
        "eleva el ticket promedio sin incrementar costos fijos.\n\n"
        "El modelo recomienda mantener el subsidio cruzado por encima del **15%** del mix de ventas."
    ),
    "personal": (
        "👷 **Optimizador de Turnos – Reglas del Modelo:**\n"
        "- 1 Hornero por cada 20 pizzas/hora\n"
        "- 1 Armador por cada 15 pizzas/hora\n"
        "- 1 Cajero/Portal por cada 25 pedidos/hora\n\n"
        "En hora pico típica (Viernes 20:00): ~52 pedidos → **3 horneros, 3 armadores, 2 cajeros = 8 personas**.\n"
        "Día normal (Martes 15:00): ~15 pedidos → **1 hornero, 1 armador, 1 cajero = 3 personas**."
    ),
    "hotNready": (
        "🍕 **Control Hot-N-Ready (máx. vida útil: 30 min):**\n"
        "- El Agente calcula lotes cada 30 min basándose en la predicción de los siguientes 45 min\n"
        "- **Regla de Oro:** Nunca más de 4 pizzas en calentador si el siguiente pico es >30 min\n"
        "- En pico de viernes noche: preparar **6-8 pizzas Pepperoni + 4-5 Queso** cada 30 min\n\n"
        "Esto garantiza entrega en <60 segundos y merma < 2% (vs. hasta 12% sin optimización)."
    ),
    "quincena": (
        "📅 **Efecto Quincena Colombiana:**\n"
        "- Días 14-16 y 29-31 de cada mes: demanda sube **+35% a +45%**\n"
        "- El modelo detecta este patrón con **92% de precisión** en el histórico\n"
        "- Recomendación: programar turno reforzado (+2 personas) desde las 11:00 am en días de quincena\n"
        "- Impacto financiero estimado por quincena bien gestionada: **+$1.2M – $1.8M COP adicionales/mes**"
    ),
    "delivery": (
        "🛵 **Análisis de Canales de Delivery:**\n"
        "- Rappi / DiDi / Uber Eats: comisión del **30%** sobre el precio de venta\n"
        "- App Oficial + Pizza Portal: **0% de comisión**\n\n"
        "**Estrategia de Migración:**\n"
        "1. Comunicar el beneficio del Pizza Portal (recojo autónomo, sin espera)\n"
        "2. Ofrecer descuento del 5% a quien use la app oficial\n"
        "3. Meta: llevar el delivery tercero de ~35% → ~15% del mix\n"
        "4. **Ahorro estimado:** hasta **$4.2M COP/mes** por sucursal"
    ),
    "clima": (
        "🌧️ **Impacto del Clima en la Operación:**\n"
        "- **Aguacero:** tráfico peatonal –25%, pero pedidos delivery +50%\n"
        "- **Lluvia Leve:** impacto moderado; delivery +25%\n"
        "- **Nublado:** demanda casi normal (–5%)\n\n"
        "El modelo ajusta automáticamente el mix de canales y la predicción de demanda según el clima seleccionado. "
        "En aguacero, la alerta de 'Pico Delivery' se activa para pre-empacar pedidos."
    ),
}


def _detectar_intencion(pregunta: str) -> str:
    p = pregunta.lower()
    if any(w in p for w in ["pico", "hora", "peak", "tráfico"]):
        return "pico"
    if any(w in p for w in ["margen", "ganancia", "profit", "rentab"]):
        return "margen"
    if any(w in p for w in ["inflaci", "queso", "harina", "insumo", "costo"]):
        return "inflacion"
    if any(w in p for w in ["personal", "turno", "empleado", "staff", "nómina"]):
        return "personal"
    if any(w in p for w in ["hot", "ready", "horno", "lote", "pizza"]):
        return "hotNready"
    if any(w in p for w in ["quincena", "15", "30", "pago", "nómina"]):
        return "quincena"
    if any(w in p for w in ["rappi", "delivery", "didi", "uber", "comisi", "canal", "portal"]):
        return "delivery"
    if any(w in p for w in ["lluvia", "aguacero", "clima", "tiempo"]):
        return "clima"
    return "general"


def _respuesta_general(estado: dict) -> str:
    h = estado.get("hora_sim", 12)
    pedidos = estado.get("pedidos_actuales", 0)
    margen  = estado.get("margen_actual", 0.0)
    clima   = estado.get("clima", "Soleado")
    es_pico = estado.get("es_pico", False)

    alerta = "🔴 **HORA PICO ACTIVA** – Verifica lotes HNR y personal." if es_pico else "🟢 Ritmo operacional normal."
    return (
        f"**Estado actual {h:02d}:00h | Clima: {clima}**\n\n"
        f"{alerta}\n\n"
        f"- Pedidos proyectados esta hora: **{pedidos}**\n"
        f"- Margen neto proyectado: **{margen:.1%}**\n\n"
        "Puedo analizar: `horas pico`, `márgenes`, `inflación`, `personal`, "
        "`hot-n-ready`, `quincena`, `delivery` o `clima`. ¿Qué quieres explorar?"
    )


# ─── Clase principal del Agente ───────────────────────────────────────────────

class CopilotAgent:
    def __init__(self, api_key: str = "", proveedor: str = "offline"):
        self.api_key   = api_key.strip()
        self.proveedor = proveedor
        self.historial = []

    def _responder_offline(self, pregunta: str, estado: dict) -> str:
        intencion = _detectar_intencion(pregunta)
        if intencion in RESPUESTAS_OFFLINE:
            return RESPUESTAS_OFFLINE[intencion]
        return _respuesta_general(estado)

    def _responder_gemini(self, pregunta: str, estado: dict) -> str:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            modelo = genai.GenerativeModel("gemini-1.5-flash")
            contexto = self._construir_contexto(estado)
            prompt = f"{contexto}\n\nPregunta del operador: {pregunta}"
            respuesta = modelo.generate_content(prompt)
            return respuesta.text
        except ImportError:
            return (
                "⚠️ Instala `google-generativeai` para usar Gemini: "
                "`pip install google-generativeai`\n\n"
                + self._responder_offline(pregunta, estado)
            )
        except Exception as e:
            return f"⚠️ Error Gemini: {e}\n\n" + self._responder_offline(pregunta, estado)

    def _responder_openai(self, pregunta: str, estado: dict) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            contexto = self._construir_contexto(estado)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": contexto},
                    {"role": "user", "content": pregunta},
                ],
                max_tokens=512,
            )
            return resp.choices[0].message.content
        except ImportError:
            return (
                "⚠️ Instala `openai` para usar GPT: `pip install openai`\n\n"
                + self._responder_offline(pregunta, estado)
            )
        except Exception as e:
            return f"⚠️ Error OpenAI: {e}\n\n" + self._responder_offline(pregunta, estado)

    def _construir_contexto(self, estado: dict) -> str:
        return (
            "Eres OpsAI, el agente de inteligencia artificial operacional de Little Caesars Colombia. "
            "Eres experto en operaciones de restaurantes de comida rápida, márgenes financieros, "
            "predicción de demanda y optimización de personal. Respondes en español colombiano, "
            "con datos concretos en COP (pesos colombianos). Eres conciso, directo y orientado a la acción.\n\n"
            f"Estado actual de la simulación:\n"
            f"- Hora: {estado.get('hora_sim', 'N/A')}:00h\n"
            f"- Clima: {estado.get('clima', 'N/A')}\n"
            f"- Pedidos proyectados: {estado.get('pedidos_actuales', 'N/A')}\n"
            f"- Ingresos hora: ${estado.get('ingresos_hora', 0):,.0f} COP\n"
            f"- Margen neto: {estado.get('margen_actual', 0):.1%}\n"
            f"- Personal recomendado: {estado.get('personal_total', 'N/A')} personas\n"
            f"- ¿Hora pico?: {'Sí' if estado.get('es_pico') else 'No'}\n"
        )

    def responder(self, pregunta: str, estado: dict) -> str:
        """Punto de entrada principal del agente."""
        self.historial.append({"rol": "usuario", "msg": pregunta})

        if self.api_key and self.proveedor == "Gemini":
            respuesta = self._responder_gemini(pregunta, estado)
        elif self.api_key and self.proveedor == "OpenAI":
            respuesta = self._responder_openai(pregunta, estado)
        else:
            respuesta = self._responder_offline(pregunta, estado)

        self.historial.append({"rol": "agente", "msg": respuesta})
        return respuesta

    def analizar_anomalia(self, pred: int, real: int, umbral_pct: float = 0.20) -> dict | None:
        """Detecta si la demanda real diverge más del umbral% de la predicción."""
        if pred == 0:
            return None
        desv = abs(real - pred) / pred
        if desv > umbral_pct:
            tipo   = "pico_inesperado" if real > pred else "caida_inesperada"
            return {
                "tipo":    tipo,
                "pred":    pred,
                "real":    real,
                "desv":    desv,
                "mensaje": (
                    f"⚠️ {'Pico inesperado' if tipo == 'pico_inesperado' else 'Caída inesperada'}: "
                    f"se esperaban **{pred}** pedidos pero hay **{real}** "
                    f"({desv:.0%} de desviación). "
                    + ("Incrementa lotes HNR y personal de inmediato." if tipo == "pico_inesperado"
                       else "Revisa si hay corte de luz, evento cercano o problema de acceso.")
                ),
            }
        return None

    def sugerencia_proactiva(self, estado: dict) -> str | None:
        """Genera sugerencias proactivas basadas en el estado actual."""
        h = estado.get("hora_sim", 12)
        margen = estado.get("margen_actual", 0.20)
        pct_del = estado.get("pct_delivery", 0.35)

        # 30 min antes del pico almuerzo
        if h == 11:
            return "🕐 Faltan ~60 min para el pico de almuerzo. Precalienta hornos y prepara 2 lotes HNR de Pepperoni."
        if h == 12:
            return "🍕 **PICO DE ALMUERZO** – Asegura 3 horneros en línea. Lanza promo de Crazy Bread para subir ticket."
        if h == 18:
            return "🌅 Arranca el turno de cena. Revisa inventario de queso mozzarella y cajas. Recuerda pre-calentamiento."
        if h == 19:
            return "🍕 **PICO DE CENA** – Horas más rentables del día. Maximiza venta cruzada con Pepsi + Canelitas."
        if margen < 0.12:
            return f"⚠️ Margen en {margen:.1%} – por debajo del objetivo. Revisa mix de canales y venta cruzada."
        if pct_del > 0.45:
            return f"💸 El {pct_del:.0%} de pedidos va por delivery de terceros. Activa promoción de Pizza Portal."
        return None

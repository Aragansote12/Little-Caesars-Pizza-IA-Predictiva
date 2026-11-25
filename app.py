"""
app.py – Little Caesars OpsAI Colombia
Dashboard principal con simulación en tiempo real, 4 pestañas estratégicas y Copilot IA.
"""

import time
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ─── Configuración de página (DEBE SER PRIMERA LLAMADA) ───────────────────────
st.set_page_config(
    page_title="Little Caesars OpsAI Colombia",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Imports locales ──────────────────────────────────────────────────────────
from data_generator import cargar_o_generar, PRECIOS
from ml_engine import LCMLEngine
from copilot_agent import CopilotAgent

# ═══════════════════════════════════════════════════════════════════════════════
# CSS GLOBAL – Estética SaaS moderna con identidad Little Caesars
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root variables ── */
:root {
  --lc-orange:    #FF671F;
  --lc-peach:     #FFF2EB;
  --lc-cream:     #FAF8F5;
  --lc-white:     #FFFFFF;
  --lc-stone:     #F3EFEA;
  --lc-border:    #ECE7E0;
  --lc-grid:      #EAE6DF;
  --lc-carbon:    #1E232B;
  --lc-green:     #10B981;
  --lc-red:       #EF4444;
  --lc-amber:     #F59E0B;
  --shadow-sm:    0 2px 8px rgba(30,35,43,0.06);
  --shadow-md:    0 4px 20px rgba(30,35,43,0.08);
  --radius:       16px;
  --radius-sm:    10px;
}

/* ── Global base (forzar tema claro incluso sin config.toml) ── */
:root { color-scheme: light !important; }
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[class*="css"] {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  background-color: var(--lc-cream) !important;
  color: var(--lc-carbon) !important;
}
h1, h2, h3, h4, h5, h6,
.stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
.lc-card, .lc-card-sm {
  color: var(--lc-carbon) !important;
}

/* ── Ocultar decoraciones Streamlit por defecto ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2rem 2rem 2rem !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--lc-stone) !important;
  border-right: 1px solid var(--lc-border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1rem 1rem 2rem 1rem !important; }

/* ── Métricas como tarjetas ── */
[data-testid="stMetric"] {
  background: var(--lc-white) !important;
  border: 1px solid var(--lc-border) !important;
  border-top: 3px solid var(--lc-orange) !important;
  border-radius: var(--radius-sm) !important;
  padding: 1rem 1.2rem !important;
  box-shadow: var(--shadow-sm) !important;
}
[data-testid="stMetric"] label {
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: #6B7280 !important;
}
[data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 1.55rem !important;
  font-weight: 600 !important;
  color: var(--lc-carbon) !important;
}
[data-testid="stMetricDelta"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.8rem !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 4px !important;
  background: var(--lc-stone) !important;
  border-radius: 10px !important;
  padding: 4px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  padding: 0.5rem 1rem !important;
  color: #6B7280 !important;
  background: transparent !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  background: var(--lc-white) !important;
  color: var(--lc-carbon) !important;
  box-shadow: var(--shadow-sm) !important;
}

/* ── Buttons ── */
.stButton > button {
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  border: 1.5px solid var(--lc-border) !important;
  background: var(--lc-white) !important;
  color: var(--lc-carbon) !important;
  transition: all 0.15s ease !important;
}
.stButton > button:hover {
  border-color: var(--lc-orange) !important;
  color: var(--lc-orange) !important;
  box-shadow: 0 0 0 3px rgba(255,103,31,0.08) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
  background: var(--lc-orange) !important;
  border-color: var(--lc-orange) !important;
}

/* ── Info/Success/Warning boxes ── */
.stAlert { border-radius: var(--radius-sm) !important; border-left-width: 4px !important; }

/* ── Expanders ── */
details summary { font-weight: 600 !important; }

/* ── Cards genéricas (via HTML) ── */
.lc-card {
  background: var(--lc-white);
  border: 1px solid var(--lc-border);
  border-radius: var(--radius);
  padding: 1.4rem 1.6rem;
  box-shadow: var(--shadow-md);
  margin-bottom: 1rem;
}
.lc-card-sm {
  background: var(--lc-white);
  border: 1px solid var(--lc-border);
  border-radius: var(--radius-sm);
  padding: 1rem 1.2rem;
  box-shadow: var(--shadow-sm);
  margin-bottom: 0.6rem;
}
.lc-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--lc-peach);
  color: var(--lc-orange);
  border-radius: 20px;
  padding: 3px 10px;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;
}
.lc-badge-green {
  background: #D1FAE5; color: #065F46;
}
.lc-badge-red {
  background: #FEE2E2; color: #991B1B;
}
.dot-green {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--lc-green);
  display: inline-block;
  box-shadow: 0 0 0 3px rgba(16,185,129,0.25);
  animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
  0%,100% { box-shadow: 0 0 0 3px rgba(16,185,129,0.25); }
  50%      { box-shadow: 0 0 0 6px rgba(16,185,129,0.10); }
}
.mono { font-family: 'JetBrains Mono', monospace; }
.lc-divider { border: none; border-top: 1px solid var(--lc-border); margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
def render_header(hora_sim: int, minuto_sim: int, dia_tipo: str, clima: str):
    now_str = f"{hora_sim:02d}:{minuto_sim:02d}"
    html = (
        f'<div style="background:#FFFFFF;border:1px solid #ECE7E0;border-radius:16px;'
        f'overflow:hidden;box-shadow:0 4px 20px rgba(30,35,43,0.07);margin-bottom:1.4rem;">'
        f'<div style="height:4px;background:linear-gradient(90deg,#FF671F 0%,#FF9357 100%);"></div>'
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'padding:1rem 1.6rem;flex-wrap:wrap;gap:0.8rem;">'
        f'<div style="display:flex;align-items:center;gap:14px;">'
        f'<div style="width:44px;height:44px;background:#FF671F;border-radius:10px;'
        f'display:flex;align-items:center;justify-content:center;font-size:1.5rem;line-height:1;">🍕</div>'
        f'<div>'
        f'<div style="font-size:1.15rem;font-weight:800;color:#1E232B;letter-spacing:-0.02em;">'
        f'Little Caesars <span style="color:#FF671F;">OpsAI</span></div>'
        f'<div style="font-size:0.72rem;color:#6B7280;font-weight:500;margin-top:1px;">'
        f'Bogot&aacute; D.C. &nbsp;&bull;&nbsp; Gesti&oacute;n Predictiva Hot-N-Ready</div>'
        f'</div></div>'
        f'<div style="text-align:center;">'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:2rem;font-weight:700;'
        f'color:#1E232B;letter-spacing:0.04em;line-height:1;">{now_str}</div>'
        f'<div style="font-size:0.68rem;color:#9CA3AF;font-weight:500;text-transform:uppercase;'
        f'letter-spacing:0.08em;margin-top:2px;">Hora Simulada &nbsp;|&nbsp; {dia_tipo} &nbsp;|&nbsp; {clima}</div>'
        f'</div>'
        f'<div>'
        f'<span style="display:inline-flex;align-items:center;gap:7px;background:#F0FDF4;'
        f'color:#166534;border:1px solid #BBF7D0;border-radius:20px;padding:5px 14px;'
        f'font-size:0.72rem;font-weight:700;letter-spacing:0.04em;">'
        f'<span style="width:7px;height:7px;border-radius:50%;background:#22C55E;'
        f'box-shadow:0 0 0 3px rgba(34,197,94,0.25);display:inline-block;"></span>'
        f'Agente Activo &nbsp;&bull;&nbsp; Simulaci&oacute;n en Vivo</span>'
        f'</div></div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS DE PLOTLY
# ═══════════════════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#1E232B", size=12),
    xaxis=dict(
        showgrid=True, gridcolor="#EAE6DF", gridwidth=1,
        zeroline=False, linecolor="#ECE7E0", tickfont=dict(size=11),
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#EAE6DF", gridwidth=1,
        zeroline=False, linecolor="#ECE7E0", tickfont=dict(size=11),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)", bordercolor="#ECE7E0",
        borderwidth=1, font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor="#FFFFFF", bordercolor="#ECE7E0",
        font=dict(family="Plus Jakarta Sans", size=12, color="#1E232B"),
    ),
)

COLORES = {
    "naranja":  "#FF671F",
    "durazno":  "#FF9357",
    "verde":    "#10B981",
    "rojo":     "#EF4444",
    "azul":     "#3B82F6",
    "ambar":    "#F59E0B",
    "gris":     "#9CA3AF",
    "morado":   "#8B5CF6",
}


def fig_base() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**PLOTLY_LAYOUT, margin=dict(l=10, r=10, t=36, b=10))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "hora_sim":        11,
        "minuto_sim":      0,
        "simulando":       False,
        "historial_horas": [],
        "chat_history":    [],
        "ultimo_tick":     0.0,
        "df_hist":         None,
        "ml":              None,
        "agent":           None,
        "entrenado":       False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# CARGA Y ENTRENAMIENTO (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def cargar_sistema():
    df = cargar_o_generar()
    ml = LCMLEngine()
    metricas = ml.entrenar(df)
    return df, ml, metricas

with st.spinner("⚙️ Inicializando OpsAI – Cargando datos y entrenando modelos…"):
    df_hist, ml_engine, metricas_ml = cargar_sistema()

if not st.session_state.entrenado:
    st.session_state.df_hist = df_hist
    st.session_state.ml      = ml_engine
    st.session_state.entrenado = True


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:0.5rem 0 1rem;'>
      <span style='font-size:1.8rem;'>🍕</span>
      <div style='font-size:0.75rem; font-weight:700; color:#FF671F;
                  letter-spacing:0.08em; text-transform:uppercase; margin-top:4px;'>
        PANEL DE CONTROL
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Control simulación ────────────────────────────────────────────────────
    st.markdown("#### ⏱️ Simulación en Tiempo Real")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶ Iniciar" if not st.session_state.simulando else "⏸ Pausar",
                     use_container_width=True):
            st.session_state.simulando = not st.session_state.simulando
    with col_b:
        if st.button("⏭ +1 Hora", use_container_width=True):
            st.session_state.hora_sim = (st.session_state.hora_sim - 11 + 1) % 11 + 11
            if st.session_state.hora_sim > 21:
                st.session_state.hora_sim = 11

    if st.button("🔄 Reiniciar Día", use_container_width=True):
        st.session_state.hora_sim = 11
        st.session_state.minuto_sim = 0
        st.session_state.simulando = False
        st.session_state.historial_horas = []

    # ── Selector de día ───────────────────────────────────────────────────────
    st.markdown("<hr class='lc-divider'>", unsafe_allow_html=True)
    st.markdown("#### 📅 Contexto del Día")
    dia_tipo = st.selectbox("Tipo de día", [
        "Martes Normal",
        "Viernes de Quincena",
        "Sábado Festivo",
        "Domingo de Puente con Partido",
    ])
    clima_sel = st.selectbox("🌤️ Clima actual", [
        "Soleado", "Nublado", "Lluvia Leve", "Aguacero"
    ])

    # Mapeo tipo-día → parámetros
    DIA_MAP = {
        "Martes Normal":                    dict(dow=1, dia=10, es_festivo=False, es_partido=False, es_qncn=False, es_fin=False),
        "Viernes de Quincena":              dict(dow=4, dia=15, es_festivo=False, es_partido=False, es_qncn=True,  es_fin=True),
        "Sábado Festivo":                   dict(dow=5, dia=7,  es_festivo=True,  es_partido=False, es_qncn=False, es_fin=True),
        "Domingo de Puente con Partido":    dict(dow=6, dia=20, es_festivo=True,  es_partido=True,  es_qncn=False, es_fin=True),
    }
    ctx = DIA_MAP[dia_tipo]

    # ── Palancas estratégicas ──────────────────────────────────────────────────
    st.markdown("<hr class='lc-divider'>", unsafe_allow_html=True)
    st.markdown("#### ⚡ Palancas Estratégicas")

    inflacion_pct = st.slider(
        "📈 Inflación Queso & Harina (%)", 0, 25, 0,
        help="Simula el impacto de inflación en materias primas sobre el margen neto."
    ) / 100.0

    pct_app = st.slider(
        "📱 Migración → App Oficial / Pizza Portal (%)", 0, 100, 25,
        help="Mueve pedidos desde Rappi/Delivery (30% comisión) a canal propio (0%)."
    ) / 100.0

    impulso_vc = st.slider(
        "🍞 Impulso Venta Cruzada (%)", 0, 100, 0,
        help="Aumenta la tasa de complementos (Borde, Crazy Bread, Pepsi) por pedido."
    ) / 100.0

    # ── API Key opcional ───────────────────────────────────────────────────────
    st.markdown("<hr class='lc-divider'>", unsafe_allow_html=True)
    with st.expander("🔑 API Key (opcional)"):
        proveedor = st.radio("Proveedor IA", ["Offline", "Gemini", "OpenAI"], index=0)
        api_key   = st.text_input("API Key", type="password", placeholder="sk-... / AI...")

    # Inicializar agente
    if st.session_state.agent is None:
        st.session_state.agent = CopilotAgent(api_key, proveedor)
    else:
        st.session_state.agent.api_key   = api_key
        st.session_state.agent.proveedor = proveedor


# ═══════════════════════════════════════════════════════════════════════════════
# LÓGICA DE SIMULACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
hora_sim   = st.session_state.hora_sim
minuto_sim = st.session_state.minuto_sim

# Tick automático cada ~2 segundos
if st.session_state.simulando:
    ahora = time.time()
    if ahora - st.session_state.ultimo_tick >= 2.0:
        st.session_state.ultimo_tick = ahora
        st.session_state.minuto_sim += 60   # avanza 1 hora
        if st.session_state.minuto_sim >= 60:
            st.session_state.minuto_sim = 0
            st.session_state.hora_sim += 1
            if st.session_state.hora_sim > 21:
                st.session_state.hora_sim  = 11
                st.session_state.simulando = False
        hora_sim   = st.session_state.hora_sim
        minuto_sim = st.session_state.minuto_sim
        st.rerun()

# Predicción hora actual
pred_actual = ml_engine.predecir_hora(
    hora_sim, ctx["dow"], ctx["dia"], 3,
    clima_sel, ctx["es_festivo"], ctx["es_partido"],
    ctx["es_qncn"], ctx["es_fin"],
    inflacion_pct, pct_app, impulso_vc,
)

# Acumular historial de horas simuladas con demanda real simulada
hh = st.session_state.historial_horas
if hh and hora_sim < hh[-1]["hora"]:
    hh = []
if not hh or hh[-1]["hora"] != hora_sim:
    factor_real = float(np.random.default_rng(hora_sim * 31 + ctx["dia"]).uniform(0.90, 1.12))
    pedidos_real = max(1, int(round(pred_actual["pedidos_pred"] * factor_real)))
    hh.append({**pred_actual, "hora": hora_sim, "pedidos_real": pedidos_real})
    if len(hh) > 11:
        hh = hh[-11:]
    st.session_state.historial_horas = hh

# Predicción del día completo
df_dia = ml_engine.predecir_dia(
    ctx["dow"], ctx["dia"], 3,
    clima_sel, ctx["es_festivo"], ctx["es_partido"],
    ctx["es_qncn"], ctx["es_fin"],
    inflacion_pct, pct_app, impulso_vc,
)

# Estado para el agente copilot
estado_actual = {
    "hora_sim":        hora_sim,
    "clima":           clima_sel,
    "pedidos_actuales": pred_actual["pedidos_pred"],
    "ingresos_hora":   pred_actual["ingresos_pred"],
    "margen_actual":   pred_actual["margen_pred"],
    "personal_total":  pred_actual["personal_total"],
    "es_pico":         pred_actual["es_pico"],
    "pct_delivery":    max(0, 0.35 - pct_app * 0.6),
}


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
render_header(hora_sim, minuto_sim, dia_tipo, clima_sel)

# ─── Sugerencia proactiva del agente ─────────────────────────────────────────
sug = st.session_state.agent.sugerencia_proactiva(estado_actual)
if sug:
    st.info(sug)

# ─── KPIs principales ─────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

ingresos_dia  = int(df_dia["ingresos_pred"].sum())
utilidad_dia  = int(df_dia["utilidad_pred"].sum())
pedidos_dia   = int(df_dia["pedidos_pred"].sum())
margen_dia    = utilidad_dia / ingresos_dia if ingresos_dia > 0 else 0
ahorro_total  = int(df_dia["ahorro_comision"].sum())

with k1:
    st.metric("💰 Ingreso Proyectado / Día",
              f"${ingresos_dia/1_000_000:.2f}M COP",
              f"+{(pct_app*ahorro_total/ingresos_dia*100):.1f}% vs sin App" if pct_app > 0.1 else None)
with k2:
    delta_margen = f"{margen_dia:.1%}"
    st.metric("📊 Margen Neto Día",
              delta_margen,
              f"{'▲' if inflacion_pct == 0 else '▼'} {abs(inflacion_pct*3.5):.1f}pp inflación")
with k3:
    st.metric("🛒 Pedidos Proyectados",
              f"{pedidos_dia:,}",
              f"Hora actual: {pred_actual['pedidos_pred']}")
with k4:
    color_ahorro = "normal" if ahorro_total > 0 else "off"
    st.metric("💸 Ahorro en Comisiones",
              f"${ahorro_total/1_000:.0f}K COP",
              f"App Oficial {pct_app:.0%}")
with k5:
    st.metric("👷 Personal Esta Hora",
              f"{pred_actual['personal_total']} personas",
              f"{'🔴 PICO' if pred_actual['es_pico'] else '🟢 Normal'}")


# ═══════════════════════════════════════════════════════════════════════════════
# PESTAÑAS PRINCIPALES
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🍕 Horas Pico & Hot-N-Ready",
    "💰 Márgenes & Canal Mix",
    "📊 Análisis Histórico",
    "🤖 Copilot IA",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICCIÓN HORAS PICO, TURNOS Y HOT-N-READY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_main, col_side = st.columns([2, 1], gap="large")

    with col_main:
        st.markdown("### 📈 Demanda en Vivo vs. Predicción IA")

        # Gráfico principal: demanda hora por hora
        fig_dem = fig_base()

        horas_x    = df_dia["hora"].tolist()
        pred_y     = df_dia["pedidos_pred"].tolist()

        # Relleno de fondo bajo la curva predicha
        fig_dem.add_trace(go.Scatter(
            x=horas_x, y=pred_y,
            fill="tozeroy",
            fillcolor="rgba(255,103,31,0.07)",
            line=dict(color=COLORES["naranja"], width=2.5, shape="spline"),
            mode="lines",
            name="Predicción IA",
        ))

        # Demanda real acumulada hasta la hora simulada
        if st.session_state.historial_horas:
            real_x = [r["hora"] for r in st.session_state.historial_horas]
            real_y = [r["pedidos_real"] for r in st.session_state.historial_horas]
            fig_dem.add_trace(go.Scatter(
                x=real_x, y=real_y,
                mode="lines+markers",
                line=dict(color="#1E232B", width=2.5, shape="spline"),
                marker=dict(size=7, color="#1E232B", line=dict(width=1.5, color="white")),
                name="Demanda Real (En Vivo)",
            ))

        # Picos marcados
        pico_x = [h for h, p in zip(horas_x, df_dia["es_pico"].tolist()) if p]
        pico_y = [p for p, es in zip(pred_y, df_dia["es_pico"].tolist()) if es]
        if pico_x:
            fig_dem.add_trace(go.Scatter(
                x=pico_x, y=pico_y,
                mode="markers",
                marker=dict(size=10, color=COLORES["naranja"],
                            line=dict(width=2, color="white"),
                            symbol="circle"),
                name="⚡ Hora Pico",
            ))

        # Hora actual (línea vertical)
        if 11 <= hora_sim <= 21:
            fig_dem.add_vline(
                x=hora_sim,
                line_dash="dot", line_color=COLORES["ambar"],
                line_width=2,
                annotation_text=f" Ahora {hora_sim}:00",
                annotation_font_color=COLORES["ambar"],
                annotation_font_size=11,
            )

        # Franjas de pico almuerzo y cena
        for x0, x1, label in [(12, 14.5, "🍽️ Pico Almuerzo"), (19, 21, "🌙 Pico Cena")]:
            fig_dem.add_vrect(
                x0=x0, x1=x1,
                fillcolor="rgba(255,103,31,0.06)",
                layer="below", line_width=0,
                annotation_text=label,
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="#FF671F",
            )

        fig_dem.update_layout(
            height=320,
            xaxis_title="Hora del Día",
            yaxis_title="Pedidos Proyectados",
            xaxis=dict(tickvals=list(range(11, 22)), ticktext=[f"{h}:00" for h in range(11, 22)]),
            showlegend=True,
        )
        st.plotly_chart(fig_dem, use_container_width=True, config={"displayModeBar": False})

        # ── Optimizador de Turnos ──────────────────────────────────────────────
        st.markdown("### 👷 Planificación Óptima de Turnos")
        fig_staff = fig_base()
        fig_staff.add_trace(go.Bar(
            x=horas_x,
            y=df_dia["horneros"].tolist(),
            name="🔥 Horneros",
            marker_color=COLORES["naranja"],
            marker_opacity=0.85,
        ))
        fig_staff.add_trace(go.Bar(
            x=horas_x,
            y=df_dia["armadores"].tolist(),
            name="🤝 Armadores",
            marker_color=COLORES["durazno"],
            marker_opacity=0.85,
        ))
        fig_staff.add_trace(go.Bar(
            x=horas_x,
            y=df_dia["cajeros"].tolist(),
            name="💳 Cajeros/Portal",
            marker_color=COLORES["ambar"],
            marker_opacity=0.85,
        ))
        fig_staff.update_layout(
            height=240,
            barmode="stack",
            xaxis=dict(tickvals=list(range(11, 22)), ticktext=[f"{h}:00" for h in range(11, 22)]),
            yaxis_title="Personas",
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_staff, use_container_width=True, config={"displayModeBar": False})

    with col_side:
        st.markdown("### 🍕 Control Hot-N-Ready")
        st.markdown("""
        <div class='lc-card-sm' style='border-top:3px solid #FF671F;'>
          <div style='font-size:0.68rem;font-weight:700;color:#9CA3AF;letter-spacing:.06em;
                      text-transform:uppercase;margin-bottom:.5rem;'>
            🔥 Lotes para los próximos 30 min
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Lotes recomendados para las próximas 3 horas
        horas_prox = [h for h in range(hora_sim, min(hora_sim + 4, 22))]
        for h in horas_prox:
            if h > 21:
                break
            pr = ml_engine.predecir_hora(
                h, ctx["dow"], ctx["dia"], 3,
                clima_sel, ctx["es_festivo"], ctx["es_partido"],
                ctx["es_qncn"], ctx["es_fin"],
                inflacion_pct, pct_app, impulso_vc,
            )
            pico_badge = "🔴" if pr["es_pico"] else "🟢"
            curr = "▶ " if h == hora_sim else ""
            st.markdown(f"""
            <div class='lc-card-sm' style='{"border-left:3px solid #FF671F;" if h == hora_sim else ""}'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='font-weight:700;font-size:0.85rem;'>{curr}{h}:00h {pico_badge}</span>
                <span class='lc-badge' style='font-size:0.65rem;'>
                  {pr["pedidos_pred"]} pedidos
                </span>
              </div>
              <div style='margin-top:.5rem;display:flex;gap:.5rem;flex-wrap:wrap;'>
                <div style='background:#FFF2EB;border-radius:8px;padding:5px 10px;font-size:0.78rem;'>
                  🍕 <b>{pr["lotes_pepp"]}</b> lotes Pepperoni
                </div>
                <div style='background:#FFF9EE;border-radius:8px;padding:5px 10px;font-size:0.78rem;'>
                  🧀 <b>{pr["lotes_queso"]}</b> lotes Queso
                </div>
              </div>
              <div style='margin-top:.4rem;display:flex;gap:.4rem;flex-wrap:wrap;font-size:0.72rem;color:#6B7280;'>
                <span>🔥 {pr["horneros"]} horneros</span>
                <span>• 🤝 {pr["armadores"]} armadores</span>
                <span>• 💳 {pr["cajeros"]} cajeros</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Indicador tipo cronómetro
        st.markdown("""
        <div class='lc-card-sm' style='background:#FFF9F5;border:1.5px dashed #FF671F;text-align:center;margin-top:.5rem;'>
          <div style='font-size:1.5rem;'>⏱️</div>
          <div style='font-size:0.72rem;font-weight:700;color:#FF671F;margin-top:4px;'>
            HOT-N-READY TIMER
          </div>
          <div style='font-family:\"JetBrains Mono\",monospace;font-size:1.2rem;
                      font-weight:700;color:#1E232B;margin-top:4px;'>
            30:00 min
          </div>
          <div style='font-size:0.65rem;color:#9CA3AF;margin-top:3px;'>
            Vida útil máxima en calentador
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Resumen de personal turno completo
        personal_dia = df_dia["personal_total"].max()
        costo_nom_hr = pred_actual["costo_nomina"]
        st.markdown(f"""
        <div class='lc-card-sm' style='margin-top:.5rem;'>
          <div style='font-size:0.68rem;font-weight:700;color:#9CA3AF;text-transform:uppercase;
                      letter-spacing:.06em;margin-bottom:.6rem;'>💰 Nómina Esta Hora</div>
          <div style='font-family:\"JetBrains Mono\",monospace;font-size:1.3rem;
                      font-weight:700;color:#1E232B;'>
            ${costo_nom_hr:,.0f}
          </div>
          <div style='font-size:0.72rem;color:#6B7280;margin-top:3px;'>COP nómina + operación</div>
          <div style='margin-top:.5rem;font-size:0.78rem;'>
            Pico del día: <b>{personal_dia}</b> personas
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: MÁRGENES, ESCENARIOS Y CANAL MIX
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown("### 💰 Cascada de Ingresos y Márgenes (Día Completo)")

        # Cascada financiera
        ingresos_hr   = df_dia["ingresos_pred"].tolist()
        utilidad_hr   = df_dia["utilidad_pred"].tolist()
        costo_mp_hr   = df_dia["costo_mp"].tolist()
        costo_nom_hr  = df_dia["costo_nomina"].tolist()
        horas_x       = df_dia["hora"].tolist()

        fig_fin = fig_base()
        fig_fin.add_trace(go.Scatter(
            x=horas_x, y=[v / 1000 for v in ingresos_hr],
            name="Ingreso Bruto",
            fill="tozeroy", fillcolor="rgba(255,103,31,0.08)",
            line=dict(color=COLORES["naranja"], width=2.5, shape="spline"),
        ))
        fig_fin.add_trace(go.Scatter(
            x=horas_x, y=[v / 1000 for v in utilidad_hr],
            name="Utilidad Neta",
            fill="tozeroy", fillcolor="rgba(16,185,129,0.1)",
            line=dict(color=COLORES["verde"], width=2.5, shape="spline"),
        ))
        fig_fin.add_trace(go.Scatter(
            x=horas_x, y=[v / 1000 for v in costo_mp_hr],
            name="Costo Materia Prima",
            line=dict(color=COLORES["rojo"], width=1.5, dash="dot", shape="spline"),
        ))
        fig_fin.update_layout(
            height=300,
            xaxis=dict(tickvals=list(range(11, 22)), ticktext=[f"{h}:00" for h in range(11, 22)]),
            yaxis_title="Miles COP",
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_fin, use_container_width=True, config={"displayModeBar": False})

        # Escenario "¿Qué pasa si?" Comparativa
        st.markdown("### ⚡ Simulador de Escenarios: ¿Qué Pasa Si...?")

        # Calculamos 3 escenarios para comparar
        def calc_escenario(inf, app, vc):
            d = ml_engine.predecir_dia(ctx["dow"], ctx["dia"], 3, clima_sel,
                                       ctx["es_festivo"], ctx["es_partido"],
                                       ctx["es_qncn"], ctx["es_fin"], inf, app, vc)
            ing = d["ingresos_pred"].sum()
            util = d["utilidad_pred"].sum()
            return dict(ingresos=ing, utilidad=util, margen=util/ing if ing > 0 else 0)

        esc_base    = calc_escenario(0, 0.10, 0)
        esc_actual  = calc_escenario(inflacion_pct, pct_app, impulso_vc)
        esc_optimo  = calc_escenario(0, 0.70, 0.60)

        nombres_esc = ["Base (sin IA)", "Actual (tus palancas)", "Óptimo (IA máx.)"]
        ingresos_e  = [esc_base["ingresos"], esc_actual["ingresos"], esc_optimo["ingresos"]]
        utilidad_e  = [esc_base["utilidad"], esc_actual["utilidad"], esc_optimo["utilidad"]]
        margen_e    = [esc_base["margen"],   esc_actual["margen"],   esc_optimo["margen"]]

        fig_esc = fig_base()
        colores_esc = [COLORES["gris"], COLORES["naranja"], COLORES["verde"]]
        fig_esc.add_trace(go.Bar(
            x=nombres_esc,
            y=[v / 1_000_000 for v in ingresos_e],
            name="Ingreso (M COP)",
            marker_color=colores_esc, marker_opacity=0.75,
        ))
        fig_esc.add_trace(go.Bar(
            x=nombres_esc,
            y=[v / 1_000_000 for v in utilidad_e],
            name="Utilidad Neta (M COP)",
            marker_color=colores_esc, marker_opacity=0.95,
            marker_pattern_shape="x",
        ))
        fig_esc.update_layout(
            height=260, barmode="group",
            yaxis_title="Millones COP",
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_esc, use_container_width=True, config={"displayModeBar": False})

        # Tabla comparativa
        df_esc = pd.DataFrame({
            "Escenario":        nombres_esc,
            "Ingresos (M COP)": [f"${v/1_000_000:.2f}M" for v in ingresos_e],
            "Utilidad (M COP)": [f"${v/1_000_000:.2f}M" for v in utilidad_e],
            "Margen Neto":      [f"{v:.1%}" for v in margen_e],
        })
        st.dataframe(df_esc, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("### 📡 Mix de Canales de Venta")

        pct_del_actual = max(0.05, 0.35 - pct_app * 0.6)
        pct_app_actual = min(0.60, 0.25 + pct_app * 0.35)
        pct_most       = max(0.05, 1.0 - pct_del_actual - pct_app_actual)
        total_c = pct_del_actual + pct_app_actual + pct_most
        pct_del_actual /= total_c; pct_app_actual /= total_c; pct_most /= total_c

        fig_canal = go.Figure(go.Pie(
            labels=["🛵 Delivery Terceros", "📱 App Oficial / Portal", "🏪 Mostrador"],
            values=[pct_del_actual, pct_app_actual, pct_most],
            hole=0.62,
            marker_colors=[COLORES["rojo"], COLORES["verde"], COLORES["naranja"]],
            textinfo="label+percent",
            textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
        ))
        fig_canal.update_layout(
            **PLOTLY_LAYOUT,
            height=240,
            showlegend=False,
            annotations=[dict(
                text=f"<b>{pct_del_actual:.0%}</b><br><span style='font-size:9px'>Delivery</span>",
                x=0.5, y=0.5, font_size=14, showarrow=False, font_color=COLORES["rojo"],
            )],
        )
        st.plotly_chart(fig_canal, use_container_width=True, config={"displayModeBar": False})

        # Impacto financiero de comisiones
        ing_total_dia = df_dia["ingresos_pred"].sum()
        comision_del  = ing_total_dia * pct_del_actual * 0.30
        ahorro_vs_base = max(0, ing_total_dia * (0.35 - pct_del_actual) * 0.30)

        st.markdown(f"""
        <div class='lc-card'>
          <div style='font-size:0.68rem;font-weight:700;color:#9CA3AF;letter-spacing:.06em;
                      text-transform:uppercase;margin-bottom:.8rem;'>
            💸 Impacto Comisiones Hoy
          </div>
          <div style='display:flex;flex-direction:column;gap:.6rem;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
              <span style='font-size:.82rem;'>Comisión delivery (30%)</span>
              <span class='mono' style='font-weight:700;color:#EF4444;'>
                -${comision_del/1000:.0f}K
              </span>
            </div>
            <div style='border-top:1px dashed #ECE7E0;padding-top:.6rem;
                        display:flex;justify-content:space-between;align-items:center;'>
              <span style='font-size:.82rem;font-weight:600;'>✅ Ahorro vs. 100% delivery</span>
              <span class='mono' style='font-weight:700;color:#10B981;'>
                +${ahorro_vs_base/1000:.0f}K
              </span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Desglose de costos
        st.markdown("### 🧮 Estructura de Costos")
        costo_mp_total  = df_dia["costo_mp"].sum()
        costo_nom_total = df_dia["costo_nomina"].sum()
        costo_fijo_tot  = 18_500 * 11
        costo_com_tot   = comision_del
        ing_tot         = df_dia["ingresos_pred"].sum()

        fig_costs = go.Figure(go.Treemap(
            labels=["Total Costos", "Materia Prima\n(32.4%)", "Nómina\n(31.7%)",
                    "Comisiones\nDelivery", "Costos Fijos"],
            parents=["", "Total Costos", "Total Costos", "Total Costos", "Total Costos"],
            values=[costo_mp_total + costo_nom_total + costo_com_tot + costo_fijo_tot,
                    costo_mp_total, costo_nom_total, costo_com_tot, costo_fijo_tot],
            marker_colorscale=[[0, "#FFF2EB"], [0.5, "#FF9357"], [1, "#FF671F"]],
            textfont=dict(size=11, color="#1E232B"),
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f} COP<extra></extra>",
        ))
        fig_costs.update_layout(**PLOTLY_LAYOUT, height=220)
        st.plotly_chart(fig_costs, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ANÁLISIS HISTÓRICO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Análisis del Histórico 6 Meses — Bogotá Zona T")

    res = ml_engine.resumen_historico()
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    with col_h1:
        st.metric("📦 Total Pedidos Históricos", f"{res['total_pedidos']:,}")
    with col_h2:
        st.metric("💰 Ingresos Totales",   f"${res['ingreso_total']/1_000_000:.1f}M COP")
    with col_h3:
        st.metric("📈 Margen Promedio",    f"{res['margen_promedio']:.1%}")
    with col_h4:
        DIAS_NOMBRES = ["Lunes", "Martes", "Miércoles", "Jueves",
                        "Viernes", "Sábado", "Domingo"]
        st.metric("🏆 Día Más Rentable",   DIAS_NOMBRES[res["dia_top"]])

    col_h_a, col_h_b = st.columns(2, gap="large")

    with col_h_a:
        # Heatmap de demanda por hora y día de semana
        st.markdown("#### 🌡️ Mapa de Calor: Demanda (Hora × Día)")
        pivot = df_hist.groupby(["dia_semana", "hora"])["pedidos_total"].mean().reset_index()
        pivot_m = pivot.pivot(index="dia_semana", columns="hora", values="pedidos_total").fillna(0)
        pivot_m.index = [DIAS_NOMBRES[i] for i in pivot_m.index]

        fig_heat = go.Figure(go.Heatmap(
            z=pivot_m.values,
            x=[f"{h}:00" for h in pivot_m.columns],
            y=pivot_m.index.tolist(),
            colorscale=[[0, "#FFF2EB"], [0.5, "#FF9357"], [1, "#FF671F"]],
            showscale=True,
            hovertemplate="<b>%{y} %{x}</b><br>Pedidos prom: %{z:.0f}<extra></extra>",
        ))
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=280, margin=dict(l=80, r=10, t=10, b=40))
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

        # Efecto quincena
        st.markdown("#### 📅 Efecto Quincena Colombiana")
        df_qncn = df_hist.groupby("es_quincena")["pedidos_total"].mean().reset_index()
        df_qncn["Período"] = df_qncn["es_quincena"].map({False: "Días Normales", True: "Días Quincena"})
        fig_q = go.Figure(go.Bar(
            x=df_qncn["Período"],
            y=df_qncn["pedidos_total"],
            marker_color=[COLORES["gris"], COLORES["naranja"]],
            marker_opacity=0.85,
            text=[f"{v:.0f}" for v in df_qncn["pedidos_total"]],
            textposition="outside",
        ))
        fig_q.update_layout(**PLOTLY_LAYOUT, height=200, yaxis_title="Pedidos Prom./hora",
                            showlegend=False)
        st.plotly_chart(fig_q, use_container_width=True, config={"displayModeBar": False})

    with col_h_b:
        # Mix de productos histórico
        st.markdown("#### 🍕 Mix de Productos Vendidos (6 meses)")
        cols_prod = ["unid_pizza_pepp_hot", "unid_pizza_queso_hot", "unid_pizza_borde",
                     "unid_crazy_bread", "unid_canelitas", "unid_pepsi"]
        nombres_prod = ["Pizza Pepperoni H-N-R", "Pizza Queso H-N-R", "Pizza Borde Relleno",
                        "Crazy Bread", "Canelitas", "Pepsi"]
        totales = [df_hist[c].sum() for c in cols_prod]
        colores_prod = [COLORES["naranja"], COLORES["durazno"], COLORES["ambar"],
                        COLORES["verde"], COLORES["morado"], COLORES["azul"]]

        fig_prod = go.Figure(go.Bar(
            x=nombres_prod, y=totales,
            marker_color=colores_prod, marker_opacity=0.85,
            text=[f"{v:,.0f}" for v in totales],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Total: %{y:,.0f}<extra></extra>",
        ))
        fig_prod.update_layout(**PLOTLY_LAYOUT, height=240,
                               yaxis_title="Unidades Vendidas",
                               xaxis_tickangle=-25, showlegend=False)
        st.plotly_chart(fig_prod, use_container_width=True, config={"displayModeBar": False})

        # Margen por canal
        st.markdown("#### 💸 Impacto del Canal en Rentabilidad")
        fig_chan = go.Figure()
        canales  = ["Mostrador\n(0% comisión)", "App Oficial\n(0% comisión)", "Delivery Terceros\n(30% comisión)"]
        margenes = [res["margen_promedio"] + 0.05,
                    res["margen_promedio"] + 0.03,
                    res["margen_promedio"] - 0.09]
        bar_colors = [COLORES["verde"], COLORES["naranja"], COLORES["rojo"]]
        fig_chan.add_trace(go.Bar(
            x=canales, y=margenes,
            marker_color=bar_colors, marker_opacity=0.85,
            text=[f"{v:.1%}" for v in margenes],
            textposition="outside",
        ))
        fig_chan.update_layout(
            **PLOTLY_LAYOUT, height=230,
            yaxis_title="Margen Neto Est.", yaxis_tickformat=".0%",
            showlegend=False,
        )
        st.plotly_chart(fig_chan, use_container_width=True, config={"displayModeBar": False})

        # Feature importances del modelo
        st.markdown("#### 🤖 Variables Clave del Modelo RF")
        fi_df = ml_engine.importancias()
        if not fi_df.empty:
            fi_df["feature_label"] = fi_df["feature"].map({
                "hora": "Hora del día", "dia_semana": "Día de semana",
                "es_quincena": "Es Quincena", "es_fin_semana": "Es Fin de Semana",
                "clima_enc": "Clima", "es_festivo": "Es Festivo",
                "es_partido": "Es Partido", "dia_mes": "Día del mes",
                "mes": "Mes",
            }).fillna(fi_df["feature"])
            fig_fi = fig_base()
            fig_fi.add_trace(go.Bar(
                x=fi_df["importance"].head(7),
                y=fi_df["feature_label"].head(7),
                orientation="h",
                marker_color=COLORES["naranja"],
                marker_opacity=0.85,
            ))
            fig_fi.update_layout(
                height=200,
                xaxis_title="Importancia (RF)",
                showlegend=False,
            )
            fig_fi.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

    # Métricas del modelo
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🎯 MAE Demanda (RF)",   f"±{metricas_ml['mae_demanda']:.1f} pedidos")
    with c2:
        st.metric("💡 MAE Ingresos (GBM)", f"±${metricas_ml['mae_ingresos']/1000:.1f}K COP")
    with c3:
        st.metric("⚡ Umbral Hora Pico",   f"{metricas_ml['umbral_pico']:.0f} pedidos/h")
    with c4:
        st.metric("📂 Registros Históricos", f"{metricas_ml['registros']:,}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: COPILOT IA
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    col_chat, col_ctx = st.columns([2, 1], gap="large")

    with col_chat:
        st.markdown("### 🤖 OpsAI Copilot — Pregunta al Agente")
        st.markdown("""
        <div class='lc-card-sm' style='background:#F0FDF4;border-color:#BBF7D0;'>
          <span class='dot-green'></span>&nbsp;
          <span style='font-size:.78rem;font-weight:600;color:#166534;'>
            Agente activo · Modo: {}  · Análisis sobre datos reales del simulador
          </span>
        </div>
        """.format("Offline" if not api_key else proveedor), unsafe_allow_html=True)

        # Renderizar historial de chat
        for msg in st.session_state.chat_history:
            if msg["rol"] == "usuario":
                st.markdown(f"""
                <div style='display:flex;justify-content:flex-end;margin:.4rem 0;'>
                  <div style='background:#FF671F;color:white;border-radius:14px 14px 2px 14px;
                              padding:.7rem 1rem;max-width:80%;font-size:.85rem;'>
                    {msg["msg"]}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='display:flex;justify-content:flex-start;margin:.4rem 0;'>
                  <div style='background:white;border:1px solid #ECE7E0;
                              border-radius:14px 14px 14px 2px;
                              padding:.7rem 1rem;max-width:80%;font-size:.85rem;
                              box-shadow:0 2px 8px rgba(30,35,43,.05);'>
                    {msg["msg"]}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # Preguntas sugeridas
        st.markdown("**💡 Preguntas rápidas:**")
        sugerencias = [
            "¿Cuáles son las horas pico de hoy?",
            "¿Cómo afecta la inflación al margen?",
            "¿Cuánto ahorro si migro al Pizza Portal?",
            "¿Qué pasa si llueve a las 7pm?",
            "¿Cuánto personal necesito en hora pico?",
            "Explícame el efecto quincena",
        ]
        cols_sug = st.columns(3)
        for i, sug in enumerate(sugerencias):
            with cols_sug[i % 3]:
                if st.button(f"💬 {sug}", key=f"sug_{i}", use_container_width=True):
                    respuesta = st.session_state.agent.responder(sug, estado_actual)
                    st.session_state.chat_history.append({"rol": "usuario", "msg": sug})
                    st.session_state.chat_history.append({"rol": "agente", "msg": respuesta})
                    st.rerun()

        # Input de chat
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Escribe tu pregunta al Agente OpsAI…",
                placeholder="Ej: ¿Qué lotes Hot-N-Ready debo tener listos a las 7pm del viernes?",
                label_visibility="collapsed",
            )
            enviado = st.form_submit_button("Enviar →", use_container_width=True)

        if enviado and user_input.strip():
            respuesta = st.session_state.agent.responder(user_input.strip(), estado_actual)
            st.session_state.chat_history.append({"rol": "usuario", "msg": user_input.strip()})
            st.session_state.chat_history.append({"rol": "agente", "msg": respuesta})
            st.rerun()

    with col_ctx:
        st.markdown("### 📡 Contexto en Vivo")

        # Estado actual del simulador
        pico_color = "#EF4444" if pred_actual["es_pico"] else "#10B981"
        pico_texto = "HORA PICO 🔴" if pred_actual["es_pico"] else "Ritmo Normal 🟢"
        st.markdown(f"""
        <div class='lc-card'>
          <div style='font-size:.68rem;font-weight:700;color:#9CA3AF;text-transform:uppercase;
                      letter-spacing:.06em;margin-bottom:.8rem;'>Estado del Simulador</div>
          <div style='display:flex;flex-direction:column;gap:.5rem;'>
            <div style='display:flex;justify-content:space-between;'>
              <span style='font-size:.82rem;'>🕐 Hora simulada</span>
              <span class='mono' style='font-weight:700;'>{hora_sim:02d}:00h</span>
            </div>
            <div style='display:flex;justify-content:space-between;'>
              <span style='font-size:.82rem;'>🌤️ Clima</span>
              <span style='font-weight:600;'>{clima_sel}</span>
            </div>
            <div style='display:flex;justify-content:space-between;'>
              <span style='font-size:.82rem;'>🛒 Pedidos/hora</span>
              <span class='mono' style='font-weight:700;'>{pred_actual["pedidos_pred"]}</span>
            </div>
            <div style='display:flex;justify-content:space-between;'>
              <span style='font-size:.82rem;'>💰 Ingresos/hora</span>
              <span class='mono' style='font-weight:700;'>${pred_actual["ingresos_pred"]/1000:.0f}K COP</span>
            </div>
            <div style='display:flex;justify-content:space-between;'>
              <span style='font-size:.82rem;'>📊 Margen neto</span>
              <span class='mono' style='font-weight:700;color:{"#10B981" if pred_actual["margen_pred"] > 0.15 else "#EF4444"};'>
                {pred_actual["margen_pred"]:.1%}
              </span>
            </div>
            <div style='display:flex;justify-content:space-between;'>
              <span style='font-size:.82rem;'>👷 Personal</span>
              <span class='mono' style='font-weight:700;'>{pred_actual["personal_total"]} personas</span>
            </div>
            <div style='border-top:1px dashed #ECE7E0;margin-top:.4rem;padding-top:.5rem;'>
              <span style='font-size:.85rem;font-weight:700;color:{pico_color};'>{pico_texto}</span>
            </div>
          </div>
        </div>

        <div class='lc-card' style='margin-top:.5rem;'>
          <div style='font-size:.68rem;font-weight:700;color:#9CA3AF;text-transform:uppercase;
                      letter-spacing:.06em;margin-bottom:.8rem;'>⚡ Palancas Activas</div>
          <div style='display:flex;flex-direction:column;gap:.5rem;'>
            <div style='display:flex;justify-content:space-between;'>
              <span style='font-size:.78rem;'>📈 Inflación</span>
              <span class='lc-badge {"lc-badge-red" if inflacion_pct > 0.10 else ""}'
                    style='font-size:.68rem;'>
                +{inflacion_pct:.0%}
              </span>
            </div>
            <div style='display:flex;justify-content:space-between;'>
              <span style='font-size:.78rem;'>📱 App Oficial</span>
              <span class='lc-badge {"lc-badge-green" if pct_app > 0.40 else ""}'
                    style='font-size:.68rem;'>
                {pct_app:.0%}
              </span>
            </div>
            <div style='display:flex;justify-content:space-between;'>
              <span style='font-size:.78rem;'>🍞 Venta Cruzada</span>
              <span class='lc-badge' style='font-size:.68rem;'>+{impulso_vc:.0%}</span>
            </div>
            <div style='display:flex;justify-content:space-between;'>
              <span style='font-size:.78rem;'>💸 Ahorro comis.</span>
              <span class='mono' style='font-weight:700;color:#10B981;font-size:.82rem;'>
                +${pred_actual["ahorro_comision"]/1000:.0f}K COP/h
              </span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Lotes HNR actuales
        st.markdown(f"""
        <div class='lc-card-sm' style='border-top:3px solid #FF671F;'>
          <div style='font-size:.68rem;font-weight:700;color:#9CA3AF;text-transform:uppercase;
                      letter-spacing:.06em;margin-bottom:.6rem;'>🍕 Hot-N-Ready Ahora</div>
          <div style='display:flex;gap:.5rem;flex-wrap:wrap;'>
            <div style='background:#FFF2EB;border-radius:8px;padding:6px 12px;'>
              🍕 <b>{pred_actual["lotes_pepp"]}</b> lotes Pepperoni
            </div>
            <div style='background:#FFF9EE;border-radius:8px;padding:6px 12px;'>
              🧀 <b>{pred_actual["lotes_queso"]}</b> lotes Queso
            </div>
          </div>
          <div style='margin-top:.5rem;font-size:.75rem;color:#9CA3AF;'>
            Horneros: {pred_actual["horneros"]} · Armadores: {pred_actual["armadores"]} · Cajeros: {pred_actual["cajeros"]}
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-RERUN para simulación en vivo
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.simulando:
    time.sleep(2)
    st.rerun()

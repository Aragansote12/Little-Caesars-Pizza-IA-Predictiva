"""
data_generator.py
Genera dataset histórico sintético calibrado a Colombia (6 meses, ~4,320 registros/hora).
Sucursal simulada: Little Caesars Bogotá – Zona T / Plaza de las Américas
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

RUTA_DATOS = "data/historico_lc_colombia.parquet"

# ─── Precios en COP ────────────────────────────────────────────────────────────
PRECIOS = {
    "pizza_pepp_hot":    24_900,
    "pizza_queso_hot":   24_900,
    "pizza_borde":       36_900,
    "crazy_bread":        9_900,
    "canelitas":         10_900,
    "pepsi":              6_500,
}

# ─── Costos de producción (% sobre precio venta) ───────────────────────────────
COSTO_MATERIA_PRIMA_PCT = 0.324   # 32.4%
COSTO_NOMINA_PCT        = 0.317   # 31.7%
COSTO_FIJO_HORA         = 18_500  # COP/hora (arriendo, servicios, amortización)

# ─── Comisiones por canal ─────────────────────────────────────────────────────
COMISION_DELIVERY       = 0.30    # Rappi / DiDi / Uber Eats
COMISION_APP_OFICIAL    = 0.00
COMISION_MOSTRADOR      = 0.00

# ─── Festivos colombianos aproximados ─────────────────────────────────────────
FESTIVOS_COLOMBIA = {
    (1, 1), (1, 6), (3, 25), (4, 17), (4, 18),
    (5, 1), (6, 2), (6, 23), (6, 30), (7, 20),
    (8, 7), (8, 18), (10, 13), (11, 3), (11, 17),
    (12, 8), (12, 25),
}

# ─── Días de partido (simulados) ─────────────────────────────────────────────
DIAS_PARTIDO = {
    (3, 15), (3, 22), (4, 5), (4, 19), (5, 10),
    (5, 24), (6, 7), (6, 21), (7, 12), (7, 26),
    (8, 9), (8, 23), (9, 6), (9, 20),
}

CLIMAS = ["Soleado", "Nublado", "Lluvia Leve", "Aguacero"]
PESOS_CLIMA_BASE = [0.35, 0.30, 0.25, 0.10]


def _factor_quincena(dia: int) -> float:
    if dia in (14, 15, 16):
        return np.random.uniform(1.35, 1.45)
    if dia in (29, 30, 31):
        return np.random.uniform(1.30, 1.40)
    return 1.0


def _factor_dia_semana(dow: int) -> float:
    # 0=Lunes … 6=Domingo
    factores = {0: 0.75, 1: 0.70, 2: 0.80, 3: 0.85, 4: 1.10, 5: 1.30, 6: 1.20}
    return factores.get(dow, 1.0)


def _factor_hora(hora: int) -> float:
    """Pico almuerzo y cena colombianos."""
    curva = {
        11: 0.50, 12: 0.90, 13: 1.20, 14: 1.10, 15: 0.70,
        16: 0.55, 17: 0.65, 18: 0.85, 19: 1.15, 20: 1.25,
        21: 0.95, 22: 0.00,
    }
    return curva.get(hora, 0.0)


def _factor_clima(clima: str) -> float:
    base = {"Soleado": 1.0, "Nublado": 0.95, "Lluvia Leve": 0.90, "Aguacero": 0.75}
    return base.get(clima, 1.0)


def _delivery_boost_clima(clima: str) -> float:
    """Cuando llueve, sube delivery."""
    boost = {"Soleado": 1.0, "Nublado": 1.05, "Lluvia Leve": 1.25, "Aguacero": 1.50}
    return boost.get(clima, 1.0)


def _mix_canal(clima: str) -> dict:
    """Distribución de canal según clima."""
    boost = _delivery_boost_clima(clima)
    raw_delivery = 0.35 * boost
    raw_app      = 0.25
    raw_mostrador = max(0.05, 1.0 - raw_delivery - raw_app)
    total = raw_delivery + raw_app + raw_mostrador
    return {
        "mostrador":   raw_mostrador / total,
        "app_oficial": raw_app / total,
        "delivery":    raw_delivery / total,
    }


def _mix_productos(hora_pico: bool, es_quincena: bool) -> dict:
    """Probabilidades de producto por pedido."""
    if hora_pico:
        return {
            "pizza_pepp_hot":  0.38,
            "pizza_queso_hot": 0.22,
            "pizza_borde":     0.15,
            "crazy_bread":     0.12,
            "canelitas":       0.06,
            "pepsi":           0.07,
        }
    if es_quincena:
        return {
            "pizza_pepp_hot":  0.30,
            "pizza_queso_hot": 0.18,
            "pizza_borde":     0.20,
            "crazy_bread":     0.14,
            "canelitas":       0.09,
            "pepsi":           0.09,
        }
    return {
        "pizza_pepp_hot":  0.35,
        "pizza_queso_hot": 0.25,
        "pizza_borde":     0.12,
        "crazy_bread":     0.14,
        "canelitas":       0.07,
        "pepsi":           0.07,
    }


def generar_datos(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    registros = []

    inicio = datetime(2025, 1, 1, 11, 0)
    fin    = datetime(2025, 7, 1,  0, 0)
    hora_actual = inicio

    while hora_actual < fin:
        h   = hora_actual.hour
        dow = hora_actual.weekday()
        dia = hora_actual.day
        mes = hora_actual.month

        if h < 11 or h >= 22:
            hora_actual += timedelta(hours=1)
            continue

        # ── Contexto ──────────────────────────────────────────────────────────
        es_festivo  = (mes, dia) in FESTIVOS_COLOMBIA
        es_partido  = (mes, dia) in DIAS_PARTIDO
        es_quincena = dia in (14, 15, 16, 29, 30, 31)
        es_puente   = es_festivo and dow == 6  # domingo previo a lunes festivo
        es_fin_sem  = dow >= 4  # Viernes-Domingo

        # ── Clima ─────────────────────────────────────────────────────────────
        pesos_clima = list(PESOS_CLIMA_BASE)
        if es_puente or es_fin_sem:
            pesos_clima[2] += 0.05
            pesos_clima[3] += 0.05
        pesos_clima = np.array(pesos_clima)
        pesos_clima /= pesos_clima.sum()
        clima = rng.choice(CLIMAS, p=pesos_clima)

        # ── Demanda base ──────────────────────────────────────────────────────
        demanda_base = 42  # pedidos/hora en hora pico normal
        f_hora    = _factor_hora(h)
        f_dia     = _factor_dia_semana(dow)
        f_qncn    = _factor_quincena(dia)
        f_clima   = _factor_clima(clima)
        f_festivo = 1.30 if es_festivo else 1.0
        f_partido = 1.40 if es_partido else 1.0
        f_puente  = 1.25 if es_puente else 1.0

        demanda_esperada = (demanda_base * f_hora * f_dia * f_qncn *
                            f_clima * f_festivo * f_partido * f_puente)
        ruido = rng.normal(0, demanda_esperada * 0.08)
        pedidos = max(0, int(round(demanda_esperada + ruido)))

        if pedidos == 0:
            hora_actual += timedelta(hours=1)
            continue

        # ── Canal mix ─────────────────────────────────────────────────────────
        canal_mix = _mix_canal(clima)
        pedidos_mostrador   = int(pedidos * canal_mix["mostrador"])
        pedidos_app         = int(pedidos * canal_mix["app_oficial"])
        pedidos_delivery    = pedidos - pedidos_mostrador - pedidos_app

        # ── Productos ─────────────────────────────────────────────────────────
        hora_pico = f_hora >= 1.0
        mix_prod  = _mix_productos(hora_pico, es_quincena)
        productos = list(mix_prod.keys())
        probs     = np.array(list(mix_prod.values()))
        probs    /= probs.sum()

        prod_vendidos = {p: 0 for p in productos}
        for _ in range(pedidos):
            prod = rng.choice(productos, p=probs)
            prod_vendidos[prod] += 1

        # ── Ingresos ──────────────────────────────────────────────────────────
        ingreso_bruto = sum(prod_vendidos[p] * PRECIOS[p] for p in productos)
        costo_mp      = ingreso_bruto * COSTO_MATERIA_PRIMA_PCT
        costo_nomina  = ingreso_bruto * COSTO_NOMINA_PCT
        comisiones    = (pedidos_delivery * (ingreso_bruto / max(pedidos, 1))
                         * COMISION_DELIVERY * canal_mix["delivery"])
        costo_total   = costo_mp + costo_nomina + comisiones + COSTO_FIJO_HORA
        utilidad_neta = ingreso_bruto - costo_total
        margen_neto   = utilidad_neta / ingreso_bruto if ingreso_bruto > 0 else 0

        # ── Personal necesario ────────────────────────────────────────────────
        # Reglas: 1 hornero cada 20 pizzas/hora, 1 armador cada 15, 1 cajero cada 25
        pizzas_hora = (prod_vendidos["pizza_pepp_hot"] + prod_vendidos["pizza_queso_hot"]
                       + prod_vendidos["pizza_borde"])
        horneros  = max(1, int(np.ceil(pizzas_hora / 20)))
        armadores = max(1, int(np.ceil(pizzas_hora / 15)))
        cajeros   = max(1, int(np.ceil(pedidos / 25)))
        personal_total = horneros + armadores + cajeros

        # ── Lotes Hot-N-Ready (cada 30 min → dividimos por 2) ────────────────
        lotes_pepp  = max(1, int(np.ceil(prod_vendidos["pizza_pepp_hot"] / 2)))
        lotes_queso = max(1, int(np.ceil(prod_vendidos["pizza_queso_hot"] / 2)))

        registro = {
            "timestamp":        hora_actual,
            "hora":             h,
            "dia_semana":       dow,
            "dia_mes":          dia,
            "mes":              mes,
            "clima":            clima,
            "es_festivo":       es_festivo,
            "es_partido":       es_partido,
            "es_quincena":      es_quincena,
            "es_fin_semana":    es_fin_sem,
            "pedidos_total":    pedidos,
            "pedidos_mostrador": pedidos_mostrador,
            "pedidos_app":      pedidos_app,
            "pedidos_delivery": pedidos_delivery,
            **{f"unid_{p}": prod_vendidos[p] for p in productos},
            "ingreso_bruto_cop": ingreso_bruto,
            "costo_mp_cop":     costo_mp,
            "costo_nomina_cop": costo_nomina,
            "comisiones_cop":   comisiones,
            "costo_fijo_cop":   COSTO_FIJO_HORA,
            "costo_total_cop":  costo_total,
            "utilidad_neta_cop": utilidad_neta,
            "margen_neto":      margen_neto,
            "horneros":         horneros,
            "armadores":        armadores,
            "cajeros":          cajeros,
            "personal_total":   personal_total,
            "lotes_pepp_30min": lotes_pepp,
            "lotes_queso_30min": lotes_queso,
            "hora_pico":        hora_pico,
            "f_demanda_hora":   f_hora,
        }
        registros.append(registro)
        hora_actual += timedelta(hours=1)

    df = pd.DataFrame(registros)
    return df


def cargar_o_generar() -> pd.DataFrame:
    if os.path.exists(RUTA_DATOS):
        return pd.read_parquet(RUTA_DATOS)
    os.makedirs("data", exist_ok=True)
    print("⏳ Generando datos históricos (6 meses)...")
    df = generar_datos()
    df.to_parquet(RUTA_DATOS, index=False)
    print(f"✅ Dataset generado: {len(df):,} registros → {RUTA_DATOS}")
    return df


if __name__ == "__main__":
    df = cargar_o_generar()
    print(df.head())
    print(df.dtypes)
    print(f"\nRegistros totales: {len(df):,}")
    print(f"Ingresos promedio/hora: ${df['ingreso_bruto_cop'].mean():,.0f} COP")
    print(f"Margen neto promedio:   {df['margen_neto'].mean():.1%}")

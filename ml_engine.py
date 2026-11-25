"""
ml_engine.py
Motor de ML liviano: entrena modelos predictivos sobre el histórico generado.
Modelos:
  1. Predictor de demanda (pedidos/hora) → RandomForestRegressor
  2. Clasificador de Hora Pico            → basado en percentil dinámico
  3. Optimizador de personal              → derivado del predictor de demanda
  4. Calculador de lotes Hot-N-Ready      → regla matemática + predicción
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


FEATURES = [
    "hora", "dia_semana", "dia_mes", "mes",
    "clima_enc", "es_festivo", "es_partido",
    "es_quincena", "es_fin_semana",
]


def _preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    df["clima_enc"] = le.fit_transform(df["clima"])
    for col in ["es_festivo", "es_partido", "es_quincena", "es_fin_semana"]:
        df[col] = df[col].astype(int)
    return df, le


class LCMLEngine:
    """Encapsula todos los modelos predictivos de OpsAI Colombia."""

    def __init__(self):
        self.modelo_demanda  = None
        self.modelo_ingresos = None
        self.le_clima        = None
        self.umbral_pico     = None
        self.mae_demanda     = None
        self.mae_ingresos    = None
        self.df_hist         = None

    # ─── Entrenamiento ────────────────────────────────────────────────────────
    def entrenar(self, df: pd.DataFrame) -> dict:
        self.df_hist = df.copy()
        df_prep, self.le_clima = _preparar_features(df)

        X = df_prep[FEATURES].values
        y_pedidos  = df_prep["pedidos_total"].values
        y_ingresos = df_prep["ingreso_bruto_cop"].values

        X_tr, X_te, yp_tr, yp_te, yi_tr, yi_te = train_test_split(
            X, y_pedidos, y_ingresos, test_size=0.15, random_state=42
        )

        # Modelo 1: Demanda (pedidos/hora)
        self.modelo_demanda = RandomForestRegressor(
            n_estimators=120, max_depth=8, min_samples_leaf=4,
            n_jobs=-1, random_state=42
        )
        self.modelo_demanda.fit(X_tr, yp_tr)
        pred_p = self.modelo_demanda.predict(X_te)
        self.mae_demanda = mean_absolute_error(yp_te, pred_p)

        # Modelo 2: Ingresos/hora
        self.modelo_ingresos = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.08,
            random_state=42
        )
        self.modelo_ingresos.fit(X_tr, yi_tr)
        pred_i = self.modelo_ingresos.predict(X_te)
        self.mae_ingresos = mean_absolute_error(yi_te, pred_i)

        # Umbral dinámico de hora pico (percentil 70)
        self.umbral_pico = np.percentile(y_pedidos, 70)

        return {
            "mae_demanda":  self.mae_demanda,
            "mae_ingresos": self.mae_ingresos,
            "umbral_pico":  self.umbral_pico,
            "registros":    len(df),
        }

    # ─── Predicción por hora ──────────────────────────────────────────────────
    def predecir_hora(
        self, hora: int, dia_semana: int, dia_mes: int, mes: int,
        clima: str, es_festivo: bool, es_partido: bool,
        es_quincena: bool, es_fin_semana: bool,
        inflacion_pct: float = 0.0,
        pct_app: float = 0.25,
        impulso_venta_cruzada: float = 0.0,
    ) -> dict:
        """Devuelve predicciones completas para una hora específica."""
        try:
            clima_enc = self.le_clima.transform([clima])[0]
        except Exception:
            clima_enc = 0

        X = np.array([[
            hora, dia_semana, dia_mes, mes,
            clima_enc,
            int(es_festivo), int(es_partido),
            int(es_quincena), int(es_fin_semana),
        ]])

        pedidos_pred  = max(0, self.modelo_demanda.predict(X)[0])
        ingresos_pred = max(0, self.modelo_ingresos.predict(X)[0])

        # Ajuste por inflación (comprime márgenes)
        ingresos_ajust = ingresos_pred * (1 + inflacion_pct * 0.1)
        costo_mp_ajust = ingresos_ajust * (0.324 + inflacion_pct * 0.015)

        # Ajuste por migración a app oficial (reduce comisiones)
        pct_delivery_actual = max(0, 0.35 - pct_app * 0.6)
        ahorro_comision = ingresos_ajust * pct_delivery_actual * 0.30 * (pct_app / 0.5)

        # Impulso venta cruzada (sube ticket promedio)
        ticket_boost = 1 + (impulso_venta_cruzada * 0.12)
        ingresos_final = ingresos_ajust * ticket_boost

        costo_nomina = ingresos_final * 0.317
        costo_total  = costo_mp_ajust + costo_nomina + 18_500 - ahorro_comision
        utilidad     = ingresos_final - costo_total
        margen       = utilidad / ingresos_final if ingresos_final > 0 else 0

        # Personal óptimo
        pizzas_est = pedidos_pred * 0.75
        horneros   = max(1, int(np.ceil(pizzas_est / 20)))
        armadores  = max(1, int(np.ceil(pizzas_est / 15)))
        cajeros    = max(1, int(np.ceil(pedidos_pred / 25)))

        # Hot-N-Ready lotes por 30 min
        lotes_pepp  = max(1, int(np.ceil(pedidos_pred * 0.37 / 2)))
        lotes_queso = max(1, int(np.ceil(pedidos_pred * 0.24 / 2)))

        es_pico = pedidos_pred >= self.umbral_pico

        return {
            "pedidos_pred":    round(pedidos_pred),
            "ingresos_pred":   round(ingresos_final),
            "utilidad_pred":   round(utilidad),
            "margen_pred":     margen,
            "ahorro_comision": round(ahorro_comision),
            "horneros":        horneros,
            "armadores":       armadores,
            "cajeros":         cajeros,
            "personal_total":  horneros + armadores + cajeros,
            "lotes_pepp":      lotes_pepp,
            "lotes_queso":     lotes_queso,
            "es_pico":         es_pico,
            "costo_mp":        round(costo_mp_ajust),
            "costo_nomina":    round(costo_nomina),
        }

    # ─── Predicción para un día completo (11-22h) ─────────────────────────────
    def predecir_dia(
        self, dia_semana: int, dia_mes: int, mes: int,
        clima: str, es_festivo: bool, es_partido: bool,
        es_quincena: bool, es_fin_semana: bool,
        inflacion_pct: float = 0.0,
        pct_app: float = 0.25,
        impulso_vc: float = 0.0,
    ) -> pd.DataFrame:
        horas   = list(range(11, 22))
        filas   = []
        for h in horas:
            pred = self.predecir_hora(
                h, dia_semana, dia_mes, mes, clima,
                es_festivo, es_partido, es_quincena, es_fin_semana,
                inflacion_pct, pct_app, impulso_vc,
            )
            pred["hora"] = h
            filas.append(pred)
        return pd.DataFrame(filas)

    # ─── Feature importances ──────────────────────────────────────────────────
    def importancias(self) -> pd.DataFrame:
        if self.modelo_demanda is None:
            return pd.DataFrame()
        fi = self.modelo_demanda.feature_importances_
        return (
            pd.DataFrame({"feature": FEATURES, "importance": fi})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ─── Resumen estadístico del histórico ────────────────────────────────────
    def resumen_historico(self) -> dict:
        df = self.df_hist
        return {
            "total_pedidos":     int(df["pedidos_total"].sum()),
            "ingreso_total":     int(df["ingreso_bruto_cop"].sum()),
            "utilidad_total":    int(df["utilidad_neta_cop"].sum()),
            "margen_promedio":   df["margen_neto"].mean(),
            "hora_top":          int(df.groupby("hora")["pedidos_total"].mean().idxmax()),
            "dia_top":           int(df.groupby("dia_semana")["pedidos_total"].mean().idxmax()),
            "pct_delivery":      df["pedidos_delivery"].sum() / df["pedidos_total"].sum(),
            "pct_app":           df["pedidos_app"].sum() / df["pedidos_total"].sum(),
            "pct_mostrador":     df["pedidos_mostrador"].sum() / df["pedidos_total"].sum(),
        }

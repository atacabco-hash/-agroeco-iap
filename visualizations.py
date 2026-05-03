"""
utils/visualizations.py
Gráficas Plotly para AgroEco IA — ATACABCO
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paleta de colores ATACABCO
COLORES_DIAGNOSTICO = {
    "Suelo Muy Saludable":           "#2e7d32",
    "Suelo Saludable":               "#558b2f",
    "Suelo Moderadamente Saludable": "#f9a825",
    "Suelo con Deficiencias":        "#e65100",
    "Suelo Degradado":               "#b71c1c",
}

PALETA = ["#4a6741", "#5c3d2e", "#a0522d", "#d4a96a", "#2e7d32", "#e65100"]

FONT = dict(family="DM Sans, sans-serif", color="#1a1209")


def _base_layout(fig, titulo=""):
    fig.update_layout(
        title=dict(text=titulo, font=dict(family="Playfair Display, serif", size=18, color="#5c3d2e")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(245,239,230,0.5)",
        font=FONT,
        margin=dict(t=50, b=30, l=10, r=10),
    )
    return fig


# ─────────────────────────────────────────────
# 1. DISTRIBUCIÓN DE DIAGNÓSTICOS (donut)
# ─────────────────────────────────────────────
def distribucion_diagnosticos(df: pd.DataFrame) -> go.Figure:
    conteo = df["Diagnostico_Agroecologico"].value_counts().reset_index()
    conteo.columns = ["Diagnóstico", "Muestras"]
    colores = [COLORES_DIAGNOSTICO.get(d, "#888") for d in conteo["Diagnóstico"]]

    fig = go.Figure(go.Pie(
        labels=conteo["Diagnóstico"],
        values=conteo["Muestras"],
        hole=0.45,
        marker=dict(colors=colores, line=dict(color="white", width=2)),
        textinfo="percent+label",
        textfont=dict(size=11),
    ))
    _base_layout(fig, "Distribución de Diagnósticos")
    fig.update_layout(showlegend=False, height=320)
    return fig


# ─────────────────────────────────────────────
# 2. SCATTER pH vs Materia Orgánica
# ─────────────────────────────────────────────
def scatter_ph_materia_organica(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="pH",
        y="Materia_Organica_pct",
        color="Diagnostico_Agroecologico",
        color_discrete_map=COLORES_DIAGNOSTICO,
        hover_data=["ID_Muestra", "Tipo_Manejo", "Ubicacion"],
        labels={
            "pH": "pH del Suelo",
            "Materia_Organica_pct": "Materia Orgánica (%)",
            "Diagnostico_Agroecologico": "Diagnóstico",
        },
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
    _base_layout(fig, "pH vs. Materia Orgánica")
    fig.update_layout(
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, font=dict(size=10)),
    )
    return fig


# ─────────────────────────────────────────────
# 3. RADAR de Zonas Cromatográficas
# ─────────────────────────────────────────────
def radar_zonas(muestra: dict, similares: pd.DataFrame) -> go.Figure:
    categorias = ["Central\n(Física)", "Interna\n(Química)", "Media\n(Biológica)", "Externa\n(Enzimática)"]
    campos = ["Zona_Central_Fisica", "Zona_Interna_Quimica", "Zona_Media_Biologica", "Zona_Externa_Enzimatica"]

    vals_muestra = [muestra.get(c, 5) for c in campos]
    vals_similares = [similares[c].mean() for c in campos if c in similares.columns]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=vals_muestra + [vals_muestra[0]],
        theta=categorias + [categorias[0]],
        fill="toself",
        name="Tu muestra",
        line_color="#4a6741",
        fillcolor="rgba(74,103,65,0.25)",
        line_width=2,
    ))

    if vals_similares:
        fig.add_trace(go.Scatterpolar(
            r=vals_similares + [vals_similares[0]],
            theta=categorias + [categorias[0]],
            fill="toself",
            name="Promedio similares",
            line_color="#d4a96a",
            fillcolor="rgba(212,169,106,0.15)",
            line_width=2,
            line_dash="dot",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=9)),
            bgcolor="rgba(245,239,230,0.5)",
        ),
        title=dict(text="Perfil de Zonas Cromatográficas",
                   font=dict(family="Playfair Display, serif", size=16, color="#5c3d2e")),
        paper_bgcolor="rgba(0,0,0,0)",
        font=FONT,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        height=380,
        margin=dict(t=50, b=60, l=40, r=40),
    )
    return fig


# ─────────────────────────────────────────────
# 4. BOXPLOT por Tipo de Manejo
# ─────────────────────────────────────────────
def boxplot_por_manejo(df: pd.DataFrame, variable: str) -> go.Figure:
    orden = (
        df.groupby("Tipo_Manejo")[variable]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )

    fig = px.box(
        df,
        x=variable,
        y="Tipo_Manejo",
        color="Diagnostico_Agroecologico",
        color_discrete_map=COLORES_DIAGNOSTICO,
        category_orders={"Tipo_Manejo": orden},
        labels={
            variable: variable.replace("_", " "),
            "Tipo_Manejo": "Tipo de Manejo",
            "Diagnostico_Agroecologico": "Diagnóstico",
        },
        points="outliers",
    )
    _base_layout(fig, f"{variable.replace('_', ' ')} por Tipo de Manejo")
    fig.update_layout(
        height=520,
        showlegend=False,
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


# ─────────────────────────────────────────────
# 5. HEATMAP de Correlaciones
# ─────────────────────────────────────────────
def heatmap_correlaciones(df: pd.DataFrame) -> go.Figure:
    cols = [
        "pH", "Materia_Organica_pct", "Carbono_Organico_Total_pct",
        "Nitrogeno_Total_pct", "Biomasa_Microbiana_C_mg_kg",
        "Zona_Central_Fisica", "Zona_Interna_Quimica",
        "Zona_Media_Biologica", "Zona_Externa_Enzimatica",
        "Puntuacion_Calidad_General",
    ]
    etiquetas = [
        "pH", "M.O.(%)", "C.Org.(%)", "N Total",
        "Biomasa Mic.", "Z.Central", "Z.Interna",
        "Z.Media", "Z.Externa", "Calidad",
    ]
    corr = df[cols].corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=etiquetas,
        y=etiquetas,
        colorscale=[
            [0.0, "#b71c1c"], [0.35, "#f5efe6"],
            [0.65, "#f5efe6"], [1.0, "#2e7d32"]
        ],
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        textfont=dict(size=9),
        showscale=True,
    ))
    _base_layout(fig, "Correlaciones entre Variables del Suelo")
    fig.update_layout(height=460, xaxis=dict(tickangle=-35))
    return fig


# ─────────────────────────────────────────────
# 6. MAPA DE CALIDAD (placeholder geográfico)
# ─────────────────────────────────────────────
def mapa_calidad(df: pd.DataFrame) -> go.Figure:
    """
    Figura de barras horizontales mostrando calidad promedio por ubicación.
    (Reemplazable por mapa real con coordenadas geográficas en versión futura)
    """
    top_ubic = (
        df.groupby("Ubicacion")["Puntuacion_Calidad_General"]
        .mean()
        .sort_values(ascending=True)
        .tail(15)
        .reset_index()
    )

    fig = px.bar(
        top_ubic,
        x="Puntuacion_Calidad_General",
        y="Ubicacion",
        orientation="h",
        color="Puntuacion_Calidad_General",
        color_continuous_scale=["#b71c1c", "#f9a825", "#2e7d32"],
        labels={
            "Puntuacion_Calidad_General": "Puntuación Calidad",
            "Ubicacion": "Ubicación",
        },
    )
    _base_layout(fig, "Calidad del Suelo por Ubicación (Top 15)")
    fig.update_layout(height=460, coloraxis_showscale=False)
    return fig

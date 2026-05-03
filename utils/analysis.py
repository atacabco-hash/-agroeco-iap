"""
utils/analysis.py
Funciones de análisis agroecológico para AgroEco IA — ATACABCO
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


# ─────────────────────────────────────────────
# ÍNDICE DE SALUD DEL SUELO
# ─────────────────────────────────────────────

DIAGNOSTICOS = {
    "Suelo Muy Saludable":           ("muy-saludable",  10),
    "Suelo Saludable":               ("saludable",       7.5),
    "Suelo Moderadamente Saludable": ("moderado",        5.5),
    "Suelo con Deficiencias":        ("deficiencias",    3.5),
    "Suelo Degradado":               ("degradado",       1.5),
}

def calcular_indice_salud(muestra: dict) -> tuple[float, str, str]:
    """
    Calcula un índice compuesto de salud del suelo (0-10) a partir de
    las zonas cromatográficas, pH y materia orgánica.
    Retorna (indice, diagnostico_texto, clase_css)
    """
    # Normalización de zonas (escala 1-10 → 0-1)
    zonas = [
        muestra.get("Zona_Central_Fisica", 5),
        muestra.get("Zona_Interna_Quimica", 5),
        muestra.get("Zona_Media_Biologica", 5),
        muestra.get("Zona_Externa_Enzimatica", 5),
    ]
    score_zonas = np.mean(zonas) / 10  # 0–1

    # pH: óptimo 6.0–7.0
    ph = muestra.get("pH", 6.5)
    score_ph = max(0, 1 - abs(ph - 6.5) / 3)

    # Materia orgánica: >5% excelente, <1% crítico
    mo = muestra.get("Materia_Organica_pct", 3.0)
    score_mo = min(mo / 5.0, 1.0)

    # Índice compuesto (ponderado)
    indice = round((score_zonas * 0.5 + score_ph * 0.25 + score_mo * 0.25) * 10, 1)

    # Diagnóstico
    if indice >= 8.0:
        diagnostico, clase = "Suelo Muy Saludable", "muy-saludable"
    elif indice >= 6.5:
        diagnostico, clase = "Suelo Saludable", "saludable"
    elif indice >= 5.0:
        diagnostico, clase = "Suelo Moderadamente Saludable", "moderado"
    elif indice >= 3.0:
        diagnostico, clase = "Suelo con Deficiencias", "deficiencias"
    else:
        diagnostico, clase = "Suelo Degradado", "degradado"

    return indice, diagnostico, clase


# ─────────────────────────────────────────────
# COMPARACIÓN CON DATASET (KNN)
# ─────────────────────────────────────────────

FEATURES_KNN = [
    "Zona_Central_Fisica", "Zona_Interna_Quimica",
    "Zona_Media_Biologica", "Zona_Externa_Enzimatica",
    "pH", "Materia_Organica_pct",
]

def comparar_con_dataset(df: pd.DataFrame, muestra: dict, n_vecinos: int = 5) -> pd.DataFrame:
    """
    Encuentra los N casos más similares en el dataset usando KNN.
    """
    df_limpio = df[FEATURES_KNN].dropna()
    indices_validos = df_limpio.index

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df_limpio)

    muestra_vec = np.array([[
        muestra.get(f, df[f].mean()) for f in FEATURES_KNN
    ]])
    muestra_scaled = scaler.transform(muestra_vec)

    knn = NearestNeighbors(n_neighbors=min(n_vecinos, len(X)), metric="euclidean")
    knn.fit(X)
    distancias, vecinos_idx = knn.kneighbors(muestra_scaled)

    idx_reales = indices_validos[vecinos_idx[0]]
    similares = df.loc[idx_reales].copy()
    similares["Similitud_pct"] = (1 - distancias[0] / (distancias[0].max() + 1e-8)) * 100
    return similares.reset_index(drop=True)


# ─────────────────────────────────────────────
# GENERACIÓN DE RECOMENDACIONES
# ─────────────────────────────────────────────

def generar_recomendaciones(muestra: dict, similares: pd.DataFrame,
                             altitud: int = 1500, cultivo: str = "café") -> list[dict]:
    """
    Genera recomendaciones agroecológicas contextualizadas.
    """
    recomendaciones = []
    ph  = muestra.get("pH", 6.5)
    mo  = muestra.get("Materia_Organica_pct", 3.0)
    zb  = muestra.get("Zona_Media_Biologica", 5)
    ze  = muestra.get("Zona_Externa_Enzimatica", 5)
    zc  = muestra.get("Zona_Central_Fisica", 5)

    # ── Materia orgánica ──
    if mo < 2.5:
        recomendaciones.append({
            "titulo": "🌱 Incrementar Materia Orgánica (URGENTE)",
            "descripcion": (
                f"Tu suelo tiene {mo}% de materia orgánica, por debajo del mínimo recomendado (3%). "
                "Esto limita la vida microbiana, la retención de agua y la disponibilidad de nutrientes."
            ),
            "practica": (
                "Aplicar compost maduro (3–5 ton/ha), establecer cultivos de cobertura "
                "(canavalia, mucuna, trébol) e incorporar abonos verdes cada 6 meses."
            ),
        })
    elif mo >= 5.0:
        recomendaciones.append({
            "titulo": "✅ Excelente Contenido de Materia Orgánica",
            "descripcion": f"Tu suelo presenta {mo}% de materia orgánica, indicador de alta fertilidad y actividad biológica. Mantener las prácticas actuales.",
            "practica": "Continuar con incorporación de residuos de cosecha y cobertura vegetal.",
        })
    else:
        recomendaciones.append({
            "titulo": "📈 Materia Orgánica en Nivel Aceptable",
            "descripcion": f"El {mo}% de materia orgánica es adecuado, pero hay margen de mejora para alcanzar suelos de alta calidad (>5%).",
            "practica": "Aplicar vermicompost (2 ton/ha/año) y mantener cobertura vegetal permanente.",
        })

    # ── pH ──
    if ph < 5.5:
        recomendaciones.append({
            "titulo": "⚠️ pH Muy Ácido — Corrección Necesaria",
            "descripcion": f"El pH de {ph} limita severamente la disponibilidad de fósforo, calcio y magnesio, y puede generar toxicidad por aluminio.",
            "practica": f"Aplicar cal dolomita (500–1000 kg/ha según pH objetivo). Para cultivo de {cultivo} a {altitud} msnm, pH óptimo: 5.8–6.5.",
        })
    elif ph > 7.5:
        recomendaciones.append({
            "titulo": "⚠️ pH Alcalino — Monitoreo Necesario",
            "descripcion": f"El pH de {ph} puede reducir disponibilidad de micronutrientes (hierro, manganeso, zinc).",
            "practica": "Incorporar materia orgánica ácida (turba, compost de hojarasca) y evitar encalado adicional.",
        })
    else:
        recomendaciones.append({
            "titulo": "✅ pH en Rango Óptimo",
            "descripcion": f"pH de {ph} — condiciones adecuadas para la mayoría de cultivos y microorganismos del suelo.",
            "practica": "Monitorear pH cada 6 meses para detectar cambios estacionales.",
        })

    # ── Zona biológica ──
    if zb < 4:
        recomendaciones.append({
            "titulo": "🔴 Baja Actividad Biológica",
            "descripcion": "La zona media del cromatograma indica poca actividad microbiana. Esto compromete el ciclo de nutrientes y la estructura del suelo.",
            "practica": "Inocular con microorganismos eficientes (EM), aplicar caldos microbianos (bocashi líquido) y eliminar el uso de agroquímicos.",
        })
    elif zb >= 7:
        recomendaciones.append({
            "titulo": "🟢 Alta Actividad Biológica",
            "descripcion": "Excelente zona biológica en el cromatograma. El suelo presenta una comunidad microbiana activa y diversa.",
            "practica": "Mantener la cobertura vegetal y el suministro constante de materia orgánica para sostener esta actividad.",
        })

    # ── Zona enzimática ──
    if ze < 4:
        recomendaciones.append({
            "titulo": "🧪 Baja Actividad Enzimática",
            "descripcion": "La zona externa débil sugiere escasa actividad de enzimas del suelo, clave para mineralización de nutrientes.",
            "practica": "Aplicar extractos de algas marinas, humus de lombriz y bioestimulantes de origen natural para activar las enzimas del suelo.",
        })

    # ── Zona central ──
    if zc < 4:
        recomendaciones.append({
            "titulo": "🪨 Estructura Mineral Deficiente",
            "descripcion": "La zona central débil puede indicar baja mineralización o suelo con poca capacidad de intercambio catiónico.",
            "practica": "Aplicar roca fosfórica, zeolita o biochar para mejorar la fracción mineral del suelo.",
        })

    # ── Recomendación contextual por altitud ──
    if altitud < 1000:
        contexto_altitud = "zona baja tropical (< 1000 msnm)"
        cultivos_sugeridos = "yuca, plátano, cacao, aguacate Hass"
    elif altitud < 2000:
        contexto_altitud = "zona media andina (1000–2000 msnm)"
        cultivos_sugeridos = "café, fríjol, plátano, cítricos, hortalizas"
    elif altitud < 3000:
        contexto_altitud = "zona alta andina (2000–3000 msnm)"
        cultivos_sugeridos = "papa, fresa, arveja, granadilla, curuba"
    else:
        contexto_altitud = "zona de páramo (> 3000 msnm)"
        cultivos_sugeridos = "papa nativa, ulluco, quinua"

    recomendaciones.append({
        "titulo": f"📍 Sistema Agroecológico para {contexto_altitud}",
        "descripcion": f"Según la altitud de {altitud} msnm, tu territorio se ubica en {contexto_altitud}. Las condiciones climáticas favorecen sistemas agroforestales diversos.",
        "practica": f"Asociaciones recomendadas: {cultivos_sugeridos}. Implementar sistema agroforestal multiestrato con plantas nativas.",
    })

    # ── Lección del dataset ──
    manejo_similares = similares["Tipo_Manejo"].mode()
    if not manejo_similares.empty:
        mejor_manejo = manejo_similares.iloc[0]
        recomendaciones.append({
            "titulo": "📚 Lección del Dataset Global",
            "descripcion": (
                f"Los {len(similares)} casos más similares al tuyo en nuestra base de datos "
                f"corresponden principalmente a sistemas de «{mejor_manejo}». "
                "Estudiar estas experiencias puede enriquecer tu práctica agroecológica."
            ),
            "practica": None,
        })

    return recomendaciones


# ─────────────────────────────────────────────
# ESTADÍSTICAS POR TIPO DE MANEJO
# ─────────────────────────────────────────────

def estadisticas_por_manejo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla resumen de indicadores promedio por tipo de manejo.
    """
    cols = ["Tipo_Manejo", "pH", "Materia_Organica_pct",
            "Biomasa_Microbiana_C_mg_kg", "Puntuacion_Calidad_General"]
    return (
        df[cols]
        .groupby("Tipo_Manejo")
        .mean()
        .round(2)
        .sort_values("Puntuacion_Calidad_General", ascending=False)
        .rename(columns={
            "pH": "pH",
            "Materia_Organica_pct": "M.O. (%)",
            "Biomasa_Microbiana_C_mg_kg": "Biomasa Mic.",
            "Puntuacion_Calidad_General": "Calidad (0-14)",
        })
    )

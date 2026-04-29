"""
AgroEco IA — Cromatografía de Suelos con Inteligencia Artificial
Desarrollado por ATACABCO · Medellín, Antioquia, Colombia
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
import io
from utils.analysis import (
    calcular_indice_salud,
    generar_recomendaciones,
    comparar_con_dataset,
    estadisticas_por_manejo,
)
from utils.visualizations import (
    radar_zonas,
    distribucion_diagnosticos,
    scatter_ph_materia_organica,
    heatmap_correlaciones,
    boxplot_por_manejo,
    mapa_calidad,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AgroEco IA · ATACABCO",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS PERSONALIZADOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --tierra:    #5c3d2e;
    --arcilla:   #a0522d;
    --musgo:     #4a6741;
    --clorofila: #6b8f5e;
    --arena:     #d4a96a;
    --crema:     #f5efe6;
    --negro-suelo: #1a1209;
    --gris-humus: #3d3530;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--crema);
    color: var(--negro-suelo);
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--tierra) !important;
}
[data-testid="stSidebar"] * {
    color: var(--crema) !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 1rem;
    padding: 6px 0;
}

/* Métricas */
[data-testid="metric-container"] {
    background: white;
    border-left: 4px solid var(--musgo);
    border-radius: 8px;
    padding: 12px 16px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* Botones */
.stButton > button {
    background-color: var(--musgo) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 10px 28px !important;
    transition: background 0.2s;
}
.stButton > button:hover {
    background-color: var(--tierra) !important;
}

/* Tarjetas de diagnóstico */
.diagnostico-card {
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
    border-left: 5px solid;
}
.muy-saludable  { background:#e8f5e9; border-color:#2e7d32; }
.saludable      { background:#f1f8e9; border-color:#558b2f; }
.moderado       { background:#fff8e1; border-color:#f9a825; }
.deficiencias   { background:#fff3e0; border-color:#e65100; }
.degradado      { background:#fce4ec; border-color:#b71c1c; }

/* Encabezado */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--tierra);
    line-height: 1.15;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    color: var(--gris-humus);
    font-weight: 300;
}
.badge {
    display: inline-block;
    background: var(--musgo);
    color: white;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 2px;
}
.divider {
    border: none;
    border-top: 2px solid var(--arena);
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    df = pd.read_csv("data/dataset_cromatografia_suelos.csv", encoding="utf-8-sig")
    numericas = [
        "pH", "Materia_Organica_pct", "Carbono_Organico_Total_pct",
        "Nitrogeno_Total_pct", "Fosforo_Asimilable_ppm", "Potasio_Intercambiable_ppm",
        "Biomasa_Microbiana_C_mg_kg", "Respiracion_Basal_mg_CO2_kg_dia",
        "Zona_Central_Fisica", "Zona_Interna_Quimica",
        "Zona_Media_Biologica", "Zona_Externa_Enzimatica",
        "Densidad_Aparente_g_cm3", "Capacidad_Retencion_Agua_pct",
        "Puntuacion_Calidad_General",
    ]
    for col in numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df = cargar_datos()


# ─────────────────────────────────────────────
# SIDEBAR — NAVEGACIÓN
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌱 AgroEco IA")
    st.markdown("*Una herramienta de ATACABCO*")
    st.markdown("---")
    pagina = st.radio(
        "Navegar",
        options=[
            "🏠  Inicio",
            "🔬  Demo IA",
            "📊  Explorador de Datos",
            "⚙️  Tecnología",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small>📍 Medellín, Antioquia<br>🌐 ATACABCO · Red de Economía Circular</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# PÁGINA: INICIO
# ─────────────────────────────────────────────
if pagina == "🏠  Inicio":
    col_txt, col_img = st.columns([3, 2], gap="large")

    with col_txt:
        st.markdown(
            '<p class="hero-title">Inteligencia Artificial<br>para el Suelo Vivo</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="hero-sub">Analizamos cromatogramas de suelo con IA para generar '
            'recomendaciones agroecológicas contextualizadas, integrando saberes ancestrales '
            'y ciencia de datos.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span class="badge">Agroecología</span>'
            '<span class="badge">Economía Circular</span>'
            '<span class="badge">Saberes Ancestrales</span>'
            '<span class="badge">IA</span>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Métricas clave del dataset
    st.markdown("### 📈 Base de Conocimiento del Sistema")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Muestras analizadas", f"{len(df)}")
    m2.metric("Tipos de manejo", f"{df['Tipo_Manejo'].nunique()}")
    m3.metric("Países representados",
              f"{df['Ubicacion'].str.extract(r',\s*(.+)$')[0].nunique() or '15+'}")
    m4.metric("pH promedio", f"{df['pH'].mean():.2f}")
    m5.metric("M.O. promedio", f"{df['Materia_Organica_pct'].mean():.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.plotly_chart(
            distribucion_diagnosticos(df),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            scatter_ph_materia_organica(df),
            use_container_width=True,
        )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 🌿 ¿Qué hace AgroEco IA?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🔬 Análisis Visual**")
        st.write("Interpreta zonas cromáticas del cromatograma: física, química, biológica y enzimática.")
    with c2:
        st.markdown("**📍 Contextualización Territorial**")
        st.write("Integra altitud, tipo de suelo, clima y prácticas de manejo del territorio.")
    with c3:
        st.markdown("**📋 Recomendaciones IA**")
        st.write("Genera planes de manejo agroecológico basados en comparación con 150 casos reales.")


# ─────────────────────────────────────────────
# PÁGINA: DEMO IA
# ─────────────────────────────────────────────
elif pagina == "🔬  Demo IA":
    st.markdown("## 🔬 Análisis de Cromatograma con IA")
    st.markdown("Carga la imagen de tu cromatograma e ingresa los datos del territorio para obtener recomendaciones personalizadas.")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("### Paso 1 · Imagen del cromatograma")
        imagen = st.file_uploader(
            "Sube la foto de tu cromatograma de suelo",
            type=["jpg", "jpeg", "png"],
            help="Fotografía tomada con el método Pfeiffer o similar",
        )
        if imagen:
            img = Image.open(imagen)
            st.image(img, caption="Cromatograma cargado", use_container_width=True)

        st.markdown("### Paso 2 · Datos del territorio")
        ubicacion = st.text_input("Ubicación", value="Medellín, Antioquia, Colombia")
        altitud   = st.number_input("Altitud (msnm)", min_value=0, max_value=5000, value=1500)
        tipo_manejo = st.selectbox("Tipo de manejo actual", sorted(df["Tipo_Manejo"].unique()))
        tipo_suelo  = st.selectbox("Tipo de suelo", sorted(df["Tipo_Suelo"].unique()))
        cultivo     = st.text_input("Cultivo principal", value="Café, plátano")
        area        = st.number_input("Área (ha)", min_value=0.1, max_value=1000.0, value=2.0, step=0.5)

        st.markdown("### Paso 3 · Lectura del cromatograma")
        st.caption("Califica cada zona del cromatograma del 1 (muy débil) al 10 (excelente)")
        z_central  = st.slider("Zona Central (física-mineral)",   1, 10, 5)
        z_interna  = st.slider("Zona Interna (química-coloidal)",  1, 10, 5)
        z_media    = st.slider("Zona Media (biológica)",           1, 10, 5)
        z_externa  = st.slider("Zona Externa (enzimática)",        1, 10, 5)

        ph_campo     = st.number_input("pH medido en campo (si disponible)", min_value=3.0, max_value=9.5, value=6.0, step=0.1)
        mo_estimada  = st.number_input("M.O. estimada (%)", min_value=0.0, max_value=15.0, value=3.0, step=0.1)

        analizar = st.button("🌱 Analizar con IA", use_container_width=True)

    with col_result:
        if analizar:
            muestra = {
                "Zona_Central_Fisica": z_central,
                "Zona_Interna_Quimica": z_interna,
                "Zona_Media_Biologica": z_media,
                "Zona_Externa_Enzimatica": z_externa,
                "pH": ph_campo,
                "Materia_Organica_pct": mo_estimada,
                "Tipo_Manejo": tipo_manejo,
                "Tipo_Suelo": tipo_suelo,
            }

            with st.spinner("Consultando base de conocimiento agroecológico..."):
                indice, diagnostico, clase_css = calcular_indice_salud(muestra)
                similares = comparar_con_dataset(df, muestra)
                recomendaciones = generar_recomendaciones(muestra, similares, altitud, cultivo)

            # Índice de salud
            st.markdown(f"""
            <div class="diagnostico-card {clase_css}">
                <h3 style="margin:0">📋 {diagnostico}</h3>
                <p style="margin:4px 0 0 0; font-size:0.9rem;">
                    Índice de salud del suelo: <strong>{indice}/10</strong> · 
                    Ubicación: {ubicacion}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Radar de zonas
            st.plotly_chart(radar_zonas(muestra, similares), use_container_width=True)

            # Recomendaciones
            st.markdown("### 🌿 Recomendaciones Agroecológicas")
            for rec in recomendaciones:
                with st.expander(rec["titulo"], expanded=True):
                    st.write(rec["descripcion"])
                    if rec.get("practica"):
                        st.info(f"💡 **Práctica sugerida:** {rec['practica']}")

            # Casos similares del dataset
            st.markdown("### 🔍 Casos similares en la base de datos")
            st.dataframe(
                similares[["ID_Muestra", "Ubicacion", "Tipo_Manejo", "pH",
                            "Materia_Organica_pct", "Diagnostico_Agroecologico"]].head(5),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#888;">
                <div style="font-size:4rem;">🪱</div>
                <p style="font-family:'Playfair Display',serif; font-size:1.3rem; margin-top:16px;">
                    Carga tu cromatograma y<br>completa los datos del territorio
                </p>
                <p style="font-size:0.9rem;">El sistema comparará tu muestra con<br>150 casos reales del dataset global</p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PÁGINA: EXPLORADOR DE DATOS
# ─────────────────────────────────────────────
elif pagina == "📊  Explorador de Datos":
    st.markdown("## 📊 Explorador de Datos — Cromatografía de Suelos")

    # Filtros
    with st.expander("🔽 Filtros", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            f_diagnostico = st.multiselect(
                "Diagnóstico",
                options=df["Diagnostico_Agroecologico"].unique(),
                default=list(df["Diagnostico_Agroecologico"].unique()),
            )
        with fc2:
            f_manejo = st.multiselect(
                "Tipo de Manejo",
                options=sorted(df["Tipo_Manejo"].unique()),
                default=list(df["Tipo_Manejo"].unique()),
            )
        with fc3:
            f_ph = st.slider("Rango de pH", float(df["pH"].min()), float(df["pH"].max()),
                             (float(df["pH"].min()), float(df["pH"].max())))

    df_filtrado = df[
        df["Diagnostico_Agroecologico"].isin(f_diagnostico) &
        df["Tipo_Manejo"].isin(f_manejo) &
        df["pH"].between(*f_ph)
    ]

    st.caption(f"Mostrando **{len(df_filtrado)}** de {len(df)} muestras")

    # Gráficas
    tab1, tab2, tab3, tab4 = st.tabs(["🥧 Diagnósticos", "📦 Por Tipo de Manejo", "🌡️ Correlaciones", "📋 Tabla"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(distribucion_diagnosticos(df_filtrado), use_container_width=True)
        with c2:
            st.plotly_chart(scatter_ph_materia_organica(df_filtrado), use_container_width=True)

    with tab2:
        var_y = st.selectbox("Variable a analizar", [
            "Materia_Organica_pct", "pH", "Biomasa_Microbiana_C_mg_kg",
            "Puntuacion_Calidad_General", "Capacidad_Retencion_Agua_pct",
            "Nitrogeno_Total_pct", "Fosforo_Asimilable_ppm",
        ])
        st.plotly_chart(boxplot_por_manejo(df_filtrado, var_y), use_container_width=True)

    with tab3:
        st.plotly_chart(heatmap_correlaciones(df_filtrado), use_container_width=True)

    with tab4:
        cols_vista = [
            "ID_Muestra", "Ubicacion", "Tipo_Manejo", "Tipo_Suelo", "pH",
            "Materia_Organica_pct", "Biomasa_Microbiana_C_mg_kg",
            "Puntuacion_Calidad_General", "Diagnostico_Agroecologico",
        ]
        st.dataframe(df_filtrado[cols_vista], hide_index=True, use_container_width=True)
        csv_export = df_filtrado.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar datos filtrados (CSV)",
            data=csv_export,
            file_name="agroeco_filtrado.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────
# PÁGINA: TECNOLOGÍA
# ─────────────────────────────────────────────
elif pagina == "⚙️  Tecnología":
    st.markdown("## ⚙️ Arquitectura del Sistema")

    st.markdown("""
    AgroEco IA combina métodos de análisis de datos, visión computacional y razonamiento basado 
    en casos para generar recomendaciones agroecológicas fundamentadas en evidencia científica 
    y saberes territoriales.
    """)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("### 🔩 Componentes Técnicos")
        componentes = {
            "Interfaz": "Streamlit · Python · Plotly",
            "Análisis de imagen": "PIL / OpenCV (preprocesamiento)",
            "Análisis de datos": "Pandas · NumPy · SciPy",
            "Visualización": "Plotly Express & Graph Objects",
            "Motor de recomendaciones": "K-Nearest Neighbors sobre dataset real",
            "Dataset": "150 muestras globales · 33 variables",
            "Despliegue": "Streamlit Cloud / Docker",
        }
        for k, v in componentes.items():
            st.markdown(f"**{k}:** {v}")

    with c2:
        st.markdown("### 🌊 Flujo de Análisis")
        flujo = [
            ("1", "Imagen del cromatograma", "Carga fotográfica del análisis Pfeiffer"),
            ("2", "Lectura de zonas", "Calificación manual de 4 zonas cromáticas"),
            ("3", "Datos territoriales", "Ubicación, altitud, manejo, cultivo"),
            ("4", "Comparación KNN", "Búsqueda de casos similares en 150 muestras"),
            ("5", "Diagnóstico IA", "Índice de salud y clasificación del suelo"),
            ("6", "Recomendaciones", "Plan agroecológico personalizado"),
        ]
        for num, titulo, desc in flujo:
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin:8px 0;align-items:flex-start">
                <div style="background:#4a6741;color:white;border-radius:50%;width:28px;height:28px;
                     display:flex;align-items:center;justify-content:center;flex-shrink:0;font-weight:700">
                     {num}
                </div>
                <div>
                    <strong>{titulo}</strong><br>
                    <small style="color:#666">{desc}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 📊 Variables del Dataset")

    grupos = {
        "Zonas Cromatográficas (1–10)": [
            "Zona_Central_Fisica", "Zona_Interna_Quimica",
            "Zona_Media_Biologica", "Zona_Externa_Enzimatica",
        ],
        "Propiedades Físico-Químicas": [
            "pH", "Materia_Organica_pct", "Carbono_Organico_Total_pct",
            "Nitrogeno_Total_pct", "Fosforo_Asimilable_ppm", "Potasio_Intercambiable_ppm",
        ],
        "Actividad Biológica": [
            "Biomasa_Microbiana_C_mg_kg", "Respiracion_Basal_mg_CO2_kg_dia",
            "Fosfatasa_Acida_mg_PNF_g_h", "Beta_Glucosidasa_mg_PNF_g_h", "Arilsulfatasa_mg_PNF_g_h",
        ],
        "Propiedades Físicas": [
            "Densidad_Aparente_g_cm3", "Capacidad_Retencion_Agua_pct", "Resistencia_Penetracion_MPa",
        ],
    }

    for grupo, vars_ in grupos.items():
        with st.expander(f"**{grupo}**"):
            stats = df[vars_].describe().T[["mean", "std", "min", "max"]].round(2)
            stats.columns = ["Media", "Desv. Est.", "Mínimo", "Máximo"]
            st.dataframe(stats, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 🔭 Próximas Mejoras")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        - 📱 App móvil para captura en campo
        - 🤖 Modelo de visión (CNN) para lectura automática de zonas
        - 🛰️ Integración con datos satelitales NDVI
        - 🗺️ Cartografía de suelos IGAC Colombia
        """)
    with col_b:
        st.markdown("""
        - 🌐 Chatbot QR accesible en campo
        - 📡 Sensores IoT para monitoreo continuo
        - 🌱 Predicción de cosechas según salud del suelo
        - 🤝 Red de agricultores colaborativa
        """)

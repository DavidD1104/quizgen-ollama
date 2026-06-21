# -*- coding: utf-8 -*-
"""
QuizGen Web — Interfaz Streamlit para el generador de preguntas QuizGen.

Amplía el proyecto quizgen-ollama añadiendo una interfaz web interactiva
que reutiliza el pipeline RAG existente (quizgen.py). No modifica el script
original: importa sus funciones y añade la capa de presentación.

Uso:
    streamlit run quizgen_web.py

Requisitos adicionales respecto al quizgen.py original:
    pip install streamlit

El resto de dependencias (langchain, faiss, ollama, pypdf) son las mismas
que ya usa quizgen.py. Ollama debe estar corriendo (ollama serve).

Autor: David Domingo
Basado en: quizgen.py (pipeline RAG original)
"""

import json
import os
import shutil
import time

import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# Importamos las funciones del quizgen.py original.
# Esto demuestra que la versión web es una AMPLIACIÓN, no una reescritura.
# El pipeline RAG (cargar PDFs, crear índice, generar preguntas) se reutiliza
# tal cual. Solo añadimos la interfaz.
# ═══════════════════════════════════════════════════════════════════════════════
from quizgen import (
    cargar_o_crear_indice,
    generar_preguntas,
    exportar_preguntas,
    CARPETA_PDFS,
    CARPETA_INDICE,
    MODELO_LLM,
    MODELO_EMBEDDINGS,
    NUM_PREGUNTAS,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE PÁGINA
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="QuizGen — Test desde tus apuntes",
    page_icon="🧠",
    layout="wide",
)

# Ocultar menú hamburguesa y footer por defecto de Streamlit (más limpio)
st.markdown(
    "<style>"
    "[data-testid='stSidebarNav'] {display: none;}"
    "#MainMenu {visibility: hidden;}"
    "footer {visibility: hidden;}"
    "</style>",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES AUXILIARES ESPECÍFICAS DE LA VERSIÓN WEB
#
#  Estas funciones NO están en quizgen.py porque son específicas de Streamlit.
#  El pipeline RAG se sigue delegando al módulo original.
# ═══════════════════════════════════════════════════════════════════════════════

def guardar_pdfs_subidos(archivos_subidos: list, carpeta_destino: str) -> list:
    """
    Guarda los PDFs subidos por el usuario en la carpeta de documentos.

    Streamlit entrega los ficheros como objetos UploadedFile en memoria.
    Para que PyPDFLoader (usado por quizgen.py) pueda leerlos, necesitamos
    escribirlos a disco en la carpeta que espera cargar_o_crear_indice().

    Devuelve la lista de nombres de ficheros nuevos (los que no existían antes).
    """
    os.makedirs(carpeta_destino, exist_ok=True)

    existentes = set(os.listdir(carpeta_destino))
    nuevos = []

    for archivo in archivos_subidos:
        if archivo.name not in existentes:
            ruta = os.path.join(carpeta_destino, archivo.name)
            with open(ruta, "wb") as f:
                f.write(archivo.getvalue())
            nuevos.append(archivo.name)

    return nuevos


def listar_pdfs_en_carpeta(carpeta: str) -> list:
    """Devuelve los nombres de los PDFs disponibles en la carpeta."""
    if not os.path.isdir(carpeta):
        return []
    return sorted(f for f in os.listdir(carpeta) if f.lower().endswith(".pdf"))


def obtener_carpeta_indice(carpeta_pdfs: str) -> str:
    """
    Genera el nombre de la carpeta del índice FAISS a partir de la carpeta de PDFs.

    Replica la lógica de main() en quizgen.py:
        carpeta_indice = f"{CARPETA_INDICE}_{os.path.basename(args.carpeta)}"

    Así ambas versiones (terminal y web) comparten el mismo índice en disco.
    """
    return f"{CARPETA_INDICE}_{os.path.basename(carpeta_pdfs)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  INICIALIZACIÓN DEL ESTADO
#
#  Streamlit re-ejecuta el script completo en cada interacción del usuario.
#  st.session_state es el diccionario persistente que sobrevive entre reruns.
#  Aquí inicializamos todas las claves que usaremos.
#
#  Referencia: s4_streamlit_estado.py del profesor (patrón de inicialización)
# ═══════════════════════════════════════════════════════════════════════════════

# Estado del quiz
if "preguntas" not in st.session_state:
    st.session_state.preguntas = []          # Lista de dicts con las preguntas generadas

if "indice_actual" not in st.session_state:
    st.session_state.indice_actual = 0       # Qué pregunta se muestra ahora

if "respuestas_usuario" not in st.session_state:
    st.session_state.respuestas_usuario = {} # {0: "A", 1: "C", ...}

if "quiz_terminado" not in st.session_state:
    st.session_state.quiz_terminado = False  # True cuando se han respondido todas

if "fase" not in st.session_state:
    st.session_state.fase = "configurar"     # "configurar" → "quiz" → "resultados"


# ═══════════════════════════════════════════════════════════════════════════════
#  BARRA LATERAL — CONFIGURACIÓN Y GESTIÓN DOCUMENTAL
#
#  Patrón tomado de s13_streamlit_rag_ollama.py del profesor:
#  la columna izquierda gestiona documentos, la derecha es el área principal.
#  Aquí usamos sidebar en vez de columna porque el quiz necesita todo el ancho.
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("📂 Documentos")

    carpeta_pdfs = st.text_input(
        "Carpeta de PDFs",
        value=CARPETA_PDFS,
        help="Ruta a la carpeta con tus apuntes en PDF. "
             "Es la misma carpeta que usa quizgen.py en terminal.",
    )

    # Mostrar PDFs ya disponibles en la carpeta
    pdfs_existentes = listar_pdfs_en_carpeta(carpeta_pdfs)
    if pdfs_existentes:
        with st.expander(f"{len(pdfs_existentes)} PDF(s) disponibles", expanded=False):
            for pdf in pdfs_existentes:
                st.caption(f"📄 {pdf}")
    else:
        st.info("No hay PDFs en la carpeta. Sube algunos abajo.")

    # Subida de nuevos PDFs
    archivos = st.file_uploader(
        "Subir PDFs adicionales",
        type=["pdf"],
        accept_multiple_files=True,
        help="Los PDFs se guardan en la carpeta indicada arriba. "
             "Si subes ficheros nuevos, reindexar para incluirlos.",
    )

    if archivos:
        nuevos = guardar_pdfs_subidos(archivos, carpeta_pdfs)
        if nuevos:
            st.success(f"{len(nuevos)} PDF(s) nuevos guardados.")

    st.divider()

    # ── Configuración del modelo ──
    st.header("⚙️ Modelo")

    modelo_llm = st.text_input("Modelo LLM (Ollama)", value=MODELO_LLM)
    num_preguntas = st.slider("Número de preguntas", 1, 20, NUM_PREGUNTAS)

    st.divider()

    # ── Gestión del índice ──
    carpeta_idx = obtener_carpeta_indice(carpeta_pdfs)
    indice_existe = os.path.exists(carpeta_idx)

    if indice_existe:
        st.caption(f"✅ Índice FAISS disponible")
    else:
        st.caption("⬜ Sin índice (se creará al generar)")

    if st.button("🔄 Forzar reindexación", use_container_width=True):
        if os.path.exists(carpeta_idx):
            shutil.rmtree(carpeta_idx)
        st.success("Índice eliminado. Se creará de nuevo al generar.")

    st.divider()

    # ── Reiniciar quiz ──
    if st.button("🗑️ Reiniciar quiz", use_container_width=True):
        st.session_state.preguntas = []
        st.session_state.indice_actual = 0
        st.session_state.respuestas_usuario = {}
        st.session_state.quiz_terminado = False
        st.session_state.fase = "configurar"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  ÁREA PRINCIPAL — TRES FASES DEL FLUJO
#
#  El quiz tiene un flujo lineal de tres pasos:
#
#    1. CONFIGURAR  → El usuario ve los ajustes y pulsa "Generar Quiz"
#    2. QUIZ        → Se presentan las preguntas una a una
#    3. RESULTADOS  → Puntuación final, revisión de fallos, exportar JSON
#
#  Cada fase se renderiza condicionalmente según st.session_state.fase.
#  Este patrón de "máquina de estados" con session_state es la forma
#  estándar de manejar flujos multipaso en Streamlit.
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🧠 QuizGen")
st.caption("Genera preguntas tipo test a partir de tus apuntes con IA local")


# ─────────────────────────────────────────────────────────────────────────────
#  FASE 1: CONFIGURAR — Pantalla inicial
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.fase == "configurar":

    # Verificaciones previas
    pdfs = listar_pdfs_en_carpeta(carpeta_pdfs)

    if not pdfs:
        st.warning(
            "No hay PDFs disponibles. Sube archivos desde la barra lateral "
            "o indica una carpeta que contenga PDFs."
        )

    else:
        st.write(
            f"Se generarán **{num_preguntas} preguntas** a partir de "
            f"**{len(pdfs)} PDF(s)** usando el modelo **{modelo_llm}**."
        )

        if st.button("🚀 Generar Quiz", type="primary", use_container_width=True):

            # ── Paso 1: Crear/cargar índice FAISS ──
            # Reutilizamos cargar_o_crear_indice() de quizgen.py directamente.
            # Esta función gestiona la caché en disco: si el índice ya existe
            # lo carga, si no lo crea desde cero. Es la misma función que
            # se ejecuta cuando haces `python quizgen.py` en terminal.

            with st.spinner("Indexando documentos..."):
                try:
                    vectorstore = cargar_o_crear_indice(carpeta_pdfs, carpeta_idx)
                except Exception as e:
                    st.error(
                        f"Error al indexar: {e}\n\n"
                        "Comprueba que Ollama esté corriendo (`ollama serve`) "
                        f"y que el modelo de embeddings ({MODELO_EMBEDDINGS}) esté descargado."
                    )
                    st.stop()

            # ── Paso 2: Generar preguntas ──
            # Reutilizamos generar_preguntas() de quizgen.py.
            # Esta función:
            #   1. Elige un tema semilla aleatorio
            #   2. Usa el retriever FAISS para buscar los 3 fragmentos más relevantes
            #   3. Construye un prompt con el contexto recuperado
            #   4. Llama al LLM local y parsea el JSON de respuesta
            #   5. Valida campos obligatorios (pregunta, opciones, respuesta, explicación)
            #
            # El progreso se muestra con un spinner porque la generación puede
            # tardar 1-3 minutos dependiendo del modelo y del hardware.

            with st.spinner(
                f"Generando {num_preguntas} preguntas con {modelo_llm}... "
                "Esto puede tardar un par de minutos."
            ):
                try:
                    preguntas = generar_preguntas(
                        vectorstore, num_preguntas, modelo=modelo_llm
                    )
                except Exception as e:
                    st.error(
                        f"Error al generar preguntas: {e}\n\n"
                        f"Comprueba que el modelo {modelo_llm} esté descargado en Ollama."
                    )
                    st.stop()

            if not preguntas:
                st.error(
                    "No se pudieron generar preguntas. Posibles causas:\n\n"
                    "- Ollama no está corriendo\n"
                    f"- El modelo {modelo_llm} no está descargado\n"
                    "- Los PDFs no contienen texto extraíble"
                )
                st.stop()

            # ── Guardar en estado y cambiar de fase ──
            st.session_state.preguntas = preguntas
            st.session_state.indice_actual = 0
            st.session_state.respuestas_usuario = {}
            st.session_state.quiz_terminado = False
            st.session_state.fase = "quiz"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  FASE 2: QUIZ — Preguntas una a una
#
#  Diseño: una pregunta por pantalla (no todas a la vez) para que el usuario
#  se concentre y la experiencia sea más parecida a un examen real.
#  Se muestra una barra de progreso arriba y navegación abajo.
# ─────────────────────────────────────────────────────────────────────────────

elif st.session_state.fase == "quiz":

    preguntas = st.session_state.preguntas
    total = len(preguntas)
    idx = st.session_state.indice_actual

    # ── Barra de progreso ──
    respondidas = len(st.session_state.respuestas_usuario)
    st.progress(respondidas / total, text=f"Pregunta {idx + 1} de {total}")

    # ── Pregunta actual ──
    p = preguntas[idx]

    st.subheader(f"Pregunta {idx + 1}")
    st.markdown(f"**{p['pregunta']}**")

    # ── Opciones ──
    # Usamos st.radio para las opciones A/B/C/D.
    # La key incluye el índice de la pregunta para que Streamlit mantenga
    # cada radio independiente (si no, todas las preguntas compartirían
    # el mismo widget y se pisarían los valores).

    opciones_formateadas = [
        f"{letra}) {texto}" for letra, texto in p["opciones"].items()
    ]

    # Si el usuario ya respondió esta pregunta, pre-seleccionamos su respuesta
    respuesta_previa = st.session_state.respuestas_usuario.get(idx)
    indice_previo = None
    if respuesta_previa:
        letras = list(p["opciones"].keys())
        if respuesta_previa in letras:
            indice_previo = letras.index(respuesta_previa)

    seleccion = st.radio(
        "Elige tu respuesta:",
        opciones_formateadas,
        index=indice_previo,
        key=f"radio_pregunta_{idx}",
        label_visibility="collapsed",
    )

    # Extraer la letra (A, B, C o D) de la selección
    letra_seleccionada = seleccion.split(")")[0].strip() if seleccion else None

    # ── Botón de confirmar respuesta ──
    ya_respondida = idx in st.session_state.respuestas_usuario

    if not ya_respondida:
        if st.button("Confirmar respuesta", type="primary", use_container_width=True):
            st.session_state.respuestas_usuario[idx] = letra_seleccionada
            st.rerun()

    # ── Feedback tras responder ──
    if ya_respondida:
        respuesta_dada = st.session_state.respuestas_usuario[idx]
        correcta = p["respuesta_correcta"]
        es_correcta = respuesta_dada == correcta

        if es_correcta:
            st.success("✅ ¡Correcto!")
        else:
            st.error(
                f"❌ Incorrecto. Tu respuesta: {respuesta_dada}) — "
                f"Correcta: {correcta}) {p['opciones'][correcta]}"
            )

        # Mostrar explicación del LLM
        with st.expander("💡 Explicación", expanded=es_correcta):
            st.write(p["explicacion"])

        # ── Navegación ──
        col_prev, col_next = st.columns(2)

        with col_prev:
            if idx > 0:
                if st.button("← Anterior", use_container_width=True):
                    st.session_state.indice_actual = idx - 1
                    st.rerun()

        with col_next:
            if idx < total - 1:
                if st.button("Siguiente →", use_container_width=True):
                    st.session_state.indice_actual = idx + 1
                    st.rerun()
            else:
                # Última pregunta → botón para ver resultados
                if len(st.session_state.respuestas_usuario) == total:
                    if st.button("📊 Ver resultados", type="primary", use_container_width=True):
                        st.session_state.quiz_terminado = True
                        st.session_state.fase = "resultados"
                        st.rerun()
                else:
                    st.info(
                        f"Faltan {total - len(st.session_state.respuestas_usuario)} "
                        "preguntas por responder."
                    )

    # ── Mapa de preguntas (navegación rápida) ──
    # Permite saltar a cualquier pregunta. Las respondidas se marcan.
    with st.expander("🗂️ Mapa de preguntas"):
        cols = st.columns(min(total, 10))
        for i in range(total):
            col = cols[i % len(cols)]
            respondida = i in st.session_state.respuestas_usuario

            if respondida:
                correcta = (
                    st.session_state.respuestas_usuario[i]
                    == preguntas[i]["respuesta_correcta"]
                )
                etiqueta = f"{'✅' if correcta else '❌'} {i+1}"
            else:
                etiqueta = f"⬜ {i+1}"

            if col.button(etiqueta, key=f"nav_{i}", use_container_width=True):
                st.session_state.indice_actual = i
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  FASE 3: RESULTADOS — Puntuación y revisión
# ─────────────────────────────────────────────────────────────────────────────

elif st.session_state.fase == "resultados":

    preguntas = st.session_state.preguntas
    total = len(preguntas)

    # ── Calcular puntuación ──
    aciertos = sum(
        1
        for i, p in enumerate(preguntas)
        if st.session_state.respuestas_usuario.get(i) == p["respuesta_correcta"]
    )
    porcentaje = (aciertos / total) * 100 if total > 0 else 0

    # ── KPIs ──
    # Patrón de st.metric en columnas, tomado de s1_streamlit_escritura.py
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Puntuación", f"{aciertos}/{total}")
    with col2:
        st.metric("Porcentaje", f"{porcentaje:.0f}%")
    with col3:
        if porcentaje == 100:
            st.metric("Valoración", "🏆 Perfecto")
        elif porcentaje >= 70:
            st.metric("Valoración", "👍 Buen resultado")
        elif porcentaje >= 50:
            st.metric("Valoración", "📖 Repasa un poco")
        else:
            st.metric("Valoración", "📚 Necesitas repasar")

    # ── Barra visual de progreso ──
    st.progress(porcentaje / 100)

    st.divider()

    # ── Revisión detallada ──
    # Dos pestañas: una para los fallos (lo que importa repasar) y otra
    # con todas las preguntas. Patrón de st.tabs tomado de s9.
    tab_fallos, tab_todas = st.tabs(["❌ Preguntas falladas", "📋 Todas las preguntas"])

    with tab_fallos:
        fallos = [
            (i, p)
            for i, p in enumerate(preguntas)
            if st.session_state.respuestas_usuario.get(i) != p["respuesta_correcta"]
        ]

        if not fallos:
            st.success("¡No has fallado ninguna pregunta!")
        else:
            for i, p in fallos:
                resp_usuario = st.session_state.respuestas_usuario.get(i, "?")
                correcta = p["respuesta_correcta"]

                st.markdown(f"**P{i+1}. {p['pregunta']}**")
                st.markdown(
                    f"Tu respuesta: **{resp_usuario})** {p['opciones'].get(resp_usuario, '')} — "
                    f"Correcta: **{correcta})** {p['opciones'][correcta]}"
                )
                st.caption(f"💡 {p['explicacion']}")
                st.divider()

    with tab_todas:
        for i, p in enumerate(preguntas):
            resp_usuario = st.session_state.respuestas_usuario.get(i, "?")
            correcta = p["respuesta_correcta"]
            acertada = resp_usuario == correcta

            icono = "✅" if acertada else "❌"
            st.markdown(f"**{icono} P{i+1}. {p['pregunta']}**")

            for letra, texto in p["opciones"].items():
                marcador = ""
                if letra == correcta:
                    marcador = " ✅"
                elif letra == resp_usuario and not acertada:
                    marcador = " ❌"
                st.caption(f"  {letra}) {texto}{marcador}")

            st.caption(f"💡 {p['explicacion']}")
            st.divider()

    # ── Acciones finales ──
    st.subheader("Acciones")

    col_exportar, col_nuevo = st.columns(2)

    with col_exportar:
        # Exportar a JSON — reutilizamos el formato de exportar_preguntas()
        # de quizgen.py, pero aquí usamos st.download_button para que el
        # usuario descargue el fichero directamente desde el navegador.
        datos_export = json.dumps(
            st.session_state.preguntas, ensure_ascii=False, indent=2
        )
        st.download_button(
            "📥 Exportar preguntas (JSON)",
            data=datos_export,
            file_name="quizgen_preguntas.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_nuevo:
        if st.button("🔄 Nuevo quiz", use_container_width=True):
            st.session_state.preguntas = []
            st.session_state.indice_actual = 0
            st.session_state.respuestas_usuario = {}
            st.session_state.quiz_terminado = False
            st.session_state.fase = "configurar"
            st.rerun()

# -*- coding: utf-8 -*-
"""
QuizGen - Generador de preguntas tipo test a partir de PDFs usando Ollama + RAG

Carga documentos PDF, los indexa con FAISS y genera preguntas de opción múltiple
usando un LLM local vía Ollama.

Uso:
    python quizgen.py                          # Usa la carpeta por defecto
    python quizgen.py --carpeta "mis_apuntes"  # Carpeta personalizada
    python quizgen.py --num 10                 # Genera 10 preguntas
    python quizgen.py --exportar quiz.json     # Exporta las preguntas a JSON
"""

import argparse
import json
import os
import random
import re
import sys

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN POR DEFECTO ---
CARPETA_PDFS = "Documentacion RAG Preguntas"
CARPETA_INDICE = "quizgen_vectorstore"
MODELO_LLM = "llama3.1"
MODELO_EMBEDDINGS = "mxbai-embed-large"
NUM_PREGUNTAS = 5


def cargar_o_crear_indice(carpeta_pdfs, carpeta_indice):
    """Carga el índice FAISS existente o lo crea a partir de los PDFs."""
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDINGS)

    if os.path.exists(carpeta_indice):
        print(f"[*] Cargando índice existente desde '{carpeta_indice}'...")
        return FAISS.load_local(
            carpeta_indice, embeddings, allow_dangerous_deserialization=True
        )

    print(f"[*] Indexando PDFs de '{carpeta_pdfs}'...")
    archivos = [
        os.path.join(carpeta_pdfs, f)
        for f in os.listdir(carpeta_pdfs)
        if f.lower().endswith(".pdf")
    ]

    if not archivos:
        print(f"[!] No se encontraron PDFs en '{carpeta_pdfs}'.")
        sys.exit(1)

    documentos = []
    for pdf in archivos:
        print(f"    - {os.path.basename(pdf)}")
        loader = PyPDFLoader(pdf)
        documentos.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    fragmentos = splitter.split_documents(documentos)

    print(f"[*] Generando embeddings para {len(fragmentos)} fragmentos...")
    vectorstore = FAISS.from_documents(fragmentos, embeddings)
    vectorstore.save_local(carpeta_indice)
    print("[*] Índice guardado.")
    return vectorstore


def generar_preguntas(vectorstore, num_preguntas, modelo=MODELO_LLM):
    """Genera preguntas tipo test usando RAG sobre los documentos indexados."""
    llm = ChatOllama(model=modelo, temperature=0.4)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    temas_semilla = [
        "conceptos fundamentales y definiciones",
        "algoritmos y técnicas principales",
        "aplicaciones prácticas y ejemplos",
        "diferencias entre métodos o enfoques",
        "ventajas y limitaciones",
        "parámetros e hiperparámetros",
        "arquitecturas y modelos",
        "métricas de evaluación",
        "proceso de entrenamiento",
        "preprocesamiento de datos",
    ]

    preguntas = []
    intentos = 0
    max_intentos = num_preguntas * 3

    while len(preguntas) < num_preguntas and intentos < max_intentos:
        intentos += 1
        tema = random.choice(temas_semilla)
        docs = retriever.invoke(f"Explica {tema}")

        if not docs:
            continue

        contexto = "\n\n".join(d.page_content for d in docs[:3])

        prompt = f"""A partir del siguiente contenido académico, genera UNA pregunta tipo test de opción múltiple.

REGLAS:
- La pregunta debe evaluar comprensión, no memorización literal.
- Exactamente 4 opciones: A, B, C, D.
- Solo una respuesta correcta.
- Las opciones incorrectas deben ser plausibles.
- Responde EXCLUSIVAMENTE con JSON válido, sin texto extra ni markdown.

FORMATO JSON exacto:
{{
  "pregunta": "texto de la pregunta",
  "opciones": {{
    "A": "opción A",
    "B": "opción B",
    "C": "opción C",
    "D": "opción D"
  }},
  "respuesta_correcta": "A",
  "explicacion": "breve explicación de por qué es correcta"
}}

CONTENIDO:
{contexto[:3000]}"""

        try:
            response = llm.invoke(prompt)
            texto = response.content.strip()

            # Extraer JSON de la respuesta
            match = re.search(r"\{.*\}", texto, re.DOTALL)
            if not match:
                continue

            pregunta = json.loads(match.group(0))

            campos = ["pregunta", "opciones", "respuesta_correcta", "explicacion"]
            if not all(c in pregunta for c in campos):
                continue
            if pregunta["respuesta_correcta"] not in ["A", "B", "C", "D"]:
                continue
            if len(pregunta["opciones"]) != 4:
                continue

            preguntas.append(pregunta)
            print(f"    Pregunta {len(preguntas)}/{num_preguntas} generada.")

        except (json.JSONDecodeError, Exception):
            continue

    return preguntas


def ejecutar_quiz(preguntas):
    """Ejecuta el quiz de forma interactiva en la terminal."""
    aciertos = 0
    total = len(preguntas)

    print(f"\n{'='*50}")
    print(f"  QUIZ - {total} preguntas")
    print(f"{'='*50}\n")

    for i, p in enumerate(preguntas, 1):
        print(f"--- Pregunta {i}/{total} ---")
        print(f"\n{p['pregunta']}\n")

        for letra, texto in p["opciones"].items():
            print(f"  {letra}) {texto}")

        while True:
            respuesta = input("\nTu respuesta (A/B/C/D): ").strip().upper()
            if respuesta in ["A", "B", "C", "D"]:
                break
            print("Opción no válida. Escribe A, B, C o D.")

        correcta = p["respuesta_correcta"]
        if respuesta == correcta:
            aciertos += 1
            print(f"\nCORRECTO!")
        else:
            print(f"\nINCORRECTO. La respuesta era: {correcta}) {p['opciones'][correcta]}")

        print(f"Explicación: {p['explicacion']}\n")

    # Resultado final
    porcentaje = (aciertos / total) * 100 if total > 0 else 0
    print(f"{'='*50}")
    print(f"  RESULTADO: {aciertos}/{total} ({porcentaje:.0f}%)")

    if porcentaje == 100:
        print("  Perfecto!")
    elif porcentaje >= 70:
        print("  Buen resultado.")
    elif porcentaje >= 50:
        print("  Aprobado justo, repasa un poco más.")
    else:
        print("  Necesitas repasar. Vuelve a intentarlo.")

    print(f"{'='*50}")
    return aciertos, total


def exportar_preguntas(preguntas, ruta):
    """Guarda las preguntas generadas en un archivo JSON."""
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(preguntas, f, ensure_ascii=False, indent=2)
    print(f"[*] {len(preguntas)} preguntas exportadas a '{ruta}'.")


def main():
    parser = argparse.ArgumentParser(
        description="QuizGen - Genera preguntas tipo test a partir de PDFs con Ollama"
    )
    parser.add_argument(
        "--carpeta", default=CARPETA_PDFS,
        help=f"Carpeta con los PDFs (default: {CARPETA_PDFS})"
    )
    parser.add_argument(
        "--num", type=int, default=NUM_PREGUNTAS,
        help=f"Número de preguntas a generar (default: {NUM_PREGUNTAS})"
    )
    parser.add_argument(
        "--exportar", default=None,
        help="Exportar preguntas a archivo JSON (ej: quiz.json)"
    )
    parser.add_argument(
        "--modelo", default=MODELO_LLM,
        help=f"Modelo de Ollama a usar (default: {MODELO_LLM})"
    )
    parser.add_argument(
        "--reindexar", action="store_true",
        help="Forzar reindexación de los PDFs"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.carpeta):
        print(f"[!] La carpeta '{args.carpeta}' no existe.")
        sys.exit(1)

    carpeta_indice = f"{CARPETA_INDICE}_{os.path.basename(args.carpeta)}"

    if args.reindexar and os.path.exists(carpeta_indice):
        import shutil
        shutil.rmtree(carpeta_indice)
        print("[*] Índice anterior eliminado.")

    # 1. Cargar/crear índice
    vectorstore = cargar_o_crear_indice(args.carpeta, carpeta_indice)

    # 2. Generar preguntas
    print(f"\n[*] Generando {args.num} preguntas con {args.modelo}...")
    preguntas = generar_preguntas(vectorstore, args.num, modelo=args.modelo)

    if not preguntas:
        print("[!] No se pudieron generar preguntas. Revisa que Ollama esté corriendo.")
        sys.exit(1)

    # 3. Exportar si se pidió
    if args.exportar:
        exportar_preguntas(preguntas, args.exportar)

    # 4. Quiz interactivo
    ejecutar_quiz(preguntas)


if __name__ == "__main__":
    main()

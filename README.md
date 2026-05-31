# QuizGen

Generador de preguntas tipo test de opción múltiple a partir de tus propios apuntes en PDF. Usa **RAG** (Retrieval-Augmented Generation) para indexar el contenido y un **LLM local** vía [Ollama](https://ollama.com/) para crear preguntas con 4 opciones, respuesta correcta y explicación. Todo corre en local, sin APIs de pago.

## Funcionalidades

- Indexa cualquier carpeta de PDFs con FAISS (con caché para no reindexar cada vez)
- Genera preguntas de comprensión, no de memorización literal
- Quiz interactivo en terminal con puntuación final
- Exportación de preguntas a JSON para reutilizarlas o compartirlas
- Modelo LLM configurable por línea de comandos

## Requisitos previos

- **Python 3.11**
- **Ollama** instalado y en ejecución (`ollama serve`)
- Modelos descargados:
  ```bash
  ollama pull llama3.1
  ollama pull mxbai-embed-large
  ```

## Instalación

```bash
# Crear entorno conda (recomendado)
conda create --prefix ./env python=3.11 -y
conda activate ./env

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
python quizgen.py                              # 5 preguntas de la carpeta por defecto
python quizgen.py --num 10                     # 10 preguntas
python quizgen.py --carpeta "mis_apuntes"      # carpeta de PDFs personalizada
python quizgen.py --exportar quiz.json         # guardar preguntas en JSON
python quizgen.py --reindexar                  # forzar reindexación de los PDFs
python quizgen.py --modelo llama3              # usar otro modelo de Ollama
```

La carpeta de PDFs por defecto es `Documentacion RAG Preguntas/`. En la primera ejecución se crea un directorio `quizgen_vectorstore_*/` con el índice FAISS; en ejecuciones posteriores se reutiliza para ir más rápido.

## Ejemplo de ejecución

```
$ python quizgen.py

[*] Indexando PDFs de 'Documentacion RAG Preguntas'...
    - Tema_3_Fundamentos_del_Aprendizaje_Automático.pdf
    - Tema_4_Aprendizaje_No_Supervisado_y_Reducción_Características.pdf
    - Tema_5_Shallow_Neural_Networks.pdf
    - Tema_6_Redes_Recurrentes_y_NLP.pdf
    - Tema_7_Redes_Neuronales_Convolucionales.pdf
[*] Generando embeddings para 376 fragmentos...
[*] Índice guardado.

[*] Generando 5 preguntas con llama3.1...
    Pregunta 1/5 generada.
    Pregunta 2/5 generada.
    Pregunta 3/5 generada.
    Pregunta 4/5 generada.
    Pregunta 5/5 generada.

==================================================
  QUIZ - 5 preguntas
==================================================

--- Pregunta 1/5 ---

¿Cuál es el objetivo principal del aprendizaje no supervisado?

  A) Predecir categorías de futuros ejemplos
  B) Encontrar patrones en datos sin etiquetas o respuestas predefinidas
  C) Mejorar la programación dinámica para problemas complejos
  D) Aplicar aprendizaje multi-tarea a nuevos problemas

Tu respuesta (A/B/C/D): B

CORRECTO!
Explicación: El aprendizaje no supervisado busca encontrar patrones y relaciones
en datos sin etiquetas o respuestas predefinidas.

==================================================
  RESULTADO: 4/5 (80%)
  Buen resultado.
==================================================
```

## Cómo funciona

QuizGen utiliza un pipeline **RAG (Retrieval-Augmented Generation)** para generar preguntas fundamentadas en el contenido real de tus apuntes. A continuación se explica cada etapa del proceso.

### Diagrama del pipeline

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌───────────────┐
│  PDFs de    │────>│  Chunking    │────>│  Embeddings    │────>│  Índice FAISS │
│  entrada    │     │  (1000 chars)│     │  (mxbai-embed) │     │  (vectorstore)│
└─────────────┘     └──────────────┘     └────────────────┘     └───────┬───────┘
                                                                        │
                    ┌──────────────┐     ┌────────────────┐             │
                    │  Pregunta    │<────│  LLM genera    │<────────────┘
                    │  tipo test   │     │  pregunta JSON │  retriever busca
                    └──────────────┘     └────────────────┘  los 3 fragmentos
                                                             más relevantes
```

### 1. Carga y lectura de PDFs

El primer paso es leer todos los archivos `.pdf` de la carpeta indicada. Para ello se usa `PyPDFLoader` de LangChain, que extrae el texto plano de cada página manteniendo metadatos como el nombre del archivo y el número de página de origen. Esto permite que más adelante, al recuperar fragmentos, se pueda trazar de dónde viene cada trozo de información.

### 2. Chunking (fragmentación del texto)

Un LLM no puede procesar documentos enteros de cientos de páginas de golpe: tiene un límite de contexto y, además, cuanto más texto se le pasa, menos precisa es su respuesta. Por eso se divide el texto en **chunks** (fragmentos) más pequeños.

QuizGen utiliza `RecursiveCharacterTextSplitter` con:
- **chunk_size = 1000** caracteres por fragmento
- **chunk_overlap = 150** caracteres de solapamiento entre fragmentos consecutivos

El solapamiento es importante: evita que una idea que cae justo en el límite entre dos fragmentos se pierda o quede cortada. El splitter intenta cortar por párrafos, luego por frases, luego por palabras, para que cada fragmento sea lo más coherente posible.

### 3. Embeddings (representación vectorial)

Aquí es donde entra la parte más interesante. Cada fragmento de texto se convierte en un **vector numérico** (embedding) que captura su significado semántico. Dos textos que hablan de lo mismo tendrán vectores cercanos en el espacio, aunque usen palabras distintas.

QuizGen usa el modelo **`mxbai-embed-large`** ejecutado localmente a través de Ollama. Este modelo transforma cada fragmento en un vector de alta dimensionalidad. Por ejemplo, un fragmento sobre "redes neuronales convolucionales" y otro sobre "CNN para clasificación de imágenes" producirán vectores muy próximos entre sí, porque semánticamente tratan el mismo concepto.

### 4. Indexación con FAISS

Los vectores generados se almacenan en un **índice FAISS** (Facebook AI Similarity Search). FAISS es una librería optimizada para búsqueda por similitud vectorial: dado un vector de consulta, encuentra rápidamente los *k* vectores más cercanos del índice.

El índice se guarda en disco (`quizgen_vectorstore_*/`) para no tener que recalcular los embeddings en cada ejecución. Solo se regenera si se usa `--reindexar` o si se cambia la carpeta de PDFs.

### 5. RAG: Retrieval-Augmented Generation

RAG es el patrón que conecta la búsqueda vectorial con la generación de texto del LLM. En lugar de pedirle al modelo que genere preguntas "de la nada" (lo que provocaría alucinaciones o preguntas genéricas), se le proporciona **contexto real extraído de tus apuntes**.

El proceso funciona así:

1. Se elige aleatoriamente un **tema semilla** de una lista predefinida (p. ej., "algoritmos y técnicas principales", "métricas de evaluación", "arquitecturas y modelos"...).
2. El **retriever** convierte ese tema en un embedding y busca en FAISS los **3 fragmentos más similares** (`k=3`). Esto es la fase de *retrieval*.
3. Esos fragmentos se inyectan como contexto en el **prompt** que se envía al LLM. Esto es la fase de *augmented generation*: el modelo genera basándose en información real, no inventada.

De esta forma, cada pregunta generada está anclada al contenido de tus PDFs concretos.

### 6. Generación de preguntas con el LLM

El LLM (`llama3.1` por defecto, ejecutado localmente vía Ollama con `temperature=0.4`) recibe un prompt estructurado que incluye:

- El contexto recuperado por el retriever (máximo 3000 caracteres)
- Instrucciones para generar exactamente **una pregunta tipo test** con 4 opciones (A, B, C, D)
- La regla de que la pregunta debe evaluar **comprensión**, no memorización literal
- El requisito de que las opciones incorrectas sean **plausibles** (no absurdas)
- El formato de salida en **JSON estricto** con los campos: `pregunta`, `opciones`, `respuesta_correcta` y `explicacion`

La temperatura a 0.4 equilibra entre creatividad (para no repetir siempre las mismas preguntas) y coherencia (para que las preguntas tengan sentido).

Tras recibir la respuesta del LLM, se valida el JSON: se comprueba que tenga los 4 campos obligatorios, que la respuesta correcta sea una letra válida (A-D) y que haya exactamente 4 opciones. Si la validación falla, se descarta y se reintenta (hasta un máximo de `num_preguntas * 3` intentos).

### 7. Quiz interactivo y puntuación

Las preguntas validadas se presentan una a una en la terminal. Tras cada respuesta del usuario se muestra si ha acertado o no, la respuesta correcta y la explicación generada por el LLM. Al final se calcula la puntuación total con un mensaje según el porcentaje de aciertos.

### Resumen visual

```
Tus PDFs ──> Trozos de texto ──> Vectores numéricos ──> Índice FAISS
                                                             │
Tema aleatorio ──> Embedding de consulta ────────────> Búsqueda similitud
                                                             │
                                              Top 3 fragmentos relevantes
                                                             │
                                                      Prompt + Contexto
                                                             │
                                                     LLM local (Ollama)
                                                             │
                                                  Pregunta tipo test (JSON)
                                                             │
                                                    Quiz en terminal
```

## Tecnologías

| Tecnología | Papel en el proyecto |
|---|---|
| [Python 3.11](https://www.python.org/) | Lenguaje principal |
| [Ollama](https://ollama.com/) | Motor para ejecutar LLMs en local sin APIs de pago |
| [LangChain](https://www.langchain.com/) | Orquestación del pipeline RAG (loaders, splitters, retrievers) |
| [FAISS](https://github.com/facebookresearch/faiss) | Búsqueda eficiente por similitud vectorial (Facebook AI) |
| [pypdf](https://github.com/py-pdf/pypdf) | Extracción de texto de archivos PDF |
| [mxbai-embed-large](https://ollama.com/library/mxbai-embed-large) | Modelo de embeddings para representar texto como vectores |
| [llama3.1](https://ollama.com/library/llama3.1) | LLM para generar las preguntas tipo test |

## Estructura del proyecto

```
QuizGen/
├── quizgen.py                      # Script principal
├── requirements.txt                # Dependencias Python
├── ejemplo_ejecucion.txt           # Salida de una ejecución real
├── Documentacion RAG Preguntas/    # PDFs de ejemplo (apuntes de IA)
│   ├── Tema_3_Fundamentos_del_Aprendizaje_Automático.pdf
│   ├── Tema_4_Aprendizaje_No_Supervisado_y_Reducción_Características.pdf
│   ├── Tema_5_Shallow_Neural_Networks.pdf
│   ├── Tema_6_Redes_Recurrentes_y_NLP.pdf
│   └── Tema_7_Redes_Neuronales_Convolucionales.pdf
├── .gitignore
├── CLAUDE.md
└── README.md
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**QuizGen** is a standalone tool that generates multiple-choice quiz questions from PDF documents using RAG (Retrieval-Augmented Generation) + Ollama as a local LLM runtime. It indexes PDFs with FAISS, generates questions via a local LLM, and runs an interactive quiz in the terminal. Built as part of a university course on Generative AI.

## Architecture

Single-script application (`quizgen.py`) with the following pipeline:

1. **PDF loading** — reads all PDFs from a configurable folder using `PyPDFLoader`
2. **Chunking** — splits documents into 1000-char chunks (150 overlap) with `RecursiveCharacterTextSplitter`
3. **Embedding + indexing** — generates embeddings via `mxbai-embed-large` and stores them in a FAISS index
4. **Question generation** — retrieves relevant chunks per topic seed, prompts the LLM to produce a JSON-formatted multiple-choice question
5. **Interactive quiz** — presents questions in the terminal, collects answers, scores results
6. **Export** — optionally saves generated questions to a JSON file

Key dependencies: `langchain-ollama`, `langchain-community`, `faiss-cpu`, `pypdf`

## Environment Setup

```bash
# Create conda env (Python 3.11)
conda create --prefix <path> python=3.11 -y
conda activate <path>
pip install -r requirements.txt
```

Ollama must be running locally (`ollama serve`) on `http://localhost:11434`.

## Required Ollama Models

- `llama3.1` — question generation (LLM)
- `mxbai-embed-large` — text embeddings

Pull with: `ollama pull <model-name>`

## Running

```bash
python quizgen.py                              # 5 questions from default folder
python quizgen.py --num 10                     # 10 questions
python quizgen.py --carpeta "otra_carpeta"     # custom PDF folder
python quizgen.py --exportar quiz.json         # save questions to JSON
python quizgen.py --reindexar                  # force re-index PDFs
```

The default PDF folder is `Documentacion RAG Preguntas/`. A `quizgen_vectorstore_*/` cache directory is created on first run and reused on subsequent runs unless `--reindexar` is passed.

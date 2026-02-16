# Enterprise-Ready RAG – Take Home Project

## Overview

This project implements a minimal Retrieval-Augmented Generation (RAG) pipeline designed to demonstrate how a prototype can evolve into an enterprise-ready architecture.

The system combines:

Local LLaMA (via Ollama)

Sentence-Transformer embeddings

Chroma vector database

Modular ingestion, retrieval, and response components

This implementation focuses not only on functionality, but also on scalability, modularity, and production alignment.
  
---
## Architecture

<img width="995" height="801" alt="image" src="https://github.com/user-attachments/assets/ea6d9e96-f234-4ce3-9aa4-e164eefb8dc7" />


---
## Repository Structure
```bash
data/                       # Sample dataset
01_ingestion.py             # Document loading + chunking + embedding
02_retrieval.py             # Similarity search + filtering
03_response.py              # Prompt augmentation + LLM call
rag_note_book.ipynb         # End-to-end demo
requirements.txt            # Dependencies
Enterprise_RAG_Presentation.pdf
README.md
```bash

---
## Enterprise Extension Path

This prototype can be extended to enterprise environments by:

Airflow-based ingestion orchestration

Incremental indexing

Hybrid retrieval (BM25 + dense)

Kubernetes deployment

GPU scheduling (vLLM)

Autoscaling inference services

Retrieval quality measurement (Recall@K, MRR)

---

## Requirements

Before running locally, install:

#### 1) Python (3.9+)
https://www.python.org/downloads/

#### 2) Ollama (for Local LLaMA)
https://ollama.com/download

After installing Ollama, pull the model:

```bash
ollama pull llama3
```

## Option 1 — Run in Google Colab (Quick Preview)
Open:

```bash
https://colab.research.google.com/github/kavya-18/Take_home_rag-project/blob/main/rag_note_book.ipynb

```

### Click:

1. Run anyway

2. Connect

3. Runtime → Run all

### Note:
Because this project uses a local LLaMA model, the final LLM generation step will not run in Colab.

### You can still:

Load the document

Create embeddings

Store in vector DB

Retrieve relevant chunks

## Option 2 — Run Locally (Full Functionality)
Open terminal inside the project folder.

### Step 1 — Create virtual environment
```bash
python -m venv venv
```

#### Activate:

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```


### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the notebook
```bash
jupyter notebook
```

Open:
```bash
rag_note_book.ipynb
```

Click Run All.

---

## Expected Output

When running locally, the system:

Loads and chunks documents

Stores embeddings in Chroma

Retrieves top-K relevant chunks

Generates a grounded response using LLaMA

The notebook demonstrates intermediate outputs for transparency.











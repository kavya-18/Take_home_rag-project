# Take Home RAG Project

This project demonstrates a minimal RAG pipeline using:

- Local LLaMA (via Ollama)
- Chroma vector database
- Sentence-transformer embeddings
- Modular ingestion, retrieval, and response
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/0a29b8e5-05cc-4ba3-8a36-2462eafc488c" />

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









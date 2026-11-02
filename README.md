# Bilou Camera — RAG Project

A Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions about the
**Bilou Camera** engineering documentation and get answers grounded **only** in those documents.

The pipeline uses **Ollama** locally to turn text into embeddings, **ChromaDB** to store and
search those embeddings, and the **Google Gemini API** to generate the final answer from the
retrieved context.

---

## How it works

```
data/*.txt
    │
    ▼  load_data.ipynb
Load + split into chunks (800 chars, 80 overlap)
    │
    ▼  Ollama  ·  nomic-embed-text
Embed every chunk locally  →  280 vectors
    │
    ▼
ChromaDB (chroma_db/)  ·  persistent vector store
    │
    ▼  run_rag.py
User question → embed with the same Ollama model
    │
    ▼
Top 5 most similar chunks (similarity_search_with_score, k=5)
    │
    ▼  Gemini  ·  gemini-3.8-flash  (temperature 0.2)
Answer generated from the retrieved context + list of sources
```

### 1. Ingestion — `load_data.ipynb`

1. **Load** every `.txt` file under `data/` with LangChain's `DirectoryLoader` + `TextLoader`
   → **2 documents**.
2. **Split** each document with `RecursiveCharacterTextSplitter`
   (`chunk_size=800`, `chunk_overlap=80`) → **280 chunks**.
3. **Give each chunk a deterministic ID** — `<source>:<page>:<chunk_index>`,
   e.g. `data/biloU_Camera_rapport.txt:0:12`. Re-running the notebook therefore never
   creates duplicates: existing IDs are checked first and only new chunks are inserted.
4. **Embed** each chunk with `OllamaEmbeddings(model="nomic-embed-text")`.
5. **Store** the vectors in a persistent ChromaDB database at `chroma_db/`.

### 2. Querying — `run_rag.py`

1. Embed the user's question with the **same** Ollama embedding model.
2. Retrieve the **top 5** closest chunks from ChromaDB.
3. Build a prompt containing only those chunks as `Context` and the question.
4. Send the prompt to **Gemini** (`ChatGoogleGenerativeAI`, `gemini-3.8-flash`, `temperature=0.2`)
   with the instruction to answer *from the context only, without adding or removing information*.
5. Print the generated **Answer** and the **Sources** (chunk IDs) it was built from.

---

## Why Ollama for embeddings?

Embeddings are the numeric representation of text used for similarity search. They are needed
for **every chunk and every query**, so they run thousands of times. Doing that locally with
Ollama means:

- **Free** — no per-token embedding cost, no API quota.
- **Private** — the documents never leave the machine.
- **Fast and offline** — works without an internet connection.
- **Small and specialized** — `nomic-embed-text` is a compact model that produces high-quality
  embeddings and runs comfortably on a laptop.

Install it once:

```bash
# macOS / Linux / Windows: https://ollama.com
ollama pull nomic-embed-text
```

Ollama must be running in the background when you ingest or query.

## Why Gemini for the answer?

Retrieval only *finds* relevant text; something still has to read it and write a clear,
accurate answer. That is the "generation" half of RAG, and it happens **once per question**,
so it benefits from a strong hosted model:

- **Reasoning + writing quality** — Gemini composes an answer from several retrieved chunks
  and respects the strict "context only" prompt.
- **Cheap at query time** — one call per question, not per chunk.
- **No large local model needed** — generation stays in the cloud while the heavy,
  repeated work (embeddings) stays local.

You need a Google API key:

1. Create one at <https://aistudio.google.com/app/apikey>.
2. Export it before running:

   ```bash
   export GOOGLE_API_KEY="your-key-here"
   ```

`run_rag.py` reads the key from the environment. **Never commit a real key to Git.**

---

## The data

The knowledge base is two plain-text engineering documents about Bilou Camera, an iOS
film-simulation app (Swift / SwiftUI / AVFoundation / ImageIO / Core Image / Metal).

| File | Size | Lines | What it contains |
|---|---|---|---|
| `data/BilouCamera_Final_Engineering_Architecture_v2.txt` | ~55.5k chars | 1,298 | **Final engineering architecture v2** — the implementation baseline: executive architecture decision, accept/modify/replace review, Git acquisition + licensing, final system architecture, Swift module/directory structure, film-simulation profile contract and registry, how to build `bilou.chrome.v1` and `bilou.negative.v1`, colour science and input contract, the LUT engine (parsing, sampling, baking), the recipe engine data model, testing and implementation order. |
| `data/biloU_Camera_rapport.txt` | ~105.8k chars | 1,821 | **Comprehensive architectural & implementation report** — an audit of the real GitHub repositories (`fujilab`, `Fuji-Recipe-Converter`, `fuji-recipes`, the Higher Source LUT pack), the photo pipeline architecture, Classic Chrome and Classic Negative step-by-step builds, the recipe engine on top of film simulation, suggestions and fixes, the photo roadmap, and the first execution tasks. |

Both documents cover the same project from different angles: **v2** is the consolidated
specification, the **report** is the audit and reasoning behind it. Together they are
~161k characters → **280 retrievable chunks**.

### Legal context stored in the data

The documents repeatedly state the product is an **independent approximation** inspired by
popular Fujifilm-style simulations — not an official or identical Fujifilm rendering. No
proprietary LUTs or RAF decoding are used, and every asset must carry its source, licence
and checksum.

---

## Results

The two screenshots below show the pipeline answering a question from the documentation.

**Retrieval and answer**

![RAG result 1](Screenshot%202026-11-02%20at%2009.18.38.png)

**Answer with sources**

![RAG result 2](Screenshot%202026-11-02%20at%2009.19.10.png)

---

## Project structure

```
RAG_project/
├── data/
│   ├── BilouCamera_Final_Engineering_Architecture_v2.txt
│   └── biloU_Camera_rapport.txt
├── chroma_db/            # persistent ChromaDB vector store (280 chunks)
├── load_data.ipynb       # ingestion: load → split → embed → store
├── run_rag.py            # query: retrieve → prompt → generate → print sources
├── Screenshot ...png     # result screenshots
└── README.md
```

## Setup

```bash
# 1. Ollama (embeddings)
ollama pull nomic-embed-text

# 2. Python dependencies
pip install langchain langchain-community langchain-chroma \
            langchain-google-genai langchain-text-splitters \
            chromadb python-dotenv notebook

# 3. Gemini (generation)
export GOOGLE_API_KEY="your-key-here"
```

## Usage

```bash
# Build / update the vector database (run load_data.ipynb top to bottom)
jupyter notebook load_data.ipynb

# Ask a question
python run_rag.py
```

To ask a different question, edit `question` at the bottom of `run_rag.py`, or import
`query_rag` and call it directly.

## Notes

- `run_rag.py` uses `k=5` retrieved chunks and `temperature=0.2` to keep answers close to
  the source text.
- Embeddings must use the **same model** at ingestion and query time, otherwise the vectors
  are not comparable.
- The vector database is committed so the project runs immediately; delete `chroma_db/` and
  re-run the notebook to rebuild it from the text files.
- `OllamaEmbeddings` from `langchain_community` is deprecated; `langchain-ollama` is the
  long-term replacement.

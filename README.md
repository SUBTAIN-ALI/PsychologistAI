# Psychologist AI Assistant

An intelligent platform that automates lead generation workflows and provides a continuously learning research assistant using Retrieval-Augmented Generation (RAG).

## Features

This system is composed of two main subsystems:

### 1. Visibility System (AI Lead Generation Automation)
A multi-agent workflow powered by LangGraph that orchestrates the following:
* **Manager Agent**: Controls the flow of conversation and logic.
* **Lead Generation Agent**: Analyzes potential leads and qualifies them for therapy support.
* **Message Writing Agent**: Generates empathetic and supportive initial outreach messages.
* **Conversation Agent**: Handles multi-turn chat intelligently.
* **Scheduling Agent**: Integrates with calendar availability to offer and book consultation slots.

### 2. Research System (AI Research Assistant using RAG)
A powerful RAG pipeline utilizing Qdrant and LangChain to provide context-aware insights from psychological literature.
* **Document Ingestion**: Recursively chunks and indexes PDFs or Text documents.
* **Semantic Search**: Fast and accurate retrieval of psychology knowledge using OpenAI Embeddings.
* **Research Assistant QA**: Provides scientifically backed and professional answers to complex therapy questions.

---

## Tech Stack
* **Language:** Python
* **Orchestration:** LangGraph & LangChain
* **Vector Database:** Qdrant
* **LLM & Embeddings:** OpenAI (`gpt-4o`, `text-embedding-3-small`)
* **API Framework:** FastAPI

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd PsychologistAI
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   A `.env` file is required in the root directory. Configure your OpenAI API Key and (optionally) your external Qdrant URL.
   ```ini
   OPENAI_API_KEY=your-openai-key-here
   QDRANT_URL=":memory:" # Uses local memory by default
   VECTOR_CHUNK_SIZE=800
   VECTOR_CHUNK_OVERLAP=150
   EMBEDDING_MODEL=text-embedding-3-small
   LLM_MODEL=gpt-4o
   ```

---

## Usage

### 1. Start the API Server

The project exposes a FastAPI backend containing routes for both visibility actions and research questions.

```bash
uvicorn main:app --reload
```
Once running, you can interact with the API via the interactive Swagger UI at:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

### 2. Testing the Workflow Locally

To test the LangGraph state machine natively without hitting the HTTP endpoints, run the CLI test script.

```bash
python test_run.py
```

### 3. Uploading PDFs into the RAG Knowledge Base

You can add your PDF (or `.txt`) documents to the RAG index in two ways:

**Option A – Via API (recommended)**  
With the server running (`uvicorn main:app --reload`), use the upload endpoint:

- **Endpoint:** `POST /api/rag/upload`
- **Body:** `multipart/form-data` with a single file field named `file`
- **Allowed types:** `.pdf`, `.txt`

Example with **curl**:
```bash
curl -X POST "http://localhost:8000/api/rag/upload" \
  -H "accept: application/json" \
  -F "file=@/path/to/your/document.pdf"
```

You can also use the **Swagger UI** at [http://localhost:8000/docs](http://localhost:8000/docs): open `POST /api/rag/upload`, click “Try it out”, choose your file, and execute.

**Option B – From Python (script or REPL)**  
Use the ingestion helper with a local file path:

```python
from rag.document_ingestion import ingest_document

# Add a research paper or text file to the vector store
ingest_document("path/to/paper.pdf")
# or
ingest_document("path/to/notes.txt")
```

Documents are chunked, embedded with your configured embedding model, and stored in the Qdrant collection `psychology_knowledge`. After uploading, query them via `POST /api/research/query`.

---

## Project Structure

```
PsychologistAI/
├── agents/                  # Multi-agent system (Lead, Message, Conversation, Scheduler, Manager)
├── api/                     # FastAPI route definitions
├── config/                  # Global settings and environment loading
├── data/                    # Synthetic datasets (mock social media posts)
├── rag/                     # RAG pipeline logic (Chunking, Embeddings, Vector Store, Retrievers)
├── tools/                   # Shared utility tools and data loaders
├── workflows/               # LangGraph state configurations and Chains
├── main.py                  # API server entrypoint
├── requirements.txt         # Pip dependencies
├── test_run.py              # CLI testing script
└── .env                     # Local environment keys
```

## Note on Safe Use
This application is a demonstration. AI must *not* provide formal medical diagnoses. The system's prompts have been designed strictly to offer supportive, empathetic dialogue, and refer out to a human clinician.

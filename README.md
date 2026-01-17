# Excel RAG System

A complete end-to-end Retrieval-Augmented Generation (RAG) system for querying Excel data using natural language. Built with ChromaDB for vector storage, Streamlit for the UI, and Google Gemini as the LLM.

## Features

- **Excel-Only Storage**: No SQL database needed - all data stored in Excel files
- **Multi-Sheet Support**: Upload Excel files with multiple sheets, automatically merged into a canonical dataset
- **Incremental Vector Updates**: Only embeds new/changed rows, no full re-embedding required
- **Column Normalization**: Automatic typo fixes and standardization of column names
- **Deduplication**: Stable row IDs using SHA256 for reliable deduplication
- **Natural Language Queries**: Ask questions about your data in plain English
- **Source Attribution**: See which rows were used to generate answers

## Project Structure

```
Temp2/
├── app.py              # Streamlit UI application
├── config.py           # Configuration settings
├── excel_handler.py    # Excel file operations
├── gemini_client.py    # Google Gemini API wrapper
├── vector_store.py     # ChromaDB operations
├── rag_pipeline.py     # RAG orchestration
├── requirements.txt    # Dependencies
├── data/               # Excel data storage (created automatically)
└── chroma_db/          # ChromaDB persistence (created automatically)
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

Set your Google Gemini API key as an environment variable:

```bash
# Windows
set GEMINI_API_KEY=your-api-key-here

# Linux/Mac
export GEMINI_API_KEY=your-api-key-here
```

Or enter it directly in the Streamlit sidebar.

### 3. Run the Application

```bash
streamlit run app.py
```

## Usage

### Uploading Data

1. Go to the **Upload** tab
2. Upload an Excel file (.xlsx)
3. Preview the merged data from all sheets
4. Click **Save/Update Excel Store** to persist the data

### Updating the Vector Index

- **Incremental Update**: Only adds new rows (recommended for most updates)
- **Rebuild Index**: Completely rebuilds the index from scratch

### Querying Data

1. Go to the **Query** tab
2. Enter your question in natural language
3. Click **Search** to get an answer
4. Optionally view the retrieved source rows

### Manual Entry

1. Go to the **Manual Entry** tab
2. Fill in the fields for your new row
3. Click **Add Row** to save to Excel and embed

## Configuration

Edit `config.py` to customize:

- `GEMINI_CHAT_MODEL`: Chat model (default: gemini-2.0-flash)
- `GEMINI_EMBEDDING_MODEL`: Embedding model (default: text-embedding-004)
- `TOP_K_RESULTS`: Number of results to retrieve (default: 5)
- `COLUMN_TYPO_FIXES`: Mapping of typos to correct column names

## How It Works

### Data Flow

1. **Upload**: Excel file → Parse sheets → Normalize columns → Merge into canonical DataFrame
2. **Store**: Add row IDs (SHA256) → Deduplicate → Save to `data_store.xlsx`
3. **Index**: Convert rows to text → Check for existing embeddings → Only embed new rows → Store in ChromaDB
4. **Query**: Embed question → Retrieve top-k similar rows → Build context → Ask Gemini → Return answer

### Incremental Embedding

The system uses a stable `row_id` computed from:
- SHA256 hash of normalized row JSON
- Publication type (if present)

Before embedding, the system checks which row IDs already exist in ChromaDB and only embeds missing ones. This makes updates fast and cost-effective.

### Column Normalization

Columns are automatically normalized:
- Stripped of whitespace
- Converted to lowercase
- Spaces replaced with underscores
- Known typos corrected (e.g., `title_of_publicaion` → `title`)

## API Reference

### `gemini_client.py`

```python
embed_text(text: str) -> List[float]
embed_query(query: str) -> List[float]
answer_with_gemini(query: str, context: str) -> str
```

### `excel_handler.py`

```python
process_uploaded_file(file_buffer) -> Tuple[Dict, DataFrame]
save_to_excel_store(df: DataFrame) -> bool
append_row_to_store(row_data: Dict) -> Tuple[bool, str, DataFrame]
load_existing_store() -> Optional[DataFrame]
```

### `vector_store.py`

```python
add_documents(documents: List[Dict]) -> Tuple[int, int]  # (added, skipped)
query(query_text: str, top_k: int) -> List[Dict]
rebuild_collection(documents: List[Dict]) -> int
```

### `rag_pipeline.py`

```python
query(question: str, top_k: int) -> Dict  # {answer, sources, context}
update_index_incremental() -> Tuple[int, int]
rebuild_index() -> int
```

## License

MIT

[English](README_EN.md) | [中文](README.md)

# Local Vector Knowledge Base

[![macOS](https://img.shields.io/badge/platform-macOS-orange)](https://www.apple.com/macos)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A macOS-based local vector knowledge base management tool with support for multi-format file import, vector storage, and semantic search.

[Quick Start](#quick-start) • [Features](#features) • [API Reference](#api-reference) • [Configuration](#configuration)

---

## Quick Start

**Step 1: Install Dependencies**

```bash
# Clone the project
git clone https://github.com/your-repo/WuBuQiLu_Local-Vector-Knowledge.git
cd WuBuQiLu_Local-Vector-Knowledge

# Install Python dependencies
pip3 install -r requirements.txt
```

**Step 2: Start Service**

```bash
python3 web_ui.py
```

**Step 3: Access**

Open browser and visit http://localhost:8000

---

## Features

### Multi-format Support

Supports importing various file formats with automatic text extraction and vectorization:

| Format | Support | Description |
|--------|---------|-------------|
| 📄 PDF | ✅ | Streaming parse, low memory usage |
| 📝 Markdown | ✅ | Convert to plain text |
| 📊 Excel | ✅ | Merge text by worksheet |
| 🖼️ Images | ✅ | OCR text recognition |
| 📃 TXT | ✅ | Split by line |
| 📄 Word | ✅ | Paragraph + table parsing |
| 📽️ PPT | ✅ | Split by slide |

### Smart Search

- **Hybrid Search**: Vector similarity + keyword matching, combined scoring
- **Keyword Highlighting**: Highlight matched keywords in results
- **Type Filtering**: Filter results by file type

### MacOS Optimization

- **MPS Acceleration**: Apple Silicon devices automatically use GPU
- **Quantized Models**: 8bit quantization reduces memory by 50%
- **Incremental Update**: Auto-detect file changes, only update modified files

### Web Management Interface

- 📤 File upload & indexing
- 🔍 Semantic search
- 📁 File management
- 📊 Statistics charts
- 🌙 Dark mode

---

## Project Structure

```
local_vector_kb/
├── setup_macos.sh          # macOS one-click installation script
├── config.yaml             # Configuration file
├── LocalVectorKB.py        # Core knowledge base class
├── web_ui.py               # Web service entry point
├── templates/
│   └── index.html          # Web frontend
├── static/                 # Static files directory
├── temp_uploads/          # Temporary files directory
├── requirements.txt       # Python dependencies
├── README.md              # Chinese documentation
└── README_EN.md           # English documentation
```

---

## Configuration

### Config File (config.yaml)

```yaml
# Knowledge base configuration
kb:
  path: "./local_vector_db"        # Vector database storage path
  collection_name: "multi_file_kb" # Collection name
  chunk_size: 500                  # Chunk size
  incremental_update: true         # Enable incremental update

# Model configuration
model:
  embed_model: "BAAI/bge-m3"       # Embedding model
  device: "auto"                   # auto=auto-detect MPS/CPU
  quantization: true               # Enable quantization

# ChromaDB configuration
chroma:
  database_impl: "leveldb"         # LevelDB backend

# Web service configuration
web:
  host: "0.0.0.0"
  port: 8000
  api_key: "your-secret-key"       # API authentication key
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KB_API_KEY` | API authentication key | `local_kb_2026_secure_key` |

---

## API Reference

All APIs require `X-API-Key` in request header.

### API List

| Method | Path | Description | Rate Limit |
|--------|------|-------------|------------|
| GET | / | Management interface | - |
| POST | /api/v1/upload_async | Async file upload | 60/min |
| GET | /api/v1/task/{task_id} | Get task status | 120/min |
| POST | /api/v1/search | Semantic search | 60/min |
| POST | /api/v1/delete | Delete file | 30/min |
| POST | /api/v1/batch_delete | Batch delete | 10/min |
| POST | /api/v1/clear | Clear knowledge base | 5/min |
| GET | /api/v1/stats | Get statistics | 30/min |
| POST | /api/v1/batch_upload | Batch upload | 10/min |
| POST | /api/v1/batch_search | Batch search | 30/min |

### Examples

```bash
# Search
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: local_kb_2026_secure_key" \
  -F "query=your query" \
  -F "top_k=5"

# Get statistics
curl http://localhost:8000/api/v1/stats \
  -H "X-API-Key: local_kb_2026_secure_key"

# Upload file
curl -X POST http://localhost:8000/api/v1/upload_async \
  -H "X-API-Key: local_kb_2026_secure_key" \
  -F "file=@/path/to/file.pdf"
```

---

## Tech Stack

- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (LevelDB backend)
- **Embedding Model**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- **Web Framework**: [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn
- **Frontend**: HTML5 + ECharts
- **OCR**: Tesseract

---

## Performance Optimization

| Optimization | Description |
|--------------|-------------|
| MPS Acceleration | Apple Silicon devices auto-use GPU |
| Quantized Models | 8bit quantization reduces memory |
| Batch Processing | Support batch upload and search |
| Incremental Update | Only process modified files |
| LevelDB | 10x faster than SQLite |

---

## Notes

1. First launch will download embedding model, which takes time
2. Large file indexing takes longer, recommend using async upload
3. Please keep your API Key secure
4. Clearing knowledge base is irreversible, please use with caution

---

## FAQ

**Q: Getting "No module named 'xxx'" error?**
> A: Run `pip3 install -r requirements.txt` to install all dependencies

**Q: How to change API key?**
> A: Edit `web.api_key` in `config.yaml`

**Q: Does it support Windows/Linux?**
> A: Current version is optimized for macOS, other systems need to adjust Tesseract path

**Q: How to view logs?**
> A: Logs are saved to `kb_logs.log` by default

---

## License

MIT License

---

<div align="center">

**If you like this project, feel free to ⭐ Star**

</div>

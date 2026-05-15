# AI Roadmap & Project Research Copilot

A Streamlit AI/ML research copilot that searches current web sources, indexes retrieved content in a vector database, and generates citation-backed answers for roadmaps, project ideas, and learning plans.

## Features

- Source-backed research answers with citations.
- Roadmap validation, project ideas, and learning plan modes.
- Lightweight vector retrieval with semantic embeddings.
- Source cards with relevance and source type.
- Evaluation dashboard with retrieval diagnostics, source mix, citation coverage, evidence accuracy, and an evidence matrix.
- Markdown report export.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your local API keys:

```text
EXA_API_KEY=...
GOOGLE_API_KEY=...
```

## Run

```powershell
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io).
3. Create a new app from this repository.
4. Set the main file path to `app.py`.
5. Add these secrets in the app settings:

```toml
EXA_API_KEY = "your_key"
GOOGLE_API_KEY = "your_key"
```

Do not commit real API keys to GitHub.

## Test

```powershell
pytest -q
```

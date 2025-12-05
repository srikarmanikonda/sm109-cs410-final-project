# Clinical Trials Search Engine

A search engine for clinical trials that uses BM25 text retrieval with query-aware boosting for structured fields like Phase, Status, and Location.

---

## Table of Contents

1. [Overview](#overview)
2. [How to Use](#how-to-use)
   - [Installation](#installation)
   - [Fetching Data](#fetching-data)
   - [Running the Web App](#running-the-web-app)
   - [Running Evaluation](#running-evaluation)
3. [Implementation Details](#implementation-details)
   - [Architecture](#architecture)
   - [Search Algorithm](#search-algorithm)
   - [Evaluation Methodology](#evaluation-methodology)

---

## Overview

This project implements a clinical trials search engine that allows users to search for trials using natural language queries. It fetches data from the [ClinicalTrials.gov API](https://clinicaltrials.gov/), indexes it using BM25, and provides a Streamlit web app for searching.


## How to Use

### Installation

1. Clone or download this project.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Dependencies include:
   - `pandas` - Data manipulation
   - `rank_bm25` - BM25 retrieval algorithm
   - `streamlit` - Web interface
   - `numpy` - Numerical operations
   - `requests` - API calls

### Fetching Data

The project includes sample data in `data/sample_trials.csv`. To fetch fresh data from ClinicalTrials.gov:

```bash
cd src
python fetch_data.py
```

This fetches 100 recent clinical trials and saves them to `data/sample_trials.csv`.

### Running the Web App

Launch the Streamlit app:

```bash
cd src
streamlit run app.py
```

Open your browser to the displayed URL (typically `http://localhost:8501`).



### Running Evaluation

To evaluate the search system:

1. Generate the judgment template (if not already present):
   ```bash
   cd src
   python evaluate.py
   ```

2. Fill in the `relevance` column in `data/evaluation_judgments.csv` with:
   - `1` for relevant results
   - `0` for non-relevant results

3. Run evaluation again to compute metrics:
   ```bash
   python -c "from evaluate import compute_metrics; compute_metrics()"
   ```

---

## Project Implementation Details

### Architecture

The system consists of four main components:

| File | Purpose |
|------|---------|
| `fetch_data.py` | Fetches clinical trial data from the ClinicalTrials.gov API |
| `search_engine.py` | Core search logic with BM25 and boosting |
| `app.py` | Streamlit web interface |
| `evaluate.py` | Evaluation framework for measuring retrieval quality |

**Data Flow:**
```
ClinicalTrials.gov API → fetch_data.py → sample_trials.csv → search_engine.py → app.py
```

### Search Algorithm

The search engine combines two scoring mechanisms:

#### 1. BM25 Text Retrieval

Documents are indexed using BM25Okapi on a concatenation of:
- `BriefTitle`
- `BriefSummary`
- `Condition`

Preprocessing involves lowercasing and removing non-alphanumeric characters.

#### 2. Query-Aware Boosting

The system parses the query to detect  filters and applies boosts:

| Filter | Boost Value | Example Query |
|--------|-------------|---------------|
| Phase | +5.0 | "phase 2", "phase II" |
| Status | +5.0 | "recruiting", "completed" |
| Location | +3.0 | "New York", "California" |

Phase matching handles both numeric (1, 2, 3, 4) and Roman numeral (I, II, III, IV) formats.

#### Final Scoring

```
final_score = bm25_score + boost_score
```

Results are sorted by `final_score` in descending order.

### Evaluation Methodology

The evaluation compares two configurations:
- **Baseline**: BM25 only (no boosting)
- **System**: BM25 + boosting

**Metrics:**
- **Precision@5 (P@5)**: Fraction of relevant documents in the top 5 results
- **nDCG@5**: Normalized Discounted Cumulative Gain at rank 5, which accounts for the position of relevant documents

**Test Queries:**
The evaluation uses 10 predefined queries covering various conditions (lung cancer, diabetes, asthma, etc.) and filters (phase, recruiting status).

---

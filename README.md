# Book Recommendation System

An end-to-end collaborative filtering book recommendation system built with Python, scikit-learn, and Streamlit. The system ingests the Book-Crossing dataset, preprocesses and validates it, trains a K-Nearest Neighbors model, and serves recommendations through an interactive web interface.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Pipeline Stages](#pipeline-stages)
- [Model](#model)
- [Web Application](#web-application)
- [Installation](#installation)
- [Usage](#usage)
- [Docker](#docker)
- [Configuration](#configuration)
- [Logging and Exception Handling](#logging-and-exception-handling)

---

## Overview

The system recommends books based on collaborative filtering. It identifies users with similar reading and rating patterns, then surfaces books that users with overlapping tastes have rated highly. Given a book title, the model returns the 5 most similar books along with their cover images.

The pipeline is separated into discrete stages — ingestion, validation, transformation, and training — each of which can be triggered independently or run in sequence through the training pipeline.

---

## Architecture

```
Raw Dataset (ZIP)
      |
      v
Stage 0: Data Ingestion       -- Downloads and extracts the dataset
      |
      v
Stage 1: Data Validation      -- Cleans, filters, and merges ratings with book metadata
      |
      v
Stage 2: Data Transformation  -- Builds a user-book pivot table; serializes objects
      |
      v
Stage 3: Model Trainer        -- Fits KNN on sparse pivot matrix; saves model
      |
      v
Streamlit Web App             -- Loads serialized objects; serves recommendations
```

---

## Project Structure

```
recommendation_system/
├── app.py                          # Streamlit web application entry point
├── main.py                         # Script to run the training pipeline directly
├── setup.py                        # Package setup
├── pyproject.toml
├── requirements.txt
├── Dockerfile
│
├── config/
│   └── config.yaml                 # All pipeline configuration (paths, URLs, filenames)
│
├── book_recommender/
│   ├── components/
│   │   ├── stage_00_data_ingestion.py
│   │   ├── stage_01_data_validation.py
│   │   ├── stage_02_data_transformation.py
│   │   └── stage_03_model_trainer.py
│   ├── config/
│   │   └── configuration.py        # Reads config.yaml; returns typed config objects
│   ├── constant/
│   │   └── __init__.py             # CONFIG_FILE_PATH constant
│   ├── entity/
│   │   └── config_entity.py        # Named tuples for each stage's config
│   ├── exception/
│   │   └── exception_handler.py    # Custom AppException with file/line diagnostics
│   ├── logger/
│   │   └── log.py                  # Timestamped log file written to logs/
│   ├── pipeline/
│   │   └── training_pipeline.py    # Orchestrates all four stages in sequence
│   └── utils/
│       └── util.py                 # YAML file reader
│
├── artifacts/                      # Auto-generated; all stage outputs written here
│   ├── dataset/
│   │   ├── raw_data/               # Downloaded ZIP file
│   │   ├── ingested_data/          # Extracted CSVs
│   │   ├── clean_data/             # clean_data.csv from validation stage
│   │   └── transformed_data/       # transformed_data.pkl (pivot table)
│   ├── serialized_objects/         # book_names.pkl, book_pivot.pkl, final_rating.pkl
│   └── trained_model/              # model.pkl
│
├── logs/                           # Timestamped log files
└── notebook/                       # Exploratory notebooks and raw CSVs
    ├── BX-Books.csv
    ├── BX-Book-Ratings.csv
    └── BX-Users.csv
```

---

## Dataset

The project uses the **Book-Crossing** dataset, downloaded automatically during the ingestion stage from the configured URL.

| File | Description |
|---|---|
| `BX-Books.csv` | Book metadata: ISBN, title, author, year, publisher, image URL |
| `BX-Book-Ratings.csv` | User ratings: user ID, ISBN, rating (0-10) |
| `BX-Users.csv` | User demographics (not used by the model) |

Separator: semicolon (`;`). Encoding: latin-1.

---

## Pipeline Stages

### Stage 0 — Data Ingestion (`stage_00_data_ingestion.py`)

- Downloads the dataset ZIP from the URL in `config.yaml` using `urllib.request.urlretrieve`.
- Saves the ZIP to `artifacts/dataset/raw_data/`.
- Extracts the contents to `artifacts/dataset/ingested_data/`.

### Stage 1 — Data Validation (`stage_01_data_validation.py`)

Preprocessing steps applied to produce a clean dataset:

1. Reads `BX-Books.csv` and `BX-Book-Ratings.csv`.
2. Retains only the relevant book columns: `ISBN`, title, author, year, publisher, `Image-URL-L`.
3. Renames columns to snake_case (`title`, `author`, `year`, `publisher`, `image_url`, `user_id`, `rating`).
4. Filters to users who have rated more than 200 books.
5. Merges ratings with book metadata on `ISBN`.
6. Filters to books that have received at least 50 ratings.
7. Drops duplicate `(user_id, title)` pairs.
8. Saves the result as `clean_data.csv` and as `final_rating.pkl` (serialized for the web app).

### Stage 2 — Data Transformation (`stage_02_data_transformation.py`)

1. Reads `clean_data.csv`.
2. Builds a pivot table with `title` as the index, `user_id` as columns, and `rating` as values.
3. Fills missing entries with `0`.
4. Serializes the pivot table as `transformed_data.pkl`.
5. Serializes the book title index as `book_names.pkl` and the pivot table as `book_pivot.pkl` for use in the web app.

### Stage 3 — Model Trainer (`stage_03_model_trainer.py`)

1. Loads `transformed_data.pkl`.
2. Converts the pivot table to a sparse CSR matrix using `scipy.sparse.csr_matrix`.
3. Fits a `sklearn.neighbors.NearestNeighbors` model with the brute-force algorithm.
4. Serializes the trained model to `artifacts/trained_model/model.pkl`.

---

## Model

**Algorithm:** K-Nearest Neighbors (brute-force distance search)

The model operates on a user-book interaction matrix where each row is a book and each column is a user. Ratings are the cell values; missing ratings are treated as 0. At inference time, the model computes the 5 nearest neighbors to a query book vector using Euclidean distance and returns those books as recommendations.

**Why brute force:** The dataset size after filtering is small enough that brute-force search is fast and avoids approximation errors introduced by tree-based methods.

---

## Web Application

The Streamlit interface (`app.py`) exposes two actions:

- **Train Recommender System** — triggers the full training pipeline from within the UI.
- **Show Recommendation** — accepts a book title from a dropdown (populated from `book_names.pkl`) and displays the 5 recommended books with their cover images.

The `Recommendation` class loads the three serialized objects (`book_names.pkl`, `book_pivot.pkl`, `final_rating.pkl`) and the trained model (`model.pkl`) at startup. Cover images are fetched from the `image_url` field stored in `final_rating.pkl`.

---

## Installation

**Prerequisites:** Python 3.11+

```bash
# Clone the repository
git clone https://github.com/SujeethHR/recommendation_system.git
cd recommendation_system

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

The package itself is installed in editable mode via `-e .` in `requirements.txt`.

---

## Usage

### Run the training pipeline

```bash
python main.py
```

This downloads the dataset, runs all four pipeline stages, and writes all artifacts to the `artifacts/` directory.

### Launch the web application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in a browser. If the model has not been trained yet, click **Train Recommender System** first. Once training completes, select a book from the dropdown and click **Show Recommendation**.

---

## Docker

The included `Dockerfile` builds a self-contained image that serves the Streamlit app on port 8501.

```bash
# Build the image
docker build -t book-recommender .

# Run the container
docker run -p 8501:8501 book-recommender
```

The application will be accessible at `http://localhost:8501`.

Base image: `python:3.11-slim`. System dependencies installed: `build-essential`, `software-properties-common`, `git`.

---

## Configuration

All paths and URLs are defined in `config/config.yaml`. No hardcoded paths exist in the source code.

```yaml
artifacts_config:
  artifacts_dir: artifacts

data_ingestion_config:
  dataset_download_url: "<url to dataset ZIP>"
  dataset_dir: dataset
  ingested_dir: ingested_data
  raw_data_dir: raw_data

data_validation_config:
  clean_data_dir: clean_data
  serialized_objects_dir: serialized_objects
  books_csv_file: BX-Books.csv
  ratings_csv_file: BX-Book-Ratings.csv

data_transformation_config:
  transformed_data_dir: transformed_data

model_trainer_config:
  trained_model_dir: trained_model
  trained_model_name: model.pkl
```

`AppConfiguration` reads this file at startup and constructs typed `namedtuple` config objects for each stage. The path to `config.yaml` is resolved relative to the working directory via the `CONFIG_FILE_PATH` constant in `book_recommender/constant/__init__.py`.

---

## Logging and Exception Handling

**Logging** (`book_recommender/logger/log.py`)

- Log files are written to `logs/` with the filename `log_YYYY-MM-DD-HH-MM-SS.log`.
- A new log file is created each time the application starts.
- Format: `[timestamp] logger_name - LEVEL - message`

**Exception Handling** (`book_recommender/exception/exception_handler.py`)

- `AppException` wraps any exception and enriches the message with the source file name and line number extracted from the traceback.
- All pipeline stages and web app methods catch bare exceptions and re-raise as `AppException`, ensuring consistent error reporting throughout the system.

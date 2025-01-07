# News Retriever

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/unwale/news-retrieval.git
   cd news-retrieval
   ```

2. **Set up the Python environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set environment variables**:
   Create a `.env` file by copying the example configuration:
   ```bash
   cp .env.example .env
   ```

---

## Usage

### Data Preparation

1. **Scrape Data**:
   To fetch raw data, use:
   ```bash
   python -m scripts/scraping/scrape.py
   ```
   This will save the data to `data/raw/data.jsonl`.

2. **Preprocess Data**:
   Use the following scripts for cleaning and preparing your data:
   - Fill missing values:
     ```bash
     python -m scripts/preprocessing/fillna.py
     ```
   - Split data into training and testing sets:
     ```bash
     python -m scripts/preprocessing/split_train_test.py
     ```
   - Clean and lemmatize:
     ```bash
     python -m spacy download ru_core_news_sm
     python -m scripts/preprocessing/clean_lemmatize.py \
       --input data/raw/train.jsonl \
       --output data/processed/train.jsonl
     python -m scripts/preprocessing/clean_lemmatize.py \
       --input data/raw/test.jsonl \
       --output data/processed/test.jsonl
     ```

### Training Models

To train all classification models, run:
```bash
. scripts/training/classification/train_all.sh
```
This will save the trained models in the `model/saved/` directory.

### Evaluation

1. **Classification Evaluation**:
   ```bash
   . scripts/evaluation/classification/eval_all.sh
   ```

2. **Topic Matching Evaluation**:
   ```bash
   . scripts/evaluation/topic_matching/eval_all.sh
   ```

### Metrics and Results

- All evaluation results will be saved in the `results/` directory as `.csv` files.
- Key metrics include:
  - News classification metrics: `classification_metrics.csv`
  - Topic matching metrics: `matching_metrics.csv`
---

## Running Shell Scripts

To ensure scripts execute correctly, run them from the root of the project. For example:
```bash
. scripts/training/classification/train_all.sh
```

---

## License

This project is licensed under the [MIT License](LICENSE).

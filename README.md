# portuguese-fake-news-detection

Binary classification of Portuguese news (real vs fake) using Fake.br-Corpus.
This repository follows a notebook-first workflow focused on [v4.ipynb](v4.ipynb).

## Project Goal

Build and compare two approaches for Portuguese fake news detection:
- traditional NLP/ML pipeline;
- Transformer-based models.

Target labels:
- `label = 0`: real news
- `label = 1`: fake news

## Repository Structure

- [v4.ipynb](v4.ipynb): main notebook with setup, corpus consolidation, EDA, preprocessing, and analysis.
- [AGENTS.md](AGENTS.md): workflow and project operating rules.

## Dataset

- Primary corpus: Fake.br-Corpus
- Expected Google Drive location in Colab:
  - `/content/drive/MyDrive/Fake.br-Corpus/`

## Execution Environment

- Google Colab
- Python 3.x
- Key libraries used in the notebook include:
  - pandas
  - seaborn
  - matplotlib
  - nltk
  - spacy (`pt_core_news_sm`)

## How To Reproduce

1. Open [v4.ipynb](v4.ipynb) in Google Colab.
2. Mount Google Drive when prompted.
3. Confirm corpus path exists.
4. Run notebook cells in order (Setup -> Corpus -> EDA -> Preprocessing).

## Current Phase

The project currently has:
- environment setup and dependency loading;
- corpus consolidation with metadata and labels;
- preprocessing and exploratory diagnostics.

Next phase:
- train/evaluate baseline ML models (for example TF-IDF + linear classifier);
- compare against Transformer models with consistent metrics.

## Suggested Metrics

- F1-score
- Precision
- Recall
- Confusion matrix

## License

Define a license before external distribution.
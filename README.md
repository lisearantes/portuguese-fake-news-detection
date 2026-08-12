# classificador-fake-br

Análise e classificação binária de notícias falsas em português brasileiro usando o Fake.br-Corpus, comparando pipelines tradicionais de NLP/ML (TF-IDF + SVM, Regressão Logística) e modelos baseados em Transformers.

Este repositório segue um fluxo de trabalho centrado no notebook [main.ipynb](main.ipynb).

## Objetivo

Construir e comparar duas abordagens para detecção de fake news em português:
- pipeline tradicional de NLP/ML (TF-IDF + classificadores lineares);
- modelos baseados em Transformers.

Rótulos do corpus:
- `label = 0`: notícia verdadeira
- `label = 1`: notícia falsa

## Estrutura do Repositório

- [main.ipynb](main.ipynb): notebook principal com setup, carregamento do corpus, EDA, pré-processamento, vetorização TF-IDF e classificação.
- [AGENTS.md](AGENTS.md): regras de fluxo de trabalho e convenções do projeto.

## Dataset

- Corpus: [Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus)
- Caminho esperado no Google Drive (Colab):
  - `/content/drive/MyDrive/Fake.br-Corpus/`

## Ambiente de Execução

- Google Colab
- Python 3.x
- Principais bibliotecas:
  - `pandas`, `numpy`
  - `seaborn`, `matplotlib`
  - `nltk`, `spacy` (`pt_core_news_sm`)
  - `scikit-learn` (TF-IDF, LinearSVC, LogisticRegression)

## Como Reproduzir

1. Abra [main.ipynb](main.ipynb) no Google Colab.
2. Monte o Google Drive quando solicitado.
3. Confirme que o caminho do corpus existe.
4. Execute as células em ordem: Setup → Corpus → EDA → Pré-processamento → Modelagem.

## Fase Atual

O projeto conta com:
- setup do ambiente e carregamento de dependências;
- consolidação do corpus com metadados e rótulos;
- análise exploratória (EDA) com visualizações;
- pré-processamento de texto (limpeza, remoção de stopwords, lematização);
- vetorização TF-IDF com unigramas e bigramas;
- classificação binária com **Linear SVC** e **Regressão Logística**.

Próxima fase:
- comparação com modelos baseados em Transformers (ex.: BERTimbau).

## Métricas Avaliadas

- Acurácia
- F1-Score (ponderado)
- Precisão
- Recall
- Confusion matrix

# classificador-fake-br

Classificação binária de notícias (real vs. fake) em português brasileiro, com o [Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus). O projeto compara duas abordagens no mesmo conjunto de treino/teste:

- **Tradicional:** TF-IDF + Linear SVC e Regressão Logística, sobre texto limpo (`clean_text`)
- **Transformer:** fine-tuning do BERTimbau (`neuralmind/bert-base-portuguese-cased`), sobre o texto original (`texto_completo`)

O fluxo é notebook-first no **Google Colab**, com lógica reutilizável em `src/` e artefatos persistidos no Google Drive.

## Rótulos

| `label` | Classe              |
|---------|---------------------|
| `0`     | Notícia verdadeira  |
| `1`     | Notícia falsa       |

## Pipeline (ordem obrigatória)

Execute os notebooks **nesta ordem**. Cada etapa depende dos arquivos gravados pela anterior.

```text
1. 1_preprocessamento.ipynb
        │
        │  gera clean_text, divide o corpus e grava splits/
        ▼
2. 2_treinamento.ipynb   (GPU recomendada)
        │
        │  treina clássicos + BERTimbau e grava modelos/
        ▼
3. 3_avaliacao.ipynb
           avalia no mesmo teste e grava resultados/
```

| Ordem | Notebook | Função |
|-------|----------|--------|
| 1 | [1_preprocessamento.ipynb](1_preprocessamento.ipynb) | Carrega o corpus, cria `clean_text`, faz split estratificado e salva treino/teste |
| 2 | [2_treinamento.ipynb](2_treinamento.ipynb) | Treina TF-IDF + SVC/LR e faz fine-tuning do BERTimbau |
| 3 | [3_avaliacao.ipynb](3_avaliacao.ipynb) | Carrega splits e modelos, avalia qualidade e custo, gera comparação |

Não pule etapas: o treinamento não refaz o split, e a avaliação não treina de novo — ambos leem o que já está no Drive.

## Como rodar no Google Colab

### Pré-requisitos

1. Conta Google com o corpus em  
   `/content/drive/MyDrive/Fake.br-Corpus/`  
   (estrutura original do Fake.br-Corpus: pastas de textos reais/falsos e metadados).
2. Este repositório disponível no ambiente Colab (clone, upload ou atalho no Drive), **incluindo a pasta `src/`** junto dos notebooks.
3. Dependências do [requirements.txt](requirements.txt) (os notebooks instalam o necessário nas células de setup).

### Passo a passo

1. **Pré-processamento** — abra `1_preprocessamento.ipynb`
   - Runtime: CPU basta
   - Monte o Drive e execute todas as células
   - Saídas esperadas no Drive:
     - `Fake.br-Corpus/splits/treino.csv`
     - `Fake.br-Corpus/splits/teste.csv`
     - `Fake.br-Corpus/fake.br-preprocessed.csv`

2. **Treinamento** — abra `2_treinamento.ipynb`
   - Runtime: **GPU** (obrigatório/recomendado para o BERTimbau)
   - Monte o Drive e execute todas as células
   - Saídas esperadas:
     - `Fake.br-Corpus/modelos/` — `vectorizer_tfidf.joblib`, `modelo_svc.joblib`, `modelo_lr.joblib`
     - `Fake.br-Corpus/bertimbau_fake_br/modelo_final/`
     - `Fake.br-Corpus/resultados/custo_treino.csv`

3. **Avaliação** — abra `3_avaliacao.ipynb`
   - Runtime: GPU recomendada para inferência do BERTimbau
   - Monte o Drive e execute todas as células
   - Saídas esperadas:
     - `Fake.br-Corpus/resultados/comparacao.csv`
     - matrizes de confusão e tabelas de custo no notebook / pasta `resultados/`

### Observações importantes

- Sessões do Colab são efêmeras: o estado do kernel some ao reiniciar; os artefatos ficam no Drive.
- Sem montar o Drive, caminhos e leituras falham.
- Se mudar a limpeza de texto, rode de novo o pré-processamento e, em seguida, treino e avaliação.
- Modelos tradicionais usam `clean_text`; BERTimbau usa `texto_completo` — a mesma partição serve aos dois.

## Estrutura do repositório

```text
.
├── 1_preprocessamento.ipynb   # etapa 1 — corpus e split
├── 2_treinamento.ipynb        # etapa 2 — treino
├── 3_avaliacao.ipynb                    # etapa 3 — avaliação
├── requirements.txt
├── AGENTS.md                     # convenções para agentes/edição
├── README.md
└── src/
    ├── data.py                   # carga, preprocessamento e splits
    ├── preprocessing.py          # limpeza de texto (PT)
    ├── train_modelos_tradicionais.py
    ├── bert_finetuning.py
    ├── evaluation.py             # métricas e persistência
    ├── recursos.py               # tempo, RAM, VRAM, custos
    ├── visualization.py
    └── pipeline.py               # utilitário de conexão ao Drive
```

## Layout esperado no Google Drive

```text
/content/drive/MyDrive/Fake.br-Corpus/
├── ...                          # corpus original (textos + metadados)
├── fake.br-preprocessed.csv     # corpus completo com clean_text
├── splits/
│   ├── treino.csv               # texto_completo, clean_text, label
│   └── teste.csv
├── modelos/                     # vetorizador + SVC + LR
├── bertimbau_fake_br/
│   └── modelo_final/
└── resultados/
    ├── custo_treino.csv
    └── comparacao.csv
```

## Modelos e métricas

**Modelos**

- TF-IDF (unigramas/bigramas) + Linear SVC  
- TF-IDF + Regressão Logística  
- BERTimbau fine-tuned (`bert-base-portuguese-cased`)

**Métricas de qualidade:** acurácia, precisão, recall, F1 (ponderado), matriz de confusão.

**Custos medidos:** tempo de treino/inferência, uso de RAM e VRAM (quando há GPU), tamanho dos artefatos.

## Dependências principais

Listadas em [requirements.txt](requirements.txt): `pandas`, `numpy`, `scikit-learn`, `nltk`, `spacy`, `transformers`, `torch`, `datasets`, `evaluate`, `matplotlib`, `seaborn`, `joblib`, `psutil`.

No Colab, o modelo spaCy `pt_core_news_sm` e recursos NLTK são baixados nas células de setup dos notebooks.

## Dataset

- Fonte: [Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus)
- Caminho padrão no Colab: `/content/drive/MyDrive/Fake.br-Corpus/`

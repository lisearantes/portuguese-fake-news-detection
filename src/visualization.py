import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def configurar_fonte_arial():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']


def _fonte_eixos():
    return {'family': 'Arial', 'size': 11, 'color': 'black'}


def _formatar_ponto_milhar(x, pos):
    return f"{int(x):,}".replace(',', '.')


def _aplicar_estilo_tcc(ax, fig):
    ax.grid(False)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for eixo in ['bottom', 'left']:
        ax.spines[eixo].set_color('black')
        ax.spines[eixo].set_linewidth(1.5)
        ax.spines[eixo].set_linestyle('-')
    ax.tick_params(axis='both', colors='black', width=1.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(10)
        label.set_color('black')


def plot_preenchimento(df):
    preenchimento = (df.notnull().sum() / len(df)) * 100
    fig, ax = plt.subplots(figsize=(9, len(preenchimento) * 0.4))
    barras = ax.barh(preenchimento.index, preenchimento.values, color='#A9C1D9', height=0.6)
    _aplicar_estilo_tcc(ax, fig)
    ax.set_xlabel('Percentual de preenchimento (%)', fontdict=_fonte_eixos())
    ax.set_ylabel('Colunas do Corpus', fontdict=_fonte_eixos())
    for barra, pct in zip(barras, preenchimento.values):
        ax.text(pct + 1, barra.get_y() + barra.get_height() / 2,
                f"{pct:.1f}%", va='center', ha='left',
                fontsize=10, fontname='Arial', color='black')
    plt.tight_layout()
    plt.show()


def plot_distribuicao_classes(df):
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.countplot(x=df['label'], color='steelblue', ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Verdadeira', 'Falsa'])
    ax.set_yticks([0, 600, 1600, 2600, 3600])
    ax.set_ylim(0, df['label'].value_counts().max() * 1.15)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_formatar_ponto_milhar))
    _aplicar_estilo_tcc(ax, fig)
    ax.set_xlabel('Tipo de Notícia', fontdict=_fonte_eixos())
    ax.set_ylabel('Quantidade', fontdict=_fonte_eixos())
    ax.set_title('')
    plt.tight_layout()
    plt.show()


def plot_distribuicao_categorias(df):
    fig, ax = plt.subplots(figsize=(6, 2))
    sns.countplot(y=df['categoria'], color='steelblue', ax=ax)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels(['Política', 'TV & Celebridades', 'Sociedade & Cotidiano',
                        'Ciência & Tecnologia', 'Economia', 'Religião'])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_formatar_ponto_milhar))
    _aplicar_estilo_tcc(ax, fig)
    ax.set_xlabel('Quantidade de Artigos', fontdict=_fonte_eixos())
    ax.set_ylabel('Categorias', fontdict=_fonte_eixos())
    ax.set_title('')
    plt.tight_layout()
    plt.show()

# TODO: Adicionar docstring

def plot_boxplot_tamanho(df):
    if 'num_palavras_texto' not in df.columns:
        df['num_palavras_texto'] = df['texto_completo'].fillna('').astype(str).str.split().str.len()
    df['num_palavras_limpo'] = df['clean_text'].fillna('').astype(str).str.split().str.len()
    dados = pd.DataFrame({
        'Tamanho': df['num_palavras_texto'].tolist() + df['num_palavras_limpo'].tolist(),
        'Corpus': ['Original'] * len(df) + ['Pré-processado'] * len(df),
    })
    fig, ax = plt.subplots(figsize=(7, 4))
    tons = sns.light_palette("steelblue", n_colors=2, reverse=True)
    sns.boxplot(x='Tamanho', y='Corpus', data=dados, hue='Corpus',
                palette=tons, legend=False, ax=ax)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_formatar_ponto_milhar))
    _aplicar_estilo_tcc(ax, fig)
    ax.set_xlabel('Tamanho da Sentença (número de palavras)', fontdict=_fonte_eixos())
    ax.set_ylabel('Estado do Corpus', fontdict=_fonte_eixos())
    ax.set_title('')
    plt.tight_layout()
    plt.show()


def plot_esparsidade_tfidf(X_train_tfidf):
    amostra = X_train_tfidf[:200, :400].toarray()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.spy(amostra, markersize=2, color='steelblue', aspect='auto')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for eixo in ['bottom', 'left']:
        ax.spines[eixo].set_color('black')
        ax.spines[eixo].set_linewidth(1.5)
    ax.tick_params(axis='both', colors='black', width=1.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(10)
        label.set_color('black')
    ax.set_xlabel('Índice da Feature (Termos do Vocabulário)', fontdict=_fonte_eixos())
    ax.set_ylabel('Índice da Amostra (Documentos)', fontdict=_fonte_eixos())
    ax.set_title('')
    densidade = (X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])) * 100
    print(f"Densidade Global: {densidade:.2f}% | Esparsidade: {100 - densidade:.1f}%")
    fig.tight_layout()
    plt.show()


def plot_heatmap_tfidf(X_train_tfidf, tfidf_terms):
    n_docs = min(15, X_train_tfidf.shape[0])
    sample = X_train_tfidf[:n_docs]
    top_idx = np.argsort(np.asarray(sample.sum(axis=0)).ravel())[::-1][:20]
    heat_data = sample[:, top_idx].toarray()
    heat_terms = [tfidf_terms[i] for i in top_idx]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(heat_data, cmap='Blues', cbar_kws={'label': 'Peso TF-IDF'},
                xticklabels=heat_terms,
                yticklabels=[f'doc_{i}' for i in range(n_docs)], ax=ax)
    ax.set_title('')
    ax.set_xlabel('Termos do Vocabulário', fontdict=_fonte_eixos())
    ax.set_ylabel('Documentos', fontdict=_fonte_eixos())
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(9)
        label.set_color('black')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Peso TF-IDF', fontname='Arial', fontsize=11, color='black')
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(9)
        label.set_color('black')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_impacto_vetorizacao_transformer(
    model,
    tokenizer,
    texts,
    labels,
    output_path=None,
    max_samples=500,
    batch_size=16,
    max_length=512,
):
    """Visualiza a tokenizacao e as representacoes contextuais do Transformer."""
    import torch

    texts = pd.Series(texts).fillna('').astype(str).reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)
    valid = texts.str.strip().ne('') & labels.notna()
    texts = texts[valid]
    labels = labels[valid]

    if len(texts) == 0:
        raise ValueError('Nao ha textos validos para visualizar.')

    if len(texts) > max_samples:
        sample_indices = np.linspace(0, len(texts) - 1, max_samples, dtype=int)
        texts = texts.iloc[sample_indices]
        labels = labels.iloc[sample_indices]

    device = next(model.parameters()).device
    model.eval()
    embeddings = []
    token_lengths = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts.iloc[start:start + batch_size].tolist()
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        attention_mask = encoded['attention_mask']

        with torch.inference_mode():
            outputs = model.base_model(
                **encoded,
                return_dict=True,
            )

        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embeddings.append(pooled.cpu().numpy())
        token_lengths.extend(attention_mask.sum(dim=1).cpu().numpy().tolist())

    embeddings = np.vstack(embeddings)
    token_lengths = np.asarray(token_lengths)
    if len(embeddings) < 2:
        raise ValueError('Sao necessarios pelo menos dois textos validos.')

    projection = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    class_names = np.where(labels.to_numpy().astype(int) == 1, 'Falsa', 'Verdadeira')
    colors = {'Verdadeira': 'steelblue', 'Falsa': '#c44e52'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for class_name in ['Verdadeira', 'Falsa']:
        class_mask = class_names == class_name
        axes[0].scatter(
            projection[class_mask, 0],
            projection[class_mask, 1],
            s=24,
            alpha=0.65,
            color=colors[class_name],
            label=class_name,
        )
        axes[1].hist(
            token_lengths[class_mask],
            bins=30,
            alpha=0.6,
            color=colors[class_name],
            label=class_name,
        )

    _aplicar_estilo_tcc(axes[0], fig)
    _aplicar_estilo_tcc(axes[1], fig)
    axes[0].set_xlabel('Componente principal 1', fontdict=_fonte_eixos())
    axes[0].set_ylabel('Componente principal 2', fontdict=_fonte_eixos())
    axes[0].set_title('Representacoes contextuais (PCA)')
    axes[0].legend(frameon=False)
    axes[1].set_xlabel('Quantidade de tokens', fontdict=_fonte_eixos())
    axes[1].set_ylabel('Quantidade de noticias', fontdict=_fonte_eixos())
    axes[1].set_title('Distribuicao apos a tokenizacao')
    axes[1].legend(frameon=False)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig, {
        'embeddings': embeddings,
        'projection': projection,
        'token_lengths': token_lengths,
        'labels': labels.to_numpy(),
    }


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["#ffffff", "steelblue"])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                cbar=False, ax=ax,
                annot_kws={"fontname": "Arial", "fontsize": 12, "color": "black"})
    ax.set_title('')
    ax.set_xlabel('Classe Predita', fontdict=_fonte_eixos())
    ax.set_ylabel('Classe Real', fontdict=_fonte_eixos())
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(10)
        label.set_color('black')
    for spine in ax.spines.values():
        spine.set_visible(False)
    for eixo in ['bottom', 'left']:
        ax.spines[eixo].set_visible(True)
        ax.spines[eixo].set_color('black')
        ax.spines[eixo].set_linewidth(1.5)
    plt.tight_layout()
    plt.show()

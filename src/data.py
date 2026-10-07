from pathlib import Path

import pandas as pd


COLUNAS_CORPUS = {'texto_completo', 'label'}
COLUNAS_CHAVE_SPLIT = ['id_original', 'label']
COLUNAS_SUBCONJUNTO = {'id_original', 'label', 'texto_completo', 'clean_text'}
ARQUIVO_TREINO = 'treino.csv'
ARQUIVO_TESTE = 'teste.csv'


def carregar_corpus(caminho_csv, colunas_esperadas=None):
    caminho_csv = Path(caminho_csv)
    if not caminho_csv.exists():
        raise FileNotFoundError(f'Arquivo do corpus não encontrado: {caminho_csv}')

    df = pd.read_csv(caminho_csv)
    esperadas = set(colunas_esperadas or COLUNAS_CORPUS)
    faltantes = esperadas - set(df.columns)
    if faltantes:
        raise ValueError(
            f'Colunas esperadas ausentes: {sorted(faltantes)}. '
            f'Colunas encontradas: {list(df.columns)}'
        )
    return df


def preprocessar_corpus(df: pd.DataFrame):
    faltantes = COLUNAS_CORPUS - set(df.columns)
    if faltantes:
        raise ValueError(
            f'Colunas esperadas ausentes para o pré-processamento: {sorted(faltantes)}. '
            f'Colunas encontradas: {list(df.columns)}'
        )

    from .preprocessing import text_cleaning

    df_processado = df.copy()
    df_processado['clean_text'] = (
        df_processado['texto_completo'].fillna('').astype(str).apply(text_cleaning)
    )
    return df_processado


def _chaves_split(df):
    faltantes = set(COLUNAS_CHAVE_SPLIT) - set(df.columns)
    if faltantes:
        raise ValueError(
            f'Colunas esperadas ausentes para a divisão: {sorted(faltantes)}. '
            f'Colunas encontradas: {list(df.columns)}'
        )
    return df['id_original'].astype(str) + '\0' + df['label'].astype(str)


def _validar_chave_unica(df):
    chaves = _chaves_split(df)
    if chaves.duplicated().any():
        raise ValueError(
            'id_original + label não identificam cada notícia de forma única.'
        )
    return chaves


def dividir_corpus(df, test_size=0.2, random_state=42):
    faltantes = COLUNAS_CORPUS - set(df.columns)
    if faltantes:
        raise ValueError(
            f'Colunas esperadas ausentes para a divisão: {sorted(faltantes)}. '
            f'Colunas encontradas: {list(df.columns)}'
        )
    _validar_chave_unica(df)

    from sklearn.model_selection import train_test_split

    X = df['texto_completo']
    y = df['label']
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def salvar_split(diretorio, df, X_train, X_test):
    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)
    faltantes = COLUNAS_SUBCONJUNTO - set(df.columns)
    if faltantes:
        raise ValueError(
            f'Colunas esperadas ausentes para gravar a divisão: {sorted(faltantes)}. '
            'Acrescente clean_text antes de gravar a divisão.'
        )
    _validar_chave_unica(df)

    indices_treino = list(X_train.index)
    indices_teste = list(X_test.index)
    if set(indices_treino) & set(indices_teste):
        raise ValueError('Treino e teste se sobrepõem.')
    if set(indices_treino) | set(indices_teste) != set(df.index):
        raise ValueError('Treino e teste não cobrem o corpus.')

    df.loc[indices_treino].to_csv(diretorio / ARQUIVO_TREINO, index=False)
    df.loc[indices_teste].to_csv(diretorio / ARQUIVO_TESTE, index=False)
    return diretorio / ARQUIVO_TREINO, diretorio / ARQUIVO_TESTE


def _ler_subconjunto(caminho, nome):
    if not caminho.exists():
        raise FileNotFoundError(
            f'Subconjunto de {nome} não encontrado: {caminho}. '
            'Execute main_preprocessamento.ipynb antes desta etapa.'
        )
    parte = pd.read_csv(caminho)
    faltantes = COLUNAS_SUBCONJUNTO - set(parte.columns)
    if faltantes:
        raise ValueError(
            f'Colunas esperadas ausentes em {nome}: {sorted(faltantes)}. '
            'Execute main_preprocessamento.ipynb para gravar os subconjuntos.'
        )
    if _chaves_split(parte).duplicated().any():
        raise ValueError(f'O arquivo de {nome} contém notícias repetidas.')
    return parte


def carregar_split(diretorio):
    diretorio = Path(diretorio)
    treino = _ler_subconjunto(diretorio / ARQUIVO_TREINO, 'treino')
    teste = _ler_subconjunto(diretorio / ARQUIVO_TESTE, 'teste')

    if set(_chaves_split(treino)) & set(_chaves_split(teste)):
        raise ValueError('Treino e teste se sobrepõem.')

    return treino, teste

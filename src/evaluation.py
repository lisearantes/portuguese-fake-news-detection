from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)

from src.recursos import (
    carregar_custos,
    montar_comparacao,
    persistir_comparacao as _persistir_comparacao_csv,
)


def extrair_metricas(y_true, y_pred, nome_modelo):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {
        'Modelo': nome_modelo,
        'Precisão': p,
        'Recall': r,
        'F1-Score': f,
        'Acurácia': accuracy_score(y_true, y_pred),
    }


def tabela_resultados(y_test, predicoes, extras=None, formatar=True):
    """predicoes: lista de tuplas (y_pred, nome_modelo). extras: DataFrame ou lista de dicts com coluna Modelo."""
    df = pd.DataFrame([extrair_metricas(y_test, pred, nome) for pred, nome in predicoes])
    if extras is not None:
        df_extras = pd.DataFrame(extras) if not isinstance(extras, pd.DataFrame) else extras
        df = df.merge(df_extras, on='Modelo', how='left')
    if formatar:
        for col in ['Precisão', 'Recall', 'F1-Score', 'Acurácia']:
            if col in df.columns:
                df[col] = df[col].map(lambda valor: f'{valor:.4f}' if isinstance(valor, float) else valor)
    return df


def persistir_comparacao(diretorio, y_test=None, predicoes=None, df_qualidade=None,
                         custos_inferencia=None, custos_treino=None):
    """Grava comparacao.csv juntando qualidade e custos. Aceita predicoes ou um DataFrame já montado."""
    if df_qualidade is None:
        if y_test is None or predicoes is None:
            raise ValueError('Informe df_qualidade ou (y_test e predicoes).')
        df_qualidade = tabela_resultados(y_test, predicoes, formatar=False)
    custos = carregar_custos(diretorio)
    if custos_inferencia is None:
        custos_inferencia = custos['inferencia']
    if custos_treino is None:
        custos_treino = custos['treino']
    df = montar_comparacao(df_qualidade, custos_inferencia, custos_treino)
    return df, _persistir_comparacao_csv(diretorio, df)


def relatorio_completo(y_test, y_pred, nome_modelo):
    print(f"\n### {nome_modelo}")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")


ARQUIVOS_MODELOS_CLASSICOS = {
    'vectorizer': 'vectorizer_tfidf.joblib',
    'svc': 'modelo_svc.joblib',
    'lr': 'modelo_lr.joblib',
}


def persistir_modelos_classicos(diretorio, vectorizer, modelo_svc, modelo_lr):
    """Grava o vetorizador TF-IDF, o Linear SVC e a regressão logística."""
    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)
    artefatos = {
        'vectorizer': vectorizer,
        'svc': modelo_svc,
        'lr': modelo_lr,
    }
    caminhos = {}
    for chave, objeto in artefatos.items():
        caminho = diretorio / ARQUIVOS_MODELOS_CLASSICOS[chave]
        joblib.dump(objeto, caminho)
        caminhos[chave] = caminho
    return caminhos

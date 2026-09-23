from pathlib import Path

import joblib
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)


def treinar_svc(X_train, y_train, X_test):
    modelo = LinearSVC()
    modelo.fit(X_train, y_train)
    return modelo, modelo.predict(X_test)


def treinar_lr(X_train, y_train, X_test):
    modelo = LogisticRegression(max_iter=1000, solver='liblinear')
    modelo.fit(X_train, y_train)
    return modelo, modelo.predict(X_test)


def extrair_metricas(y_true, y_pred, nome_modelo):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {
        'Modelo': nome_modelo,
        'Precisão': p,
        'Recall': r,
        'F1-Score': f,
        'Acurácia': accuracy_score(y_true, y_pred),
    }


def tabela_resultados(y_test, predicoes):
    """predicoes: lista de tuplas (y_pred, nome_modelo)"""
    df = pd.DataFrame([extrair_metricas(y_test, pred, nome) for pred, nome in predicoes])
    for col in ['Precisão', 'Recall', 'F1-Score', 'Acurácia']:
        df[col] = df[col].map('{:.4f}'.format)
    return df


def relatorio_completo(y_test, y_pred, nome_modelo):
    print(f"\n### {nome_modelo}")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")


ARQUIVOS_MODELOS_CLASSICOS = {
    'vectorizer': 'vectorizer_tfidf.joblib',
    'svc': 'modelo_svc.joblib',
    'lr': 'modelo_lr.joblib',
}
MODELOS_CLASSICOS_DIR = Path(__file__).resolve().parents[1] / 'results' / 'modelos'


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

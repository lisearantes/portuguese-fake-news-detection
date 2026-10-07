from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from src.evaluation import persistir_modelos_classicos
from src.recursos import (
    combinar_custos,
    medir_recursos,
    persistir_custos,
    snapshot_ambiente,
    tamanho_artefato,
)


def treinar_modelos_tradicionais(X_train, X_test, y_train, modelos_dir, resultados_dir=None):
    n_treino = len(X_train)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    with medir_recursos('TF-IDF', 'treino', n_amostras=n_treino) as custo_tfidf:
        X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    with medir_recursos('Linear SVC', 'treino', n_amostras=n_treino) as custo_fit_svc:
        modelo_svc = LinearSVC()
        modelo_svc.fit(X_train_tfidf, y_train)

    with medir_recursos('Logistic Regression', 'treino', n_amostras=n_treino) as custo_fit_lr:
        modelo_lr = LogisticRegression(max_iter=1000, solver='liblinear')
        modelo_lr.fit(X_train_tfidf, y_train)

    pred_svc = modelo_svc.predict(X_test_tfidf)
    pred_lr = modelo_lr.predict(X_test_tfidf)

    caminhos = persistir_modelos_classicos(modelos_dir, vectorizer, modelo_svc, modelo_lr)
    custo_tfidf['tamanho_artefato_mb'] = tamanho_artefato(caminhos['vectorizer'])
    custo_fit_svc['tamanho_artefato_mb'] = tamanho_artefato(caminhos['svc'])
    custo_fit_lr['tamanho_artefato_mb'] = tamanho_artefato(caminhos['lr'])

    linhas_treino = [
        combinar_custos(custo_tfidf, custo_fit_svc, nome='Linear SVC'),
        combinar_custos(custo_tfidf, custo_fit_lr, nome='Logistic Regression'),
    ]

    if resultados_dir is not None:
        persistir_custos(
            resultados_dir,
            ambiente=snapshot_ambiente(),
            linhas_treino=linhas_treino,
        )

    return {
        'vectorizer': vectorizer,
        'modelo_svc': modelo_svc,
        'modelo_lr': modelo_lr,
        'caminhos': caminhos,
        'custos': linhas_treino,
        'pred_svc': pred_svc,
        'pred_lr': pred_lr,
    }


if __name__ == '__main__':
    treinar_modelos_tradicionais()

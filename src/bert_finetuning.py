import inspect
import os

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.recursos import medir_recursos, persistir_custos, snapshot_ambiente, tamanho_artefato

def _validar_gpu():
    # 1) Diagnóstico e validação do ambiente de execução.
    # Como o fine-tuning de BERT exige processamento em GPU, este bloco verifica
    # se o ambiente tem suporte CUDA antes de iniciar o treinamento.
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"cuda built: {torch.backends.cuda.is_built()}")
    print(f"cuda available: {torch.cuda.is_available()}")

    try:
        import subprocess

        smi = subprocess.run(
            ['nvidia-smi', '-L'], capture_output=True, text=True, check=False
        )
        if smi.returncode == 0 and smi.stdout.strip():
            print('nvidia-smi:', smi.stdout.strip())
        else:
            print('nvidia-smi: não disponível neste kernel.')
    except FileNotFoundError:
        print('nvidia-smi: comando não encontrado neste kernel.')

    if not torch.cuda.is_available():
        # Se a GPU não estiver disponível, interrompemos cedo para evitar execução
        # lenta em CPU ou erros posteriores no treinamento do modelo.
        raise RuntimeError(
            'GPU não detectada neste kernel. Se você já ativou T4 no Colab, reinicie o runtime e execute novamente desde a célula 1. '
            'Se estiver rodando no VS Code local, este kernel não usa a GPU do Colab.'
        )

    print(f'GPU ativa: {torch.cuda.get_device_name(0)}')


def _preparar_split_e_datasets(df, X_train=None, X_test=None, y_train=None, y_test=None):
    # 2) Validação dos dados e da divisão treino/teste já gravada.
    # O fine-tuning reutiliza a partição de 1_preprocessamento.ipynb e não cria outra.
    # X_train e X_test são o texto original, não o clean_text dos modelos clássicos.
    if 'df' not in globals() and df is None:
        raise RuntimeError(
            "A variável 'df' não foi encontrada nesta sessão. "
            "Execute o carregamento da partição antes do fine-tuning."
        )

    if df is not None:
        if 'texto_completo' not in df.columns or 'label' not in df.columns:
            raise RuntimeError(
                "O dataframe não contém as colunas esperadas ('texto_completo' e 'label'). "
                "O BERTimbau usa o texto original da partição."
            )

    if X_train is None or X_test is None or y_train is None or y_test is None:
        raise RuntimeError(
            'A divisão treino/teste deve ser fornecida. '
            'Execute 1_preprocessamento.ipynb para gravar Fake.br-Corpus/splits/.'
        )

    # 3) Mapeamento de rótulos para o formato esperado pelo Transformers.
    # Isso permite que o modelo entenda as classes no formato indexado do Hugging Face.
    unique_labels = sorted(df['label'].dropna().unique().tolist())
    label2id = {int(label): idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: str(label) for label, idx in label2id.items()}

    # 4) Reaproveita o mesmo split para comparação justa com os modelos tradicionais.
    train_df = pd.DataFrame({
        'texto_completo': X_train.fillna('').astype(str).reset_index(drop=True),
        'label': y_train.reset_index(drop=True).map(label2id).astype(int),
    })
    test_df = pd.DataFrame({
        'texto_completo': X_test.fillna('').astype(str).reset_index(drop=True),
        'label': y_test.reset_index(drop=True).map(label2id).astype(int),
    })

    # 5) Criação do dataset do Hugging Face em formato padronizado.
    hf_dataset = {
        'train': Dataset.from_pandas(train_df),
        'test': Dataset.from_pandas(test_df),
    }

    return hf_dataset, label2id, id2label, X_train, X_test, y_train, y_test


def finetune_bertimbau(
    df,
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    model_name='neuralmind/bert-base-portuguese-cased',
    output_dir=None,
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    resultados_dir=None,
):
    """
    Realiza o fine-tuning do BERTimbau para classificação binária de notícias reais vs falsas.

    Retorna um dicionário com o Trainer, o modelo, o tokenizer e os artefatos finais.
    """
    # 1) Validação do ambiente de treinamento antes de iniciar o experimento.
    if not output_dir:
        raise ValueError(
            'Informe output_dir (ex.: Fake.br-Corpus/bertimbau_fake_br no Google Drive).'
        )
    _validar_gpu()
    if df is None:
        raise RuntimeError('O dataframe df deve ser fornecido para o fine-tuning.')

    if 'texto_completo' not in df.columns or 'label' not in df.columns:
        raise RuntimeError(
            "O dataframe não contém as colunas esperadas ('texto_completo' e 'label'). "
            "O BERTimbau usa o texto original da partição."
        )

    # 2) Prepara split, rótulos e datasets do Hugging Face.
    hf_dataset, label2id, id2label, X_train, X_test, y_train, y_test = _preparar_split_e_datasets(
        df, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    # 6) Carrega o tokenizer do modelo BERTimbau em português.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 7) Tokenização em lote para truncar textos longos em 512 tokens.
    def tokenize_func(examples):
        return tokenizer(examples['texto_completo'], truncation=True, max_length=512)

    tokenized_datasets = {
        split: ds.map(tokenize_func, batched=True)
        for split, ds in hf_dataset.items()
    }

    # 8) Carrega o modelo para classificação seqüencial.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id={str(k): v for k, v in label2id.items()},
    )

    # 9) Cria a pasta de saída para salvar checkpoints e modelo final.
    os.makedirs(output_dir, exist_ok=True)

    # 10) Define hiperparâmetros de treinamento.
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        logging_steps=50,
        report_to='none',
    )

    # 11) Métricas de avaliação: F1 ponderado e accuracy.
    metric_f1 = evaluate.load('f1')
    metric_acc = evaluate.load('accuracy')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        f1 = metric_f1.compute(predictions=predictions, references=labels, average='weighted')
        acc = metric_acc.compute(predictions=predictions, references=labels)
        return {'f1': f1['f1'], 'accuracy': acc['accuracy']}

    # 12) Monta o Trainer com modelo, datasets, tokenizer e métricas.
    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets['train'],
        'eval_dataset': tokenized_datasets['test'],
        'data_collator': DataCollatorWithPadding(tokenizer=tokenizer),
        'compute_metrics': compute_metrics,
    }

    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if 'tokenizer' in trainer_signature:
        trainer_kwargs['tokenizer'] = tokenizer
    elif 'processing_class' in trainer_signature:
        trainer_kwargs['processing_class'] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    # 13) Execução do fine-tuning e salvamento do melhor modelo.
    print('Iniciando fine-tuning do BERTimbau...')
    n_treino = len(tokenized_datasets['train'])
    with medir_recursos('BERTimbau', 'treino', n_amostras=n_treino, gpu=True) as custo_treino:
        train_output = trainer.train()

    final_model_dir = os.path.join(output_dir, 'modelo_final')
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f'Modelo salvo com sucesso em: {final_model_dir}')

    custo_treino['hf_train_runtime'] = train_output.metrics.get('train_runtime')
    custo_treino['tamanho_artefato_mb'] = tamanho_artefato(final_model_dir)
    if resultados_dir is not None:
        persistir_custos(
            resultados_dir,
            ambiente=snapshot_ambiente(),
            linhas_treino=[custo_treino],
        )

    return {
        'trainer': trainer,
        'model': trainer.model,
        'tokenizer': tokenizer,
        'final_model_dir': final_model_dir,
        'train_output': train_output,
        'custos': [custo_treino],
    }


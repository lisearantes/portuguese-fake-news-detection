from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import threading
import time

import pandas as pd
import psutil


ARQUIVO_AMBIENTE = 'ambiente.json'
ARQUIVO_CUSTO_TREINO = 'custo_treino.csv'
ARQUIVO_CUSTO_INFERENCIA = 'custo_inferencia.csv'
ARQUIVO_COMPARACAO = 'comparacao.csv'
CHAVE_MODELO = 'Modelo'


def _bytes_para_mb(valor):
    if valor is None:
        return None
    return round(float(valor) / (1024 ** 2), 4)


def _rss_bytes():
    return psutil.Process(os.getpid()).memory_info().rss


def _modulo_cuda():
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return torch
    return None


def _resetar_pico_vram():
    torch = _modulo_cuda()
    if torch is None:
        return
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def _pico_vram_mb():
    torch = _modulo_cuda()
    if torch is None:
        return None
    torch.cuda.synchronize()
    return _bytes_para_mb(torch.cuda.max_memory_allocated())


def snapshot_ambiente():
    torch = _modulo_cuda()
    gpu_nome = None
    if torch is not None:
        gpu_nome = torch.cuda.get_device_name(0)
    memoria = psutil.virtual_memory()
    return {
        'capturado_em': datetime.now(timezone.utc).isoformat(),
        'cpu_count': os.cpu_count(),
        'ram_total_mb': _bytes_para_mb(memoria.total),
        'ram_disponivel_mb': _bytes_para_mb(memoria.available),
        'cuda_disponivel': torch is not None,
        'gpu_nome': gpu_nome,
    }


def tamanho_artefato(caminho):
    caminho = Path(caminho)
    if not caminho.exists():
        return None
    if caminho.is_file():
        return _bytes_para_mb(caminho.stat().st_size)
    total = 0
    for arquivo in caminho.rglob('*'):
        if arquivo.is_file():
            total += arquivo.stat().st_size
    return _bytes_para_mb(total)


def _arredondar_registro(registro):
    saida = dict(registro)
    for chave, valor in list(saida.items()):
        if isinstance(valor, float):
            saida[chave] = round(valor, 4)
    return saida


@contextmanager
def medir_recursos(nome, etapa, n_amostras=None, intervalo_pico=0.05, gpu=False):
    """Mede tempo, CPU, RAM RSS e, se gpu=True, pico de VRAM ao redor de um bloco."""
    registro = {
        CHAVE_MODELO: nome,
        'etapa': etapa,
        'n_amostras': n_amostras,
    }
    stop = threading.Event()
    pico_rss = [_rss_bytes()]

    def _amostrar():
        while not stop.wait(intervalo_pico):
            pico_rss[0] = max(pico_rss[0], _rss_bytes())

    amostrador = threading.Thread(target=_amostrar, daemon=True)
    ram_antes = _rss_bytes()
    pico_rss[0] = ram_antes
    if gpu:
        _resetar_pico_vram()
    amostrador.start()
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        yield registro
    finally:
        tempo_s = time.perf_counter() - t0
        tempo_cpu_s = time.process_time() - cpu0
        stop.set()
        amostrador.join(timeout=1)
        ram_depois = _rss_bytes()
        pico_rss[0] = max(pico_rss[0], ram_depois)
        registro['tempo_s'] = tempo_s
        registro['tempo_cpu_s'] = tempo_cpu_s
        registro['ram_antes_mb'] = _bytes_para_mb(ram_antes)
        registro['ram_depois_mb'] = _bytes_para_mb(ram_depois)
        registro['ram_delta_mb'] = _bytes_para_mb(ram_depois - ram_antes)
        registro['ram_pico_mb'] = _bytes_para_mb(pico_rss[0])
        registro['vram_pico_mb'] = _pico_vram_mb() if gpu else None
        if n_amostras:
            registro['tempo_por_noticia_ms'] = (tempo_s / n_amostras) * 1000
        registro.update(_arredondar_registro(registro))


def combinar_custos(*registros, nome=None):
    """Soma tempos e usa o pico de memória entre medições sequenciais (ex.: TF-IDF + classificador)."""
    validos = [dict(reg) for reg in registros if reg]
    if not validos:
        return {}
    combinado = dict(validos[0])
    if nome is not None:
        combinado[CHAVE_MODELO] = nome
    for extra in validos[1:]:
        for chave in ('tempo_s', 'tempo_cpu_s', 'ram_delta_mb'):
            if combinado.get(chave) is not None or extra.get(chave) is not None:
                combinado[chave] = (combinado.get(chave) or 0) + (extra.get(chave) or 0)
        for chave in ('ram_pico_mb', 'vram_pico_mb'):
            valores = [valor for valor in (combinado.get(chave), extra.get(chave)) if valor is not None]
            if valores:
                combinado[chave] = max(valores)
        if combinado.get('tamanho_artefato_mb') is not None or extra.get('tamanho_artefato_mb') is not None:
            combinado['tamanho_artefato_mb'] = (
                (combinado.get('tamanho_artefato_mb') or 0)
                + (extra.get('tamanho_artefato_mb') or 0)
            )
        if extra.get('ram_depois_mb') is not None:
            combinado['ram_depois_mb'] = extra['ram_depois_mb']
        if extra.get('hf_train_runtime') is not None:
            combinado['hf_train_runtime'] = extra['hf_train_runtime']
    n_amostras = combinado.get('n_amostras')
    if n_amostras and combinado.get('tempo_s') is not None:
        combinado['tempo_por_noticia_ms'] = (combinado['tempo_s'] / n_amostras) * 1000
    return _arredondar_registro(combinado)


def _upsert_csv(caminho, linhas):
    caminho = Path(caminho)
    df_novo = pd.DataFrame(linhas)
    if df_novo.empty:
        return df_novo
    if CHAVE_MODELO not in df_novo.columns:
        raise ValueError(f'As linhas de custo precisam da coluna {CHAVE_MODELO}.')
    if caminho.exists():
        df_antigo = pd.read_csv(caminho)
        if CHAVE_MODELO in df_antigo.columns:
            df_antigo = df_antigo[~df_antigo[CHAVE_MODELO].isin(df_novo[CHAVE_MODELO])]
            df = pd.concat([df_antigo, df_novo], ignore_index=True)
        else:
            df = df_novo
    else:
        df = df_novo
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False)
    return df


def persistir_custos(diretorio, ambiente=None, linhas_treino=None, linhas_inferencia=None):
    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)
    ambiente = ambiente or snapshot_ambiente()
    with open(diretorio / ARQUIVO_AMBIENTE, 'w', encoding='utf-8') as arquivo:
        json.dump(ambiente, arquivo, ensure_ascii=False, indent=2)

    gravados = {'ambiente': diretorio / ARQUIVO_AMBIENTE}
    if linhas_treino:
        gravados['treino'] = diretorio / ARQUIVO_CUSTO_TREINO
        _upsert_csv(gravados['treino'], linhas_treino)
    if linhas_inferencia:
        gravados['inferencia'] = diretorio / ARQUIVO_CUSTO_INFERENCIA
        _upsert_csv(gravados['inferencia'], linhas_inferencia)
    return gravados


def _ler_csv_opcional(caminho):
    caminho = Path(caminho)
    if not caminho.exists():
        return None
    return pd.read_csv(caminho)


def carregar_custos(diretorio):
    diretorio = Path(diretorio)
    ambiente = None
    caminho_ambiente = diretorio / ARQUIVO_AMBIENTE
    if caminho_ambiente.exists():
        with open(caminho_ambiente, encoding='utf-8') as arquivo:
            ambiente = json.load(arquivo)
    return {
        'ambiente': ambiente,
        'treino': _ler_csv_opcional(diretorio / ARQUIVO_CUSTO_TREINO),
        'inferencia': _ler_csv_opcional(diretorio / ARQUIVO_CUSTO_INFERENCIA),
        'comparacao': _ler_csv_opcional(diretorio / ARQUIVO_COMPARACAO),
    }


def _prefixar(df, prefixo):
    if df is None or df.empty:
        return None
    colunas = {col: f'{prefixo}_{col}' for col in df.columns if col != CHAVE_MODELO}
    return df.rename(columns=colunas)


def montar_comparacao(df_qualidade, custos_inferencia=None, custos_treino=None):
    df = df_qualidade.copy()
    inferencia = custos_inferencia
    treino = custos_treino
    if isinstance(inferencia, list):
        inferencia = pd.DataFrame(inferencia)
    if isinstance(treino, list):
        treino = pd.DataFrame(treino)
    inferencia = _prefixar(inferencia, 'inferencia')
    treino = _prefixar(treino, 'treino')
    if inferencia is not None:
        df = df.merge(inferencia, on=CHAVE_MODELO, how='left')
    if treino is not None:
        df = df.merge(treino, on=CHAVE_MODELO, how='left')
    return df


def persistir_comparacao(diretorio, df_comparacao):
    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)
    caminho = diretorio / ARQUIVO_COMPARACAO
    df_comparacao.to_csv(caminho, index=False)
    return caminho

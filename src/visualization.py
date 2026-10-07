import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import confusion_matrix


def _fonte_eixos():
    return {'family': 'Arial', 'size': 11, 'color': 'black'}


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


def _desenhar_matriz_confusao(ax, y_test, y_pred, fig=None, rotulo_painel=None):
    cm = confusion_matrix(y_test, y_pred)
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["#ffffff", "steelblue"])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                cbar=False, ax=ax,
                annot_kws={"fontname": "Arial", "fontsize": 12, "color": "black"})
    ax.set_title('')
    ax.set_xlabel('Classe Predita', fontdict=_fonte_eixos())
    ax.set_ylabel('Classe Real', fontdict=_fonte_eixos())
    if fig is None:
        fig = ax.figure
    _aplicar_estilo_tcc(ax, fig)
    if rotulo_painel:
        ax.text(
            -0.08, 1.08, str(rotulo_painel),
            transform=ax.transAxes, ha='left', va='top',
            fontname='Arial', fontsize=11, color='black', fontweight='bold',
        )


def plot_confusion_matrices(y_test, predicoes, figsize=(14, 4)):
    """predicoes: lista de tuplas (y_pred, nome). O nome aparece no canto superior esquerdo de cada painel."""
    n = len(predicoes)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, (y_pred, nome) in zip(axes, predicoes):
        _desenhar_matriz_confusao(ax, y_test, y_pred, fig=fig, rotulo_painel=nome)
    plt.tight_layout()
    plt.show()

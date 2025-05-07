import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(ground_truths, predictions, classes, path, figsize=(16,16), num_size=12, order_by_classes=False):
    '''
    Creates and plots a confusion matrix given the ground truths and the predictions of the classification model.

    Args:
        ground_truths (torch.tensor): ground truth (correct) target values.
        predictions (torch.tensor): estimated targets as returned by the model.
        classes (list): list of the classes labels.

    Returns:
        fig (matplotlib.figure.Figure): figure object.
    '''

    labels = classes if order_by_classes else None
    cm = confusion_matrix(ground_truths, predictions, labels=labels)
    cm_norm = confusion_matrix(ground_truths, predictions, labels=labels, normalize='true')

    df = pd.DataFrame(cm, index=classes, columns=classes)
    df_norm = pd.DataFrame(cm_norm, index=classes, columns=classes)

    plt.figure(figsize = figsize)
    ax = sns.heatmap(df_norm, annot=df, fmt='d', linewidths=0.5, linecolor='black', cmap='YlGn', vmin=0, vmax=1, annot_kws={"color": "black", "size": num_size})

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)

    cbar = ax.collections[0].colorbar
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1.5)

    ax.set_title('Confusion Matrix', fontdict={'fontsize': 32, 'fontweight': 'medium'})
    ax.set_xlabel('Predicted class', fontsize=18)
    ax.set_ylabel('True class', fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=12)
    fig = ax.get_figure()
    plt.savefig(f'{path}confusion.png', bbox_inches='tight')

    return fig


def plot_losses(learn, path):
    '''
    Creates and plots a figure with the training and validation losses curves.

    Args:
        learn (fastai.learner.Learner): trained learner object.

    Returns:
        fig (matplotlib.figure.Figure): figure object.
    '''

    rec = learn.recorder
    train_losses = np.array(rec.losses)
    train_iters = np.linspace(0, learn.n_epoch, len(train_losses))
    valid_losses = [v[1] for v in rec.values]
    valid_iters = np.arange(1, learn.n_epoch+1)

    plt.figure()
    sns.set(style="whitegrid")
    plot = sns.lineplot(x=train_iters, y=train_losses, label='Train', linestyle='-')
    sns.lineplot(x=valid_iters, y=valid_losses, label='Valid', marker='o', linestyle='--', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig = plot.figure
    plt.savefig(f'{path}losses.png', bbox_inches='tight')
    
    return fig


def plot_metrics(learn, path):
    '''
    Creates and plots a figure with the curves of all metrics.

    Args:
        learn (fastai.learner.Learner): trained learner object.

    Returns:
        fig (matplotlib.figure.Figure): figure object.
    '''

    valid_iters = np.arange(1, learn.n_epoch+1)
    met = np.array([v[2:] for v in learn.recorder.values])
    try:
        metrics_names = [m.func.__name__ for m in learn.metrics]
    except:
        metrics_names = [m.__name__ for m in learn.metrics]

    plt.figure()
    sns.set(style="whitegrid")
    for i in np.arange(len(metrics_names)):
        plot = sns.lineplot(x=valid_iters, y=met[:,i], label=metrics_names[i], linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()   
    fig = plot.figure
    plt.savefig(f'{path}metrics.png', bbox_inches='tight')

    return fig

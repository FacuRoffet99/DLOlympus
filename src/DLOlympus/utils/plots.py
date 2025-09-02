import torch
import numpy
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
	ground_truths: list | torch.Tensor | numpy.ndarray,
	predictions: list | torch.Tensor | numpy.ndarray,
	classes: list[str],
	path: str | None = None,
	figsize: tuple[int, int] = (16, 16),
	num_size: int = 12) -> plt.Figure:
	"""
	Plots a confusion matrix given the ground truths and the predictions of a classification model.
	It automatically detects whether the inputs are encoded (integers) or decoded (strings).

	Args:
		ground_truths (list | torch.Tensor | numpy.ndarray,): Ground truth (correct) target values.
		predictions (list | torch.Tensor | numpy.ndarray,): Predicted target values from the model.
		classes (list[str]): List of class labels, used for axis and ordering.
		path (str | None, optional): Path to save the confusion matrix image. If None, the plot is not saved. Defaults to None.
		figsize (tuple[int, int], optional): Size of the figure in inches (width, height). Defaults to (16, 16).
		num_size (int, optional): Font size for the annotation numbers in the heatmap. Defaults to 12.

	Returns:
		fig (matplotlib.figure.Figure): The matplotlib Figure object containing the confusion matrix plot.
	"""

	if len(ground_truths)==0 or len(predictions)==0 or len(classes)==0:
		raise ValueError('Inputs are empty.')

	if isinstance(ground_truths[0], str):
		labels = classes
	else:
		labels = np.arange(len(classes))

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

	if path is not None:
		fig.savefig(path, bbox_inches='tight')

	return fig
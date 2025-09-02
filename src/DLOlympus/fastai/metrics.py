import torch
import numpy as np

from typing import Literal
from fastai.vision.all import Learner, Metric

class MSEMetric(Metric):
	"""
	Computes the Mean Squared Error (MSE) metric for fastai training loops.

	Args:
		axis (int | None, optional): If specified, computes MSE for a particular axis in multi-task models.
			Use None for single-task models. Defaults to None.
		average (Literal['batch', 'all'], optional): Determines how to average the error.
			'batch' averages over the batch dimension, 'all' averages over all elements. Defaults to 'batch'.
		root (bool, optional): If True, returns the Root Mean Squared Error (RMSE) instead of MSE. Defaults to False.
		metric_name (str, optional): Custom name for the metric. Defaults to 'mse'.

	Usage:
		Add to fastai Learner metrics for regression tasks.
	"""
	def __init__(self, axis: int|None = None, average: Literal['batch', 'all'] = 'batch', root: bool = False, metric_name: str = 'mse') -> None:
		self.__name__ = metric_name
		self.axis = axis 
		self.average = average
		self.root = root
	def reset(self):
		self.total = 0.
		self.count = 0
	def accumulate(self, learn: Learner):
		# Single-task case
		if self.axis is None:
			targs = learn.y
			preds = learn.pred
		# Multi-task case
		else:
			targs = learn.y[self.axis]
			preds = learn.pred[self.axis]
		self.total += (preds.flatten() - targs.flatten()).pow(2).sum().item()
		self.count += targs.shape[0] if self.average == 'batch' else torch.prod(torch.tensor(targs.shape))
	@property
	def value(self):
		if self.count == 0: 
			return None
		mse = self.total / self.count
		return np.sqrt(mse) if self.root else mse  
	@property
	def name(self):
		return self.__name__

class MAEMetric(Metric):
	"""
	Computes the Mean Absolute Error (MAE) metric for fastai training loops.

	Args:
		axis (int | None, optional): If specified, computes MAE for a particular axis in multi-task models.
			Use None for single-task models. Defaults to None.
		average (Literal['batch', 'all'], optional): Determines how to average the error.
			'batch' averages over the batch dimension, 'all' averages over all elements. Defaults to 'batch'.
		metric_name (str, optional): Custom name for the metric. Defaults to 'mae'.

	Usage:
		Add to fastai Learner metrics for regression tasks.
	"""
	def __init__(self, axis: int|None = None, average: Literal['batch', 'all'] = 'batch', metric_name: str = 'mae') -> None:
		self.__name__ = metric_name
		self.average = average
		self.axis = axis 
	def reset(self):
		self.total = 0.
		self.count = 0
	def accumulate(self, learn: Learner):
		# Single-task case
		if self.axis is None:
			targs = learn.y
			preds = learn.pred
		# Multi-task case
		else:
			targs = learn.y[self.axis]
			preds = learn.pred[self.axis]
		self.total += (preds.flatten() - targs.flatten()).abs().sum().item()
		self.count += targs.shape[0] if self.average == 'batch' else torch.prod(torch.tensor(targs.shape))
	@property
	def value(self):
		if self.count == 0: 
			return None
		mae = self.total / self.count
		return mae  
	@property
	def name(self):
		return self.__name__

class AccuracyMetric(Metric):
	"""
	Computes classification accuracy for fastai training loops.

	Args:
		axis (list | None, optional): List of axes to compute accuracy for multi-task models.
			Use None for single-task models. Defaults to None.
		metric_name (str, optional): Custom name for the metric. Defaults to 'accuracy'.

	Usage:
		Add to fastai Learner metrics for classification tasks.
	"""
	def __init__(self, axis: list|None = None, metric_name: str = 'accuracy') -> None:
		self.__name__ = metric_name
		self.axis = axis
	def reset(self):
		self.total = 0.
		self.count = 0  
	def accumulate(self, learn: Learner) -> None:
		# Single-task case
		if self.axis is None:
			targs = learn.y
			preds = learn.pred.argmax(dim=-1)
			self.total += (preds == targs).sum().item()
			self.count += targs.shape[0] 
		# Multi-task case
		else:
			corrects_per_task = torch.stack([(learn.pred[i].argmax(dim=-1) == learn.y[i]) for i in self.axis])
			jointly_correct = corrects_per_task.all(dim=0)
			self.total += jointly_correct.sum().item()
			self.count += learn.y[self.axis[0]].shape[0] 
	@property
	def value(self):
		return self.total / self.count if self.count > 0 else None
	@property
	def name(self):
		return self.__name__

class F1ScoreMetric(Metric):
	"""
	Computes the F1-score for classification tasks in fastai training loops.

	Args:
		average (Literal['macro', 'micro'], optional): 
			'macro' computes the unweighted mean F1-score across all classes.
			'micro' computes F1-score globally by counting total true positives, false negatives, and false positives.
			Defaults to 'macro'.
		axis (int, optional): If specified, computes F1 for a particular axis in multi-task models (multiple axes are not supported).
			Use None for single-task models. Defaults to None.
		metric_name (str, optional): Custom name for the metric. Defaults to 'f1_score'.

	Usage:
		Add to fastai Learner metrics for classification tasks.
	"""
	def __init__(self, average: Literal['macro', 'micro'] = 'macro', axis: int = None, metric_name: str = 'f1_score'):
		self.__name__ = metric_name
		self.axis = axis
		self.average = average
	def reset(self):
		self.seen_classes = set() 
		self.tp, self.fp, self.fn = {}, {}, {}
	def accumulate(self, learn: Learner):
		# Single-task case
		if self.axis is None:
			targs = learn.y
			probs = learn.pred
		# Multi-task case
		else:
			targs = learn.y[self.axis]
			probs = learn.pred[self.axis]
		preds = probs.argmax(dim=-1)
		classes_in_batch = [int(c.item()) for c in torch.unique(torch.cat([targs, preds]))]
		for cls in classes_in_batch:
			# Add new classes
			self.seen_classes.add(cls)
			# Init dicts for each class
			if cls not in self.tp: 
				self.tp[cls] = 0
			if cls not in self.fp: 
				self.fp[cls] = 0
			if cls not in self.fn: 
				self.fn[cls] = 0
			# Convert to a binary problem for each class
			targ_mask = (targs == cls)  
			pred_mask = (preds == cls)
			self.tp[cls] += int((targ_mask & pred_mask).sum().item())
			self.fp[cls] += int((~targ_mask & pred_mask).sum().item())
			self.fn[cls] += int((targ_mask & ~pred_mask).sum().item())
	@property
	def value(self):
		if self.average == 'micro':
			tp, fp, fn = sum(self.tp.values()), sum(self.fp.values()), sum(self.fn.values())
			precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
			recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
			f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
			return f1
		elif self.average == 'macro':
			f1_scores = []
			for cls in sorted(self.seen_classes):
				tp = self.tp.get(cls, 0)
				fp = self.fp.get(cls, 0)
				fn = self.fn.get(cls, 0)
				precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
				recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
				f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
				f1_scores.append(f1)
			return float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0
	@property
	def name(self):
		return self.__name__
import torch.nn as nn


class SummedWeightedLoss(nn.Module):
	"""
	Combines multiple loss functions into a single weighted sum.

	Args:
		loss_functions (list): List of loss functions to be applied.
		loss_weights (list): List of weights for each loss function.
		reduction (str, optional): Reduction to apply to the output. Defaults to 'mean'.

	Methods:
		forward(preds_tuple, *targs_tuple):
			Computes the weighted sum of losses for the given predictions and targets.

		activation(preds_tuple):
			Applies the corresponding activation functions to each prediction.

		decodes(preds_tuple):
			Applies the corresponding decoder functions to each prediction.
	"""
	def __init__(self, loss_functions: list, loss_weights: list, reduction: str = 'mean'):
		super().__init__()
		self.loss_functions = loss_functions
		self.loss_weights = loss_weights
		self.reduction = reduction
	def forward(self, preds_tuple, *targs_tuple, **kwargs):	
		total_loss = 0
		for f, w, p, t in zip(self.loss_functions, self.loss_weights, preds_tuple, targs_tuple):
			total_loss += f(p, t) * w
		if self.reduction=='sum': 
			total_loss = total_loss.sum()
		elif self.reduction=='mean':
			total_loss = total_loss.mean()
		return total_loss
	def activation(self, preds_tuple):
		res = []
		for loss, p in zip(self.loss_functions, preds_tuple):
			act = getattr(loss, "activation", None)
			res.append(act(p) if callable(act) else p)
		return tuple(res)
	def decodes(self, preds_tuple):
		res = []
		for loss, p in zip(self.loss_functions, preds_tuple):
			dec = getattr(loss, "decodes", None)
			res.append(dec(p) if callable(dec) else p)
		return tuple(res)
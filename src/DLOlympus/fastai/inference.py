import dill
import torch
import warnings

from functools import partial
from fastai.vision.all import L, load_learner

from ..utils.inference_base import ModelInferencer


class FastAIInferencer(ModelInferencer):
	""" Class for doing inference with a fastai model. """
	def init(self, preds_postprocessing_fns: list[callable]|None = None):
		"""
		Args:
			preds_postprocessing_fns (list[callable] | None, optional): List of postprocessing functions to apply to each model output.
																		For example, the function for a classification task should convert class ids to class names.
																		If not provided, functions will be tried to be inferred. Defaults to None.	
		"""
		def f(v,x): return v[x]
		# If available use gpu, else cpu
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		# Load learner
		self.learn = load_learner(self.checkpoint_file, cpu=self.device=='cpu', pickle_module=dill).to_fp16()
		self.n_outputs = len(self.learn.dls.c) if isinstance(self.learn.dls.c, L) else 1
		# Set or create postprocessing functions
		if preds_postprocessing_fns is None:
			warnings.warn('List of functions to postprocess outputs not provided, inferring from dataloaders.', UserWarning)
			if self.n_outputs == 1:
				preds_postprocessing_fns = [partial(f,self.learn.dls.vocab)] if hasattr(self.learn.dls, 'vocab') else [lambda x: x]
			else:
				preds_postprocessing_fns = [partial(f,v) for v in self.learn.dls.vocab]
		if preds_postprocessing_fns is not None:
			if len(preds_postprocessing_fns) < self.n_outputs:
				warnings.warn(f'The number of postprocessing functions given or inferred {len(preds_postprocessing_fns)} is less than the number of model outputs {self.n_outputs}, identity functions added.', UserWarning)
				preds_postprocessing_fns += [lambda x: x for i in range(self.n_outputs - len(preds_postprocessing_fns))]
			elif len(preds_postprocessing_fns) > self.n_outputs:
				warnings.warn(f'The number of postprocessing functions given or inferred {len(preds_postprocessing_fns)} is greater than the number of model outputs {self.n_outputs}, postprocessing disabled.', UserWarning)
				preds_postprocessing_fns = None
		self.preds_postprocessing_fns = preds_postprocessing_fns

	def process(self, items: list) -> tuple:
		""" 
		Get predictions for items.

		Args:
			items (list): Items to process. Depending on the dataloader, could be a list of paths, a list of tensors, etc.
			with_raw (bool, optional): Whether to also return the raw predictions from the model (useful for getting confidence scores). Defaults to False. 
		Returns:
			tuple: A tuple containing:
				- The raw predictions from the model.
				- The decoded predictions (the result of applying the decoding functions of the loss to the raw predictions).
				- The final outputs (the result of applying the postprocessing functions to the decoded predictions).
		"""
		# Create dataloader
		test_dl = self.learn.dls.test_dl(items, device=self.device, num_workers=0)
		# Predict
		raw_preds, _, decoded_preds = self.learn.get_preds(dl=test_dl, with_decoded=True)  
		# Apply postprocessing
		outputs = None
		if self.preds_postprocessing_fns is not None:
			if self.n_outputs == 1:
				decoded_preds = (decoded_preds,)
			outputs = tuple(fn(dp) for fn,dp in zip(self.preds_postprocessing_fns, decoded_preds))
			if self.n_outputs == 1:
				decoded_preds = decoded_preds[0]
		return raw_preds, decoded_preds, outputs

	def export(self, input_size: list|tuple, save_folder: str = ''):
		""" 
		Exports trained model to ONNX format.

		Args:
			input_size (list | tuple): The shape of the input tensor (excluding batch size), e.g., (channels, height, width).
			save_folder (str, optional): The folder path where the ONNX model will be saved. Defaults to '' (current directory).
		"""
		dummy_input = torch.randn(1, *input_size).cpu()
		model = self.learn.model.eval().cpu()
		output_names = [f'out{i}' for i in range(self.n_outputs)]
		torch.onnx.export(
			model,
			dummy_input,
			save_folder+'model.onnx',
			input_names=['input'],
			output_names=output_names,
			dynamic_axes={'input': {0: 'batch_size'}, **{name: {0: 'batch_size'} for name in output_names}}
		)		
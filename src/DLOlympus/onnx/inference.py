import torch
import pathlib
import numpy as np
import onnxruntime as ort

from PIL import Image
from tqdm import tqdm

from ..utils.inference_base import ModelInferencer


class ONNXInferencer(ModelInferencer):
	""" Class for doing inference with a onnx model. """
	def init(self, transforms):
		""" 
		Initializes the inferencer.
		
		Args:
			transforms (albumentations.Compose): A pre-configured albumentations transform pipeline.
		"""
		self.transform = transforms
		providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
		self.session = ort.InferenceSession(str(self.checkpoint_file), providers=providers)
		self.input_name = self.session.get_inputs()[0].name
		print(f"ONNX session created on {self.session.get_providers()}.")

	def _preprocess_batch(self, batch_items):
		""" Pre-processes a list of image paths into a single batch tensor. """
		batch = []
		for item_path in batch_items:
			img = np.array(Image.open(item_path).convert('RGB'))
			transformed_img = self.transform(image=img)['image']
			batch.append(transformed_img)
		return torch.stack(batch)

	def _process_one_batch(self, batch_items):
		""" Internal method to run inference on a single, prepared batch. """
		input_tensor = self._preprocess_batch(batch_items)
		input_np = input_tensor.numpy() if not input_tensor.device.type == 'cpu' else input_tensor.cpu().numpy()
		raw_outputs = self.session.run(None, {self.input_name: input_np})
		return raw_outputs

	def process(self, items, batch_size=32):
		""" 
		Runs predictions on a list of items, managing batches automatically.

		Args:
			items (list): A list of image file paths.
			batch_size (int): The number of items to process in each batch.
		"""
		num_items = len(items)
		all_results = None

		for i in tqdm(range(0, num_items, batch_size), desc="Processing Batches"):
			batch = items[i:i + batch_size]
			batch_results = self._process_one_batch(batch)
			if all_results is None:
				all_results = [batch_results[j] for j in range(len(batch_results))]
			else:
				for j in range(len(batch_results)):
					all_results[j] = np.concatenate([all_results[j], batch_results[j]], axis=0)

		return all_results
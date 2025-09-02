import pathlib
from abc import ABC, abstractmethod


class ModelInferencer(ABC):
	"""	Base (abstract) class for implementing a model inferencer.

	Args:
		checkpoint_file (str): Path to the file where the model weights are stored.
	"""
	def __init__(self, checkpoint_file):
		self.checkpoint_file = pathlib.Path(checkpoint_file)
	@abstractmethod
	def init(self):
		pass
	@abstractmethod
	def process(self, imgs_source):
		pass	  
from abc import ABC, abstractmethod
import pathlib

class ModelInferencer(ABC):
    '''
    Base (abstract) class for implementing a model inferencer.

    Args:
        checkpoint_file (str): Path to the .pth file where the model weights are stored.
    '''

    def __init__(self, checkpoint_file):
        self.checkpoint_file = pathlib.path(checkpoint_file)

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def process(self, imgs_source):
        pass      
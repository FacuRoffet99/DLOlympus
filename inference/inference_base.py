from abc import ABC, abstractmethod

class ModelInferencer(ABC):
    '''
    Base (abstract) class for implementing a model inferencer.

    Args:
        checkpoint_file (str): Path to the .pkl file where the model weights are stored.
        labels (list): List containing the names of the classes to predict (in order). 
    '''

    def __init__(self, checkpoint_file, labels):
        self.checkpoint_file = checkpoint_file
        self.labels = labels

    @abstractmethod
    def init(self, cpu):
        pass

    @abstractmethod
    def set_params(self, param_dict):
        pass

    @abstractmethod
    def process(self, imgs_source):
        pass
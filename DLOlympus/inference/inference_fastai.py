from fastai.vision.all import *
import pathlib, torch
from .inference_base import ModelInferencer
from ..training.custom_model import custom_model_learner

class FastAIInferencer(ModelInferencer):
    '''
    Base (abstract) class for implementing an inferencer with FastAI.

    Args:
        checkpoint_file (str): Path to the .pth file where the model weights are stored.
    '''

    def init(self, block, model, bs, **kwargs):
        '''
        Initialization method to create a learner with the model weights an configs.

        Args:
            block (fastai.data.block.DataBlock): FastAI datablock as used for training, but without get_x, get_y, splitter and data augmentation.
            model (torch.nn.modules): Torch model as used for training.
            bs (int): Batch size.
            kwargs: Keywords arguments for 'custom_model_learner' (it is recommended to include at least 'pretrained' and 'loss_func'). 
        '''
        # If available use gpu, else cpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Create dummy dataloader
        dls = DataLoaders.from_dblock(block, [0], bs=bs, num_workers=0)
        # Load model
        self.learn = custom_model_learner(dls, model, model_dir=self.checkpoint_file.parent, **kwargs).to_fp16()
        self.learn.load(str(self.checkpoint_file.with_suffix('')), device=self.device, strict=True, weights_only=False)         

    def process(self, items, **kwargs):
        '''
        Get predictions for items.

        Args:
            items (list): Items to process. Depending on the dataloader, could be a list of paths, a list of tensors, etc.
            kwargs: Keywords arguments for 'get_preds'.

        Returns:
            results (dict): The raw prediction results.
        '''
        # Create dataloader
        test_dl = self.learn.dls.test_dl(items, device=self.device, num_workers=0)
        # Predict
        preds, _ = self.learn.get_preds(dl=test_dl, **kwargs)  
        return preds

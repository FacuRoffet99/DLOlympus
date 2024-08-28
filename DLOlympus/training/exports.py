from fastai.vision.all import *
import torch
import torchvision.transforms as transforms

def fastai2onnx(learn, path, height, width):
    '''
    Converts a FastAI model from a Learner object to an ONNX model, and then saves it to disk.

    Args:
        learn (fastai.learner.Learner): trained learner object.
        path (str): folder where to save the 'model.onnx' file.
        height: input height for the model.
        width: input width for the model.
    '''

    # Extract pytorch model from the learner and add missing layers
    softmax_layer = torch.nn.Softmax(dim=1)
    normalization_layer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pytorch_model = torch.nn.Sequential(
        normalization_layer,
        learn.model.eval(),
        softmax_layer
    )

    # Export from pytorch to onnx 
    torch.onnx.export(
        pytorch_model,
        torch.randn(1, 3, height, width).to('cuda'),
        f'{path}/model.onnx',
        do_constant_folding=True,
        export_params=True,
        input_names=['in'],
        output_names=['out'],
        opset_version=17,
        dynamic_axes={'in' : {0 : 'batch_size'},    # variable length axes
                    'out' : {0 : 'batch_size'}}
    )    
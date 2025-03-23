import sys
from fastai.vision.all import *

def get_model(hyperparameters):
    '''
    Returns the apropiate torchvision or timm model.
    '''
    if hyperparameters['ARCH_TYPE'] == 'torchvision':
        model = getattr(sys.modules[__name__], hyperparameters['ARCH'])
    if hyperparameters['ARCH_TYPE'] == 'timm':
        model = hyperparameters['ARCH']
    return model

def get_metrics(learn, with_tta=True, **tta_kwargs):
    '''
    Returns a dictionary with the names and values of the metrics.
    '''
    
    try:
        names = [m.func.__name__ for m in learn.metrics]
    except:
        names = [m.__name__ for m in learn.metrics]
    values = learn.validate()[1:]

    if with_tta:
        names += [m.func.__name__+'_tta' for m in learn.metrics]
        results = learn.tta(**tta_kwargs)
        for m in learn.metrics:
            if type(m) == AvgMetric:
                values += m.func(results[0], results[1]).item()
            if type(m) == AccumMetric:
                values += m.func(results[0].argmax(axis=1), results[1], average='macro')
        
    metrics = dict(zip(names, values))

    return metrics

from fastai.vision.learner import TimmBody, _add_norm, _timm_norm, model_meta, _default_meta
from fastai.vision.learner import create_body as create_body_torchvision
from fastai.callback.hook import num_features_model
from torch import nn
import timm
from .utils import get_model

def create_torchvision_body(arch, pretrained=True, n_in=3):
	'''Create a body from a torchvision model object.'''
	meta = model_meta.get(arch, _default_meta)
	model = arch(pretrained=pretrained)
	body = create_body_torchvision(model, n_in, pretrained, meta['cut'])
	nf = num_features_model(nn.Sequential(*body.children()))
	return body, nf

def create_timm_body(arch, pretrained=True, n_in=3, **kwargs):
	'''Create a body from a timm model string.'''
	model = timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=n_in, **kwargs)
	body = TimmBody(model, pretrained, None, n_in)
	nf = body.model.num_features
	return body, nf, model.default_cfg

def create_body(dls, hyperparameters, n_in=3, normalize=True):
	'''Create a torchvision or timm body.'''
	arch = get_model(hyperparameters)
	pretrained = hyperparameters['PRETRAINED']   
	meta = model_meta.get(arch, _default_meta)
	if isinstance(arch, str):
		body, nf, cfg = create_timm_body(arch, pretrained=pretrained, n_in=n_in)
		if normalize: _timm_norm(dls, cfg, pretrained, n_in)
	else:
		if normalize: _add_norm(dls, meta, pretrained, n_in)
		body, nf = create_torchvision_body(arch, pretrained=pretrained, n_in=n_in)
	return body, nf


from fastai.layers import AdaptiveConcatPool2d, LinBnDrop, Flatten, SigmoidRange 
from fastai.torch_core import apply_init

def create_pre_head(n_in, init=nn.init.kaiming_normal_, concat_pool=True):
    if concat_pool: n_in *= 2
    layers = []
    layers.append(AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1))
    layers.append(Flatten())
    pre_head = nn.Sequential(*layers)
    if init is not None: apply_init(pre_head, init)
    return pre_head, n_in

def create_head(n_in, n_out, n_features=[512], init=nn.init.kaiming_normal_, p=0.5, y_range=None):
    "Model head that takes `n_in` features, runs through `n_features`, and out `n_out` classes."
    n_features = [n_in] + n_features + [n_out]
    layers = []
    for ni, no in zip(n_features[:-2], n_features[1:-1]):
        layers += LinBnDrop(ni, no, bn=True, p=p/2, act=nn.ReLU(inplace=True), lin_first=False)
    layers += LinBnDrop(n_features[-2], n_features[-1], bn=True, p=p, act=None, lin_first=False)
    if y_range is not None: layers.append(SigmoidRange(*y_range))
    head = nn.Sequential(*layers)
    if init is not None: apply_init(head, init)
    return nn.Sequential(*layers)


from fastai.vision.learner import Learner
from fastai.optimizer import Adam

def custom_model_learner(dls, model, pretrained=True, loss_func=None, opt_func=Adam, lr=1e-3, splitter=None, cbs=None, 
						 metrics=None, path=None, model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95,0.85,0.95)):
	'''Create a Learner object with for a custom model.'''
	learn = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                   metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn, moms=moms)
	if pretrained: learn.freeze()
	return learn

from fastai.torch_core import PrePostInitMeta
from itertools import islice
from collections import OrderedDict
from typing import Union
import operator

class IndexableModule(nn.Module, metaclass=PrePostInitMeta):
    "Same as `nn.Module`, but no need for subclasses to call `super().__init__`"
    def __pre_init__(self, *args, **kwargs): super().__init__()
    def __init__(self): pass
    def __len__(self) -> int:
        return len(self._modules)
    def _get_item_by_idx(self, iterator, idx):  # type: ignore[misc, type-var]
        """Get the idx-th item of the iterator."""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f'index {idx} is out of range')
        idx %= size
        return next(islice(iterator, idx, None))
    def __getitem__(self, idx: Union[slice, int]):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)
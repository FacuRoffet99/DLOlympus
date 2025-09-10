import timm
import operator
import torch.nn as nn
import torchvision.models

from collections import OrderedDict
from itertools import islice
from typing import Literal, Union

from fastai.vision.all import AdaptiveConcatPool2d, L, LinBnDrop, PrePostInitMeta, TimmBody
from fastai.vision.all import apply_init, create_body, Flatten, num_features_model, params, SigmoidRange, model_meta
from fastai.vision.learner import _default_meta


def create_torchvision_backbone(arch: str, 
								pretrained: bool = True, 
								n_in: int = 3) -> tuple[nn.Module, int, dict]:
	""" 
	Creates a backbone from a torchvision model.

	Args:
		arch (function): Torchvision model name.
		pretrained (bool, optional): Defaults to True.
		n_in (int, optional): Number of input channels. Defaults to 3.

	Returns:
		Tuple[nn.Module, int, dict]: Backbone module, number of features, and meta dictionary.
	"""
	arch = getattr(torchvision.models, arch)
	meta = model_meta.get(arch, _default_meta)
	model = arch(pretrained=pretrained)
	bbone = create_body(model, n_in, pretrained, meta['cut'])
	nf = num_features_model(nn.Sequential(*bbone.children()))
	return bbone, nf, meta

def create_timm_backbone(arch: str, 
						 pretrained: bool = True, 
						 n_in: int = 3, 
						 **kwargs) -> tuple[nn.Module, int, dict]:
	""" 
	Creates a backbone from a timm model.

	Args:
		arch (str): Timm model name.
		pretrained (bool, optional): Defaults to True.
		n_in (int, optional): Number of input channels. Defaults to 3.

	Returns:
		Tuple[nn.Module, int, dict]: Backbone module, number of features, and meta dictionary.
	"""
	model = timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=n_in, **kwargs)
	bbone = TimmBody(model, pretrained, None, n_in)
	nf = bbone.model.num_features
	meta = model.default_cfg
	return bbone, nf, meta

def create_backbone(arch: str, 
					arch_type: Literal['timm', 'torchvision'], 
					pretrained: bool = True, 
					n_in: int = 3) -> tuple[nn.Module, int, list[float], list[float]]:
	""" 
	Creates a backbone from a torchvision or timm model.

	Args:
		arch (str): Model name.
		arch_type (Literal['timm', 'torchvision']): Model library to use.
		pretrained (bool, optional): Defaults to True.
		n_in (int, optional): Number of input channels. Defaults to 3.

	Returns:
		Tuple[nn.Module, int, list[float], list[float]]: Backbone module, number of features, mean stats, std stats.
	"""
	match arch_type:
		case 'timm':
			bbone, nf, meta = create_timm_backbone(arch, pretrained=pretrained, n_in=n_in)
			stats_mean, stats_std = meta['mean'], meta['std']
		case 'torchvision':
			bbone, nf, meta = create_torchvision_backbone(arch, pretrained=pretrained, n_in=n_in)
			if meta.get('stats') is None:
				stats_mean, stats_std = None, None
			else:
				stats_mean, stats_std = meta.get('stats')
		case _:
			raise ValueError('Unsupported or invalid arch type.')

	return bbone, nf, stats_mean, stats_std

def create_head(nf: int,
				n_out: int,
				lin_ftrs: list[int] = [512],
				ps: float | list[float] = 0.5,
				pool: bool = True,
				concat_pool: bool = True,
				first_bn: bool = True,
				bn_final: bool = False,
				lin_first: bool = False,
				y_range: tuple[float, float] | None = None,
				init: callable = nn.init.kaiming_normal_) -> nn.Sequential:
	"""
	Creates a model head for a neural network, consisting of optional pooling, flattening, 
	linear layers, batch normalization, dropout, activation functions, and optional output range scaling.

	Args:
		nf (int): Number of input features to the head.
		n_out (int): Number of output classes or regression targets.
		lin_ftrs (list[int], optional): List of hidden layer sizes for the linear layers. Defaults to [512].
		ps (float or list[float], optional): Dropout probability or list of probabilities for each layer. Defaults to 0.5.
		pool (bool, optional): If True, applies pooling before the linear layers. Defaults to True.
		concat_pool (bool, optional): If True, uses adaptive concatenated pooling; otherwise, uses average pooling. Defaults to True.
		first_bn (bool, optional): If True, applies batch normalization to the first linear layer. Defaults to True.
		bn_final (bool, optional): If True, applies batch normalization to the final output. Defaults to False.
		lin_first (bool, optional): If True, applies dropout before the first linear layer. Defaults to False.
		y_range (tuple[float, float] or None, optional): If provided, constrains the output to the given range using a sigmoid. Defaults to None.
		init (callable, optional): Initialization function for the layers. Defaults to nn.init.kaiming_normal_.

	Returns:
		nn.Sequential: A sequential container of the constructed head layers.
	"""
	if pool and concat_pool: 
		nf *= 2
	lin_ftrs = [nf] + lin_ftrs + [n_out]
	bns = [first_bn] + [True]*len(lin_ftrs[1:])
	ps = L(ps)
	if len(ps) == 1: 
		ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
	actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
	layers = []
	if pool:
		pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
		layers += [pool, Flatten()]
	if lin_first: 
		layers.append(nn.Dropout(ps.pop(0)))
	for ni,no,bn,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], bns, ps, actns):
		layers += LinBnDrop(ni, no, bn=bn, p=p, act=actn, lin_first=lin_first)
	if lin_first: 
		layers.append(nn.Linear(lin_ftrs[-2], n_out))
	if bn_final: 
		layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
	if y_range is not None: 
		layers.append(SigmoidRange(*y_range))
	head = nn.Sequential(*layers)
	if init is not None: 
		apply_init(head, init)
	return head

class IndexableModule(nn.Module, metaclass=PrePostInitMeta):
	""" Same as fastai Module, but with indexing. """
	def __pre_init__(self, *args, **kwargs): super().__init__()
	def __init__(self): pass
	def __len__(self) -> int:
		return len(self._modules)
	def _get_item_by_idx(self, iterator, idx):  # type: ignore[misc, type-var]
		""" Get the idx-th item of the iterator. """
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
		
class MultiheadModel(IndexableModule):
	def __init__(self, backbone, *heads):
		self.backbone = backbone
		self.heads = nn.ModuleList(heads)
	def forward(self, x):
		if isinstance(x, tuple) and len(x)==1:
			x = x[0]
		x = self.backbone(x)
		x = tuple(h(x) for h in self.heads)
		if len(self.heads) == 1:
			x = x[0]
		return x  
	
def multihead_splitter(model):
	return [params(model.backbone), sum([params(h) for h in model.heads], [])]
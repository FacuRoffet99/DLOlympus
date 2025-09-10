import torch
from plum import dispatch
from torchvision.transforms.functional import resize

from fastai.vision.all import fastuple, IntToFloatTensor, PILImage, TensorImage, TransformBlock, get_grid, show_image


class ImageTuple(fastuple):
	@classmethod
	def create(cls, fns):
		return cls(tuple(PILImage.create(f) for f in fns.split(', ')))
	def show(self, ctx=None, **kwargs):
		return show_image(torch.cat(self, dim=2), ctx=ctx, **kwargs)
@dispatch
def show_batch(x:ImageTuple, y, samples, ctxs=None, max_n=10, nrows=None, ncols=None, figsize=None, **kwargs):
	if ctxs is None: 
		ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
	ctxs = show_batch._resolve_method_with_cache(object)[0](x, y, samples, ctxs=ctxs, max_n=max_n, **kwargs)
	return ctxs
def ImageTupleBlock(): 
	return TransformBlock(type_tfms=ImageTuple.create, batch_tfms=IntToFloatTensor)


class MultipleTensor(fastuple):
	@classmethod
	def create(cls, *fns): return cls(tuple(*fns))	
	def _preprocess_tensor(self, o:torch.Tensor, one_dim_thickness:int = 10) -> torch.Tensor:
		""" Converts any input tensor into a 3-channel (C, H, W) image tensor. """
		if o.ndim > 0:
			min_val, max_val = o.min(), o.max()
			if max_val > min_val:
				o = (o.float() - min_val) / (max_val - min_val)
		match o.ndim:
			case 1:
				o = o.unsqueeze(-1).repeat(3, 1, one_dim_thickness) # (H) -> (3,H,one_dim_thickness)
			case 2:
				o = o.unsqueeze(0).repeat(3, 1, 1) # (H, W)-> (3,H,W)
			case 3:
				if o.shape[0] == 1:
					o = o.repeat(3, 1, 1) # (1,H,W) -> (3,H,W)
		return o
	def show(self, ctx=None, one_dim_thickness=10, **kwargs):
		# Preprocess all tensors into a standard (3, H, W) format
		processed_imgs = [self._preprocess_tensor(o, one_dim_thickness=one_dim_thickness) for o in self]
		# Find maximum height
		max_h = max(o.shape[1] for o in processed_imgs)
		# Create a white separator bar
		separator = torch.ones(3, max_h, 10)
		# Resize and add separators
		final_tensors = []
		for i, img in enumerate(processed_imgs):
			c, h, w = img.shape
			new_w = int(w * (max_h / h))
			resized_img = resize(img, [max_h, new_w])
			final_tensors.append(resized_img)
			if i < len(processed_imgs) - 1:
				final_tensors.append(separator)
		# Combine all tensors horizontally
		combined = torch.cat(final_tensors, dim=2).permute(1, 2, 0)
		return show_image(TensorImage(combined), ctx=ctx, **kwargs)
@dispatch
def show_batch(x: MultipleTensor, y, samples, show_y=True, one_dim_thickness=10, ctxs=None, max_n=10, nrows=None, ncols=None, figsize=None, **kwargs):
	if ctxs is None: 
		ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
	for i, (s, c) in enumerate(zip(samples, ctxs)):
		title = str(s[1:]) if show_y else None
		s[0].show(ctx=c, title=title, one_dim_thickness=one_dim_thickness, **kwargs)
	return ctxs
def MultipleTensorBlock(): 
	return TransformBlock(type_tfms=MultipleTensor.create)
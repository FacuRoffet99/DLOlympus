from fastai.vision.all import *

class ImageTuple(fastuple):
    @classmethod
    def create(cls, fns):
        return cls(tuple(PILImage.create(f) for f in fns.split(', ')))
    def show(self, ctx=None, **kwargs):
        return show_image(torch.cat(self, dim=2), ctx=ctx, **kwargs)

def image_tuple_show_batch(x:ImageTuple, y, samples, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):
    if figsize is None: figsize = (ncols*10, max_n//ncols * 3)
    if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
    ctxs = show_batch[object](x, y, samples, ctxs=ctxs, max_n=max_n, **kwargs)
    return ctxs

def ImageTupleBlock(): 
    show_batch.add(image_tuple_show_batch)
    return TransformBlock(type_tfms=ImageTuple.create, batch_tfms=IntToFloatTensor)

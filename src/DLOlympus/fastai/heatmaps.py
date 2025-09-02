# This code was stolen from https://elte.me/2021-03-10-keypoint-regression-fastai

import itertools
import pathlib
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image, ImageDraw, ImageFont

from fastcore.transform import Transform
from fastai.torch_core import tensor, TensorImageBase
from fastai.vision.core import TensorPoint, TensorBase


# A gaussian kernel cache, so we don't have to regenerate them every time.
# This is only a small optimization, generating the kernels is pretty fast.
_gaussians = {}

def generate_gaussian(t, x, y, sigma=10):
	"""
	Generates a 2D Gaussian point at location x,y in tensor t.

	x should be in range (-1, 1) to match the output of fastai's PointScaler.

	sigma is the standard deviation of the generated 2D Gaussian.
	"""
	h,w = t.shape

	# Heatmap pixel per output pixel
	mu_x = int(0.5 * (x + 1.) * w)
	mu_y = int(0.5 * (y + 1.) * h)

	tmp_size = sigma * 3

	# Top-left
	x1,y1 = int(mu_x - tmp_size), int(mu_y - tmp_size)

	# Bottom right
	x2, y2 = int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)
	if x1 >= w or y1 >= h or x2 < 0 or y2 < 0:
		return t

	size = 2 * tmp_size + 1
	tx = np.arange(0, size, 1, np.float32)
	ty = tx[:, np.newaxis]
	x0 = y0 = size // 2

	# The gaussian is not normalized, we want the center value to equal 1
	g = _gaussians[sigma] if sigma in _gaussians \
				else tensor(np.exp(- ((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma ** 2)))
	_gaussians[sigma] = g

	# Determine the bounds of the source gaussian
	g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
	g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1

	# Image range
	img_x_min, img_x_max = max(0, x1), min(x2, w)
	img_y_min, img_y_max = max(0, y1), min(y2, h)

	t[img_y_min:img_y_max, img_x_min:img_x_max] = \
	  g[g_y_min:g_y_max, g_x_min:g_x_max]

	return t


# FastAIs `TensorPoint` class comes with a default `PointScaler` transform
# that scales the points to (-1, 1). This little function does the same
# given a point / tensor of points and original size.
def _scale(p, s): return 2 * (p / s) - 1


def heatmap2argmax(heatmap, scale=False):
	N, C, H, W = heatmap.shape
	index = heatmap.view(N,C,1,-1).argmax(dim=-1)
	pts = torch.cat([index%W, index//W], dim=2)
	if scale:
		scale = tensor([W,H], device=heatmap.device)
		pts = _scale(pts, scale)
	return pts


class Heatmap(TensorImageBase):
	"Heatmap tensor, we can use the type to modify how we display things"
	pass


class HeatmapPoint(TensorPoint):
	"""
	A class that mimics TensorPoint, but wraps it so
	we'll be able to override `show` methods with
	a different type.
	"""
	pass


class HeatmapTransform(Transform):
	"""
	A batch transform that turns TensorPoint instances into Heatmap instances,
	and Heatmap instances into HeatmapPoint instances.

	Used as the last transform after all other transformations.
	"""
	# We want the heat map transformation to happen last, so give it a high order value
	order=999

	def __init__(self, heatmap_size, sigma=10, **kwargs):
		"""
		heatmap_size: Size of the heatmap to be created
		sigma: Standard deviation of the Gaussian kernel
		"""
		super().__init__(**kwargs)
		self.sigma = sigma
		self.size = heatmap_size

	def encodes(self, x:TensorPoint):
		# The shape of x is (batch x n_points x 2)
		num_imgs = x.shape[0]
		num_points = x.shape[1]
		maps = Heatmap(torch.zeros(num_imgs, num_points, *self.size, device=x.device))
		for b,c in itertools.product(range(num_imgs), range(num_points)):
			# Note that our point is already scaled to (-1, 1) by PointScaler
			point = x[b][c]
			generate_gaussian(maps[b][c], point[0], point[1], sigma=self.sigma)
		return maps

	def decodes(self, x:Heatmap):
		"""
		Decodes a heat map back into a set of points by finding
		the coordinates of their argmax.

		This returns a HeatmapPoint class rather than a TensorPoint
		class, so we can modify how we display the output.
		"""
		# Flatten the points along the final axis,
		# and find the argmax per channel
		xy = heatmap2argmax(x, scale=True)
		return HeatmapPoint(xy, source_heatmap=x)


def coord2heatmap(x, y, w, h, heatmap):
	"""
	Inserts a coordinate (x,y) from a picture with
	original size (w x h) into a heatmap, by randomly assigning
	it to one of its nearest neighbor coordinates, with a probability
	proportional to the coordinate error.

	Arguments:
	x: x coordinate
	y: y coordinate
	w: original width of picture with x coordinate
	h: original height of picture with y coordinate
	"""
	# Get scale
	oh,ow = heatmap.shape
	sx = ow / w
	sy = oh / h

	# Unrounded target points
	px = x * sx
	py = y * sy

	# Truncated coordinates
	nx,ny = int(px), int(py)

	# Coordinate error
	ex,ey = px - nx, py - ny

	xyr = torch.rand(2, device=heatmap.device)
	xx = (ex >= xyr[0]).long()
	yy = (ey >= xyr[1]).long()
	heatmap[min(ny + yy, heatmap.shape[0] - 1),
			min(nx+xx, heatmap.shape[1] - 1)] = 1
	return heatmap


def heatmap2coord(heatmap, topk=9):
	N, C, H, W = heatmap.shape
	score, index = heatmap.view(N,C,1,-1).topk(topk, dim=-1)
	coord = torch.cat([index%W, index//W], dim=2)
	return (coord*F.softmax(score, dim=-1)).sum(-1)


def topk_heatmap2coord(heatmap, topk=9, scale=False):
	coord = heatmap2coord(heatmap, topk)
	if scale:
		_, _, H, W = heatmap.shape
		scale = tensor([W,H], device=heatmap.device)
		coord = _scale(coord, scale)
	return coord


class RandomBinaryHeatmapTransform(Transform):
	order=999

	def __init__(self, heatmap_size, topk=9, **kwargs):
		super().__init__(**kwargs)
		self.size = tensor(heatmap_size)
		self.topk=topk

	def encodes(self, x:TensorPoint):
		# The shape of x is (batch x n_points x 2)
		num_imgs = x.shape[0]
		num_points = x.shape[1]
		maps = Heatmap(torch.zeros(num_imgs, num_points, *self.size, dtype=torch.long,
								   device=x.device))
		for b,c in itertools.product(range(num_imgs), range(num_points)):
			heatmap = maps[b][c]
			# Note that our point is already scaled to (-1, 1) by PointScaler.
			# We pretend here it's in range 0...2
			point = x[b][c] + 1.
			coord2heatmap(point[0], point[1], 2., 2., heatmap)
		return maps

	def decodes(self, x:Heatmap):
		"""
		Decodes a batch of binary heatmaps back into a set of
		TensorPoints.
		"""
		if x.dtype == torch.long:
			# If the target heatmap is an annotation heatmap, our
			# decoding procedure is different - we need to directly
			# retrieve the argmax.
			return HeatmapPoint(heatmap2argmax(x, scale=True),
							   source_heatmap=x)
		return HeatmapPoint(topk_heatmap2coord(x, topk=self.topk, scale=True),
						   source_heatmap=x)


class BinaryHeadBlock(nn.Module):
	def __init__(self, in_channels, proj_channels, out_channels, **kwargs):
		super(BinaryHeadBlock, self).__init__()
		self.layers = nn.Sequential(
			nn.Conv2d(in_channels, proj_channels, 1, bias=False),
			nn.BatchNorm2d(proj_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(proj_channels, out_channels, 1, bias=False),
		)

	def forward(self, input):
		return self.layers(input)


def binary_heatmap_loss(preds, targs, pos_weight=None, topk=9):
	preds = TensorBase(preds)
	targs = TensorBase(targs).float()
	if pos_weight is not None:
		_,p,h,w=preds.shape
		pos_weight=torch.tensor(pos_weight, device=preds.device).expand(p, h, w)
	return F.binary_cross_entropy_with_logits(preds, targs, pos_weight=pos_weight)



def nmae_topk(preds, targs, topk=9):
	# Note that our function is passed two heat maps, which we'll have to
	# decode to get our points. Adding one and dividing by 2 puts us
	# in the range 0...1 so we don't have to rescale for our percentual change.
	preds = 0.5 * (TensorBase(topk_heatmap2coord(preds, topk=topk, scale=True)) + 1)
	targs = 0.5 * (TensorBase(heatmap2argmax(targs, scale=True)) + 1)

	return ((preds-targs).abs()).mean()


def draw_prediction(img, keypoints, labels, img_size, size=4, font_size=15):
	'''
	Draws the predicted keypoints of an image over it.

	Args:
		i (int): The index of the image to use.
		size (int): Radius of the points that will be drawn (default: 4).
		font_size (int): Size of the font for the annotations (default: 15).

	Returns:
		temp (PIL.Image): The annotated image.
	'''
	if isinstance(img, np.ndarray):
		img = Image.fromarray(img).resize(img_size)
	elif isinstance(img, str) or isinstance(img, pathlib.PosixPath):
		img = Image.open(img).resize(img_size)
	temp = img.copy()
	draw = ImageDraw.Draw(temp)
	font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", font_size)
	if keypoints.ndim == 1:
		kp = keypoints
		draw.ellipse((kp[0]-size, kp[1]-size, kp[0]+size, kp[1]+size), fill=(255,0,0))
	else:
		for tn, kp in zip(labels, keypoints):
			draw.ellipse((kp[0]-size, kp[1]-size, kp[0]+size, kp[1]+size), fill=(255,0,0))
			draw.text((kp[0], kp[1]), tn, font=font, fill='black', anchor='ma')
	return temp
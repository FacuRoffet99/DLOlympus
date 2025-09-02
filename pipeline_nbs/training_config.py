import torch
import albumentations
from types import SimpleNamespace

from fastai.callback.progress import ShowGraphCallback
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.wandb import WandbCallback
from fastai.data.block import DataBlock
from fastai.data.transforms import ColReader, Normalize, ColSplitter
from fastai.losses import LabelSmoothingCrossEntropyFlat, MSELossFlat
from fastai.optimizer import Adam
from fastai.vision.data import ImageBlock, CategoryBlock, RegressionBlock

from DLOlympus.fastai.transforms import AlbumentationsTrainTransform, AlbumentationsValidTransform
from DLOlympus.fastai.metrics import AccuracyMetric, F1ScoreMetric, MSEMetric, MAEMetric
from DLOlympus.fastai.losses import SummedWeightedLoss
from DLOlympus.fastai.models import MultiheadModel, multihead_splitter, create_backbone, create_head


# ------------------------------- METRICS -------------------------------

metrics = [
	AccuracyMetric(axis=[0], metric_name='acc_breed'), 
	AccuracyMetric(axis=[1], metric_name='acc_animal'),
	F1ScoreMetric(axis=0, metric_name='f1_breed'),
	F1ScoreMetric(axis=1, metric_name='f1_animal'),
	AccuracyMetric(axis=[0, 1], metric_name='acc_multi'),
	MSEMetric(axis=2, root=True, metric_name='rmse'),
	MAEMetric(axis=2),
]

# ------------------------------- CALLBACKS -------------------------------

callbacks = [
	SaveModelCallback(monitor='acc_multi', with_opt=True), 
	ShowGraphCallback,
    WandbCallback(log=None, log_preds=False, log_model=False),
]

# ------------------------------- LOSS -------------------------------

class_weights_config = [
    {'type': 'IFW', 'axis': 0},
    {'type': 'ENoS', 'axis': 1, 'k': 0.99},
    None,
]

loss = SummedWeightedLoss(
	loss_functions=[LabelSmoothingCrossEntropyFlat(reduction='none'), LabelSmoothingCrossEntropyFlat(reduction='none'), MSELossFlat(reduction='none')],
	loss_weights=[1, 1, 0.01],
)

# ------------------------------- OPTIMIZER -------------------------------

optimizer = Adam

# ------------------------------- MODEL -------------------------------

pretrained = True
bbone, nf, stats_mean, stats_std = create_backbone(arch='resnet34', arch_type='torchvision', pretrained=pretrained, n_in=3)
head_breed = create_head(nf, 36)
head_animal = create_head(nf, 2)
head_group = create_head(nf, 1, y_range=(0,280))
model = MultiheadModel(bbone, head_breed, head_animal, head_group)
if torch.cuda.is_available():
    model = model.cuda()

# ------------------------------- SPLITTER -------------------------------

splitter = multihead_splitter

# ------------------------------- TRANSFORMS -------------------------------

h, w = 64, 64

train_transforms = albumentations.Compose([
	# 1) Crop & resize -> MANDATORY
	albumentations.RandomResizedCrop(size=(h,w), scale=(0.7, 1.0), ratio=(w/h, w/h), p=1.0),
	
	# 2) Geometric -> MANDATORY, SELECT ONE
	albumentations.HorizontalFlip(p=0.5), # Option 2A: natural images
	# albumentations.SquareSymmetry(p=0.5), # Option 2B: rotational symmetry
	
	# 3) Occlusions -> HIGH IMPACT
	albumentations.OneOf([
		albumentations.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2), p=1.0),
		albumentations.GridDropout(ratio=0.3, p=1.0)
	], p=0.5),            

	# # 4) Color space reduction -> ONLY IF NECESSARY
	# albumentations.OneOf([
	# 	albumentations.ToGray(p=1.0),
	# 	albumentations.ChannelDropout(channel_drop_range=(1, 1), p=1.0)
	# ], p=0.1),

	# 5) Lighting -> HIGH IMPACT
	albumentations.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
	
	# # 6) Blur -> ONLY IF NECESSARY
	# albumentations.OneOf([
	# 	albumentations.GaussianBlur(blur_limit=(3, 7), p=1.0),
	# 	albumentations.MotionBlur(blur_limit=(3, 7), p=1.0),
	# ], p=0.2),
])

valid_transforms = albumentations.Compose([
	# 1) Deterministic resize & crop -> MANDATORY
	albumentations.SmallestMaxSize(max_size=max(h,w), p=1.0),
	albumentations.CenterCrop(height=h, width=w, p=1.0),
])


# ------------------------------- DATABLOCK -------------------------------

datablock = DataBlock(
	blocks=(ImageBlock, CategoryBlock, CategoryBlock, RegressionBlock(n_out=1)),
	n_inp=1,
	get_x=ColReader('file_path'),
	get_y=[ColReader(i) for i in ['label_breed', 'label_animal', 'group']],
	splitter=ColSplitter(col='is_valid'),
    item_tfms=[AlbumentationsTrainTransform(train_transforms), AlbumentationsValidTransform(valid_transforms)],
    batch_tfms=[Normalize.from_stats(stats_mean, stats_std)],
)

# ------------------------------- FULL CONFIGS -------------------------------

config = SimpleNamespace(
    DATABLOCK = datablock,
	METRICS = metrics,
	CALLBACKS = callbacks,
    CLASS_WEIGHTS_CONFIGS = class_weights_config,
    OPTIMIZER = optimizer,
	LOSS = loss,
	MODEL = model,
    SPLITTER = splitter,
    PRETRAINED = pretrained,
    
	BS = 32,
	EPOCHS = 4,
	WD = None,
	OVERSAMPLING_LABEL = 'label_breed',
	SEED = 18,
)
import numpy as np

from fastai.vision.all import DisplayedTransform, PILImage, store_attr


class AlbumentationsTrainTransform(DisplayedTransform):
    split_idx,order=0,2
    def __init__(self, train_aug): store_attr()
    
    def encodes(self, img: PILImage):
        aug_img = self.train_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)
    
class AlbumentationsValidTransform(DisplayedTransform):
    split_idx,order=1,2
    def __init__(self, valid_aug): store_attr()
    
    def encodes(self, img: PILImage):
        aug_img = self.valid_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)
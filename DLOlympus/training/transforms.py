from fastai.vision.all import *


class AlbumentationsTransform(DisplayedTransform):
    '''
    Class that allows the use of Albumentations transforms in FastAI.
    '''

    split_idx,order=0,2
    def __init__(self, train_aug): store_attr()

    def encodes(self, img: PILImage):
        aug_img = self.train_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)
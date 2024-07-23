import numpy as np
import pandas as pd
import torch
from fastai.vision.all import *
from sklearn.utils.class_weight import compute_class_weight

def get_weights(dls):
    y = L(map(lambda x: int(x[1]), dls.train_ds))
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    weights /= weights.min()
    weights = (weights-1)/100 + 1
    # weights = np.ones_like(weights)
    print({c: w for c,w in zip(dls.vocab, weights)})
    weights = torch.FloatTensor(weights).to(dls.device)
    return weights

def oversampled_epoch(self):
    item_weights = self.items.label.apply(lambda x: pd.DataFrame(1 / np.sqrt(self.items.label.value_counts())).to_dict()['count'][x])
    oversampled_idxs = self.items.sample(n=self.n, weights=item_weights, replace=True).index
    return [np.where(self.items.index == i)[0][0] for i in oversampled_idxs]
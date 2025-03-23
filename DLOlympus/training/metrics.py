from fastai.vision.all import *

class MSEMetric(Metric):
    '''
    MSE metric for training in FastAI.
 
    Args:
        axis (int): Axis to compute MSE for multi-task models (-1 for single-task).
        output_pos (int): Output position of predictions for custom models with multiple outputs.
        count_type (str): 'bs' to average over batch size, 'all' to average over every dimension.
        root (bool): Whether to take the square root of the MSE
        metric_name (str): Name of the metric to display.
    '''
    def __init__(self, 
                 axis=-1, 
                 output_pos=None, 
                 count_type='bs', 
                 root=False, 
                 metric_name='mse'):
        self.__name__ = metric_name
        self.axis = axis 
        self.output_pos = output_pos
        self.count_type = count_type
        self.root = root
    def reset(self):
        self.total = 0.
        self.count = 0
    def accumulate(self, learn):
        targs = learn.y[:, self.axis]
        preds = learn.pred[self.output_pos][:, self.axis] if self.output_pos is not None else learn.pred[:, self.axis]
        self.total += (preds - targs).pow(2).sum().item()
        self.count += targs.shape[0] if self.count_type == 'bs' else torch.prod(tensor(targs.shape))
    @property
    def value(self):
        if self.count == 0: return None
        mse = self.total / self.count
        return np.sqrt(mse) if self.root else mse      

class MAEMetric(Metric):
    '''
    MAE metric for training in FastAI.
 
    Args:
        axis (int): Axis to compute MAE for multi-task models (-1 for single-task).
        output_pos (int): Output position of predictions for custom models with multiple outputs.
        metric_name (str): Name of the metric to display.
    '''
    def __init__(self, 
                 axis=-1, 
                 output_pos=None, 
                 metric_name='mae'):
        self.__name__ = metric_name
        self.axis = axis 
        self.output_pos = output_pos
    def reset(self):
        self.total = 0.
        self.count = 0
    def accumulate(self, learn):
        targs = learn.y[:, self.axis]
        preds = learn.pred[self.output_pos][:, self.axis] if self.output_pos is not None else learn.pred[:, self.axis]
        self.total += (preds - targs).abs().sum().item()
        self.count += targs.shape[0]
    @property
    def value(self):
        if self.count == 0: return None
        mae = self.total / self.count
        return mae  

class AccuracyMetric(Metric):
    '''
    Accuracy metric for training in FastAI.
 
    Args:
        axis (int): Axis to compute accuracy for multi-task models (-1 for single-task).
        probs_pos (int or slice): Index to slice probabilities for multi-task models.
        metric_name (str): Name of the metric to display.
    '''
    def __init__(self, axis=-1, multi=False, metric_name='accuracy'):
        self.__name__ = metric_name
        self.axis = axis 
        self.multi = multi
    def reset(self):
        self.total = 0.
        self.count = 0  
    def accumulate(self, learn):
        n_classes = [learn.dls.c] if isinstance(learn.dls.c, int) else learn.dls.c
        # Iterate over all tasks if multi, else iterate over the axis task
        ids = range(len(n_classes)) if self.multi else [self.axis]
        for i in ids:
            targs = learn.y[i]
            probs = learn.pred[:, sum(n_classes[:i]):sum(n_classes[:i+1])]
            preds = probs.argmax(dim=1)
            self.total += (preds == targs).sum().item()
            self.count += targs.shape[0]   

    @property
    def value(self):
        return self.total / self.count if self.count > 0 else None

class F1ScoreMetric(Metric):
    '''
    F1-score metric for training in FastAI.

    Args:
        average (str): 'macro' to compute F1 per class and average them, 'micro' to compute global F1.
        axis (int): Axis to compute F1 for multi-task models (-1 for single-task).
        metric_name (str): Name of the metric to display.
    '''
    def __init__(self, average='macro', axis=-1, probs_pos=-1, metric_name='f1_score'):
        self.__name__ = metric_name
        self.axis = axis
        self.average = average
    def reset(self):
        self.count = 0
        self.unique_classes = [] 
        self.tp, self.fp, self.fn = {}, {}, {}
    def accumulate(self, learn):
        n_classes = [learn.dls.c] if isinstance(learn.dls.c, int) else learn.dls.c
        targs = learn.y[self.axis]
        probs = learn.pred[:, sum(n_classes[:self.axis]):sum(n_classes[:self.axis+1])]
        preds = probs.argmax(dim=1)
        if not self.unique_classes:
            self.unique_classes = torch.unique(targs) # Get all class labels in dataset        
        for cls in self.unique_classes:
            cls = cls.item()
            # Init dicts for each class
            if cls not in self.tp: self.tp[cls] = 0
            if cls not in self.fp: self.fp[cls] = 0
            if cls not in self.fn: self.fn[cls] = 0             
            # Convert to a binary problem for each class
            targ_mask = (targs == cls)  
            pred_mask = (preds == cls)
            self.tp[cls] += (targ_mask & pred_mask).sum().item()
            self.fp[cls] += (~targ_mask & pred_mask).sum().item()
            self.fn[cls] += (targ_mask & ~pred_mask).sum().item()
        self.count += targs.shape[0]
    @property
    def value(self):
        if self.count == 0:
            return None
        if self.average == 'micro':
            tp, fp, fn = sum(self.tp.values()), sum(self.fp.values()), sum(self.fn.values())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            return f1
        elif self.average == 'macro':
            f1_scores = []
            for cls in self.unique_classes:
                tp, fp, fn = self.tp[cls], self.fp[cls], self.fn[cls]
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                f1_scores.append(f1)
            return torch.tensor(f1_scores).mean().item()
import torch, sys
from fastai.metrics import accuracy
from fastai.torch_core import flatten_check

# Helper function to create each loss function
def create_loss_func(loss, start, end, idx):
    def loss_func(inp, *args):
        return loss(inp[:, start:end], args[idx])
    return loss_func

# Helper function to create each accuracy function
def create_acc_func(start, end, idx):
    def acc(probs, *gts):
        return accuracy(probs[:, start:end], gts[idx])
    return acc

def multi_acc(probs, *gts, axis=-1, n_classes=None):
    # Initialize a list to store the prediction results for each task
    preds_matches = []

    # Start index for slicing
    start_idx = 0
    
    # Loop over each task
    for i in range(len(gts)):
        # Calculate the end index for the current task
        end_idx = start_idx + n_classes[i]
        
        # Get the predicted and ground truth labels for the current task
        pred, gt = flatten_check(probs[:, start_idx:end_idx].argmax(dim=axis), gts[i])
        
        # Check if predictions match ground truths
        preds_matches.append(pred == gt)
        
        # Update the start index for the next task
        start_idx = end_idx
    
    # Combine all task matches using logical AND, then calculate the mean accuracy
    return torch.stack(preds_matches).all(dim=0).float().mean()
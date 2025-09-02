from types import SimpleNamespace
import pytest
import torch
import numpy as np

from src.DLOlympus.fastai.metrics import (
    MSEMetric,
    MAEMetric,
    AccuracyMetric,
    F1ScoreMetric
)

# Helper function to simulate a training loop for a metric
def run_metric_on_batches(metric, batches):
    """Resets, accumulates batches, and returns the final metric value."""
    metric.reset()
    for learn in batches:
        metric.accumulate(learn)
    return metric.value

# Mock Learner factory
def mock_learn(preds, targs):
    """Creates a lightweight mock Learner object."""
    return SimpleNamespace(pred=preds, y=targs)


# #############################
#  Tests for Regression Metrics
# #############################

@pytest.mark.parametrize("average, root, expected", [
    ('batch', False, 0.05), # MSE
    ('all',   False, 0.025), # MSE over all elements
    ('batch', True,  0.223607), # RMSE
])
def test_mse_metric(average, root, expected):
    preds = torch.tensor([[1.1, 2.3], [2.9, 4.2]])
    targs = torch.tensor([[1.0, 2.5], [3.0, 4.0]])
    batches = [mock_learn(preds, targs)]
    
    metric = MSEMetric(average=average, root=root)
    value = run_metric_on_batches(metric, batches)
    
    assert np.isclose(value, expected, atol=1e-5)

@pytest.mark.parametrize("average, expected", [
    ('batch', 0.3), # MAE
    ('all',   0.15), # MAE over all elements
])
def test_mae_metric(average, expected):
    preds = torch.tensor([[1.1, 2.3], [2.9, 4.2]])
    targs = torch.tensor([[1.0, 2.5], [3.0, 4.0]])
    batches = [mock_learn(preds, targs)]
    
    metric = MAEMetric(average=average)
    value = run_metric_on_batches(metric, batches)
    
    assert np.isclose(value, expected, atol=1e-5)

# #############################
#  Tests for Classification Metrics
# #############################

def test_accuracy_single_task():
    preds = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.6, 0.4]])
    targs = torch.tensor([0, 1, 0, 0])
    batches = [mock_learn(preds, targs)]
    
    metric = AccuracyMetric()
    value = run_metric_on_batches(metric, batches)
    
    assert value == 0.75

def test_accuracy_multi_task():
    # Sample 0: correct on task 0, wrong on task 1
    # Sample 1: correct on task 0, correct on task 1 -> JOINTLY CORRECT
    # Sample 2: wrong on task 0, correct on task 1
    # Expected: 1 jointly correct / 3 samples = 0.333...
    preds = (
        torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]), # Task 0 preds
        torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.1, 0.9]])  # Task 1 preds
    )
    targs = (
        torch.tensor([0, 0, 1]), # Task 0 targs
        torch.tensor([1, 1, 1])  # Task 1 targs
    )
    batches = [mock_learn(preds, targs)]
    
    metric = AccuracyMetric(axis=[0, 1])
    value = run_metric_on_batches(metric, batches)
    
    assert np.isclose(value, 1/3)


@pytest.mark.parametrize("average, expected", [
    ('macro', 0.555555),
    ('micro', 0.666666)
])
def test_f1_score_multi_batch(average, expected):
    # Batch 1:
    b1_preds = torch.tensor([[.7, .3], [.3, .7], [.3, .7], [.3, .7], [.3, .7], [.3, .7]])
    b1_targs = torch.tensor([0, 0, 0, 1, 1, 1])

    # Batch 2:
    b2_preds = torch.tensor([[.3, .7], [.3, .7], [.3, .7], [.7, .3], [.3, .7], [.3, .7]])
    b2_targs = torch.tensor([1, 0, 1, 1, 1, 1])
    
    batches = [mock_learn(b1_preds, b1_targs), mock_learn(b2_preds, b2_targs)]
    
    metric = F1ScoreMetric(average=average)
    value = run_metric_on_batches(metric, batches)
    
    assert np.isclose(value, expected, atol=1e-5)
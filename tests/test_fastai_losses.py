import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from src.DLOlympus.fastai.losses import SummedWeightedLoss

def test_summedweightedloss_forward_calculation():
    """
    Verifies that the forward pass correctly computes the weighted sum of losses.
    This test uses inner losses with `reduction='none'` to properly test the
    final reduction step of SummedWeightedLoss.
    """
    # Arrange
    # Use standard PyTorch losses where we can predict the output
    # IMPORTANT: Use reduction='none' to get per-item losses first
    loss1 = nn.MSELoss(reduction='none')  # (pred - targ)^2
    loss2 = nn.L1Loss(reduction='none')   # |pred - targ|
    loss_weights = [0.6, 0.4]

    preds_tuple = (torch.tensor([1., 2., 3.]), torch.tensor([5., 5., 5.]))
    targs_tuple = (torch.tensor([1., 3., 5.]), torch.tensor([4., 6., 5.]))
    
    expected_mean_loss = torch.tensor(1.2666666)

    # Act
    loss_func_mean = SummedWeightedLoss([loss1, loss2], loss_weights, reduction='mean')
    mean_result = loss_func_mean(preds_tuple, *targs_tuple)

    loss_func_sum = SummedWeightedLoss([loss1, loss2], loss_weights, reduction='sum')
    sum_result = loss_func_sum(preds_tuple, *targs_tuple)

    # Assert
    assert torch.allclose(mean_result, expected_mean_loss)
    assert torch.allclose(sum_result, torch.tensor(3.8))


@pytest.mark.parametrize("method_name", ["activation", "decodes"])
def test_summedweightedloss_delegates_methods(method_name):
    """
    Tests that `activation` and `decodes` methods correctly delegate the call
    to the inner loss functions with the corresponding predictions.
    """
    # Arrange
    # Create mock loss functions with mockable methods
    mock_loss1 = MagicMock()
    mock_loss2 = MagicMock()
    
    # Configure the return value of the method we are testing
    getattr(mock_loss1, method_name).return_value = "output1"
    getattr(mock_loss2, method_name).return_value = "output2"
    
    loss_functions = [mock_loss1, mock_loss2]
    preds_tuple = (torch.randn(3, 4), torch.randn(3, 5))
    
    loss_func = SummedWeightedLoss(loss_functions, loss_weights=[1, 1])

    # Act
    # Call the method on our SummedWeightedLoss instance (e.g., loss_func.activation(...))
    result = getattr(loss_func, method_name)(preds_tuple)

    # Assert
    # Check that the mock method of each inner loss was called exactly once
    getattr(mock_loss1, method_name).assert_called_once_with(preds_tuple[0])
    getattr(mock_loss2, method_name).assert_called_once_with(preds_tuple[1])
    
    # Check that the results are correctly collected in a tuple
    assert result == ("output1", "output2")
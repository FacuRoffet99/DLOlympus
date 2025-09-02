import pytest
import torch
import torch.nn as nn
from fastai.vision.all import TransformBlock, TimmBody, SigmoidRange, LinBnDrop, AdaptiveConcatPool2d

from src.DLOlympus.fastai.models import (
    create_backbone,
    MultiheadModel,
    multihead_splitter,
)

# ######################################
#  Tests for Backbone Creators
# ######################################

class MockModel(nn.Module):
    """A mock model to simulate timm and torchvision models."""
    def __init__(self, num_features=128):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 32, 3), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10) # The part that should be "cut"
        self.num_features = num_features # For timm compatibility
        self.default_cfg = {'mean': (0.5,), 'std': (0.5,)} # For timm compatibility

    def forward(self, x):
        return self.fc(self.avgpool(self.features(x)))

@pytest.mark.parametrize("arch_type", ["timm", "torchvision"])
def test_create_backbone(mocker, arch_type):
    """
    Tests that create_backbone correctly calls the appropriate helper
    and returns the expected types, using mocks to prevent network calls.
    """
    # Arrange
    if arch_type == "timm":
        # Patch the function that actually creates the model
        mocker.patch('timm.create_model', return_value=MockModel())
    else: # torchvision
        # Patch the model class itself
        mocker.patch('torchvision.models.resnet18', return_value=MockModel())
        
    # Act
    bbone, nf, stats_mean, stats_std = create_backbone(
        arch='resnet18', arch_type=arch_type, pretrained=False
    )
    
    # Assert
    assert isinstance(bbone, nn.Module)
    assert isinstance(nf, int)
    assert isinstance(stats_mean, (list, tuple)) or stats_mean is None
    assert isinstance(stats_std, (list, tuple)) or stats_std is None
    if arch_type == "timm":
        assert nf == 128 # From our MockModel

def test_create_backbone_raises_error_on_invalid_type():
    """Ensures a ValueError is raised for an unsupported arch_type."""
    with pytest.raises(ValueError):
        create_backbone(arch='resnet18', arch_type='invalid_type')


# ######################################
#  Tests for MultiheadModel and Splitter
# ######################################

@pytest.fixture
def dummy_multihead_model():
    """Provides a simple MultiheadModel for testing."""
    backbone = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
    head1 = nn.Sequential(nn.Linear(20, 5)) # Classification head
    head2 = nn.Sequential(nn.Linear(20, 1)) # Regression head
    return MultiheadModel(backbone, head1, head2)

def test_multihead_model_forward_multi_head(dummy_multihead_model):
    """Tests that forward returns a tuple for multiple heads."""
    inp = torch.randn(4, 10)
    out = dummy_multihead_model(inp)
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (4, 5)
    assert out[1].shape == (4, 1)

def test_multihead_model_forward_single_head():
    """Tests that forward returns a single tensor for a single head."""
    backbone = nn.Sequential(nn.Linear(10, 20))
    head = nn.Sequential(nn.Linear(20, 5))
    model = MultiheadModel(backbone, head)
    
    inp = torch.randn(4, 10)
    out = model(inp)
    
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 5)

def test_multihead_splitter(dummy_multihead_model):
    """Tests that the splitter correctly separates backbone and head parameters."""
    splitter_result = multihead_splitter(dummy_multihead_model)
    
    backbone_params = list(dummy_multihead_model.backbone.parameters())
    heads_params = list(dummy_multihead_model.heads.parameters())
    
    assert len(splitter_result) == 2
    assert len(splitter_result[0]) == len(backbone_params)
    assert len(splitter_result[1]) == len(heads_params)
    # Check if the parameter objects are the same
    assert all(p1 is p2 for p1, p2 in zip(splitter_result[0], backbone_params))
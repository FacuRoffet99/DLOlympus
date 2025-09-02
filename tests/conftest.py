import dill
import pytest
import torch
import pandas as pd
import numpy as np
from torch import nn
from fastai.vision.all import ColReader, DataBlock, Learner, TransformBlock, CategoryBlock, RegressionBlock

from src.DLOlympus.fastai.models import MultiheadModel 

@pytest.fixture(scope="session")
def single_output_learner_path(tmp_path_factory):
    """Exports a dummy single-output Learner and returns the file path."""
    df = pd.DataFrame({
        'data': [torch.randn(1, 8, 8) for _ in range(10)],
        'label': np.random.choice(['A', 'B', 'C'], 10)
    })
    dblock = DataBlock(
        blocks=(TransformBlock, CategoryBlock),
        get_x=ColReader('data'), get_y=ColReader('label'),
        splitter=lambda _: (range(len(df)), range(len(df)))
    )
    dls = dblock.dataloaders(df, bs=4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(8*8, len(dls.vocab)))
    learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss())
    
    model_path = tmp_path_factory.mktemp("models") / "single_output_model.pkl"
    learn.export(model_path, pickle_module=dill)
    return model_path

@pytest.fixture(scope="session")
def multi_output_learner_path(tmp_path_factory):
    """
    Exports a dummy multi-output Learner (2 classification, 1 regression)
    and returns the file path.
    """
    df = pd.DataFrame({
        'data': [torch.randn(1, 8, 8) for _ in range(10)],
        'label1': np.random.choice(['a', 'b'], 10),
        'label2': np.random.choice(['X', 'Y', 'Z'], 10),
        'label3': [np.random.rand(1, 2).astype(np.float32)for i in range(10)] # Regression target
    })
    dblock = DataBlock(
        blocks=(TransformBlock, CategoryBlock, CategoryBlock, RegressionBlock(n_out=2)),
        get_x=ColReader('data'),
        get_y=[ColReader('label1'), ColReader('label2'), ColReader('label3')],
        n_inp=1,
        splitter=lambda _: (range(len(df)), range(len(df)))
    )
    dls = dblock.dataloaders(df, bs=4)
    
    from src.DLOlympus.fastai.models import MultiheadModel 
    backbone = nn.Sequential(nn.Flatten(), nn.Linear(8*8, 16))
    head1 = nn.Linear(16, 2)
    head2 = nn.Linear(16, 3)
    head3 = nn.Linear(16, 2)
    model = MultiheadModel(backbone, head1, head2, head3)
    
    learn = Learner(dls, model, loss_func=nn.MSELoss()) # Loss func no importa para el test
    
    model_path = tmp_path_factory.mktemp("models") / "multi_output_model.pkl"
    learn.export(model_path, pickle_module=dill)
    return model_path

@pytest.fixture(params=[
    pytest.param("single", id="Single-Output"),
    pytest.param("multi", id="Multi-Output")
])
def learner_scenario(request, single_output_learner_path, multi_output_learner_path):
    """Meta-fixture to provide different learner scenarios for parametrization."""
    if request.param == "single":
        return {
            "path": single_output_learner_path,
            "n_outputs": 1,
            "mock_decoded_preds": (torch.tensor([2, 1, 0])), # C, B, A
            "expected_postprocessed": (['C', 'B', 'A'],)
        }
    elif request.param == "multi":
        return {
            "path": multi_output_learner_path,
            "n_outputs": 3,
            "mock_decoded_preds": (
                torch.tensor([1, 0]), # -> b, a
                torch.tensor([2, 1]), # -> Z, Y
                torch.tensor([[0.1, 0.2], [0.3, 0.4]]) # Regression output
            ),
            "expected_postprocessed": (
                ['b', 'a'], 
                ['Z', 'Y'], 
                torch.tensor([[0.1, 0.2], [0.3, 0.4]])
            )
        }
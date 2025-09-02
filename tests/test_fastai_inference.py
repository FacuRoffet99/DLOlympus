import pytest
import torch
import warnings

from src.DLOlympus.fastai.inference import FastAIInferencer

@pytest.fixture
def inferencer(learner_scenario):
    """Fixture to create an inferencer instance for the active scenario."""
    return FastAIInferencer(checkpoint_file=learner_scenario["path"])

class TestFastAIInferencerInit:
    def test_init_infers_functions(self, inferencer, learner_scenario):
        """
        Tests that postprocessing functions are correctly inferred for both
        single and multi-output learners.
        """
        # Act
        inferencer.init()
        
        # Assert
        assert inferencer.n_outputs == learner_scenario["n_outputs"]
        assert len(inferencer.preds_postprocessing_fns) == learner_scenario["n_outputs"]
        
        # Test one of the functions to ensure it's correct
        if learner_scenario["n_outputs"] == 1:
            # Test single-output vocab decoding: index 1 -> 'B'
            assert inferencer.preds_postprocessing_fns[0](1) == 'B'
        else:
            # Test multi-output: first function, index 0 -> 'a'
            assert inferencer.preds_postprocessing_fns[0](0) == 'a'
            # Test multi-output: third function (regression) should be an identity
            reg_out = torch.randn(2)
            assert torch.equal(inferencer.preds_postprocessing_fns[2](reg_out), reg_out)

    def test_init_warns_on_function_count_mismatch(self, inferencer, learner_scenario):
        """Tests that a warning is issued if the wrong number of functions is provided."""
        # Provide one more function than expected
        post_fns = [lambda x: x] * (learner_scenario["n_outputs"] + 1)
        
        with pytest.warns(Warning, match="The number of postprocessing functions given"):
            inferencer.init(preds_postprocessing_fns=post_fns)
        
        assert inferencer.preds_postprocessing_fns is None

class TestFastAIInferencerProcess:
    def test_process_with_inferred_postprocessing(self, inferencer, learner_scenario, mocker):
        """
        Tests the full process pipeline using the inferred postprocessing functions
        for both single and multi-output scenarios.
        """
        # Arrange
        inferencer.init() # Init with inferred functions
        
        # Mock the return of get_preds with data from our scenario
        raw = torch.randn(2, 3)
        expected_decoded = learner_scenario["mock_decoded_preds"]
        expected_outputs = learner_scenario["expected_postprocessed"]
        mocker.patch.object(inferencer.learn, 'get_preds', return_value=(raw, None, expected_decoded))
        
        # Act
        raw_preds, decoded_preds, outputs = inferencer.process([torch.randn(1, 8, 8), torch.randn(1, 8, 8)])
        
        # Assert
        assert torch.equal(raw_preds, raw)
        assert all(torch.equal(d1, d2) for d1, d2 in zip(decoded_preds, expected_decoded))
        print(outputs, expected_outputs)
        assert all(torch.equal(d1, d2) if isinstance(d1, torch.Tensor) else d1 == d2 for d1, d2 in zip(outputs, expected_outputs))

class TestFastAIInferencerExport:
    def test_export_calls_onnx_exporter(self, inferencer, learner_scenario, mocker):
        """Tests that export correctly configures ONNX for each learner scenario."""
        # Arrange
        mock_onnx_export = mocker.patch('torch.onnx.export')
        inferencer.init()
        
        # Act
        inferencer.export(input_size=(1, 8, 8), save_folder="/tmp/")
        
        # Assert
        mock_onnx_export.assert_called_once()
        call_kwargs = mock_onnx_export.call_args.kwargs
        
        expected_n_outputs = learner_scenario["n_outputs"]
        expected_output_names = [f'out{i}' for i in range(expected_n_outputs)]
        
        assert call_kwargs['output_names'] == expected_output_names
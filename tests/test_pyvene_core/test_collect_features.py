import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader

from experiments.pyvene_core import _collect_features, _delete_intervenable_model
from neural.model_units import AtomicModelUnit, Component
from neural.featurizers import Featurizer


class TestCollectFeatures:
    """Tests for the _collect_features function."""
    
    @pytest.fixture
    def mock_tiny_lm(self):
        """Create a mock LM pipeline."""
        pipeline = MagicMock()
        pipeline.model.device = "cpu"
        pipeline.tokenizer.pad_token_id = 0
        return pipeline
    
    @pytest.fixture 
    def model_units_list(self):
        """Create model units for testing."""
        # Create mock components
        comp1 = MagicMock()
        comp1.get_layer.return_value = 0
        comp1.index.return_value = [0, 1]
        
        comp2 = MagicMock()
        comp2.get_layer.return_value = 2
        comp2.index.return_value = [0, 1]
        
        # Create model units
        unit1 = AtomicModelUnit(
            component=comp1,
            featurizer=Featurizer(),
            id="ResidualStream(Layer:0,Token:last_token)"
        )
        unit2 = AtomicModelUnit(
            component=comp2,
            featurizer=Featurizer(),
            id="ResidualStream(Layer:2,Token:last_token)"
        )
        
        # Return nested structure: two groups, each with one unit
        return [[unit1], [unit2]]
    
    @pytest.fixture
    def mock_counterfactual_dataset(self):
        """Create a mock counterfactual dataset."""
        return [
            {"input": "input_1", "counterfactual_inputs": ["cf_1_1", "cf_1_2"]},
            {"input": "input_2", "counterfactual_inputs": ["cf_2_1", "cf_2_2"]},
            {"input": "input_3", "counterfactual_inputs": ["cf_3_1", "cf_3_2"]}
        ]
    
    @pytest.fixture
    def mock_intervenable_model(self):
        """Create a mock intervenable model."""
        model = MagicMock()
        
        # Mock the model return structure: (base_outputs, collected_activations), counterfactual_outputs
        # collected_activations should be a list of tensors
        def model_side_effect(*args, **kwargs):
            # Return 4 activation tensors (2 model units * 2 batch size)
            activations = [torch.randn(1, 32) for _ in range(4)]
            return (MagicMock(), activations), None
        
        model.side_effect = model_side_effect
        return model
    
    @pytest.fixture
    def mock_loaded_inputs(self):
        """Create mock loaded inputs from the pipeline."""
        base_loaded = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])
        }
        
        cf_loaded = [
            {
                "input_ids": torch.tensor([[7, 8, 9], [10, 11, 12]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])
            },
            {
                "input_ids": torch.tensor([[13, 14, 15], [16, 17, 18]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])
            }
        ]
        
        return base_loaded, cf_loaded
    
    def test_basic_feature_collection(self, mock_tiny_lm, model_units_list, mock_counterfactual_dataset,
                                     mock_intervenable_model, mock_loaded_inputs):
        """Test basic feature collection functionality."""
        base_loaded, cf_loaded = mock_loaded_inputs
        
        # Mock the prepare_intervenable_inputs function to avoid tokenizer errors
        with patch('experiments.pyvene_core._prepare_intervenable_model',
                  return_value=mock_intervenable_model) as mock_prepare, \
             patch('experiments.pyvene_core._prepare_intervenable_inputs',
                   return_value=(base_loaded, cf_loaded,
                                {"sources->base": ([[[0, 1]], [[0, 1]]], [[[0, 1]], [[0, 1]]])},
                                [[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]])) as mock_inputs:
            
            # Create a simple config
            config = {"batch_size": 2}
            
            # Call the function
            result = _collect_features(
                mock_counterfactual_dataset,
                mock_tiny_lm,
                model_units_list,
                config,
                verbose=False
            )
            
            # Verify that _prepare_intervenable_model was called with "collect" intervention type
            mock_prepare.assert_called_once_with(mock_tiny_lm, model_units_list, intervention_type="collect")
            
            # Verify the result structure: list of lists of tensors
            assert isinstance(result, list)
            assert len(result) == len(model_units_list)  # Should match number of model unit groups
            
            # Each result should be a list containing tensors for each model unit in the group
            for group_result in result:
                assert isinstance(group_result, list)
                for tensor in group_result:
                    assert isinstance(tensor, torch.Tensor)
    
    def test_collect_counterfactuals_flag(self, mock_tiny_lm, model_units_list, mock_counterfactual_dataset,
                                        mock_loaded_inputs):
        """Test the collect_counterfactuals flag functionality."""
        base_loaded, cf_loaded = mock_loaded_inputs
        
        # Create mock model that tracks how many times it's called
        call_count = 0
        def model_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            activations = [torch.randn(1, 32) for _ in range(4)]
            return (MagicMock(), activations), None
        
        mock_model = MagicMock(side_effect=model_side_effect)
        
        with patch('experiments.pyvene_core._prepare_intervenable_model',
                  return_value=mock_model), \
             patch('experiments.pyvene_core._prepare_intervenable_inputs',
                   return_value=(base_loaded, cf_loaded,
                                {"sources->base": ([[[0, 1]], [[0, 1]]], [[[0, 1]], [[0, 1]]])},
                                [[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]])), \
             patch('experiments.pyvene_core._delete_intervenable_model'):
            
            config = {"batch_size": 2}
            
            # Test with collect_counterfactuals=True (default)
            call_count = 0
            _collect_features(
                mock_counterfactual_dataset,
                mock_tiny_lm,
                model_units_list,
                config,
                collect_counterfactuals=True
            )
            
            # Should be called more times when collecting counterfactuals
            calls_with_cf = call_count
            assert calls_with_cf > 0
            
            # Test with collect_counterfactuals=False
            call_count = 0
            _collect_features(
                mock_counterfactual_dataset,
                mock_tiny_lm,
                model_units_list,
                config,
                collect_counterfactuals=False
            )
            
            calls_without_cf = call_count
            
            # Should be called fewer times when not collecting counterfactuals
            assert calls_without_cf < calls_with_cf
    
    def test_verbose_output(self, mock_tiny_lm, model_units_list, mock_counterfactual_dataset,
                          mock_loaded_inputs, capsys):
        """Test verbose output functionality."""
        base_loaded, cf_loaded = mock_loaded_inputs
        
        mock_model = MagicMock()
        mock_model.side_effect = lambda *args, **kwargs: (
            (MagicMock(), [torch.randn(1, 32) for _ in range(4)]), None
        )
        
        with patch('experiments.pyvene_core._prepare_intervenable_model',
                  return_value=mock_model), \
             patch('experiments.pyvene_core._prepare_intervenable_inputs',
                   return_value=(base_loaded, cf_loaded,
                                {"sources->base": ([[[0, 1]], [[0, 1]]], [[[0, 1]], [[0, 1]]])},
                                [[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]])), \
             patch('experiments.pyvene_core._delete_intervenable_model'):
            
            config = {"batch_size": 2}
            
            # Call with verbose=True
            _collect_features(
                mock_counterfactual_dataset,
                mock_tiny_lm,
                model_units_list,
                config,
                verbose=True
            )
            
            # Check that diagnostic information was printed
            captured = capsys.readouterr()
            assert "Outer length:" in captured.out
            assert "Inner lengths:" in captured.out
            assert "Datum shape:" in captured.out
    
    def test_memory_management(self, mock_tiny_lm, model_units_list, mock_counterfactual_dataset,
                              mock_loaded_inputs):
        """Test that tensors are moved to CPU for memory efficiency."""
        base_loaded, cf_loaded = mock_loaded_inputs
        
        # Create mock model that returns tensors on a specific device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mock_model = MagicMock()
        
        # Custom side effect returning tensors on device
        mock_model.side_effect = lambda inputs, unit_locations=None, **kwargs: [
            (
                MagicMock(),
                [torch.randn(1, 32, device=device) for _ in range(4)]
            ),
            None
        ]
        
        with patch('experiments.pyvene_core._prepare_intervenable_model',
              return_value=mock_model) as mock_prepare, \
            patch('experiments.pyvene_core._prepare_intervenable_inputs',
              return_value=(base_loaded, cf_loaded,
                            {"sources->base": ([[[0, 1]], [[0, 1]]], [[[0, 1]], [[0, 1]]])},
                            [[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]])) as mock_inputs, \
            patch('torch.cuda.empty_cache') as mock_empty_cache:
            
            # Config
            config = {"batch_size": 2}
            
            # Call the function
            result = _collect_features(
                mock_counterfactual_dataset,
                mock_tiny_lm,
                model_units_list,
                config,
                verbose=False
            )
            
            # Verify all tensors are on CPU
            for group_result in result:
                for tensor in group_result:
                    assert tensor.device.type == "cpu"
    
    def test_dataloader_creation(self, mock_tiny_lm, model_units_list, mock_counterfactual_dataset,
                                mock_loaded_inputs):
        """Test that DataLoader is created with correct parameters."""
        base_loaded, cf_loaded = mock_loaded_inputs
        
        mock_model = MagicMock()
        mock_model.side_effect = lambda *args, **kwargs: (
            (MagicMock(), [torch.randn(1, 32) for _ in range(4)]), None
        )
        
        # Patch DataLoader at the module level where it's imported
        with patch('experiments.pyvene_core._prepare_intervenable_model',
                  return_value=mock_model), \
             patch('experiments.pyvene_core._prepare_intervenable_inputs',
                   return_value=(base_loaded, cf_loaded,
                                {"sources->base": ([[[0, 1]], [[0, 1]]], [[[0, 1]], [[0, 1]]])},
                                [[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]])), \
             patch('experiments.pyvene_core._delete_intervenable_model'), \
             patch('experiments.pyvene_core.DataLoader') as mock_dataloader:
            
            # Mock the DataLoader to return our mock dataset
            mock_dataloader.return_value = iter([
                {"input": ["input_1"], "counterfactual_inputs": [["cf_1_1", "cf_1_2"]]},
                {"input": ["input_2"], "counterfactual_inputs": [["cf_2_1", "cf_2_2"]]},
            ])
            
            config = {"batch_size": 2}
            
            # Call the function
            _collect_features(
                mock_counterfactual_dataset,
                mock_tiny_lm,
                model_units_list,
                config
            )
            
            # Verify DataLoader was called with correct parameters
            mock_dataloader.assert_called_once()
            call_args = mock_dataloader.call_args
            assert call_args[1]["batch_size"] == 2
            assert call_args[1]["shuffle"] == False
            # Verify the collate_fn is the shallow_collate_fn
            assert call_args[1]["collate_fn"].__name__ == "shallow_collate_fn"


class TestDeleteIntervenableModel:
    """Tests for _delete_intervenable_model function."""
    
    def test_basic_cleanup(self):
        """Test basic model cleanup functionality."""
        # Create a mock intervenable model
        mock_model = MagicMock()
        
        with patch('gc.collect') as mock_gc, \
             patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('torch.cuda.is_available', return_value=True):
            
            # Call the function
            _delete_intervenable_model(mock_model)
            
            # Verify cleanup steps
            mock_model.set_device.assert_called_once_with("cpu", set_model=False)
            mock_gc.assert_called_once()
            mock_empty_cache.assert_called_once()
    
    def test_cleanup_without_cuda(self):
        """Test cleanup when CUDA is not available."""
        mock_model = MagicMock()
        
        with patch('gc.collect') as mock_gc, \
             patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('torch.cuda.is_available', return_value=False):
            
            # Call the function
            _delete_intervenable_model(mock_model)
            
            # Verify cleanup steps
            mock_model.set_device.assert_called_once_with("cpu", set_model=False)
            mock_gc.assert_called_once()
            # FIXED: empty_cache should NOT be called when CUDA is not available
            # (based on the actual implementation in _delete_intervenable_model)
            mock_empty_cache.assert_not_called()


# Run tests when file is executed directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
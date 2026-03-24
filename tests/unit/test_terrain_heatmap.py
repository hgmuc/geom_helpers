import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from geom_helpers.tiles.terrain_heatmap import (
    preprocess_elev_arr, 
    export_raw_img, 
    save_img_as_tile
)

@pytest.fixture
def sample_elev_arr():
    """Provides a basic 10x10 elevation array."""
    arr = np.linspace(800, 1200, 100).reshape(10, 10)
    return arr.astype(np.float32)

## --- Tests for preprocess_elev_arr ---

def test_preprocess_elev_arr_logic(sample_elev_arr):
    # Values <= 1000 should remain unchanged
    # Values > 1000 should be reduced by (val - 1000) * 0.85
    lim = 1000
    fct = 0.85
    result = preprocess_elev_arr(sample_elev_arr, lim=lim, fct=fct)
    
    # Check a value that was originally 1200
    # Expected: 1200 - ((1200 - 1000) * 0.85) = 1200 - 170 = 1030
    assert np.isclose(result[-1, -1], 1030.0)
    # Check a value that was originally 800 (below lim)
    assert result[0, 0] == 800.0

def test_preprocess_elev_arr_immutability(sample_elev_arr):
    original_copy = sample_elev_arr.copy()
    _ = preprocess_elev_arr(sample_elev_arr)
    # Ensure the original input array wasn't mutated
    np.testing.assert_array_equal(sample_elev_arr, original_copy)

## --- Tests for export_raw_img ---

@patch("matplotlib.pyplot.savefig")
@patch("seaborn.heatmap")
def test_export_raw_img_calls_plotting(mock_heatmap, mock_savefig, sample_elev_arr):
    mock_ax = MagicMock()
    
    export_raw_img(
        elev_arr=sample_elev_arr, 
        zoom=10, x=5, y=5, 
        ax=mock_ax, 
        kernel=3
    )
    
    # Verify heatmap was called with our array
    mock_heatmap.assert_called_once()
    assert mock_heatmap.call_args[1]['vmin'] == -500
    
    # Verify filename construction
    mock_savefig.assert_called_once_with(
        'output310-5-5.png', bbox_inches='tight', pad_inches=0.0
    )

## --- Tests for save_img_as_tile ---

@patch("imageio.v2.imread")
@patch("PIL.Image.fromarray")
@patch("os.remove")
def test_save_img_as_tile_processing(mock_remove, mock_fromarray, mock_imread):
    # 1. Setup dummy RGBA image
    dummy_img = np.zeros((10, 10, 4), dtype=np.uint8)
    dummy_img[2:8, 2:8, 3] = 255  # Square of visible pixels
    mock_imread.return_value = dummy_img
    
    # 2. Setup Mocks for PIL chain
    mock_initial_img = MagicMock(spec=Image.Image)
    mock_resized_img = MagicMock(spec=Image.Image)
    
    mock_fromarray.return_value = mock_initial_img
    # Crucial: resize() must return the next object in the chain
    mock_initial_img.resize.return_value = mock_resized_img
    
    # 3. Execute
    save_img_as_tile("test_tile.png", zoom=1, x=1, y=1, kernel=1)
    
    # 4. Verify cropping logic
    args, _ = mock_fromarray.call_args
    cropped_result = args[0]
    assert cropped_result.shape == (6, 6, 4), "Image should be cropped to 6x6"
    
    # 5. Verify the chain
    mock_initial_img.resize.assert_called_once_with((256, 256))
    mock_resized_img.save.assert_called_once_with("test_tile.png")
    mock_resized_img.close.assert_called_once()
    
    # 6. Verify cleanup
    mock_remove.assert_called_once_with('output11-1-1.png')


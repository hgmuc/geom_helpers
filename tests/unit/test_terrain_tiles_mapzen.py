import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from geom_helpers.tiles.terrain_tiles_mapzen import get_elevation_data, create_terrain_tile_from_mapzen


class TestTerrainMapzen:
    def test_get_elevation_data_calculation(self, tmp_path):
        """
        Tests the Mapzen terrarium RGB-to-Elevation formula.
        Formula: (R * 256 + G + B / 256) - 32768
        """
        fpath = str(tmp_path / "dummy_tile.png")
        
        # Create a 2x2 RGB image manually
        # Pixel [0,0]: R=128 (0.5019 in float), G=0, B=0 
        # (128 * 256 + 0 + 0) - 32768 = 0 meters
        img_data = np.zeros((2, 2, 3), dtype=np.uint8)
        img_data[0, 0, 0] = 128 
        
        # Save as a standard RGB PNG
        plt.imsave(fpath, img_data)
        
        elevations = get_elevation_data(fpath)
        
        assert elevations.shape == (2, 2)
        assert elevations[0, 0] == 0
        # Check another pixel (should be -32768 if R,G,B are 0)
        assert elevations[1, 1] == -32768


    @patch("geom_helpers.tiles.terrain_tiles_mapzen.check_path")
    def test_create_terrain_tile_path_failure(self, mock_check):
        """Tests that if base path check fails, function returns False."""
        mock_check.return_value = False
        
        result = create_terrain_tile_from_mapzen(z=1, x=1, y=1, ax=MagicMock())
        
        assert result is False

class TestDataIntegration:
    """Tests using your specific folder structure in tests/data/."""
    
    def test_load_real_sample_data(self):
        # Path based on your description: tests/data/1/...
        # We assume the test is run from project root
        base_path = os.path.join("tests", "data", "1")
        
        # This test only runs if you have actually placed a file there
        # Let's check for a hypothetical y=0.png in zoom 1, x=0
        target_file = os.path.join(base_path, "0", "0.png")
        
        if os.path.exists(target_file):
            elev = get_elevation_data(target_file)
            assert elev.shape == (256, 256)
        else:
            pytest.skip("Sample data tile not found in tests/data/1/0/0.png")
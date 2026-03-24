import os
import pytest
from unittest.mock import Mock, patch

from math import isclose
from geom_helpers.tiles.xyz_tiles import deg2num, num2deg, make_folder, check_path, download_remote_file, write_file, load_file



# Constants for testing
ZOOM_LEVEL = 10
NULL_ISLAND_LAT = 0.0
NULL_ISLAND_LON = 0.0

class TestXYZTiles:
    
    @pytest.mark.parametrize("lat, lon, zoom, expected_x, expected_y", [
        (0.0, 0.0, 0, 0, 0),             # Zoom 0 covers the whole world in one tile
        (0.0, 0.0, 10, 512, 512),       # Null Island at Zoom 10
        (48.8584, 2.2945, 14, 8296, 5636), # Eiffel Tower coordinates
        (-33.8568, 151.2153, 12, 3768, 2457), # Sydney Opera House
    ])
    def test_deg2num_specific_points(self, lat, lon, zoom, expected_x, expected_y):
        """Tests that coordinates map to the correct tile indices."""
        x, y = deg2num(lat, lon, zoom)
        assert x == expected_x
        assert y == expected_y

    @pytest.mark.parametrize("x, y, zoom, expected_lat, expected_lon", [
        (0, 0, 0, 85.0511287798, -180.0),          # Top-left of the world at zoom 0
        (512, 512, 10, 0.0, 0.0),        # Center of the world at zoom 10
    ])
    def test_num2deg_specific_tiles(self, x, y, zoom, expected_lat, expected_lon):
        """Tests that tile indices map back to the correct top-left NW corner."""
        lat, lon = num2deg(x, y, zoom)
        assert isclose(lat, expected_lat, abs_tol=1e-7)
        assert isclose(lon, expected_lon, abs_tol=1e-7)

    def test_round_trip_conversion(self):
        """
        Tests that converting deg -> tile -> deg returns a coordinate 
        consistent with the tile's top-left corner.
        """
        lat_in, lon_in = 45.0, 45.0
        zoom = 12
        
        # Get tile
        xtile, ytile = deg2num(lat_in, lon_in, zoom)
        
        # Get corner back
        lat_out, lon_out = num2deg(xtile, ytile, zoom)
        
        # The output should be the NW corner of the tile containing the input
        assert lat_out >= lat_in if lat_in < 0 else lat_out >= 0
        assert lon_out <= lon_in

    @pytest.mark.parametrize("invalid_lat", [91, -91])
    def test_deg2num_extreme_latitudes(self, invalid_lat):
        """
        Note: Mercator projection technically fails at 90 degrees.
        This test checks how the math handles edge-of-world coordinates.
        """
        # asinh/tan will become extremely large near 90 deg.
        # This is just to ensure no uncaught ZeroDivisionError or similar.
        try:
            x, y = deg2num(invalid_lat, 0.0, 10)
            assert isinstance(x, int)
            assert isinstance(y, int)
        except (ValueError, ZeroDivisionError):
            # We use a standard python 'pass' here. 
            # Static checkers prefer this over pytest.pass() inside logic blocks.
            pass

# Gravigny, Balizy Station - 48,6858378, 2,3174900
# Eiffelturm               - 48,8582094, 2,2944151
# Test                     - 48.8584,    2.2945




class TestFolderManagement:
    def test_make_folder_creates_nested_dirs(self, tmp_path):
        # tmp_path is a built-in pytest fixture for a temporary directory
        root = str(tmp_path)
        x, z = 10, 5
        
        make_folder(x, z, root=root)
        
        assert os.path.exists(os.path.join(root, "5"))
        assert os.path.exists(os.path.join(root, "5", "10"))

    def test_check_path_valid_flow(self, tmp_path):
        root = str(tmp_path)
        x, z = 1, 2
        
        # Should return True and create folders
        result = check_path(x, z, root)
        
        assert result is True
        assert os.path.isdir(os.path.join(root, "2", "1"))

    def test_check_path_nonexistent_base(self):
        # Testing a path that definitely doesn't exist
        result = check_path(0, 0, "/non/existent/path/at/all")
        assert result is False

class TestNetworkOperations:
    @patch("requests.get")
    def test_download_remote_file_success(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"\x89PNG\r\n\x1a\n" # Mocking a PNG header
        mock_get.return_value = mock_response
        
        result = download_remote_file("http://example.com/tile.png")
        assert result == b"\x89PNG\r\n\x1a\n"    
        
    @patch("requests.get")
    def test_download_remote_file_success_old(self, mock_get):
        # Setup mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake tile data"
        mock_get.return_value = mock_response
        
        result = download_remote_file("http://example.com/tile.png")
        
        assert result == b"fake tile data"
        mock_get.assert_called_once_with("http://example.com/tile.png")

    @patch("requests.get")
    def test_download_remote_file_404(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = download_remote_file("http://example.com/missing.png")
        
        assert result is None

    @patch("requests.get")
    def test_download_remote_file_exception(self, mock_get):
        mock_get.side_effect = Exception("Connection Refused")
        
        result = download_remote_file("http://offline-site.com")
        
        assert result is None

class TestFileOperations:
    def test_write_and_load_file_bytes(self, tmp_path):
        """Tests the logic for actual tile-like binary data."""
        fpath = str(tmp_path / "test.tile")
        content = b"\x00\xFF\x00\xFF" # Example binary data
        
        # This will now pass the type checker
        assert write_file(content, fpath) is True
        
        # Verify loading returns bytes, not str
        loaded = load_file(fpath)
        assert isinstance(loaded, bytes)
        assert loaded == content

    def test_write_and_load_file_bytes2(self, tmp_path):
        fpath = str(tmp_path / "test.bin")
        content = b"hello world"
        
        assert write_file(content, fpath) is True
        
        # Verify loading (note: load_file decodes utf-8)
        loaded = load_file(fpath)
        assert loaded == b"hello world"

    def test_write_file_string_encoding(self, tmp_path):
        fpath = str(tmp_path / "test.txt")
        content = "hallo"
        
        write_file(content, fpath)
        
        with open(fpath, "rb") as f:
            assert f.read() == b"hallo"

    def test_write_file_string_auto_conversion(self, tmp_path):
        """Ensures strings are still converted to bytes before writing."""
        fpath = str(tmp_path / "test.txt")
        content = "UTF-8 String"
        
        write_file(content, fpath)
        
        # Should be readable as bytes
        assert load_file(fpath) == b"UTF-8 String"

    def test_write_file_none_returns_false(self, tmp_path):
        fpath = str(tmp_path / "none.txt")
        assert write_file(None, fpath) is False

    def test_write_file_none_content(self, tmp_path):
        fpath = str(tmp_path / "none.txt")
        assert write_file(None, fpath) is False

    def test_load_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_file(str(tmp_path / "does_not_exist.png"))

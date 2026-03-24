import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from geom_helpers.tiles.terrain_tiles_copernicus import (
    get_lonlat_str, 
    get_aws_copernius_file_url,
    get_srtm_file_names,
    get_elevation_data,
    get_slice,
    stretch_empty_arr,
    increase_arr_size,
    align_arr_size,
    EXISTING_SRTM, 
)

class TestCoordinateLogic:
    @pytest.mark.parametrize("lat, lon, expected_lat, expected_lon", [
        (60.7, -0.96, "N60", "W001"),  # Note: W000 is often used for 0 to -1
        (53.02, 0.1, "N53", "E000"),
        (39.03, 1.4, "N39", "E001"),
        # (-10.5, 12.0, "S10", "E012"), # Currently out-of-scope
        (0.5, -1.5, "N00", "W002"),
    ])
    def test_get_lonlat_str(self, lat, lon, expected_lat, expected_lon):
        """Validates the string formatting for AWS URLs."""
        lats, lons = get_lonlat_str(lat, lon)
        assert lats == expected_lat
        assert lons == expected_lon

    def test_url_generation(self):
        """Checks if the URL matches the Copernicus S3 pattern."""
        url = get_aws_copernius_file_url(res=90, lat="N60", lon="W001")
        # res 90 -> res_arcsec 30
        assert "copernicus-dem-90m" in url
        assert "Copernicus_DSM_COG_30_N60_00_W001_00_DEM.tif" in url

class TestSrtmMapping:
    @patch("geom_helpers.tiles.terrain_tiles_copernicus.EXISTING_SRTM", {})
    def test_get_srtm_file_names_quadrants(self):
        """
        Tests how a bounding box that crosses integer lines 
        is split into multiple required SRTM/Copernicus tiles.
        """
        # BBox: lat_min, lon_min, lat_max, lon_max
        # Crossing N47/48 and E8/9
        bbox = (47.9, 8.9, 48.1, 9.1)
        res = 30
        
        result = get_srtm_file_names(bbox, res)
        
        # Should expect 4 keys: (47,8), (47,9), (48,8), (48,9)
        expected_keys = [
            (47, 8, 30), (47, 9, 30),
            (48, 8, 30), (48, 9, 30)
        ]
        for key in expected_keys:
            assert key in result

    @patch("geom_helpers.tiles.terrain_tiles_copernicus.EXISTING_SRTM", {})
    def test_get_srtm_file_names_quadrants1(self):
        """
        Tests how a bounding box that crosses integer lines 
        is split into multiple required SRTM/Copernicus tiles.
        """
        # BBox: lat_min, lon_min, lat_max, lon_max
        # Crossing N47/48 and E8/9
        bbox = (47.9, 8.9, 47.99, 9.1)
        res = 30
        
        result = get_srtm_file_names(bbox, res)
        
        # Should expect 4 keys: (47,8), (47,9), (48,8), (48,9)
        expected_keys = [
            (47, 8, 30), (47, 9, 30),
        ]
        for key in expected_keys:
            assert key in result

    @patch("geom_helpers.tiles.terrain_tiles_copernicus.EXISTING_SRTM", {})
    def test_get_srtm_file_names_quadrants2(self):
        """
        Tests how a bounding box that crosses integer lines 
        is split into multiple required SRTM/Copernicus tiles.
        """
        # BBox: lat_min, lon_min, lat_max, lon_max
        # Crossing N47/48 and E8/9
        bbox = (47.9, 8.9, 48.1, 8.99)
        res = 90
        
        result = get_srtm_file_names(bbox, res)
        
        # Should expect 4 keys: (47,8), (47,9), (48,8), (48,9)
        expected_keys = [
            (47, 8, 90), (48, 8, 90), 
        ]
        for key in expected_keys:
            assert key in result


class TestDataLoading:
    @patch("geom_helpers.tiles.terrain_tiles_copernicus.rasterio.open")
    def test_get_elevation_data_mocked_rasterio(self, mock_rasterio_open, tmp_path):
        """Tests loading data using the Rasterio context manager pattern."""
        
        # Setup fake file
        fake_tif = tmp_path / "fake.tif"
        fake_tif.write_text("not a real tif")

        # 2. Setup the Rasterio Mock Chain
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset
        # src.read(1) returns the numpy array
        mock_dataset.read.return_value = np.array([[10, 20], [30, 40]])

        # Mock the EXISTING_SRTM dict for this test
        with patch("geom_helpers.tiles.terrain_tiles_copernicus.EXISTING_SRTM", 
                   {(47, 8, 90): str(fake_tif)}):
            
            result = get_elevation_data(47, 8, 90)

            assert result[0, 0] == 10
            # Verify read(1) was called (rasterio bands are 1-indexed)
            mock_dataset.read.assert_called_once_with(1)

    def test_handpicked_coordinates_alignment(self):
        """
        Verifies the specific coordinates provided in the prompt
        map to the expected integer keys used in EXISTING_SRTM.
        """
        coords = [(60.7, -0.96), (53.02, 0.1), (39.03, 1.4), (39.99, 0.01), (59.99, -1.3)]
        
        # Logic: lat is floored if positive, but what about negative?
        # get_lonlat_str uses int(lat), which truncates towards zero.
        # Let's verify our expectations match your file naming.
        
        results = [get_lonlat_str(lat, lon) for lat, lon in coords]
        
        # Expected based on your filenames:
        assert results[0] == ("N60", "W001") # (60.7, -0.96) 
        assert results[1] == ("N53", "E000") # (53.02, 0.1)
        assert results[2] == ("N39", "E001") # (39.03, 1.4)
        assert results[3] == ("N39", "E000") # (39.99, 0.01)
        assert results[4] == ("N59", "W002") # (59.99, -1.3)        

class TestArrayTransformations:
    def test_stretch_empty_arr_res90(self):
        """Tests that a 10x10 empty array is stretched correctly for 90m (fct 120)."""
        arr = np.ones((10, 10)) * -10
        dims = [0]
        new_arr, dim, w_arr, new_dims = stretch_empty_arr(arr, res=90, dims=dims)
        
        # 10 * 120 = 1200
        assert dim == 1200
        assert new_arr.shape == (1200, 1200)
        assert new_dims[-1] == 1200
        assert np.all(new_arr == -10)

    def test_increase_arr_size_res30(self):
        """Tests increasing 1200x12000 to 3600x3600 for high-res."""
        arr = np.random.rand(1200, 1200)
        dims = [0]
        new_arr, dim, w_arr, new_dims = increase_arr_size(arr, 1200, dims)
        
        assert dim == 3600
        assert new_arr.shape == (3600, 3600)

    def test_align_arr_size_pad(self):
        """Tests padding when an array is slightly smaller than expected."""
        arr = np.zeros((3600, 3600))
        dims = [3601] # Goal size
        new_arr, dim, w_arr, _ = align_arr_size(arr, 3600, dims)
        
        assert dim == 3601
        assert new_arr.shape == (3601, 3601)

class TestSlicingLogic:
    def test_get_slice_single_quadrant(self):
        """Tests slicing a single tile out of a full array (n_quads=1)."""
        # Create a 100x100 array where pixels represent (y, x)
        arr = np.fromfunction(lambda y, x: y * 100 + x, (600, 600))
        # BBox: (lat2, lon1, lat1, lon2) -> lat1 is top, lon1 is left
        # Let's target the middle 10%: lat 0.45 to 0.55, lon 0.45 to 0.55
        bbox = (0.45, 0.45, 0.55, 0.55)
        dims = [600]
        
        sliced = get_slice(arr, bbox, pos_idx=0, n_quads=1, 
                           SRTM_DATA_DICT={}, dims=dims, res=90)
        
        # Expected: y starts from top (1.0). 
        # y1 (bottom) = floor(1 - (0.45-0)*100) = 55
        # y2 (top) = ceil(1 - (0.55-0)*100) = 45
        assert sliced is not None
        assert sliced.shape == (61, 61) # Due to +1 in y1+1, x2+1

    def test_get_slice_negative_coords(self):
        """Verifies that the abs/floor logic works for Southern/Western hemispheres."""
        arr = np.zeros((1200, 1200))
        # BBox in Southern/Western hemisphere: -10.5 to -10.4
        bbox = (-10.5, -1.5, -10.4, -1.4)
        dims = [1200]
        
        sliced = get_slice(arr, bbox, pos_idx=0, n_quads=1, 
                           SRTM_DATA_DICT={}, dims=dims, res=90)
        assert sliced is not None
        assert sliced.shape[0] > 0

class TestElevationDataLoading:
    #@patch("geom_helpers.tiles.terrain_tiles_copernicus.gdal.Open")
    @patch("geom_helpers.tiles.terrain_tiles_copernicus.rasterio.open")
    @patch("geom_helpers.tiles.terrain_tiles_copernicus.EXISTING_SRTM")
    @patch("geom_helpers.tiles.terrain_tiles_copernicus.SRTM_DATA_DICT")
    def test_get_elevation_data_cache_hit(self, mock_dict, mock_srtm, mock_gdal):
        """Tests that function returns from SRTM_DATA_DICT if path is cached."""
        fake_path = "C:\\tiles\\test.tif"
        mock_srtm.__getitem__.return_value = fake_path
        mock_srtm.__contains__.return_value = True
        
        cached_arr = np.array([[5, 5], [5, 5]])
        mock_dict.__contains__.return_value = True
        mock_dict.__getitem__.return_value = cached_arr
        
        result = get_elevation_data(47, 8, 90)
        
        assert np.array_equal(result, cached_arr)
        mock_gdal.assert_not_called()

    @patch("geom_helpers.tiles.terrain_tiles_copernicus.EXISTING_SRTM", {})
    def test_get_elevation_data_missing(self):
        """Tests that missing data returns the default -10 array."""
        result = get_elevation_data(0, 0, 90)
        assert result.shape == (10, 10)
        assert np.all(result == -10)


class TestSlicingPrecision:
    
    @pytest.mark.parametrize("dim", [100, 1200, 3600])
    def test_slice_dimensions_consistency(self, dim):
        """
        Tests if the slice width and height are consistent for a 0.1 x 0.1 degree box.
        In a perfect world, a 0.1 deg box in a 1.0 deg (dim x dim) tile 
        should be exactly dim/10 pixels.
        """
        # Create dummy array of size dim x dim
        arr = np.zeros((dim, dim))
        
        # BBox (lat2, lon1, lat1, lon2) -> 0.1 degree square
        # We use coordinates that are multiples of 0.1 to see how floor/ceil react
        bbox = (0.4, 0.4, 0.5, 0.5) 
        dims = [dim]
        
        sliced = get_slice(arr, bbox, pos_idx=0, n_quads=1, 
                           SRTM_DATA_DICT={}, dims=dims, res=90)
        
        assert isinstance(sliced, np.ndarray)
        h, w = sliced.shape
        print(f"DIM {dim}: Resulting shape ({h}, {w})")
        
        # At DIM 1200, 0.1 degrees should be 120 pixels.
        # Your code uses [y2:y1+1, x1:x2+1], which is inclusive.
        # If x1=40 and x2=50, the slice is 11 pixels (50-40 + 1).
        # We check if height and width are at least equal.
        assert h == w, f"Asymmetry detected at DIM {dim}: {h}x{w}"

    def test_meridian_crossing_precision(self):
        """
        Tests the specific logic for negative coordinates (Western Hemisphere).
        Your code uses: lon1 = np.abs(floor(lon1) - lon1 + ceil(lon1))
        """
        dim = 3600
        arr = np.zeros((dim, dim))
        # A box from -0.55 to -0.45 (0.1 degree wide)
        bbox = (-10.55, -1.55, -10.45, -1.45)
        dims = [dim]
        
        sliced = get_slice(arr, bbox, pos_idx=0, n_quads=1, 
                           SRTM_DATA_DICT={}, dims=dims, res=90)
        
        assert sliced is not None
        # Verify the slice isn't empty or nonsensically large
        assert 350 <= sliced.shape[0] <= 370

class TestElevationLoading:
    @patch("geom_helpers.tiles.terrain_tiles_copernicus.rasterio.open")
    def test_get_elevation_data_mocked_rasterio(self, mock_rasterio_open, tmp_path):
        """Tests loading data using the Rasterio context manager pattern."""
        # 1. Setup the fake file
        fake_tif = tmp_path / "fake.tif"
        fake_tif.write_text("not a real tif")

        # 2. Setup the Rasterio Mock Chain
        # rasterio.open(path) returns a dataset object
        mock_dataset = MagicMock()
        # The dataset is used as a context manager: with rasterio.open() as src
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset
        # src.read(1) returns the numpy array
        mock_dataset.read.return_value = np.array([[10, 20], [30, 40]])

        # 3. Mock the EXISTING_SRTM dict
        with patch("geom_helpers.tiles.terrain_tiles_copernicus.EXISTING_SRTM",
                {(47, 8, 90): str(fake_tif)}):

            result = get_elevation_data(47, 8, 90)

            # 4. Assertions
            assert result[0, 0] == 10
            # Verify read(1) was called (rasterio bands are 1-indexed)
            mock_dataset.read.assert_called_once_with(1)    
        
    def test_get_elevation_data_missing_returns_default(self):
        """Tests that if the key is not in EXISTING_SRTM, we get the -10 array."""
        # Use an empty dict for EXISTING_SRTM to ensure a miss
        with patch.dict(EXISTING_SRTM, {}, clear=True):
            result = get_elevation_data(69, 69, 90)
            assert result.shape == (10, 10)
            assert np.all(result == -10)

class TestCoordinateMapping:
    @pytest.mark.parametrize("lat, lon, expected", [
        (60.7, -0.96, ("N60", "W001")),
        (53.02, 0.1, ("N53", "E000")),
        (39.03, 1.4, ("N39", "E001")),
        (39.99, 0.01, ("N39", "E000")),
        (59.99, -1.3, ("N59", "W002")),
    ])
    def test_handpicked_coordinates(self, lat, lon, expected):
        """Verifies your coordinate-to-string logic against the prompt requirements."""
        res_lat, res_lon = get_lonlat_str(lat, lon)
        assert (res_lat, res_lon) == expected


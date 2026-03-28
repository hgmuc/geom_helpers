import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from geom_helpers.elevation.pt_elevation import (
    get_file_name,
    get_elevation_arr,
    fill_elevation_voids,
    upsample_1d_arr,
    downsample_1d_arr,
    get_elevation_from_tile, 
    get_elevation_from_tile_numba,
    compute_elevation_from_tile_numba,
    get_srtm_elevation,
    get_srtm_elevation_numba,
    compute_srtm_elevation_numba_e,
    compute_srtm_elevation_numba_w    
)

# Mocking external dependency not provided in the snippet
import sys
sys.modules['basic_helpers.file_helper'] = MagicMock()
sys.modules['basic_helpers.types_base'] = MagicMock()


@pytest.fixture
def tile_data():
    """
    Creates a 10x10 elevation tile.
    BBox: (50.0, 10.0, 51.0, 11.0) -> 1 degree square in Europe.
    """
    # Create a gradient from 100m to 200m
    elevations = np.linspace(100, 200, 100).reshape((10, 10)).astype(np.float64)
    bbox = (50.0, 10.0, 51.0, 11.0)
    return elevations, bbox

@pytest.fixture
def sample_elevations():
    """Create a 3x3 elevation grid for predictable interpolation."""
    # North-East corner of a tile
    return np.array([
        [100, 200, 300],
        [400, 500, 600],
        [700, 800, 900]
    ], dtype=np.int16)

#@pytest.fixture
#def dummy_hgt_file(tmp_path):
#    """Creates a small valid SRTM dummy file (1201x1201)."""
#    srtm_dir = tmp_path / "SRTM2"
#    srtm_dir.mkdir()
#    file_path = srtm_dir / "N52E013.hgt"
    
#    # Create a 1201x1201 array of 500m elevation
#    data = np.full((1201, 1201), 500, dtype='>i2')
#    data.tofile(file_path)
#    return str(srtm_dir)

@pytest.fixture
def dummy_hgt_file(tmp_path):
    """Creates a small valid SRTM dummy file (1201x1201)."""
    # Use / operator with tmp_path (which is a Path object)
    srtm_dir = tmp_path / "SRTM2"
    srtm_dir.mkdir(parents=True, exist_ok=True) # parents=True is safer
    file_path = srtm_dir / "N52E013.hgt"
    
    # Create a 1201x1201 array of 500m elevation
    data = np.full((1201, 1201), 500, dtype='>i2')
    data.tofile(file_path)
    
    # Return the Path object directly if possible, or a normalized string
    return str(srtm_dir.resolve())


def test_get_file_name_positive(dummy_hgt_file):
    with patch('geom_helpers.elevation.pt_elevation.get_srtm_files_directory', return_value=dummy_hgt_file):
        # 52.5 N, 13.4 E -> N52E013.hgt
        path = get_file_name(52.5, 13.4)
        assert path is not None
        assert "N52E013.hgt" in path

def test_get_file_name_not_found():
    with patch('os.path.isfile', return_value=False):
        assert get_file_name(10, 10) is None

def test_get_elevation_arr(dummy_hgt_file):
    with patch('geom_helpers.elevation.pt_elevation.get_srtm_files_directory', return_value=dummy_hgt_file):
        elevations = get_elevation_arr(52.5, 13.4)
        assert isinstance(elevations, np.ndarray)
        assert elevations.shape == (1201, 1201)
        assert elevations[0, 0] == 500

def test_fill_elevation_voids():
    # Create array with a void (-32768 is common SRTM void, code checks < 0)
    arr = np.full((1201, 1201), 100, dtype=int)
    arr[10, 10] = -1 
    
    filled = fill_elevation_voids(arr)
    assert filled is not None
    # The void at 10,10 should be interpolated (likely to 100)
    assert filled[10, 10] >= 0

def test_upsample_1d_arr():
    arr = np.array([10, 20], dtype=float)
    # Upsample from 2 to 3 elements
    upsampled = upsample_1d_arr(arr, 3, mode='linear')
    assert len(upsampled) == 3
    assert upsampled[0] == 10
    assert upsampled[2] == 20
    assert 10 < upsampled[1] < 20

def test_downsample_1d_arr():
    arr = np.array([10, 15, 20], dtype=float)
    downsampled = downsample_1d_arr(arr, 2)
    assert len(downsampled) == 2
    assert downsampled[0] == 10
    assert downsampled[1] == 20

def test_get_elevation_from_tile():
    # 10x10 elevation grid
    elevations = np.zeros((10, 10))
    elevations[5, 5] = 100
    bbox = (0.0, 0.0, 1.0, 1.0) # min_lat, min_lon, max_lat, max_lon
    
    # Check middle point
    val = get_elevation_from_tile(0.5, 0.5, bbox, elevations)
    print("val", type(val), val)
    print("elev", elevations)
    assert isinstance(val, int)
    assert val == 0

    elevations[5, 5] = 10
    val = get_elevation_from_tile(0.5, 0.5, bbox, elevations)
    assert isinstance(val, int)
    assert val > 0

@pytest.mark.parametrize("lat, lon, expected_file", [
    (52.5, 13.4, "N52E013.hgt"),
    (-10.5, 20.2, "S11E020.hgt"),
    (45.0, -122.3, "N45W123.hgt"), # Note: your code does lon = abs(lon)+1 for West
])
def test_filename_logic(lat, lon, expected_file, tmp_path):
    with patch('geom_helpers.elevation.pt_elevation.get_srtm_files_directory', return_value=str(tmp_path)):
        # Mock file existence so it returns the string
        #with patch('os.path.isfile', return_value=True):
        with patch('geom_helpers.elevation.pt_elevation.Path.exists', return_value=True):
            res = get_file_name(lat, lon)
            print("res", res, expected_file)
            assert expected_file in res

def test_get_srtm_elevation_with_provided_array():
    """Test interpolation logic directly by providing an elevation array."""
    # 3x3 grid for simplicity (DIMS=3)
    # Elevations:
    # [10, 20, 30]
    # [40, 50, 60]
    # [70, 80, 90]
    elev_arr = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ], dtype=np.int16)
    
    # Coordinates exactly on a grid point (e.g., Row 1, Col 1 -> value 50)
    # In SRTM, Lat 52.0 is Row DIMS-1, Lat 53.0 is Row 0.
    # For a 3x3 grid, 52.5 is exactly the middle row.
    result = get_srtm_elevation(lat=52.5, lon=13.5, elevations=elev_arr, DIMS=3)
    
    assert isinstance(result, int)
    # The middle of this specific grid should be 50
    assert result == 50

def test_get_srtm_elevation_from_dict_cache():
    """Test that function retrieves data from the srtm_data_dict."""
    mock_dict = {"N52E013.hgt": np.full((5, 5), 100)}
    
    with patch("geom_helpers.elevation.pt_elevation.get_file_name", return_value="N52E013.hgt"):
        # lat 52.0, lon 13.0 should hit the 100m array
        result = get_srtm_elevation(lat=52.0, lon=13.0, srtm_data_dict=mock_dict, DIMS=5)
        assert result == 100

@patch("geom_helpers.elevation.pt_elevation.get_file_name")
@patch("geom_helpers.elevation.pt_elevation.get_elevation_arr")
@patch("geom_helpers.elevation.pt_elevation.fill_elevation_voids")
@patch("geom_helpers.elevation.pt_elevation.do_pickle")
class TestSrtmLoadAndFillWithAndWithoutCacheLogic:

    def test_case_a_no_cache_provided(self, mock_pickle, mock_fill, mock_get_arr, mock_get_name):
        """
        Scenario: Volatile/Copernicus-style usage.
        srtm_data_dict is None -> Data is loaded/processed but NOT persisted.
        """
        # Setup
        mock_get_name.return_value = "N52E013.hgt"
        mock_get_arr.return_value = np.full((1201, 1201), 100, dtype=np.int16)
        mock_fill.return_value = np.full((1201, 1201), 100, dtype=np.int16)
        
        # Execution with srtm_data_dict explicitly None
        result = get_srtm_elevation(lat=52.5, lon=13.5, srtm_data_dict=None)
        
        # Assertions
        assert result == 100
        # do_pickle should NOT be called because write_srtm_data_dict should be False
        mock_pickle.assert_not_called()
        print("\nCase A: Verified no persistence when dict is None")

    def test_case_b_cache_provided_and_updated(self, mock_pickle, mock_fill, mock_get_arr, mock_get_name):
        """
        Scenario: Persistent/SRTM-style usage.
        srtm_data_dict is a dict -> Data is loaded, added to cache, and persisted.
        """
        # Setup
        mock_get_name.return_value = "N52E013.hgt"
        dummy_arr = np.full((1201, 1201), 200, dtype=np.int16)
        mock_get_arr.return_value = dummy_arr
        mock_fill.return_value = dummy_arr
        
        # Execution with an empty cache dict
        my_cache = {}
        result = get_srtm_elevation(lat=52.5, lon=13.5, srtm_data_dict=my_cache)
        
        # Assertions
        assert result == 200
        assert "N52E013.hgt" in my_cache
        # do_pickle SHOULD be called because write_srtm_data_dict should be True
        mock_pickle.assert_called_once()
        print("Case B: Verified persistence when dict is provided")


def test_get_srtm_elevation_negative_longitude():
    """Verify interpolation logic for West longitudes (lon < 0)."""
    # Create a 2x2 where West side is lower than East side
    elev_arr = np.array([
        [100, 200],
        [100, 200]
    ])
    
    # lat 45.5 (middle), lon -122.5 (middle)
    # The logic for lon < 0 calculates lon_perc differently
    result = get_srtm_elevation(lat=45.5, lon=-122.5, elevations=elev_arr, DIMS=2)
    
    assert isinstance(result, int)
    # Expected mid-point between 100 and 200
    assert 140 <= result <= 160 

def test_get_srtm_elevation_returns_none_on_missing_file():
    """Ensure None is returned if no array is provided and no file is found."""
    with patch("geom_helpers.elevation.pt_elevation.get_file_name", return_value=None):
        result = get_srtm_elevation(lat=10.0, lon=10.0, srtm_data_dict={})
        assert result is None


def test_numba_e_kernel_interpolation(sample_elevations):
    """Directly test the Numba-jitted East kernel."""
    # lat 0.5, lon 0.5 should be the center of the grid (index 1,1)
    # The math in the code maps lat 0.5 to a specific row based on YDIMS
    res = compute_srtm_elevation_numba_e(0.5, 0.5, sample_elevations, 3, 3)
    assert isinstance(res, int)
    assert res == 500

def test_numba_w_kernel_interpolation(sample_elevations):
    """Directly test the Numba-jitted West kernel."""
    # For West, lon -0.5 is treated with np.abs() and a column flip
    res = compute_srtm_elevation_numba_w(0.5, -0.5, sample_elevations, 3, 3)
    assert isinstance(res, int)
    # Based on the (XDIMS-1) - lon_col logic, it should pick the mirrored values
    assert 400 <= res <= 600

@patch("geom_helpers.elevation.pt_elevation.do_pickle")
@patch("geom_helpers.elevation.pt_elevation.get_elevation_arr")
@patch("geom_helpers.elevation.pt_elevation.fill_elevation_voids")
@patch("geom_helpers.elevation.pt_elevation.get_file_name")
def test_get_srtm_elevation_numba_dispatch_flow(mock_get_name, mock_fill, mock_get_arr, mock_pickle):
    """Test the wrapper logic: loading data and dispatching to Numba."""
    mock_get_name.return_value = "N52E013.hgt"
    dummy_arr = np.full((1201, 1201), 250, dtype=np.int16)
    mock_get_arr.return_value = dummy_arr
    mock_fill.return_value = dummy_arr
    
    srtm_dict = {}
    
    # Test Northern Hemisphere, Eastern Longitude
    result = get_srtm_elevation_numba(52.5, 13.5, srtm_data_dict=srtm_dict)
    
    assert result == 250
    assert "N52E013.hgt" in srtm_dict
    mock_pickle.assert_called_once()


def test_get_srtm_elevation_numba_with_provided_array(sample_elevations):
    """Test the 'volatile' path where an array is passed directly."""
    # lat 0.5, lon 0.5 on our 3x3 sample
    result = get_srtm_elevation_numba(0.5, 0.5, elevations=sample_elevations, DIMS=3)
    assert result == 500

def test_get_srtm_elevation_numba_auto_dims():
    """Ensure DIMS='auto' correctly detects shape for the Numba kernel."""
    elev_arr = np.full((10, 10), 75, dtype=np.int16)
    # result should be 75 regardless of coord because array is uniform
    result = get_srtm_elevation_numba(0.2, 0.2, elevations=elev_arr, DIMS='auto')  # type: ignore
    assert result == 75

def test_compute_elevation_from_tile_numba_center(tile_data):
    """Test the Numba kernel directly for a center-point interpolation."""
    elevations, bbox = tile_data
    # Exactly in the middle of the BBox
    lat, lon = 50.5, 10.5
    
    elev, raw = compute_elevation_from_tile_numba(lat, lon, bbox, elevations)
    
    assert isinstance(elev, int)
    assert isinstance(raw, np.ndarray)
    assert raw.shape == (2, 2)
    # Since the whole array is 100-200, the middle should be ~150
    assert 140 <= elev <= 160

def test_get_elevation_from_tile_numba_sea_level_logic():
    """
    Test the specific logic: if a neighbor is <= 0 and another is > 40,
    return the minimum (usually 0 or void).
    """
    # 2x2 grid: one corner is 0 (sea), another is 50 (cliff/land)
    elevations = np.array([
        [0, 50],
        [50, 50]
    ], dtype=np.float64)
    bbox = (50.0, 10.0, 51.0, 11.0)
    
    # Query point that would normally interpolate to something like 25
    result = get_elevation_from_tile_numba(50.5, 10.5, bbox, elevations)
    
    # Logic check: raw_elevations.min() is 0 (<=0) and max is 50 (>40)
    # Expected: return raw_elevations.min() -> 0
    assert result == 0

def test_get_elevation_from_tile_numba_none_handling():
    """Ensure None is returned if no elevation array is provided."""
    bbox = (50.0, 10.0, 51.0, 11.0)
    assert get_elevation_from_tile_numba(50.5, 10.5, bbox, None) is None

@pytest.mark.parametrize("lat, lon, expected_range", [
    (50.0, 10.0, (189, 191)),   # South-West corner
    (50.9, 10.9, (110, 120)),  # North-East corner
    (50.9, 11, (118, 120)),  # East border
    (50.9, 10, (109, 111)),  # West border
])
def test_get_elevation_from_tile_numba_boundaries(tile_data, lat, lon, expected_range):
    """Test interpolation at the edges of the provided tile."""
    elevations, bbox = tile_data
    result = get_elevation_from_tile_numba(lat, lon, bbox, elevations)
    assert expected_range[0] <= result <= expected_range[1]


def test_verify_tile_boundary_accuracy():
    """
    Instead of checking for a crash, we check if the math is actually correct
    at the boundary.
    """
    # 2x2 grid
    # [[10, 20],
    #  [30, 40]]
    # Top-Left (min_lat, min_lon) is index [1,0] in many mapping systems, 
    # but let's just see what your code does at the MAX corner.
    elevations = np.array([
#        [10.0, 20.0],
#        [30.0, 40.0]
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
        [70.0, 80.0, 90.0],
    ], dtype=np.float64)
    bbox = (50.0, 10.0, 51.0, 11.0) 

    # If we query the exact Top-Right corner (51.0, 11.0), 
    # we expect the result to be exactly 20.0 (the value at elevations[0, 1])
    max_lat = 51.0
    max_lon = 11.0
    
    result = get_elevation_from_tile_numba(max_lat, max_lon, bbox, elevations)
    
    print(f"Result at boundary: {result}")
    assert result == 30, f"Expected 30 at corner, but got {result}"

def test_verify_tile_boundary_accuracy1():
    """
    Instead of checking for a crash, we check if the math is actually correct
    at the boundary.
    """
    # 2x2 grid
    # [[10, 20],
    #  [30, 40]]
    # Top-Left (min_lat, min_lon) is index [1,0] in many mapping systems, 
    # but let's just see what your code does at the MAX corner.
    elevations = np.array([
        [10.0, 20.0, 30.0, 40.0],
        [50.0, 60.0, 70.0, 80.0],
        [90.0, 100.0, 110.0, 120.0],
        [130.0, 140.0, 160.0, 160.0],
    ], dtype=np.float64)
    bbox = (50.0, 10.0, 51.0, 11.0) 

    # If we query the exact Top-Right corner (51.0, 11.0), 
    # we expect the result to be exactly 20.0 (the value at elevations[0, 1])
    max_lat = 51.0
    max_lon = 11.0
    
    result = get_elevation_from_tile_numba(max_lat, max_lon, bbox, elevations)
    
    print(f"Result at boundary: {result}")
    assert result == 40, f"Expected 40 at corner, but got {result}"

@pytest.mark.parametrize("lat, lon, expected", [
    (50.0, 10.0, 30),   # South-West corner
    (51.0, 10.0, 10),  # North-West corner
    (51.0, 11, 20),  # North-East corner
    (50, 11, 40),  # South-East corner
])
def test_verify_tile_boundary_accuracy3(lat, lon, expected):
    """
    Instead of checking for a crash, we check if the math is actually correct
    at the boundary.
    """
    # 2x2 grid
    # [[10, 20],
    #  [30, 40]]
    # Top-Left (min_lat, min_lon) is index [1,0] in many mapping systems, 
    # but let's just see what your code does at the MAX corner.
    elevations = np.array([
        [10.0, 20.0],
        [30.0, 40.0]
    ], dtype=np.float64)
    bbox = (50.0, 10.0, 51.0, 11.0) 

    # If we query the exact Top-Right corner (51.0, 11.0), 
    # we expect the result to be exactly 20.0 (the value at elevations[0, 1])
    #max_lat = 51.0
    #max_lon = 11.0
    
    result = get_elevation_from_tile_numba(lat, lon, bbox, elevations)
    
    print(f"Result at boundary: {result}")
    assert result == expected, f"Expected {expected} at corner, but got {result}"

@pytest.mark.parametrize("lat, lon, expected", [
    (51.0, 10.0, 10),  # North-West corner
    (51.0, 11, 40),  # North-East corner
    (50, 11, 160),  # South-East corner
    (50.0, 10.0, 130),   # South-West corner
    (50.5, 10.5, 85),   # Center
    (50.6666667, 10.6666667, 70),
    (50.3333333, 10.6666667, 110),
])
def test_verify_tile_boundary_accuracy4(lat, lon, expected):
    """
    Instead of checking for a crash, we check if the math is actually correct
    at the boundary.
    """
    # 2x2 grid
    # [[10, 20],
    #  [30, 40]]
    # Top-Left (min_lat, min_lon) is index [1,0] in many mapping systems, 
    # but let's just see what your code does at the MAX corner.
    elevations = np.array([
        [10.0, 20.0, 30.0, 40.0],
        [50.0, 60.0, 70.0, 80.0],
        [90.0, 100.0, 110.0, 120.0],
        [130.0, 140.0, 160.0, 160.0],
    ], dtype=np.float64)
    bbox = (50.0, 10.0, 51.0, 11.0) 

    # If we query the exact Top-Right corner (51.0, 11.0), 
    # we expect the result to be exactly 20.0 (the value at elevations[0, 1])
    #max_lat = 51.0
    #max_lon = 11.0
    
    result = get_elevation_from_tile_numba(lat, lon, bbox, elevations)
    
    print(f"Result at boundary: {result}")
    assert result == expected, f"Expected {expected} at corner, but got {result}"


def test_verify_tile_boundary_accuracy2():
    """
    Instead of checking for a crash, we check if the math is actually correct
    at the boundary.
    """
    # 2x2 grid
    # [[10, 20],
    #  [30, 40]]
    # Top-Left (min_lat, min_lon) is index [1,0] in many mapping systems, 
    # but let's just see what your code does at the MAX corner.
    elevations = np.array([
        [10.0, 20.0],
        [30.0, 40.0]
    ], dtype=np.float64)
    bbox = (50.0, 10.0, 51.0, 11.0) 

    # If we query the exact Top-Right corner (51.0, 11.0), 
    # we expect the result to be exactly 20.0 (the value at elevations[0, 1])
    max_lat = 51.0
    max_lon = 11.0
    
    result = get_elevation_from_tile_numba(max_lat, max_lon, bbox, elevations)
    
    print(f"Result at boundary: {result}")
    assert result == 20, f"Expected 20 at corner, but got {result}"


def test_tile_numba_accuracy_interp():
    """Verifies bilinear interpolation at a non-grid point."""
    # 2x2 grid
    # [[10, 20],
    #  [30, 40]]
    elevations = np.array([[10, 20], [30, 40]], dtype=np.float64)
    bbox = (0.0, 0.0, 1.0, 1.0)
    
    # Query exactly in the middle (0.5, 0.5)
    # Expected: Average of all 4 pixels = 25
    res = get_elevation_from_tile_numba(0.5, 0.5, bbox, elevations)
    assert res == 25

def test_tile_numba_void_logic():
    """Tests the 'sea level' vs 'cliff' logic (>40 and <=0)."""
    # If one pixel is 0 and another is high, return the min (0)
    elevations = np.array([[0, 100], [100, 100]], dtype=np.float64)
    bbox = (0.0, 0.0, 1.0, 1.0)
    
    res = get_elevation_from_tile_numba(0.5, 0.5, bbox, elevations)
    assert res == 0

## --- 1D Sampling Tests ---

def test_upsample_1d_linear():
    """Tests upsampling from 2 points to 3 points."""
    arr = np.array([100, 200], dtype=np.float64)
    # New length 3 -> [100, 150, 200]
    res = upsample_1d_arr(arr, 3, mode='linear')
    assert len(res) == 3
    assert res[1] == 150

def test_upsample_1d_modes():
    """Tests min/max modes in upsampling."""
    arr = np.array([10, 50], dtype=np.float64)
    res_min = upsample_1d_arr(arr, 3, mode='min')
    res_max = upsample_1d_arr(arr, 3, mode='max')
    
    # Midpoint should follow the mode
    assert res_min[1] == 10
    assert res_max[1] == 50

def test_downsample_1d():
    """Tests downsampling logic."""
    arr = np.array([10, 20, 30, 40], dtype=np.float64)
    # Downsample to 2 points
    res = downsample_1d_arr(arr, 2)
    assert len(res) == 2
    assert res[0] == 10
    assert res[1] == 40


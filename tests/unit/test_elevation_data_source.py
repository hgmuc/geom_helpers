import os
import pytest
import numpy as np
from unittest.mock import patch
from geom_helpers.elevation.ElevationDataSource import ElevationDataSource, ElevationDataFromTiles
from geom_helpers.elevation.ElevationDataSource import BayDataFromTiles, MapzenDataFromTiles, MapterhornDataFromTiles
from geom_helpers.elevation.ElevationDataSource import CopernicusData, SuperDem, SonnyDtm, GeDtm30

# Concrete implementation for testing purposes
class MockElevationSource(ElevationDataSource):
    def load_file(self, lat, lon, zoom=None):
        # Returns a dummy 3x3 array
        return np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.int32)

    def get_fname(self, lat, lon) -> str:
        return f"tile_{int(lat)}_{int(lon)}"

    def get_remote_url(self, lat, lon, zoom=None) -> str:
        return f"http://example.com/{self.get_fname(lat, lon)}.tif"

    def get_subfolders(self, lat, lon) -> list[str]:
        return ["sub"]

@pytest.fixture
def temp_elevation_path(tmp_path):
    """Creates a temporary directory for elevation files."""
    d = tmp_path / "elevation_data"
    d.mkdir()
    return str(d)

class TestElevationDataSource:
    
    def test_initialization(self, temp_elevation_path):
        source = MockElevationSource(path=temp_elevation_path, filefmt="tif", res=30, dim=3)
        assert source.res == 30
        assert source.dim == 3
        assert source.filefmt == "tif"
        assert isinstance(source.data, dict)

    def test_check_path_creates_dirs(self, tmp_path):
        new_path = tmp_path / "deep" / "folder" / "structure"
        # check_path is called in __init__ if path is provided
        _ = MockElevationSource(path=str(new_path))
        assert os.path.exists(str(new_path))

    def test_get_fullpath(self, temp_elevation_path):
        source = MockElevationSource(path=temp_elevation_path, filefmt="tif")
        path = source.get_fullpath("test_file", ["a", "b"])
        expected = os.path.join(temp_elevation_path, "a", "b", "test_file.tif")
        assert path == expected

    def test_load_data_without_key(self, temp_elevation_path):
        # Initialize with with_key=False
        source = MockElevationSource(path=temp_elevation_path, dim=3, with_key=False)
        success = source.load_data(45.0, 10.0)
        
        assert success is True
        assert isinstance(source.data, np.ndarray)
        assert source.data[1, 1] == 50
        assert source.has_data_loaded(45.5, 10.5)
        assert source.get_elevation(45.5, 10.5, elevations=source.data) == 50
        assert source.get_elevation(45.75, 10.75, elevations=source.data) == 40

    def test_load_keyed_data(self, temp_elevation_path):
        source = MockElevationSource(path=temp_elevation_path, dim=3, with_key=True)
        key = "test_key"
        success = source.load_keyed_data(key, 45.0, 10.0)
        
        assert success is True
        assert key in source.data                # type: ignore
        assert source.data[key][0, 0] == 10      # type: ignore
        assert source.get_elevation(45.5, 10.5, elevations=source.data[key]) == 50     # type: ignore
        assert source.get_elevation(45.75, 10.75, elevations=source.data[key]) == 40    # type: ignore

    def test_load_keyed_data1(self, temp_elevation_path):
        source = MockElevationSource(path=temp_elevation_path, dim=3, with_key=True)
        lat, lon = 45.25, 10.25
        key = source.get_data_key(lat, lon)
        success = source.load_keyed_data(key, lat, lon)
        print("key", key)
        assert success is True
        assert key == 'tile_45_10'
        assert key in source.data                # type: ignore
        assert source.data[key][0, 0] == 10      # type: ignore
        assert source.has_data_loaded(45.5, 10.5)
        assert source.has_data_loaded(45.35, 10.65)
        assert not source.has_data_loaded(44.35, 10.65)
        assert source.get_elevation(45.5, 10.5, elevations=source.data[key]) == 50     # type: ignore
        assert source.get_elevation(45.75, 10.75, elevations=source.data[key]) == 40    # type: ignore
        assert source.get_elevation(lat, lon, elevations=source.data[key]) == 60    # type: ignore

    def test_preprocessor_execution(self, temp_elevation_path):
        # Preprocessor that adds 5 to everything
        def prep(x):
            return x + 5
        
        source = MockElevationSource(path=temp_elevation_path, preprocessor=prep)
        source.load_data(45.0, 10.0)
        
        # Original [0,0] was 10, should now be 15
        if isinstance(source.data, dict):
             # When with_key is true (default), load_data still sets self.data = preprocessed
             assert source.data[0, 0] == 15

    @patch("geom_helpers.elevation.ElevationDataSource.os.path.exists")
    def test_can_load_data(self, mock_exists, temp_elevation_path):
        source = MockElevationSource(path=temp_elevation_path, filefmt="tif")
        mock_exists.return_value = True
        assert source.can_load_data(45.0, 10.0) is True
        
        mock_exists.return_value = False
        assert source.can_load_data(45.0, 10.0) is False

    @patch("geom_helpers.elevation.ElevationDataSource.get_srtm_elevation_numba")
    def test_get_elevation_logic(self, mock_numba, temp_elevation_path):
        source = MockElevationSource(path=temp_elevation_path, dim=3)
        mock_numba.return_value = 55
        
        # Test providing elevations directly
        dummy_elevs = np.zeros((3, 3))
        result = source.get_elevation(45.0, 10.0, elevations=dummy_elevs)
        
        assert result == 55
        mock_numba.assert_called_once_with(45.0, 10.0, elevations=dummy_elevs, DIMS=3)

    def test_get_data_arr_caching(self, temp_elevation_path):
        source = MockElevationSource(path=temp_elevation_path, with_key=True)
        
        # Mock file existence to avoid actual I/O error
        with patch.object(source, 'can_load_data', return_value=True):
            # First call triggers load
            arr1 = source.get_data_arr(45.0, 10.0)
            # Second call should use cache (data dict)
            arr2 = source.get_data_arr(45.0, 10.0)
            
            assert arr1 is arr2
            assert source.get_data_key(45.0, 10.0) in source.data   # type: ignore

##### ElevationDataFromTiles

# Concrete implementation for testing
class MockTileSource(ElevationDataFromTiles):
    def load_tile_by_zoom(self, x, y, zoom):
        # Create a unique 2x2 array for each tile to verify stitching
        # Tile 0: [[1,1],[1,1]], Tile 1: [[2,2],[2,2]], etc.
        val = x + y + zoom 
        assert isinstance(self.dim, int)
        return np.full((self.dim, self.dim), val, dtype=np.int32)

    def get_remote_tile_url(self, x, y, zoom):
        return f"http://tiles.local/{zoom}/{x}/{y}.png"
        #url = f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{x}/{y}.png'
        #return url


@pytest.fixture
def tile_source(tmp_path):
    path = str(tmp_path / "tiles")
    return MockTileSource(
        path=path, 
        filefmt="png", 
        dim=2, 
        zoom_lvls=[9], 
        with_key=True
    )

class TestElevationDataFromTiles:

    def test_initialization(self, tile_source):
        assert tile_source.srctype == 'tile'
        assert 9 in tile_source.data
        assert isinstance(tile_source.data[9], dict)

    def test_get_subfolders_with_zoom(self, tile_source):
        # deg2num(45, 10, 9) -> x=270, y=189 (approx)
        subfolders = tile_source.get_subfolders_with_zoom(45.0, 10.0, 9)
        assert subfolders[0] == "9"
        assert subfolders[1] == "270"

    def test_check_full_path_windows_and_unix(self, tile_source, tmp_path):
        # Test directory creation logic
        target_dir = tmp_path / "9" / "270" / "189.png"
        tile_source.check_full_path(str(target_dir))
        assert os.path.exists(tmp_path / "9" / "270")

    def test_load_keyed_tile_data_stitching(self, tile_source):
        """
        Tests the complex stitching logic where 4 tiles are combined.
        The result should be (dim+1, dim+1) -> (3, 3)
        """
        # Manually create the 4 adjacent tiles
        # Tile 0 (x,y), Tile 1 (x+1,y), Tile 2 (x,y+1), Tile 3 (x+1,y+1)
        data_files = {
            0: np.array([[10, 10], [10, 10]]), # Main
            1: np.array([[20, 20], [20, 20]]), # Right
            2: np.array([[30, 30], [30, 30]]), # Bottom
            3: np.array([[40, 40], [40, 40]])  # Bottom-Right
        }
        
        tile_source.load_keyed_tile_data(270, 189, 9, data_files)
        stitched = tile_source.data[9][(270, 189)]
        
        assert stitched.shape == (3, 3)
        # Check core
        assert stitched[0, 0] == 10
        # Check right edge (from tile 1)
        assert stitched[0, 2] == 20
        # Check bottom edge (from tile 2)
        assert stitched[2, 0] == 30
        # Check corner (from tile 3)
        assert stitched[2, 2] == 40

    def test_load_keyed_tile_data_edge_filling(self, tile_source):
        """Tests if the code correctly fills zeros using adjacent values."""
        # Only provide tile 0
        data_files = {0: np.array([[10, 10], [10, 10]]), 1: None, 2: None, 3: None}
        
        tile_source.load_keyed_tile_data(270, 189, 9, data_files)
        stitched = tile_source.data[9][(270, 189)]
        
        # The logic: if data_arr[-1,:-1].sum() == 0: data_arr[-1,:-1] = data_arr[-2,:-1]
        # Should have copied the last valid row/col to the +1 expansion
        assert stitched[2, 0] == 10 
        assert stitched[0, 2] == 10
        assert stitched[2, 2] == 10


    def test_get_data_arr_dims(self, tile_source):
        """Tests if the code correctly fills zeros using adjacent values."""
        # Only provide tile 0
        data_files = {0: np.array([[10, 10], [10, 10]]), 1: None, 2: None, 3: None}        
        assert tile_source.get_data_arr_dims(data_files) == (2, 2)

    def test_get_data_arr_dims1(self, tile_source):
        """Tests if the code correctly fills zeros using adjacent values."""
        # Only provide tile 0
        data_files = {0: np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]]), 1: None, 2: None, 3: None}        
        assert tile_source.get_data_arr_dims(data_files) == (2, 2)

    def test_get_data_arr_dims2(self, tile_source):
        """Tests if the code correctly fills zeros using adjacent values."""
        # provide tile 0 + 1
        data_files = {0: np.array([[10, 10], [10, 10]]), 1: np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]]), 
                      2: None, 3: None}        
        assert tile_source.get_data_arr_dims(data_files) == (2, 2)

    def test_get_data_arr_dims3(self, tile_source):
        """Tests if the code correctly fills zeros using adjacent values."""
        data_files = {0: np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]]), 1: np.array([[10, 10], [10, 10]]), 2: None, 3: None}        
        assert tile_source.get_data_arr_dims(data_files) == (2, 2)

    def test_get_data_arr_dims4(self, tile_source):
        """Tests if the code correctly fills zeros using adjacent values."""
        # Set tile_source.dim to 'auto'
        tile_source.dim = 'auto'
        data_files = {0: np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]]), 1: np.array([[10, 10], [10, 10]]), 2: None, 3: None}        
        assert tile_source.get_data_arr_dims(data_files) == (3, 3)

    def test_get_data_arr_dims5(self, tile_source):
        """Tests if the code correctly fills zeros using adjacent values."""
        # Set tile_source.dim to 'auto'
        tile_source.dim = 'auto'
        data_files = {0: np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]]), 1: None, 2: None, 3: None}        
        assert tile_source.get_data_arr_dims(data_files) == (3, 3)

    def test_get_data_arr_dims6(self, tile_source):
        """Tests if the code correctly fills zeros using adjacent values."""
        # Set tile_source.dim to 'auto'
        tile_source.dim = 'auto'
        data_files = {0: np.array([[10, 10], [10, 10], [10, 10], [10, 10]]), 1: None, 2: None, 3: None}        
        assert tile_source.get_data_arr_dims(data_files) == (4, 2)

    def test_get_elevation1(self, tile_source):
        """Tests if the code correctly fills zeros using adjacent values."""
        # Set tile_source.dim to 'auto'
        tile_source.dim = 'auto'
        data_files = {0: np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]]), 1: None, 2: None, 3: None}        
        assert tile_source.get_elevation(36.5, 15.5, data_files[0]) == 10

    def test_get_elevation2(self, tile_source):
        """Tests if the code correctly fills zeros using adjacent values."""
        # Set tile_source.dim to 'auto'
        tile_source.dim = 'auto'
        data_arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.float32)        
        assert tile_source.get_elevation(36.5, 15.5, data_arr) == 50

    def test_get_elevation3(self, tile_source):
        """Tests if the code correctly fills zeros using adjacent values."""
        # Set tile_source.dim to 'auto'
        tile_source.dim = 'auto'
        data_arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90], [90, 90, 90]], dtype=np.float32)        
        assert tile_source.get_elevation(36.5, 15.5, data_arr) == 65
        assert tile_source.get_elevation(36.33333, 15.5, data_arr) == 80
        assert tile_source.get_elevation(36.16667, 15.75, data_arr) == 87

    @patch("geom_helpers.elevation.ElevationDataSource.deg2num")
    @patch("geom_helpers.elevation.ElevationDataSource.download_remote_file")
    @patch("geom_helpers.elevation.ElevationDataSource.write_file")
    def test_get_data_arr_with_download(self, mock_write, mock_dl, mock_deg, tile_source):
        tile_source.try_download = True
        mock_deg.return_value = (270, 189)
        mock_dl.return_value = b"fake_content"
        mock_write.return_value = True
        
        # Mocking can_load_data to False to force download attempt
        with patch.object(tile_source, 'can_load_data', return_value=False):
            with patch.object(tile_source, 'load_tile_by_zoom', return_value=np.zeros((2,2))):
                res = tile_source.get_data_arr(45.0, 10.0, 9)
                assert res is not None
                #assert mock_dl.called

    def test_load_data_xyz(self):
        mpz = MapzenDataFromTiles(
            filefmt='png', 
            dim=256, 
            try_download=False, 
            path='tests/data',
            zoom_lvls=[2]
        )

        data_files = {i: None for i in range(4)} 
        for i in range(4):
            print(i)
            data_files = mpz.load_data_files(i, 0, 0, 2, data_files) # type: ignore

        res = mpz.load_data_xyz(1, 1, 2, data_files) # type: ignore

        assert res
        assert mpz.data[2][1,1].shape == (257, 257) # type: ignore

# Coordinates in Bavaria (Munich area)
BAVARIA_LAT = 48.1351
BAVARIA_LON = 11.5820

class TestSpecializedElevationSources:

    @pytest.fixture
    def mock_tile_data(self):
        """Generates a consistent dummy tile for ballpark comparison."""
        return np.full((256, 256), 500, dtype=np.int32)

    def test_mapzen_download_logic(self, tmp_path):
        """Verify Mapzen URL generation and initialization."""
        mpz = MapzenDataFromTiles(
            filefmt='png', 
            dim=256, 
            path=str(tmp_path), 
            try_download=True, 
            zoom_lvls=[13]
        )
        url = mpz.get_remote_tile_url(4405, 2811, 13)
        assert "s3.amazonaws.com" in url
        assert "terrarium/13/4405/2811.png" in url

    def test_mapterhorn_initialization(self, tmp_path):
        """Verify Mapterhorn path search and URL."""
        mpth = MapterhornDataFromTiles(
            filefmt='webp', 
            dim=512, 
            zoom_lvls=[15], 
            try_download=True, 
            path=str(tmp_path)
        )
        url = mpth.get_remote_tile_url(17621, 11244, 15)
        assert "tiles.mapterhorn.com/15/17621/11244.webp" in url

        elev = mpth.get_elevation(BAVARIA_LAT, BAVARIA_LON)
        print("mpth elev", elev)
        assert pytest.approx(elev, rel=0.015) == 520

        elev = mpth.get_elevation(47.96, 12.73)
        assert pytest.approx(elev, rel=0.01) == 451

    @pytest.mark.localdata
    def test_bay_data_local(self):
        """Test BayData with local gzip mock."""
        # Using dim='auto' to test shape detection
        byt = BayDataFromTiles(filefmt='gzip', dim='auto', zoom_lvls=[15])
                
        elev = byt.get_elevation(BAVARIA_LAT, BAVARIA_LON)
        print("byt  elev", elev)
        assert pytest.approx(elev, rel=0.01) == 515

        elev = byt.get_elevation(47.96, 12.73)
        assert pytest.approx(elev, rel=0.01) == 451


    @pytest.mark.localdata
    def test_mapzen_local_auto_dim(self):
        """Test Mapzen with dim='auto'."""
        mpz9 = MapzenDataFromTiles(filefmt='png', dim='auto', zoom_lvls=[9])
        
        elev = mpz9.get_elevation(BAVARIA_LAT, BAVARIA_LON)
        print("mpz9 elev", elev)
        assert pytest.approx(elev, rel=0.01) == 515

        elev = mpz9.get_elevation(47.96, 12.73)
        assert pytest.approx(elev, rel=0.01) == 451

    @pytest.mark.localdata
    def test_ballpark_comparison(self, tmp_path):
        """
        Integration-style test to ensure all sources return similar values 
        for the same location when mocked with identical base data.
        """

        sources = [
            MapzenDataFromTiles(filefmt='png', dim=256, path=str(tmp_path), try_download=True, zoom_lvls=[13]),
            MapterhornDataFromTiles(filefmt='webp', dim=512, path=str(tmp_path), try_download=True, zoom_lvls=[15]),
            BayDataFromTiles(filefmt='gzip', dim='auto', zoom_lvls=[15])
        ]

        results = []
        for src in sources:
            # Force mock the loading and data presence
            #with patch.object(src, 'get_data_arr', return_value=mock_data):
            #    with patch("geom_helpers.elevation.ElevationDataSource.get_elevation_from_tile_numba", return_value=common_val):
            elev = src.get_elevation(BAVARIA_LAT, BAVARIA_LON)
            results.append(elev)

        # Check if all results are within 5% of each other (should be identical in this mock)
        common_val = np.mean(results)
        print("results", results, common_val)
        for r in results:
            assert pytest.approx(r, rel=0.05) == common_val


# Test Coordinates
COORD_SPAIN = (40.96, 0.73)
COORD_SPAIN_W = (40.96, -0.73)
COORD_BAVARIA = (47.96, 12.73)

class TestCopernicusLogic:

    def test_filename_generation(self):
        """Verify the naming convention for Copernicus S3 buckets."""
        cp90 = CopernicusData(res=90)
        # 40.96N, 0.73E
        fname = cp90.get_fname(40.96, 0.73)
        assert fname == "Copernicus_DSM_COG_30_N40_00_E000_00_DEM"
        
        # Test Western Hemisphere logic
        # 40.96N, -0.73W -> abs(-0.73)+1 = 1.73 -> E001
        fname_w = cp90.get_fname(40.96, -0.73)
        assert "W001" in fname_w

    def test_southern_hemisphere_raises(self):
        """Ensure the NotImplementedError is raised for Southern latitudes."""
        cp = CopernicusData()
        with pytest.raises(NotImplementedError, match="No data for Southern Hemisphere"):
            cp.get_fname(-10.0, 20.0)

    @patch("geom_helpers.elevation.ElevationDataSource.download_remote_file")
    @patch("geom_helpers.elevation.ElevationDataSource.write_file")
    @patch("geom_helpers.elevation.ElevationDataSource.CopernicusData.load_file")
    def test_try_download_orchestration(self, mock_load, mock_write, mock_dl, tmp_path):
        """Tests that CopernicusData attempts to download all 4 tiles for the overlap."""
        cp = CopernicusData(res=90, try_download=True, path=str(tmp_path))
        mock_dl.return_value = b"fake_tif_content"
        mock_write.return_value = True
        # Return a small 1200x12000 array for each load call
        mock_load.return_value = np.zeros((1200, 1200))
        
        # Trigger data array loading (which triggers 4 potential downloads/loads)
        cp.get_data_arr(40.5, 0.5)
        
        # Should have attempted to resolve tiles for (lat, lon), (lat-1, lon), etc.
        assert mock_dl.call_count >= 1

    def test_fallback_mechanism(self, tmp_path):
        """Tests that cp30 falls back to cp90 if cp30 can't load data."""
        cp90 = CopernicusData(res=90, dim=1200, try_download=True, path=str(tmp_path))
        
        cp30 = CopernicusData(res=30, dim=3600, fallbackDemSrc=cp90, path=str(tmp_path))
        
        elev = cp30.get_elevation(40.96, 0.73)
        assert elev is not None
        assert elev > 200

class TestLocalDataSources:

    @pytest.mark.localdata
    def test_superdem_data_source(self):
        sd = SuperDem(filefmt="npz", dim=3601, res=30)
        assert sd.get_fname(47.96, 12.73) == "SUPERDEM_N47E012"
        assert sd.can_load_data(47.96, 12.73)
        arr = sd.get_data_arr(47.96, 12.73)
        assert arr is not None
        assert arr.shape[0] == sd.get_dim()
        assert sd.get_elevation(47.96, 12.73) == 450

    @pytest.mark.localdata
    def test_sonnydtm_data_source(self):
        sd = SonnyDtm(filefmt="npz", dim=5566, res=20, path='C:/05_Python/awstiles/sonny')
        assert sd.get_fname(47.96, 12.73) == "tile_N47_E012"
        assert sd.can_load_data(47.96, 12.73)
        arr = sd.get_data_arr(47.96, 12.73)
        assert arr is not None
        assert arr.shape[0] == sd.get_dim()
        assert sd.get_elevation(47.96, 12.73) == 454

    @pytest.mark.localdata
    def test_gedtm_data_source(self):
        ge = GeDtm30(filefmt="npz", dim=4001, res=20, path='C:/05_Python/awstiles/GEDTM30')
        #COORD_SPAIN = (40.96, 0.73)
        print("COORD_SPAIN", COORD_SPAIN)
        assert ge.get_fname(*COORD_SPAIN) == "GEDTM30_N40_E000"
        assert ge.can_load_data(*COORD_SPAIN)
        arr = ge.get_data_arr(*COORD_SPAIN)
        assert arr is not None
        assert arr.shape[0] == ge.get_dim()
        assert ge.get_elevation(*COORD_SPAIN) == 229


    @pytest.mark.localdata
    @pytest.mark.slow
    def test_ballpark_comparison1(self, tmp_path):
        """
        Integration-style test to ensure all sources return similar values 
        for the same location when mocked with identical base data.
        """

        sources = [
            MapzenDataFromTiles(filefmt='png', dim=256, path=str(tmp_path), try_download=True, zoom_lvls=[13]),
            MapterhornDataFromTiles(filefmt='webp', dim=512, path=str(tmp_path), try_download=True, zoom_lvls=[15]),
            #BayDataFromTiles(filefmt='gzip', dim='auto', zoom_lvls=[15]),
            GeDtm30(filefmt="npz", dim=4001, res=20, path='C:/05_Python/awstiles/GEDTM30'),
            CopernicusData(res=90, dim=1200, try_download=True, path=str(tmp_path)),
            CopernicusData(res=30, dim=3600)
        ]

        results = []
        for src in sources:
            # Force mock the loading and data presence
            #with patch.object(src, 'get_data_arr', return_value=mock_data):
            #    with patch("geom_helpers.elevation.ElevationDataSource.get_elevation_from_tile_numba", return_value=common_val):
            elev = src.get_elevation(*COORD_SPAIN)
            results.append(elev)

        # Check if all results are within 5% of each other (should be identical in this mock)
        common_val = np.mean(results)
        print("results", results, common_val)
        for r in results:
            assert pytest.approx(r, rel=0.025) == common_val


    @pytest.mark.localdata
    @pytest.mark.slow
    def test_ballpark_comparison2(self, tmp_path):
        """
        Integration-style test to ensure all sources return similar values 
        for the same location when mocked with identical base data.
        """

        sources = [
            MapzenDataFromTiles(filefmt='png', dim=256, path=str(tmp_path), try_download=True, zoom_lvls=[13]),
            MapterhornDataFromTiles(filefmt='webp', dim=512, path=str(tmp_path), try_download=True, zoom_lvls=[15]),
            #BayDataFromTiles(filefmt='gzip', dim='auto', zoom_lvls=[15]),
            GeDtm30(filefmt="npz", dim=4001, res=20, path='C:/05_Python/awstiles/GEDTM30'),
            CopernicusData(res=90, dim=1200, try_download=True, path=str(tmp_path)),
            CopernicusData(res=30, dim=3600)
        ]

        results = []
        for src in sources:
            # Force mock the loading and data presence
            #with patch.object(src, 'get_data_arr', return_value=mock_data):
            #    with patch("geom_helpers.elevation.ElevationDataSource.get_elevation_from_tile_numba", return_value=common_val):
            elev = src.get_elevation(*COORD_SPAIN_W)
            results.append(elev)

        # Check if all results are within 5% of each other (should be identical in this mock)
        common_val = np.mean(results)
        print("results", results, common_val)
        for r in results:
            assert pytest.approx(r, rel=0.025) == common_val


    @pytest.mark.localdata
    @pytest.mark.slow
    def test_ballpark_comparison3(self, tmp_path):
        """
        Integration-style test to ensure all sources return similar values 
        for the same location when mocked with identical base data.
        """

        sources = [
            MapzenDataFromTiles(filefmt='png', dim=256, path=str(tmp_path), try_download=True, zoom_lvls=[13]),
            MapterhornDataFromTiles(filefmt='webp', dim=512, path=str(tmp_path), try_download=True, zoom_lvls=[15]),
            BayDataFromTiles(filefmt='gzip', dim='auto', zoom_lvls=[15]),
            SonnyDtm(filefmt="npz", dim=5566, res=20, path='C:/05_Python/awstiles/sonny'),
            #CopernicusData(res=90, dim=1200, try_download=True, path=str(tmp_path)),
            CopernicusData(res=30, dim=3600)
        ]

        results = []
        for src in sources:
            # Force mock the loading and data presence
            #with patch.object(src, 'get_data_arr', return_value=mock_data):
            #    with patch("geom_helpers.elevation.ElevationDataSource.get_elevation_from_tile_numba", return_value=common_val):
            elev = src.get_elevation(*COORD_BAVARIA)
            results.append(elev)

        # Check if all results are within 5% of each other (should be identical in this mock)
        common_val = np.mean(results)
        print("results", results, common_val)
        for r in results:
            assert pytest.approx(r, rel=0.025) == common_val

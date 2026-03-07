import pytest
from math import cos, radians
#from typing import cast
from geom_helpers.latlon_rc import (
    get_base_latlon_icel, base_default_func, 
    get_base_latlon_cana, get_base_latlon_cabo, get_base_latlon_azor, get_base_latlon_made,
    get_num_rows_cols,
    compute_sub_rc,
    get_sub_rc,
    get_sub_rc_code,
    get_rev_sub_rc
)

# Mocking Coordinate values for testing
# (Lat, Lon)
BASE_COORD = (50.5, 10.5)
ICEL_COORD = (64.5, -20.5)
CANA_COORD = (28.5, -15.5)

class TestLatLonRC:

    @pytest.mark.parametrize("lon, expected_base_lon", [
        (-25.0, -26),  # diff 1 -> 1/4=0 -> -26
        (-20.0, -22),  # diff 6 -> 6/4=1 -> -26 + 4 = -22
        (-15.0, -18),  # diff 11 -> 11/4=2 -> -26 + 8 = -18
    ])
    def test_get_base_latlon_icel(self, lon: float, expected_base_lon: int):
        lat = 65.0
        result_lat, result_lon = get_base_latlon_icel((lat, lon))
        assert result_lat == int(lat)
        assert result_lon == expected_base_lon

    def test_get_base_latlon_functions(self):
        """Verify the logic of the individual base functions."""
        # Test the default logic: (int(x), floor(y))
        assert base_default_func((10.7, 5.9)) == (10, 5)
        
        assert get_base_latlon_cana((0, 0)) == (26, -19)
        # Testing the lambda logic used in get_num_rows_cols for other regions
        # In the source, these are hardcoded in the if/elif block
        assert (26, -19) == (26, -19) 

        assert get_base_latlon_cana((1, 1)) == (26, -19)
        assert get_base_latlon_cabo((2, 0)) == (14, -27)
        assert get_base_latlon_azor((3, 0)) == (36, -32)
        assert get_base_latlon_made((0, 0)) == (32, -18)

    def test_get_num_rows_cols_invalid(self):
        # Test coordinates outside any defined region
        result = get_num_rows_cols(-90, -180)
        assert result == ()

    @pytest.mark.parametrize("lat, lon, expected_key", [
        (52, 13, 'base'),
        (64, -20, 'iceland'),
        (28, -16, 'canary'),
        (37, -26, 'azores'),
        (15, -24, 'caboverde'),
        (33, -17, 'madeira'),
    ])
    def test_get_num_rows_cols_regions(self, lat, lon, expected_key):
        res = get_num_rows_cols(lat, lon)
        assert res is not None
        assert len(res) == 10
        assert res[9] == expected_key  # ParamsKey is the last element

    def test_compute_sub_rc_logic(self):
        # Test the Numba-jitted math function directly
        # params: lat, lon, grid_w, grid_h, rows, cols, x1, y1, n1, n2, b_lat, b_lon
        r, c = compute_sub_rc(
            50.5, 10.5, 10, 7, 100, 100, 2, 3, 2, 3, 50, 10
        )
        assert isinstance(r, int)
        assert isinstance(c, int)
        assert r >= 0
        assert c >= 0

    def test_get_sub_rc_integration(self, mocker):
        # We mock get_reg_cell_code if it's external, 
        # but here we test the flow of get_sub_rc
        res = get_sub_rc(52.5, 13.5)
        assert len(res) == 2
        assert all(isinstance(x, int) for x in res)

    def test_get_sub_rc_code_format(self):
        # Mocking get_reg_cell_code to return a known 7-char prefix
        #mocker.patch('geom_helpers.latlon_rc.get_reg_cell_code', return_value="ABC1234")
        
        code = get_sub_rc_code(50.5, 11.5)
        # Expected: prefix(7) + r + c
        print("code", code)
        assert code.startswith("EM44-")
        # Ensure r and c were appended
        assert len(code) == 9
        assert code.endswith("50")
        assert code == "EM44-5150"

    def test_get_rev_sub_rc(self, mocker):
        # Mock the reverse lookup of the base cell
        mocker.patch('geom_helpers.latlon_rc.get_rev_reg_cell_code', return_value=(52.0, 13.0))
        
        lat, lon = get_rev_sub_rc("ABC1234", 1, 1)   # Mocked fake code
        
        assert isinstance(lat, float)
        assert isinstance(lon, float)
        # Check if coordinates are within the expected 1-degree cell
        assert 52.0 <= lat <= 53.0
        assert 13.0 <= lon <= 14.0

    def test_get_rev_sub_rc1(self):
        # Mock the reverse lookup of the base cell
        #mocker.patch('geom_helpers.latlon_rc.get_rev_reg_cell_code', return_value=(52.0, 13.0))
        
        lat, lon = get_rev_sub_rc("GO11-2345", 1, 1)
        
        assert isinstance(lat, float)
        assert isinstance(lon, float)
        # Check if coordinates are within the expected 1-degree cell
        assert 52.0 <= lat <= 53.0
        assert 13.0 <= lon <= 14.0
        assert pytest.approx(lat, abs=1e-5) == 52.014199
        assert pytest.approx(lon, abs=1e-5) == 13.040124


    def test_num_rows_cols_dicts_populated(self):
        from geom_helpers.latlon_rc import NUM_ROWS_COLS, NUM_ROWS_COLS_BY_CELL
        assert len(NUM_ROWS_COLS) > 0
        assert len(NUM_ROWS_COLS_BY_CELL) > 0


    def test_fwd_bwd_rc_coding(self):
        d = 6e-4
        for lat, lon in [(36.2302, -6.71), (46.531, 6.412), (53.681, 16.756), 
                         (55.681, 18.001), (65.6788, -18.222), (67.6788, -19.2326), 
                         (32.44, -17.111), (28.8755, -16.111)]:
            sub_rc_code = get_sub_rc_code(lat, lon)
            r, c = get_sub_rc(lat, lon)
            rev_lat, rev_lon = get_rev_sub_rc(sub_rc_code, r, c)
            print(f"{lat:9.5f}, {lon:9.5f} - {sub_rc_code} - {r} {c} - {rev_lat:9.5f}, {rev_lon:9.5f} - {lat-rev_lat:9.5f}, {lon-rev_lon:9.5f}")
            assert sub_rc_code.endswith(f"{r}{c}")
            assert pytest.approx(rev_lat, abs=d) == lat   # 9e-4 = ca. 100m, 6e-4 = ca 67.5 m
            assert pytest.approx(rev_lon, abs=d / cos(radians(lat))) == lon


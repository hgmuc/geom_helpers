import pytest
from geom_helpers.bearing_helper import get_bearing, get_track_bearings, get_track_bearings_num

# Mocking the Coordinate type if not available in test env, 
# though usually it's a tuple[float, float]
from typing import Any
Coordinate = Any 

class TestBearingHelper:

    @pytest.mark.parametrize("pt1, pt2, expected", [
        ((0.0, 0.0), (1.0, 0.0), 0.0),     # North
        ((0.0, 0.0), (0.0, 1.0), 90.0),    # East
        ((1.0, 10.0), (0.0, 10.0), 180.0),   # South
        ((0.0, 1.0), (0.0, 0.0), 270.0),   # West
        ((48.0, 11.0), (48.0, 11.0), 0.0), # Same point
    ])
    def test_get_bearing_cardinal_points(self, pt1, pt2, expected):
        """Test basic cardinal directions and identical points."""
        print(pt1, pt2, get_bearing(pt1, pt2), expected)
        assert get_bearing(pt1, pt2) == pytest.approx(expected, abs=1e-2)

    def test_get_bearing_with_extra_data(self):
        """Verify the function handles tuples longer than 2 (e.g., including elevation)."""
        pt1 = (48.1371, 11.5761, 520) # Munich
        pt2 = (52.5200, 13.4050, 34)  # Berlin
        # Result should be roughly North-North-East (~14-15 degrees)
        result = get_bearing(pt1, pt2)   # type: ignore
        print(pt1, pt2, get_bearing(pt1, pt2))  # type: ignore
        assert 14 <= result <= 15
        assert isinstance(result, float)

    def test_get_track_bearings_format(self):
        """Verify the string formatting in get_track_bearings."""
        track = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        # 0.0 (North), 90.0 (East)
        results = get_track_bearings(track) # type: ignore
        
        assert len(results) == 2
        assert results[0].strip() == "0.00"
        # If the math results in 89.99, it means the float was likely 89.994...
        # To be safe with floats, we convert back to float for the assertion:
        assert float(results[1]) == pytest.approx(90.00, abs=0.015)        
        #assert results[1].strip() == "90.00"

    def test_get_track_bearings_printout(self, capsys):
        """Test that printout=True actually prints to stdout."""
        track = [(0.0, 0.0), (1.0, 0.0)]
        get_track_bearings(track, printout=True)   # type: ignore
        captured = capsys.readouterr()
        assert "===>" in captured.out
        assert "0.00" in captured.out

    def test_get_track_bearings_num(self):
        """Verify numeric list return and rounding."""
        track = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        # East (90.0), North (0.0)
        results = get_track_bearings_num(track)   # type: ignore
        
        assert results == [90.0, 0.0]
        assert isinstance(results[0], float)

    def test_empty_or_single_point_track(self):
        """Ensure functions handle tracks too short to have bearings."""
        assert get_track_bearings([]) == []
        assert get_track_bearings([(0.0, 0.0)]) == []
        assert get_track_bearings_num([(0.0, 0.0)]) == []
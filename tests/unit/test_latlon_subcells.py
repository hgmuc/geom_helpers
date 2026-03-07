#import pytest
from unittest.mock import patch #, MagicMock
from geom_helpers.latlon_subcells import get_relevant_cells_from_latlon, get_subcells_in_bbox
from geom_helpers.osm_reader_helper import InvalidCoordinateError

class TestLatLonSubcells:
    
    @patch("geom_helpers.latlon_subcells.get_subcells_in_bbox")
    def test_get_relevant_cells_from_latlon(self, mock_get_subcells):
        """Verify it extracts keys (Cells) from the subcell dictionary."""
        # Setup mock return: { 'A1': {'01', '02'}, 'B2': {'03'} }
        mock_get_subcells.return_value = {"A1": {"01", "02"}, "B2": {"03"}}
        
        pt1 = (50.0, 10.0)
        pt2 = (50.1, 10.1)
        
        result = get_relevant_cells_from_latlon(pt1, pt2, buffer=0.1)
        
        # We expect only the keys 'A1' and 'B2'
        assert isinstance(result, list)
        assert len(result) == 2
        assert "A1" in result
        assert "B2" in result
        mock_get_subcells.assert_called_once_with(pt1, pt2, 0.1, dimx=1, dimy=1)

    def test_get_relevant_cells_from_latlon_no_mock(self):
        """Verify it extracts keys (Cells) from the subcell dictionary."""
        
        pt1 = (50.0, 10.0)
        pt2 = (50.1, 10.1)
        
        result = get_relevant_cells_from_latlon(pt1, pt2, buffer=0.1)

        assert isinstance(result, list)
        assert len(result) == 4
        assert "EL" in result
        assert "EK" in result
        assert "DL" in result
        assert "DK" in result


    @patch("geom_helpers.latlon_subcells.get_coord_code")
    def test_get_subcells_in_bbox_default(self, mock_get_coord):
        """Test default code type with mock coordinate codes."""
        # Mock behavior: return a 4+ char string for specific coordinates
        # Bounding box roughly 50.0 to 50.1 (dim 0.1)
        mock_get_coord.side_effect = lambda lat, lon: f"XY{int(lat*10)}{int(lon*10)}"
        
        pt1 = (50.0, 10.0)
        pt2 = (50.1, 10.1)
        
        # dimx/dimy default to 0.1
        result = get_subcells_in_bbox(pt1, pt2, code_type='default')
        
        # Check structure: { Cell: {Subcell} }
        assert "XY" in result
        # Based on np.arange logic, it should hit points like (50.0, 10.0)
        assert "5010"[:2] == "50" # Just checking our mock logic
        assert len(result["XY"]) > 0

    def test_get_subcells_in_bbox_default_no_mock(self):
        """Test default code type with mock coordinate codes."""        
        pt1 = (50.0, 10.0)
        pt2 = (50.15, 10.15)
        
        # dimx/dimy default to 0.1
        result = get_subcells_in_bbox(pt1, pt2, code_type='default')
        
        # Check structure: { Cell: {Subcell} }
        assert "EL" in result
        # Based on np.arange logic, it should hit points like (50.0, 10.0)
        #assert "5010"[:2] == "50" # Just checking our mock logic
        assert len(result) == 1
        assert len(result["EL"]) == 4
        assert sorted(result["EL"]) == ['00', '01', '10', '11']


    @patch("geom_helpers.latlon_subcells.get_reg_cell_code")
    def test_get_subcells_in_bbox_non_default_3char(self, mock_get_reg):
        """Test non-default code type with large dimensions resulting in 3-char codes."""
        mock_get_reg.return_value = "ABCDE" # Full code
        
        pt1 = (50.0, 10.0)
        pt2 = (50.5, 10.5)
        
        # dimx >= 1/3 triggers len_subcell_code = 3
        result = get_subcells_in_bbox(pt1, pt2, dimx=0.5, dimy=0.5, code_type='custom')
        
        # If len is 3, Cell is result[:2] ('AB'), Subcell is result[2:3] ('C')
        assert "AB" in result
        assert "C" in result["AB"]
        assert len(list(result["AB"])[0]) == 1


    def test_get_subcells_in_bbox_non_default_3char_no_mock(self):
        """Test non-default code type with large dimensions resulting in 3-char codes."""        
        pt1 = (53.0, 13.0)
        pt2 = (53.5, 13.5)
        
        # dimx >= 1/3 triggers len_subcell_code = 3
        result = get_subcells_in_bbox(pt1, pt2, dimx=0.5, dimy=0.5, code_type='custom')
        
        # If len is 3, Cell is result[:2] ('AB'), Subcell is result[2:3] ('C')
        print("res", result)
        assert "HO" in result
        assert len(result) == 1
        assert "1" in result["HO"]
        assert "2" not in result["HO"]
        assert "5" not in result["HO"]
        assert len(result["HO"]) == 2

    def test_get_subcells_in_bbox_non_default_3char_no_mock1(self):
        """Test non-default code type with large dimensions resulting in 3-char codes."""        
        pt1 = (53.0, 13.0)
        pt2 = (53.5, 13.51)
        
        # dimx >= 1/3 triggers len_subcell_code = 3
        result = get_subcells_in_bbox(pt1, pt2, dimx=0.5, dimy=0.5, code_type='custom')
        
        # If len is 3, Cell is result[:2] ('AB'), Subcell is result[2:3] ('C')
        print("res", result)
        assert "HO" in result
        assert len(result) == 1
        assert "1" in result["HO"]
        assert len(result["HO"]) == 4
        assert sorted(result["HO"]) == ['1', '2', '3', '4']
        assert "6" not in result["HO"]
        assert "5" not in result["HO"]

    def test_get_subcells_in_bbox_buffer_expansion(self):
        """Verify that buffer correctly expands the search area."""
        # Use a small range where no points would normally be found except for buffer
        pt1 = (50.0, 10.0)
        pt2 = (50.01, 10.01)
        
        with patch("geom_helpers.latlon_subcells.get_coord_code") as mock_get:
            mock_get.return_value = "AA11"
            # Large buffer to ensure it wraps more points
            get_subcells_in_bbox(pt1, pt2, buffer=1.0)
            
            # Check if any call used a buffered coordinate (e.g., 49.0)
            args, _ = mock_get.call_args_list[0]
            assert args[0] < 50.0 

    @patch("geom_helpers.latlon_subcells.get_coord_code")
    def test_get_subcells_invalid_coord_handling(self, mock_get_coord):
        """Ensure InvalidCoordinateError is caught and skipped."""
        mock_get_coord.side_effect = [InvalidCoordinateError, "BB22"]
        
        pt1 = (50.0, 10.0)
        pt2 = (50.1, 10.2)
        
        result = get_subcells_in_bbox(pt1, pt2, dimx=0.1, dimy=0.1)
        
        # Should only contain the second result
        assert "BB" in result
        assert "22" in result["BB"]

    @patch("geom_helpers.latlon_subcells.get_coord_code")
    def test_get_subcells_generic_exception_handling(self, mock_get_coord, capsys):
        """Ensure generic exceptions are logged and skipped."""
        mock_get_coord.side_effect = Exception("Critical Failure")
        
        pt1 = (50.0, 10.0)
        pt2 = (50.1, 10.1)
        
        result = get_subcells_in_bbox(pt1, pt2)
        
        captured = capsys.readouterr()
        assert "ERROR get_subcells_in_bbox" in captured.out
        assert result == {}


# type: ignore
import unittest
import math

# python -m unittest test_osm_reader_helper.py

#import osm_reader_helper
from geom_helpers.distance_helper import get_dist
from geom_helpers.osm_reader_helper import InvalidCoordinateError, InvalidCellCodeError
from geom_helpers.osm_reader_helper import F_MAP_CELL, F_MAP_CODE, ICEL_CODES, REV_MAP_CELL, REV_MAP_CODE
from geom_helpers.osm_reader_helper import (
    CABO_CELLS, MADE_CELLS, CANA_CELLS, AZOR_CELLS, ICEL_CELLS) # MEDI_CELLS, BASE_CELLS

from geom_helpers.osm_reader_helper import (
    get_coord_code_azores, get_coord_code_madeira, get_coord_code_canary, 
    get_coord_code_caboverde, get_coord_code_iceland, get_coord_code_med, 
    get_coord_code_base, get_coord_code)

from geom_helpers.osm_reader_helper import encode_id, decode_id
from geom_helpers.osm_reader_helper import (
    get_coord_cell_azores, get_coord_cell_madeira, get_coord_cell_canary, 
    get_coord_cell_caboverde, get_coord_cell_iceland, get_coord_cell_med, 
    get_coord_cell_base, get_coord_cell)


from geom_helpers.osm_reader_helper import (
    get_rev_coord_code_caboverde, get_rev_coord_code_iceland, get_rev_coord_code_madeira)
from geom_helpers.osm_reader_helper import (
    get_rev_coord_code_base, get_rev_coord_code_canary, get_rev_coord_code_azores, get_rev_coord_code)



class TestGetDist(unittest.TestCase):
    def setUp(self):
        self.M = 111320  # Approx. meters per degree latitude
    
    def test_get_dist(self):
        self.assertAlmostEqual(get_dist((0, 0), (1, 0)), self.M, delta=1)
        self.assertAlmostEqual(get_dist((0, 0), (3, 0)), 3*self.M, delta=1)
        self.assertAlmostEqual(get_dist((0, 0), (0, 1)), self.M, delta=1)
        self.assertAlmostEqual(get_dist((1, 0), (0, 1)), 
                               math.sqrt(self.M ** 2 + (self.M * math.cos(math.radians(0.5))) ** 2), 
                               f"ERR: {(self.M * math.cos(math.radians(0.5))) ** 2}", delta=1)
        self.assertAlmostEqual(get_dist((0, 45), (0, 46)), self.M)
        self.assertAlmostEqual(get_dist((41, 10), (40, 11)), 
                               math.sqrt(self.M ** 2 + (self.M * math.cos(math.radians(40.5))) ** 2), 
                               f"ERR: {(self.M * math.cos(math.radians(40.5))) ** 2}", delta=1)
        self.assertAlmostEqual(get_dist((1, -1), (0, 1)), 
                               math.sqrt(self.M ** 2 + (2 * self.M * math.cos(math.radians(0.5))) ** 2), 
                               f"ERR: {(self.M * math.cos(math.radians(0.5))) ** 2}", delta=1)

    def test_get_dist_with_elev(self):
        self.assertAlmostEqual(get_dist((0, 0, 10), (0, 0, 0)), 10, delta=1)
        self.assertAlmostEqual(get_dist((0, 0, 0), (1, 0, 10)), self.M, delta=1)
        self.assertAlmostEqual(get_dist((0, 0, 0), (1, 0, 100)), self.M, delta=1)
        self.assertAlmostEqual(get_dist((0, 0, 0), (1, 0, 1000)), self.M + 5, delta=1)
        self.assertAlmostEqual(get_dist((0, 0, 10000), (1, 0, 1000)), self.M + 363, delta=1)
        self.assertAlmostEqual(get_dist((0, 0, 10000), (1, 0, 0)), self.M + 448, delta=1)

class TestGetCoordCode(unittest.TestCase):
    def test_get_coord_code_azores(self):
        self.assertEqual(get_coord_code_azores(37.3, -25.2), '3016')
        self.assertEqual(get_coord_code_azores(38.45789, -27.11), '3024')
        #print("XXXX  ", type(get_coord_code_azores(47.3, -25.2)), "  XXXX")
        self.assertRaises(InvalidCoordinateError, get_coord_code_azores, 47.3, -25.2)
        with self.assertRaises(InvalidCoordinateError):
            get_coord_code_azores(47.3, -25.2)  # Only call it *inside* the block
    
    def test_get_coord_code_madeira(self):
        self.assertEqual(get_coord_code_madeira(32.456, -17.231), '2000')
        self.assertEqual(get_coord_code_madeira(33.186, -16.771), '2011')
        self.assertRaises(InvalidCoordinateError, get_coord_code_madeira, 47.3, -25.2)

    def test_get_coord_code_canary(self):
        self.assertEqual(get_coord_code_canary(28.0, -16.0), '1023')
        self.assertEqual(get_coord_code_canary(29.65, -15.72), '1033')
        self.assertEqual(get_coord_code_canary(29.65, -13.72), '1035')
        self.assertRaises(InvalidCoordinateError, get_coord_code_canary, 47.3, -25.2)

    def test_get_coord_code_caboverde(self):
        self.assertEqual(get_coord_code_caboverde(16.20, -24.49), '0022')
        self.assertEqual(get_coord_code_caboverde(17.440, -23.11), '0033')
        self.assertRaises(InvalidCoordinateError, get_coord_code_caboverde, 9.3, -25.2)

    def test_get_coord_code_med(self):
        self.assertEqual(get_coord_code_med(34.5, 15.6), 'yQ56')
        self.assertEqual(get_coord_code_med(35.36, 8.44), 'zJ34')
        self.assertEqual(get_coord_code_med(35.1, 35.99), 'zk19')
        self.assertRaises(InvalidCoordinateError, get_coord_code_med, 33.3, 25.2)

    def test_get_coord_code_iceland(self):
        self.assertEqual(get_coord_code_iceland(64.601, -17.77), 'S260')
        self.assertEqual(get_coord_code_iceland(65.05, -19.10), 'T107')
        self.assertEqual(get_coord_code_iceland(63.46, -13.4), 'R341')
        self.assertRaises(InvalidCoordinateError, get_coord_code_caboverde, 70.3, -17.2)

    def test_get_coord_code_base(self):
        self.assertEqual(get_coord_code_base(36.0, -11.0), '0000')
        self.assertEqual(get_coord_code_base(48.0, 12.0), 'CN00')
        self.assertEqual(get_coord_code_base(71.1, 25.4), 'Za14')
        self.assertEqual(get_coord_code_base(71.2, 26.5), 'Zb25')
        self.assertEqual(get_coord_code_base(71.35, 44.08), 'Zt30')
        self.assertRaises(InvalidCoordinateError, get_coord_code_base, 70.3, -17.2)

    def test_get_coord_code(self):
        self.assertEqual(get_coord_code(37.0, -25.0), '3017')
        self.assertEqual(get_coord_code(32.0, -17.0), '2001')
        self.assertEqual(get_coord_code(28.0, -16.0), '1023')
        self.assertEqual(get_coord_code(16.0, -24.0), '0023')
        self.assertEqual(get_coord_code(64.0, -18.0), 'S200')
        self.assertEqual(get_coord_code(34.5, 15.6), 'yQ56')
        self.assertEqual(get_coord_code(48.44, 11.50), 'CM45')
        self.assertRaises(InvalidCoordinateError, get_coord_code, 70.3, -17.2)
        self.assertRaises(InvalidCoordinateError, get_coord_code, 32.3, 17.2)


class TestGetCoordCellCode(unittest.TestCase):
    def test_get_coord_cell_azores(self):
        self.assertEqual(get_coord_cell_azores(37.0, -25.0), '30')
        self.assertEqual(get_coord_cell_azores(38.0, -27.0), '30')

    def test_get_coord_cell_madeira(self):
        self.assertEqual(get_coord_cell_madeira(32.0, -17.0), '20')
        self.assertEqual(get_coord_cell_madeira(33.0, -16.0), '20')

    def test_get_coord_cell_canary(self):
        self.assertEqual(get_coord_cell_canary(28.0, -16.0), '10')
        self.assertEqual(get_coord_cell_canary(29.0, -15.0), '10')

    def test_get_coord_cell_caboverde(self):
        self.assertEqual(get_coord_cell_caboverde(16.0, -24.0), '00')
        self.assertEqual(get_coord_cell_caboverde(17.0, -23.0), '00')

    def test_get_coord_cell_iceland(self):
        self.assertEqual(get_coord_cell_iceland(64.0, -18.0), 'S2')
        self.assertEqual(get_coord_cell_iceland(65.0, -19.0), 'T1')
        self.assertEqual(get_coord_cell_iceland(63.06, -13.4), 'R3')

    def test_get_coord_cell_med(self):
        self.assertEqual(get_coord_cell_med(34.5, 15.6), 'yQ')
        self.assertEqual(get_coord_cell_med(35.36, 8.4), 'zJ')
        self.assertEqual(get_coord_cell_med(35.1, 35.99), 'zk')

    def test_get_coord_cell_base(self):
        self.assertEqual(get_coord_cell_base(36.0, -11.0), '00')
        self.assertEqual(get_coord_cell_base(48.0, 12.0), 'CN')
        self.assertEqual(get_coord_cell_base(71.1, 25.4), 'Za')
        self.assertEqual(get_coord_cell_base(71.2, 26.5), 'Zb')
        self.assertEqual(get_coord_cell_base(71.3, 44.8), 'Zt')
        self.assertEqual(get_coord_cell_base(71.3, -2.8), 'Z8')
        self.assertEqual(get_coord_cell_base(71.3, -1.8), 'Z9')
        self.assertEqual(get_coord_cell_base(71.3, -0.8), 'ZA')
        self.assertEqual(get_coord_cell_base(71.3, 0.8), 'ZB')
        self.assertEqual(get_coord_cell_base(71.3, 1.8), 'ZC')
        self.assertEqual(get_coord_cell_base(51.3, -2.8), 'F8')
        self.assertEqual(get_coord_cell_base(51.3, -1.8), 'F9')
        self.assertEqual(get_coord_cell_base(51.3, -0.8), 'FA')
        self.assertEqual(get_coord_cell_base(51.3, 0.8), 'FB')
        self.assertEqual(get_coord_cell_base(51.3, 1.8), 'FC')

    def test_get_coord_cell(self):
        self.assertEqual(get_coord_cell(37.0, -25.0), '30')
        self.assertEqual(get_coord_cell(32.0, -17.0), '20')
        self.assertEqual(get_coord_cell(28.0, -16.0), '10')
        self.assertEqual(get_coord_cell(16.0, -24.0), '00')
        self.assertEqual(get_coord_cell(64.0, -18.0), 'S2')
        self.assertEqual(get_coord_cell(34.5, 15.6), 'yQ')
        self.assertEqual(get_coord_cell(36.0, -11.0), '00')
        self.assertEqual(get_coord_cell(47.0, 2.01), 'BD')
        self.assertEqual(get_coord_cell(53.2, 15.5), 'HQ')
        self.assertEqual(get_coord_cell(71.1, 25.4), 'Za')
        self.assertEqual(get_coord_cell(71.2, 26.5), 'Zb')
        self.assertEqual(get_coord_cell(71.3, 44.8), 'Zt')
        self.assertEqual(get_coord_cell(71.3, -2.8), 'Z8')
        self.assertEqual(get_coord_cell(71.3, -1.8), 'Z9')
        self.assertEqual(get_coord_cell(71.3, -0.8), 'ZA')
        self.assertEqual(get_coord_cell(71.3, 0.8), 'ZB')
        self.assertEqual(get_coord_cell(71.3, 1.8), 'ZC')
        self.assertEqual(get_coord_cell(51.3, -2.8), 'F8')
        self.assertEqual(get_coord_cell(51.3, -1.8), 'F9')
        self.assertEqual(get_coord_cell(51.3, -0.8), 'FA')
        self.assertEqual(get_coord_cell(51.3, 0.8), 'FB')
        self.assertEqual(get_coord_cell(51.3, 1.8), 'FC')


class TestGetCoordCell(unittest.TestCase):
    def test_get_coord_cell_azores(self):
        self.assertEqual(get_coord_code_azores(37.0, -25.0), '3017', '3017')
        self.assertEqual(get_coord_code_azores(38.0, -27.0), '3025', '3025')

    def test_get_coord_cell_madeira(self):  # mad_bbox = (-18, 32, -16, 34)
        self.assertEqual(get_coord_code_madeira(32.0, -17.0), '2001', '2001')
        self.assertEqual(get_coord_code_madeira(33.0, -16.0), '2012', '2012')

    def test_get_coord_cell_canary(self): # can_bbox = (-19, 26, -12, 31)
        self.assertEqual(get_coord_code_canary(28.0, -16.0), '1023', '1023')
        self.assertEqual(get_coord_code_canary(29.0, -15.0), '1034', '1034')

    def test_get_coord_cell_caboverde(self):  # cpv_bbox = (-27, 14, -21, 19)
        self.assertEqual(get_coord_code_caboverde(16.0, -24.0), '0023', '0023')
        self.assertEqual(get_coord_code_caboverde(17.0, -23.0), '0034', '0034')

    def test_get_coord_cell_iceland(self):   # isl_bbox = (-26, 62, -12, 68)
        self.assertEqual(get_coord_code_iceland(64.81, -18.0), 'S280', 'S280')
        self.assertEqual(get_coord_code_iceland(65.0, -19.23), 'T106', 'T106')
        self.assertEqual(get_coord_code_iceland(63.26, -13.4), 'R321', 'R321')

    def test_get_coord_cell_med(self):   # med_bbox = (-7, 34, 45, 36)
        self.assertEqual(get_coord_code_med(34.5, 15.6), 'yQ56', 'yQ56')
        self.assertEqual(get_coord_code_med(35.36, 8.4), 'zJ34', 'zJ34')
        self.assertEqual(get_coord_code_med(35.1, 35.99), 'zk19', 'zk19')

    def test_get_coord_cell_base(self):  # base_bbox = (-11, 36, 45, 72)
        self.assertEqual(get_coord_code_base(36.0, -11.0), '0000', '0000')
        self.assertEqual(get_coord_code_base(48.0, 12.0), 'CN00', 'CN00')
        self.assertEqual(get_coord_code_base(71.1, 25.4), 'Za14', 'Za14')
        self.assertEqual(get_coord_code_base(71.22, 26.57), 'Zb25', 'Zb25')
        self.assertEqual(get_coord_code_base(71.3, 44.8), 'Zt38', 'Zt38')
        self.assertEqual(get_coord_code_base(71.3, -2.8), 'Z838', 'Z838')
        self.assertEqual(get_coord_code_base(71.3, -1.8), 'Z938', 'Z938')
        self.assertEqual(get_coord_code_base(71.3, -0.8), 'ZA38', 'ZA38')
        self.assertEqual(get_coord_code_base(71.3, 0.8), 'ZB38', 'ZB38')
        self.assertEqual(get_coord_code_base(71.3, 1.8), 'ZC38', 'ZC38')
        self.assertEqual(get_coord_code_base(51.3, -2.8), 'F838', 'F838')
        self.assertEqual(get_coord_code_base(51.3, -1.8), 'F938', 'F938')
        self.assertEqual(get_coord_code_base(51.3, -0.8), 'FA38', 'FA38')
        self.assertEqual(get_coord_code_base(51.3, 0.8), 'FB38', 'FB38')
        self.assertEqual(get_coord_code_base(51.3, 1.8), 'FC38', 'FC38')

    def test_get_coord_cell(self):
        self.assertEqual(get_coord_code(37.0, -25.0), '3017', '3017')
        self.assertEqual(get_coord_code(32.0, -17.0), '2001', '2001')
        self.assertEqual(get_coord_code(28.0, -16.0), '1023', '1023')
        self.assertEqual(get_coord_code(16.0, -24.0), '0023', '0023')
        self.assertEqual(get_coord_code(64.4, -18.1), 'S149', 'S149')
        self.assertEqual(get_coord_code(64.4, -17.91), 'S240', 'S240')
        self.assertEqual(get_coord_code(34.5, 15.6), 'yQ56', 'yQ56')
        self.assertEqual(get_coord_code(36.345, -11.713), '0037', '0037')
        self.assertEqual(get_coord_code(47.987, 2.01), 'BD90', 'BD90')
        self.assertEqual(get_coord_code(53.2, 15.5), 'HQ25', 'HQ25')
        self.assertEqual(get_coord_code(36.0, -11.0), '0000', '0000')
        self.assertEqual(get_coord_code(48.0, 12.0), 'CN00', 'CN00')
        self.assertEqual(get_coord_code(71.1, 25.4), 'Za14', 'Za14')
        self.assertEqual(get_coord_code(71.2, 26.5), 'Zb25', 'Zb25')
        self.assertEqual(get_coord_code(71.3, 44.8), 'Zt38', 'Zt38')
        self.assertEqual(get_coord_code(71.3, -2.8), 'Z838', 'Z838')
        self.assertEqual(get_coord_code(71.3, -1.8), 'Z938', 'Z938')
        self.assertEqual(get_coord_code(71.3, -0.8), 'ZA38', 'ZA38')
        self.assertEqual(get_coord_code(71.3, 0.8), 'ZB38', 'ZB38')
        self.assertEqual(get_coord_code(71.3, 1.8), 'ZC38', 'ZC38')
        self.assertEqual(get_coord_code(51.3, -2.8), 'F838', 'F838')
        self.assertEqual(get_coord_code(51.3, -1.8), 'F938', 'F938')
        self.assertEqual(get_coord_code(51.3, -0.8), 'FA38', 'FA38')
        self.assertEqual(get_coord_code(51.3, 0.8), 'FB38', 'FB38')
        self.assertEqual(get_coord_code(51.3, 1.8), 'FC38', 'FC38')


class TestConstants(unittest.TestCase):
    def test_f_map_cell_has_expected_keys(self):
        # Example key you're expecting to be present
        self.assertIn((38, -27), F_MAP_CELL)
        self.assertTrue(len(F_MAP_CELL) > len(REV_MAP_CELL))

    def test_f_map_cell_function_behavior(self):
        # Check the function for a specific lat/lon returns expected output
        lat, lon = 38.5, -27.1
        self.assertIn((int(lat), int(lon)), F_MAP_CELL)
        func = F_MAP_CELL[(int(lat), int(lon))]
        result = func(lat, lon)
        self.assertEqual(result, "30")  # Replace with real expected value

    def test_f_map_code_has_expected_keys(self):
        # Example key you're expecting to be present
        self.assertIn((48, 14), F_MAP_CODE)
        self.assertTrue(len(F_MAP_CODE) > len(REV_MAP_CODE))
        self.assertTrue(len(F_MAP_CELL) == len(F_MAP_CODE))

    def test_f_map_code_function_behavior(self):
        # Check the function for a specific lat/lon returns expected output
        lat, lon = 46.52, 10.11
        self.assertIn((int(lat), int(lon)), F_MAP_CODE)
        func = F_MAP_CODE[(int(lat), int(lon))]
        result = func(lat, lon)
        self.assertEqual(result, "AL51")  # Replace with real expected value

    def test_icel_codes_length(self):
        self.assertEqual(len(ICEL_CODES), 20)

    def test_icel_codes_contains_expected(self):
        self.assertIn("R3", ICEL_CODES)  

    def test_rev_map_cell_has_expected_keys(self):
        self.assertIn("00", REV_MAP_CELL)
        self.assertIn("10", REV_MAP_CELL)
        self.assertIn("20", REV_MAP_CELL)
        self.assertIn("30", REV_MAP_CELL)
        self.assertIn("R3", REV_MAP_CELL)
    
    def test_rev_map_cell_has_expected_values(self):
        self.assertIn(sorted(CABO_CELLS)[5], REV_MAP_CELL["00"], sorted(CABO_CELLS)[5])
        self.assertIn(sorted(CANA_CELLS)[5], REV_MAP_CELL["10"], sorted(CANA_CELLS)[5])
        self.assertIn(sorted(MADE_CELLS)[2], REV_MAP_CELL["20"], sorted(MADE_CELLS)[2])
        self.assertIn(sorted(AZOR_CELLS)[6], REV_MAP_CELL["30"], sorted(AZOR_CELLS)[6])
        #for i, x in enumerate(sorted(ICEL_CELLS)):
        #    cell = get_coord_cell(x[0], x[1])
        #    print(cell, i, x, cell in REV_MAP_CELL, REV_MAP_CELL[cell])
        self.assertIn(sorted(ICEL_CELLS)[6], REV_MAP_CELL["R1"], sorted(ICEL_CELLS)[6])
        self.assertIn(sorted(ICEL_CELLS)[10], REV_MAP_CELL["R2"], sorted(ICEL_CELLS)[10])
        self.assertIn(sorted(ICEL_CELLS)[18], REV_MAP_CELL["S0"], sorted(ICEL_CELLS)[18])
        self.assertIn((48,12), REV_MAP_CELL["CN"], (48,12))
        self.assertEqual(len(CABO_CELLS), len(REV_MAP_CELL["00"])-1)
        self.assertEqual(len(CANA_CELLS), len(REV_MAP_CELL["10"])-1)
        self.assertEqual(len(MADE_CELLS), len(REV_MAP_CELL["20"])-1)
        self.assertEqual(len(AZOR_CELLS), len(REV_MAP_CELL["30"])-1)
        self.assertEqual(len(REV_MAP_CELL["S2"]), 5)
        self.assertEqual(len(REV_MAP_CELL["T1"]), 5)
        self.assertEqual(len(REV_MAP_CELL["ZH"]), 1)
        self.assertEqual(len(REV_MAP_CELL["yK"]), 1)

    def test_rev_map_code_has_expected_keys(self):
        self.assertIn("CM", REV_MAP_CODE)
        self.assertIn("Zh", REV_MAP_CODE)
        self.assertIn("10", REV_MAP_CODE)
        self.assertIn("yB", REV_MAP_CODE)

    def test_rev_map_code_has_expected_funcs(self):
        self.assertEqual(REV_MAP_CODE["00"][0], get_rev_coord_code_caboverde)
        self.assertEqual(REV_MAP_CODE["10"][0], get_rev_coord_code_canary)
        self.assertEqual(REV_MAP_CODE["20"][0], get_rev_coord_code_madeira)
        self.assertEqual(REV_MAP_CODE["30"][0], get_rev_coord_code_azores)
        self.assertEqual(REV_MAP_CODE["S2"][0], get_rev_coord_code_iceland)
        self.assertEqual(REV_MAP_CODE["BO"][0], get_rev_coord_code_base)

    def test_rev_map_cell_function_behavior(self):
        # Check the function for a specific cell returns expected output
        cell = "CM"
        self.assertIn(cell, REV_MAP_CELL, 
                      f"ERROR {cell} not found in REV_MAP_CELL - test_rev_map_cell_function_behavior")
        result = REV_MAP_CELL[cell]
        #result = func(cell)
        self.assertEqual(result, [(48, 11)], 
                         f"ERROR {cell} returned different result - test_rev_map_cell_function_behavior") 

        cell = "Cz"
        self.assertFalse(cell in REV_MAP_CELL, 
                         f"ERROR {cell} should not be in REV_MAP_CELL - test_rev_map_cell_function_behavior")
        code = "CM26"
        self.assertFalse(code in REV_MAP_CELL, 
                         f"ERROR {code} should not be in REV_MAP_CELL - test_rev_map_cell_function_behavior")
        
        for cell, num_elements in [("00", len(CABO_CELLS)+1), ("10", len(CANA_CELLS)+1), 
                                   ("20", len(MADE_CELLS)+1), ("30", len(AZOR_CELLS)+1), 
                                   ("S3", 4), ("R2", 5), ("T1", 5), ("U0", 5),  
                                   ("CM", 1)] + [(c, 1) for c in ["AB", "DG", "KJ", "yQ", "zJ", "zk"]]: 
            with self.subTest(cell=cell):
                self.assertIn(cell, REV_MAP_CELL, 
                          f"ERROR subtest {cell} not found in REV_MAP_CELL - test_rev_map_cell_function_behavior")
                result = REV_MAP_CELL[cell]
                self.assertTrue(len(result) == num_elements, 
                                f"ERROR subtest {cell} returned invalid list {result} {len(result)} {num_elements} - test_rev_map_cell_function_behavior")
    
    def test_rev_map_code_function_behavior(self):
        # Check the function for a specific code returns expected output
        code = "CM25"
        self.assertIn(code[:2], REV_MAP_CODE, 
                      f"ERROR {code[:2]} not found in REV_MAP_CODE - test_rev_map_code_function_behavior")
        func, _, _ = REV_MAP_CODE[code[:2]]
        result = func(code)
        self.assertEqual(result, (48.2, 11.5), 
                         f"ERROR {code} returned different result - test_rev_map_code_function_behavior") 

        code = "Cz"
        self.assertFalse(code[:2] in REV_MAP_CODE, 
                         f"ERROR {code[:2]} should not be in REV_MAP_CODE - test_rev_map_code_function_behavior")
        
        code = "DM46"
        self.assertTrue(code[:2] in REV_MAP_CODE, 
                         f"ERROR {code[:2]} should be in REV_MAP_CODE - test_rev_map_code_function_behavior")

    def test_get_rev_coord_code_caboverde(self):
        self.assertEqual(get_rev_coord_code_caboverde("0000"), (14, -27))
        self.assertEqual(get_rev_coord_code_caboverde("0022"), (16, -25))
        self.assertEqual(get_rev_coord_code_caboverde("0033"), (17, -24))
        self.assertRaises(InvalidCellCodeError, get_rev_coord_code_caboverde, "2080")
        self.assertRaises(InvalidCellCodeError, get_rev_coord_code_caboverde, "2038")

    def test_get_rev_coord_code(self):
        self.assertEqual(get_rev_coord_code("CM00"), (48, 11))
        self.assertEqual(get_rev_coord_code("CM36"), (48.3, 11.6))
        self.assertEqual(get_rev_coord_code("AD15"), (46.1, 2.5))
        self.assertEqual(get_rev_coord_code("1000"), (26, -19))
        self.assertEqual(get_rev_coord_code("yN90"), (34.9, 12))
        self.assertEqual(get_rev_coord_code("3b07"), (39, 26.7))
        self.assertEqual(get_rev_coord_code("1022"), (28, -17))
        self.assertEqual(get_rev_coord_code("2012"), (33, -16))
        self.assertEqual(get_rev_coord_code("3000"), (36, -32))
        self.assertEqual(get_rev_coord_code("3012"), (37, -30))
        self.assertEqual(get_rev_coord_code("R025"), (63.2, -24))
        self.assertEqual(get_rev_coord_code("R100"), (63, -22))
        self.assertEqual(get_rev_coord_code("R138"), (63.3, -18.8))

    
    def test_get_rev_coord_code_canary(self):
        self.assertEqual(get_rev_coord_code_canary("1000"), (26, -19))
        self.assertEqual(get_rev_coord_code_canary("1022"), (28, -17))
        self.assertEqual(get_rev_coord_code_canary("1033"), (29, -16))
        self.assertRaises(InvalidCellCodeError, get_rev_coord_code_canary, "3090")
        self.assertRaises(InvalidCellCodeError, get_rev_coord_code_canary, "3039")

    def test_get_rev_coord_code_madeira(self):
        self.assertEqual(get_rev_coord_code_madeira("2000"), (32, -18))
        self.assertEqual(get_rev_coord_code_madeira("2012"), (33, -16))
        self.assertEqual(get_rev_coord_code_madeira("2001"), (32, -17))

    def test_get_rev_coord_code_azores(self):
        self.assertEqual(get_rev_coord_code_azores("3000"), (36, -32))
        self.assertEqual(get_rev_coord_code_azores("3012"), (37, -30))
        self.assertEqual(get_rev_coord_code_azores("3014"), (37, -28))

    def test_get_rev_coord_code_iceland(self):
        #print("\nA1 (63, -25.8)", get_coord_code(63, -25.8))
        #print("A2 (63, -24.8)", get_coord_code(63, -24.8))
        #print("A3 (63, -23.8)", get_coord_code(63, -23.8))
        #print("A4 (63, -22.8)", get_coord_code(63, -22.8))
        #print("B2 (66, -16.28)", get_coord_code(66, -16.28))
        #print("B3 (65, -17.78)", get_coord_code(65, -17.78))
        #print("B4 (63, -22.8)", get_coord_code(64, -22.8))
        #print("V1 (67.7, -21.6)", get_coord_code(67.723, -21.523))
        self.assertEqual(get_rev_coord_code_iceland("R000"), (63, -26))
        self.assertEqual(get_rev_coord_code_iceland("R003"), (63, -24.8))
        self.assertEqual(get_rev_coord_code_iceland("R005"), (63, -24))
        self.assertEqual(get_rev_coord_code_iceland("R025"), (63.2, -24))
        self.assertEqual(get_rev_coord_code_iceland("R100"), (63, -22))
        self.assertEqual(get_rev_coord_code_iceland("R138"), (63.3, -18.8))
        self.assertEqual(get_rev_coord_code_iceland("T200"), (65, -18))  
        self.assertEqual(get_rev_coord_code_iceland("T241"), (65.4, -17.6))
        self.assertEqual(get_rev_coord_code_iceland("V171"), (67.7, -21.6))
        self.assertRaises(InvalidCellCodeError, get_rev_coord_code_iceland, "X000")
        self.assertRaises(InvalidCellCodeError, get_rev_coord_code_iceland, "W171")

class TestIDEncodeDecode(unittest.TestCase):
    def test_encode_id(self):
        self.assertEqual(encode_id(1000), '0000G8')
        self.assertEqual(encode_id(10001000), '00fxiS')

    def test_decode_id(self):
        self.assertEqual(decode_id('0000G8'), 1000)
        self.assertEqual(decode_id('00fxiS'), 10001000)


#if __name__ == '__main__':
#    unittest.main()
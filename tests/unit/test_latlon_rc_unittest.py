# type: ignore
import unittest

from geom_helpers.latlon_rc import NUM_ROWS_COLS_BY_CELL #, NUM_ROWS_COLS
from geom_helpers.latlon_rc import get_sub_rc_nghbr, get_sub_rc_nghbrs, get_sub_rc_all_nghbrs, get_sub_rc_x_nghbrs

# python -m unittest test_latlon_rc.py

class TestGetSubRCNghbr(unittest.TestCase):
    def test_get_sub_rc_nghbr1(self):
        code = 'BM33-331234'
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        for n, (args, exp_res) in enumerate([[(0, 0, -1, True), (0, 0)], 
                                             [(0, 0, 1, True), (0, 1)], 
                                             [(0, 0, -1, False), (0, 0)], 
                                             [(0, 0, 1, False), (1, 0)]]):
            res = get_sub_rc_nghbr(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbr 1 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_nghbr2(self):
        code = 'BM33-331234'
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        for n, (args, exp_res) in enumerate([[(3, 3, -1, True), (3, 2)], 
                                             [(3, 3, 1, True), (3, 4)], 
                                             [(3, 3, -1, False), (2, 3)], 
                                             [(3, 3, 1, False), (4, 3)]]):
            res = get_sub_rc_nghbr(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbr 2 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_nghbr3(self):
        code = 'BM33-331234'
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        for n, (args, exp_res) in enumerate([[(3, 0, -1, True), (3, 0)], 
                                             [(3, 0, 1, True), (3, 1)], 
                                             [(3, 0, -1, False), (2, 0)], 
                                             [(3, 0, 1, False), (4, 0)]]):
            res = get_sub_rc_nghbr(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbr 3 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_nghbr4(self):
        code = 'BM33-331234'
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        for n, (args, exp_res) in enumerate([[(7, 7, -1, True), (7, 6)], 
                                             [(7, 7, 1, True), (7, 8)], 
                                             [(7, 7, -1, False), (6, 7)], 
                                             [(7, 7, 1, False), (8, 7)]]):
            res = get_sub_rc_nghbr(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbr 4 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_nghbr5(self):
        code = 'BM33-331234'
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        for n, (args, exp_res) in enumerate([[(7, 8, -1, True), (7, 7)], 
                                             [(7, 8, 1, True), (7, 8)], 
                                             [(7, 8, -1, False), (6, 8)], 
                                             [(7, 8, 1, False), (8, 8)]]):
            res = get_sub_rc_nghbr(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbr 5 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_nghbr6(self):
        code = 'BM33-331234'
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        for n, (args, exp_res) in enumerate([[(8, 8, -1, True), (8, 7)], 
                                             [(8, 8, 1, True), (8, 8)], 
                                             [(8, 8, -1, False), (7, 8)], 
                                             [(8, 8, 1, False), (8, 8)]]):
            res = get_sub_rc_nghbr(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbr 6 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_nghbr7(self):
        code = 'BM33-331234'
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        for n, (args, exp_res) in enumerate([[(0, 8, -1, True), (0, 7)], 
                                             [(0, 8, 1, True), (0, 8)], 
                                             [(0, 8, -1, False), (0, 8)], 
                                             [(0, 8, 1, False), (1, 8)]]):
            res = get_sub_rc_nghbr(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbr 7 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_nghbr8(self):
        code = 'BM33-331234'
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        for n, (args, exp_res) in enumerate([[(8, 0, -1, True), (8, 0)], 
                                             [(8, 0, 1, True), (8, 1)], 
                                             [(8, 0, -1, False), (7, 0)], 
                                             [(8, 0, 1, False), (8, 0)]]):
            res = get_sub_rc_nghbr(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbr 8 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_nghbr9(self):
        code = 'JM33-331234' # 8, 7
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        r, c = 6, 4
        for n, (args, exp_res) in enumerate([[(r, c, -1, True), (r, c-1)], 
                                             [(r, c, 1, True), (r, c+1)], 
                                             [(r, c, -1, False), (r-1, c)], 
                                             [(r, c, 1, False), (r+1, c)]]):
            res = get_sub_rc_nghbr(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbr 9 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_nghbr10(self):
        code = 'JM33-331234' # 8, 7
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        r, c = 2, 5
        for n, (args, exp_res) in enumerate([[(r, c, -1, True), (r, c-1)], 
                                             [(r, c, 1, True), (r, c+1)], 
                                             [(r, c, -1, False), (r-1, c)], 
                                             [(r, c, 1, False), (r+1, c)]]):
            res = get_sub_rc_nghbr(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbr 10 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_nghbr11(self):
        code = 'JM33-331234' # 8, 7
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        r, c = 7, 5
        for n, (args, exp_res) in enumerate([[(r, c, -1, True), (r, c-1)], 
                                             [(r, c, 1, True), (r, c+1)], 
                                             [(r, c, -1, False), (r-1, c)], 
                                             [(r, c, 1, False), (r, c)]]):
            res = get_sub_rc_nghbr(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbr 11 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_nghbr12(self):
        code = 'JM33-331234' # 8, 7
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        r, c = 7, 6
        for n, (args, exp_res) in enumerate([[(r, c, -1, True), (r, c-1)], 
                                             [(r, c, 1, True), (r, c)], 
                                             [(r, c, -1, False), (r-1, c)], 
                                             [(r, c, 1, False), (r, c)]]):
            res = get_sub_rc_nghbr(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbr 12 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_nghbr13(self):
        code = 'ZM33-331234' # 10, 5
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        r, c = 7, 4
        for n, (args, exp_res) in enumerate([[(r, c, -1, True), (r, c-1)], 
                                             [(r, c, 1, True), (r, c)], 
                                             [(r, c, -1, False), (r-1, c)], 
                                             [(r, c, 1, False), (r+1, c)]]):
            res = get_sub_rc_nghbr(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbr 13 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_nghbr14(self):
        code = 'ZM33-331234' # 10, 5
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        r, c = 9, 4
        for n, (args, exp_res) in enumerate([[(r, c, -1, True), (r, c-1)], 
                                             [(r, c, 1, True), (r, c)], 
                                             [(r, c, -1, False), (r-1, c)], 
                                             [(r, c, 1, False), (r, c)]]):
            res = get_sub_rc_nghbr(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbr 14 #{n}: {res} vs {exp_res}")

class TestGetSubRCNghbrs(unittest.TestCase):
    def test_get_sub_rc_nghbrs1(self):
        code = 'BM33-331234'
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        for n, (args, exp_res) in enumerate([[(0, 0), [(0, 0), (1, 0), (0, 1), (1, 1)]], 
                                             [(0, 8), []], 
                                             [(2, 4), [(2, 4), (3, 4), (2, 5), (3, 5)]], 
                                             [(6, 1), [(6, 1), (7, 1), (6, 2), (7, 2)]], 
                                             [(6, 8), []], 
                                             [(8, 4), []], 
                                             [(0, 4), [(0, 4), (1, 4), (0, 5), (1, 5)]], 
                                             [(8, 0), []], 
                                             [(8, 8), []]]):
            res = get_sub_rc_nghbrs(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbrs 1 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_nghbrs2(self):
        code = 'JM33-331234'  # 8, 7
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        for n, (args, exp_res) in enumerate([[(0, 0), [(0, 0), (1, 0), (0, 1), (1, 1)]], 
                                             [(0, 6), []], 
                                             [(7, 5), []], 
                                             [(2, 4), [(2, 4), (3, 4), (2, 5), (3, 5)]], 
                                             [(5, 1), [(5, 1), (6, 1), (5, 2), (6, 2)]], 
                                             [(0, 4), [(0, 4), (1, 4), (0, 5), (1, 5)]], 
                                             [(7, 0), []], 
                                             [(7, 6), []]]):
            res = get_sub_rc_nghbrs(*args, grid_num_w, grid_num_h)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_nghbrs 2 #{n}: {res} vs {exp_res}")


class TestGetSubRCAllNghbrs(unittest.TestCase):
    def test_get_sub_rc_all_nghbrs1(self):
        code = 'BM33-331234'
        for n, (args, exp_res) in enumerate([[(code, 0, 0), [[(0, 0), (1, 0), (0, 1), (1, 1)], 
                                                             [(0, 0), (0, 1)], [(0, 0), (1, 0)]]], 
                                             [(code, 0, 8), [[(0, 8), (0, 7)], [(0, 8), (1, 8)]]], 
                                             [(code, 2, 4), [[(2, 4), (3, 4), (2, 5), (3, 5)], 
                                                             [(2, 4), (2, 3)], [(2, 4), (2, 5)], 
                                                             [(2, 4), (1, 4)], [(2, 4), (3, 4)]]], 
                                             [(code, 6, 1), [[(6, 1), (7, 1), (6, 2), (7, 2)], 
                                                             [(6, 1), (6, 0)], [(6, 1), (6, 2)], 
                                                             [(6, 1), (5, 1)], [(6, 1), (7, 1)], ]], 
                                             [(code, 6, 8), [[(6, 8), (6, 7)], [(6, 8), (5, 8)], 
                                                             [(6, 8), (7, 8)]]], 
                                             [(code, 8, 4), [[(8, 4), (8, 3)], [(8, 4), (8, 5)], 
                                                             [(8, 4), (7, 4)]]], 
                                             [(code, 0, 4), [[(0, 4), (1, 4), (0, 5), (1, 5)], 
                                                             [(0, 4), (0, 3)], [(0, 4), (0, 5)], 
                                                             [(0, 4), (1, 4)]]], 
                                             [(code, 8, 0), [[(8, 0), (8, 1)], [(8, 0), (7, 0)]]], 
                                             [(code, 8, 8), [[(8, 8), (8, 7)], [(8, 8), (7, 8)]]]]):
            res = get_sub_rc_all_nghbrs(*args)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_all_nghbrs 1 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_all_nghbrs2(self):
        code = 'BM33-331234'
        for n, (args, exp_res) in enumerate([[(code, 0, 0), [[(0, 0), (1, 0), (0, 1), (1, 1)], 
                                                             [(0, 0), (0, 1)], [(0, 0), (1, 0)]]], 
                                             [(code, 0, 8), [[(0, 8), (0, 7)], [(0, 8), (1, 8)]]], 
                                             [(code, 2, 4), [[(2, 4), (3, 4), (2, 5), (3, 5)], 
                                                             [(2, 4), (2, 3)], [(2, 4), (2, 5)], 
                                                             [(2, 4), (1, 4)], [(2, 4), (3, 4)]]], 
                                             [(code, 6, 1), [[(6, 1), (7, 1), (6, 2), (7, 2)], 
                                                             [(6, 1), (6, 0)], [(6, 1), (6, 2)], 
                                                             [(6, 1), (5, 1)], [(6, 1), (7, 1)], ]], 
                                             [(code, 6, 8), [[(6, 8), (6, 7)], [(6, 8), (5, 8)], 
                                                             [(6, 8), (7, 8)]]], 
                                             [(code, 8, 4), [[(8, 4), (8, 3)], [(8, 4), (8, 5)], 
                                                             [(8, 4), (7, 4)]]], 
                                             [(code, 0, 4), [[(0, 4), (1, 4), (0, 5), (1, 5)], 
                                                             [(0, 4), (0, 3)], [(0, 4), (0, 5)], 
                                                             [(0, 4), (1, 4)]]], 
                                             [(code, 8, 0), [[(8, 0), (8, 1)], [(8, 0), (7, 0)]]], 
                                             [(code, 8, 8), [[(8, 8), (8, 7)], [(8, 8), (7, 8)]]]]):
            res = get_sub_rc_all_nghbrs(*args)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_all_nghbrs 2 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_all_nghbrs3(self):
        code = 'JM33-331234'  # 8, 7
        #grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        for n, (args, exp_res) in enumerate([[(code, 0, 0), [[(0, 0), (1, 0), (0, 1), (1, 1)], 
                                                             [(0, 0), (0, 1)], [(0, 0), (1, 0)]]], 
                                             [(code, 0, 6), [[(0, 6), (0, 5)], [(0, 6), (1, 6)]]]]):
            res = get_sub_rc_all_nghbrs(*args)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_all_nghbrs 3 #{n}: {res} vs {exp_res}")
            
    def test_get_sub_rc_all_nghbrs4(self):
        code = 'Jk33-331234'  # 8, 7
        #grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        for n, (args, exp_res) in enumerate([[(code, 0, 0), [[(0, 0), (1, 0), (0, 1), (1, 1)], 
                                                             [(0, 0), (0, 1)], [(0, 0), (1, 0)]]], 
                                             [(code, 0, 6), [[(0, 6), (0, 5)], [(0, 6), (1, 6)]]]]):
            res = get_sub_rc_all_nghbrs(*args)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_all_nghbrs 4 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_all_nghbrs5(self):
        code = 'Kk33-331234'  # 8, 7
        #grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        for n, (args, exp_res) in enumerate([[(code, 0, 0), [[(0, 0), (1, 0), (0, 1), (1, 1)], 
                                                             [(0, 0), (0, 1)], [(0, 0), (1, 0)]]], 
                                             [(code, 0, 6), [[(0, 6), (0, 5)], [(0, 6), (1, 6)]]]]):
            res = get_sub_rc_all_nghbrs(*args)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_all_nghbrs 5 #{n}: {res} vs {exp_res}")

class TestGetSubRCXNghbrs(unittest.TestCase):
    def test_get_sub_rc_x_nghbrs1(self):
        code = 'BM33-331234'
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        for n, (args, exp_res) in enumerate([[(0, 0), [(1, 1)]], 
                                             [(0, 8), [(1, 7)]], 
                                             [(2, 4), [(1, 3), (3, 3), (3, 5), (1, 5)]], 
                                             [(6, 1), [(5, 0), (7, 0), (7, 2), (5, 2)]], 
                                             [(6, 8), [(5, 7), (7, 7)]], 
                                             [(8, 4), [(7, 3), (7, 5)]], 
                                             [(0, 4), [(1, 3), (1, 5)]], 
                                             [(8, 0), [(7, 1)]], 
                                             [(8, 8), [(7, 7)]]]):
            
            res = get_sub_rc_x_nghbrs(code, *args)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_x_nghbrs 1 #{n}: {res} vs {exp_res}")

    def test_get_sub_rc_x_nghbrs2(self):
        code = 'JM33-331234'  # 8, 7
        grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
        for n, (args, exp_res) in enumerate([[(0, 0), [(1, 1)]], 
                                             [(0, 6), [(1, 5)]], 
                                             [(7, 5), [(6, 4), (6, 6)]], 
                                             [(2, 4), [(1, 3), (3, 3), (3, 5), (1, 5)]],  
                                             [(5, 1), [(4, 0), (6, 0), (6, 2), (4, 2)]],
                                             [(0, 4), [(1, 3), (1, 5)]], 
                                             [(7, 0), [(6, 1)]], 
                                             [(7, 6), [(6, 5)]]]):
            res = get_sub_rc_x_nghbrs(code, *args)
            self.assertEqual(res, exp_res, f"ERROR get_sub_rc_x_nghbrs 2 #{n}: {res} vs {exp_res}")




'''
def get_sub_rc_all_nghbrs(code, r, c):
    nghbrs_rc = [get_sub_rc_nghbrs(code, r, c)]
    nghbr_left = get_sub_rc_nghbr(code, r, c, dval=-1, horizontal=True)
    nghbr_right = get_sub_rc_nghbr(code, r, c, dval=1, horizontal=True)
    nghbr_down = get_sub_rc_nghbr(code, r, c, dval=-1, horizontal=False)
    nghbr_up = get_sub_rc_nghbr(code, r, c, dval=1, horizontal=False)
    
    for nghbr_rc in [nghbr_left, nghbr_right, nghbr_down, nghbr_up]:
        if nghbr_rc != (r, c):
            nghbrs_rc.append([(r, c), nghbr_rc])

    return [ls for ls in nghbrs_rc if ls]
'''



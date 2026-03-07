import unittest
import numpy as np

from geom_helpers.distance_helper import get_dist
from geom_helpers.latlon_code import get_reg_cell_code, get_rev_reg_cell_code
from geom_helpers.latlon_code_numpy import (
    vec_get_char, vec_get_substring, vec_get_latlon_params_for_point, 
    get_dist_numpy, get_reg_cell_code_by_params_numpy, 
    get_rev_reg_cell_code_by_params_numpy)

from basic_helpers.config_reg_code import az_bbox, base_bbox, isl_bbox, mad_bbox, can_bbox, cpv_bbox, med_bbox

# python -m unittest test_latlon_code.py

class TestGetRegCellCode(unittest.TestCase): 
    def test_get_reg_cell_code(self):
        lat, lon = (48.5, 11.5)
        code = get_reg_cell_code(lat, lon)
        self.assertEqual(code, 'CM44-51I300', 'ERROR: CM44-51I300')
        rev_lat, rev_lon = get_rev_reg_cell_code(code)
        self.assertAlmostEqual(get_dist((lat, lon), (rev_lat, rev_lon)), 0, delta=1)

    def test_get_reg_cell_code2(self):
        base_lat, base_lon = 47, 12
        i = 0
        codes = ['BN11-110000', 'BN21-110000', 'BN31-110000', 'BN41-110000', 'BN51-110000', 'BN61-110000', ]
        print()
        for a in range(3):
            for b in range(2):
                lat = base_lat + a/3 + 1e-6
                lon = base_lon + b/2 + 1e-6
                code = get_reg_cell_code(lat, lon)
                #print(i, code)
                rev_lat, rev_lon = get_rev_reg_cell_code(code)
                self.assertEqual(code, codes[i], f"ERROR: {code} != {codes[i]} - {i} {a} {b} {lat:.7f}, {lon:.7f}")
                self.assertAlmostEqual(get_dist((lat, lon), (rev_lat, rev_lon)), 0, delta=1)
                i += 1

    def test_get_reg_cell_code3(self):
        base_lat, base_lon = 44, 14
        i = 0
        codes = ['8P18-110000', '8P28-110000', '8P38-110000', '8P48-110000', '8P58-110000', '8P68-110000', ]
        print()
        for a in range(3):
            for b in range(2):
                lat = base_lat + a/3 + 1e-6 + 2/9
                lon = base_lon + b/2 + 1e-6 + 1/6
                code = get_reg_cell_code(lat, lon)
                print(i, code)
                rev_lat, rev_lon = get_rev_reg_cell_code(code)
                self.assertEqual(code, codes[i], f"ERROR: {code} != {codes[i]} - {i} {a} {b} {lat:.7f}, {lon:.7f}")
                self.assertAlmostEqual(get_dist((lat, lon), (rev_lat, rev_lon)), 0, delta=1)
                i += 1       


    def test_get_reg_cell_code4(self):
        base_lat, base_lon = 50, 10
        i = 0
        codes = ['EL12-140000', 'EL23-170000', 'EL35-440000', 'EL46-470000', 'EL58-740000', 'EL69-770000', ]
        print()
        for a in range(3):
            for b in range(2):
                lat = base_lat + a/3 + 1e-6 + a/9 + a/27
                lon = base_lon + b/2 + 1e-6 + (b+1)/6 + (b+1)/18
                code = get_reg_cell_code(lat, lon)
                print(i, code)
                rev_lat, rev_lon = get_rev_reg_cell_code(code)
                self.assertEqual(code, codes[i], f"ERROR: {code} != {codes[i]} - {i} {a} {b} {lat:.7f}, {lon:.7f}")
                self.assertAlmostEqual(get_dist((lat, lon), (rev_lat, rev_lon)), 0, delta=1)
                i += 1       

    def test_get_reg_cell_code5(self):
        M = 111320
        base_lat, base_lon = 51, 0
        i = 0
        codes = ['FB12-142A30', 'FB23-172K60', 'FB35-444A3G', 'FB46-474K6G', 'FB58-746A3W', 'FB69-776K6W', ]
        print()
        for a in range(3):
            for b in range(2):
                lat = base_lat + a/3 + 1e-6 + a/9 + a/27 + (a+1)*76/M + (b+1)*10/M
                M_REF = np.cos(np.radians(lat)) * M
                lon = base_lon + b/2 + 1e-6 + (b+1)/6 + (b+1)/18 + (b+1)*114/M_REF + a*16/M_REF
                code = get_reg_cell_code(lat, lon)
                print(i, code)
                rev_lat, rev_lon = get_rev_reg_cell_code(code)
                self.assertEqual(code, codes[i], f"ERROR: {code} != {codes[i]} - {i} {a} {b} {lat:.7f}, {lon:.7f}")
                self.assertAlmostEqual(get_dist((lat, lon), (rev_lat, rev_lon)), 0, delta=1)
                i += 1       

    def test_get_reg_cell_code6(self):
        M = 111320
        base_lat, base_lon = 52, -1
        i = 0
        codes = ['GA12-142A30', 'GA23-172K60', 'GA35-444A3G', 'GA46-474K6G', 'GA58-746A3W', 'GA69-776K6W', ]
        print()
        for a in range(3):
            for b in range(2):
                lat = base_lat + a/3 + 1e-6 + a/9 + a/27 + (a+1)*76/M + (b+1)*10/M
                M_REF = np.cos(np.radians(lat)) * M
                lon = base_lon + b/2 + 1e-6 + (b+1)/6 + (b+1)/18 + (b+1)*114/M_REF + a*16/M_REF
                code = get_reg_cell_code(lat, lon)
                print(i, code)
                rev_lat, rev_lon = get_rev_reg_cell_code(code)
                self.assertEqual(code, codes[i], f"ERROR: {code} != {codes[i]} - {i} {a} {b} {lat:.7f}, {lon:.7f}")
                self.assertAlmostEqual(get_dist((lat, lon), (rev_lat, rev_lon)), 0, delta=1)
                i += 1       

    def test_get_reg_cell_code7(self):
        M = 111320
        base_lat, base_lon = 42, -4
        i = 0
        codes = ['6712-142A30', '6723-172K60', '6735-444A3G', '6746-474K6G', '6758-746A3W', '6769-776K6W', ]
        print()
        for a in range(3):
            for b in range(2):
                lat = base_lat + a/3 + 1e-6 + a/9 + a/27 + (a+1)*84/M + (b+1)*10/M
                M_REF = np.cos(np.radians(lat)) * M
                lon = base_lon + b/2 + 1e-6 + (b+1)/6 + (b+1)/18 + (b+1)*126/M_REF + a*16/M_REF
                code = get_reg_cell_code(lat, lon)
                print(i, code)
                rev_lat, rev_lon = get_rev_reg_cell_code(code)
                self.assertEqual(code, codes[i], f"ERROR: {code} != {codes[i]} - {i} {a} {b} {lat:.7f}, {lon:.7f}")
                self.assertAlmostEqual(get_dist((lat, lon), (rev_lat, rev_lon)), 0, delta=1)
                i += 1       

class TestGetRegCellCode2(unittest.TestCase):
    def setUp(self):
        N = 1000
        self.pts = []
        for i, bbox in enumerate([base_bbox, med_bbox, az_bbox, isl_bbox, mad_bbox, can_bbox, cpv_bbox]):
            N = 1000 if i > 0 else 2000
            base_lats = np.random.randint(bbox[1],bbox[3],N)
            base_lons = np.random.randint(bbox[0],bbox[2],N)

            arr = np.random.uniform(size=(N, 2))
            arr[:, 0] += base_lats
            arr[:, 1] += base_lons

            if i == 0:
                mask = arr[:, 1] <= -7
                mask1 = arr[:,0] >= 62
                mask2 = arr[:,0] < 40
                mask *= (mask1 + mask2)
                arr = arr[np.where(mask == 0)][:1000]

            self.pts += [tuple(arr[i]) for i in range(len(arr))]

        np.random.shuffle(self.pts)

    def test_get_reg_cell_code(self):
        #print("pts", len(self.pts))
        for i, (lat, lon) in enumerate(self.pts):
            code = get_reg_cell_code(lat, lon)
            rev_lat, rev_lon = get_rev_reg_cell_code(code)
            d = get_dist((lat, lon), (rev_lat, rev_lon))
            if d > 1:
                print(f"ERROR - get_reg_cell_code: {i:4} {code} {d:.5f} - {lat:10.6f}, {lon:10.6f} | {rev_lat:10.6f}, {rev_lon:10.6f}")
            self.assertAlmostEqual(d, 0, 
                                   #f"ERROR - get_reg_cell_code: {code} {d:.2f} - {lat:10.6f}, {lon:10.6f} | {rev_lat:10.6f}, {rev_lon:10.6f}",
                                   delta=1)

class TestGetRegCellCodeIceland(unittest.TestCase):
    def test_get_reg_cell_code_iceland(self):
        base_lat, base_lon = 64, -22
        i = 0
        codes = ['S100-110000', 'S110-110000', 'S120-110000', 'S130-110000', 'S140-110000', 'S150-110000', 
                 'S160-110000', 'S170-110000', 'S180-110000', 'S190-110000', 'S1A0-110000', 'S1B0-110000', 
                 'S1C0-110000', 'S1D0-110000', 'S1E0-110000', 'S1F0-110000', 'S1G0-110000', 'S1H0-110000', ]
        print()
        for a in range(3):
            for b in range(6):
                lat = base_lat + a/3 + 1e-6
                lon = base_lon + b*2/3 + 1e-6
                code = get_reg_cell_code(lat, lon)
                #print(f"{i:2}  {code}  {lat:.6f}, {lon:.6f}")
                rev_lat, rev_lon = get_rev_reg_cell_code(code)
                self.assertEqual(code, codes[i], f"ERROR: {code} != {codes[i]} - {i} {a} {b} {lat:.7f}, {lon:.7f}")
                self.assertAlmostEqual(get_dist((lat, lon), (rev_lat, rev_lon)), 0, delta=1)
                i += 1

    def test_get_reg_cell_code_iceland1(self):
        base_lat, base_lon = 64, -22
        i = 0
        codes = ['S100-110000', 'S113-110000', 'S126-110000', 'S130-110000', 'S143-110000', 'S156-110000', 
                 'S161-110000', 'S174-110000', 'S187-110000', 'S191-110000', 'S1A4-110000', 'S1B7-110000', 
                 'S1C2-110000', 'S1D5-110000', 'S1E8-110000', 'S1F2-110000', 'S1G5-110000', 'S1H8-110000', ]
        print()
        for a in range(3):
            for b in range(6):
                lat = base_lat + a/3 + 1e-6 + (b%3)/9
                lon = base_lon + b*2/3 + 1e-6 + (a*2)/9
                code = get_reg_cell_code(lat, lon)
                #print(f"{i:2}  {code}  {lat:.6f}, {lon:.6f}")
                rev_lat, rev_lon = get_rev_reg_cell_code(code)
                self.assertEqual(code, codes[i], f"ERROR: {code} != {codes[i]} - {i} {a} {b} {lat:.7f}, {lon:.7f}")
                self.assertAlmostEqual(get_dist((lat, lon), (rev_lat, rev_lon)), 0, delta=1)
                i += 1

    def test_get_reg_cell_code_iceland2(self):
        base_lat, base_lon = 67, -14
        i = 0
        codes = ['V300-110000', 'V313-110000', 'V326-110000', 'V330-110000', 'V343-110000', 'V356-110000', 
                 'V361-110000', 'V374-110000', 'V387-110000', 'V391-110000', 'V3A4-110000', 'V3B7-110000', 
                 'V3C2-110000', 'V3D5-110000', 'V3E8-110000', 'V3F2-110000', 'V3G5-110000', 'V3H8-110000', ]
        print()
        for a in range(3):
            for b in range(6):
                if b > 2:
                    i += 1
                    continue
                lat = base_lat + a/3 + 1e-6 + (b%3)/9
                lon = base_lon + b*2/3 + 1e-6 + (a*2)/9
                code = get_reg_cell_code(lat, lon)
                #print(f"{i:2}  {code}  {lat:.6f}, {lon:.6f}")
                rev_lat, rev_lon = get_rev_reg_cell_code(code)
                self.assertEqual(code, codes[i], f"ERROR: {code} != {codes[i]} - {i} {a} {b} {lat:.7f}, {lon:.7f}")
                self.assertAlmostEqual(get_dist((lat, lon), (rev_lat, rev_lon)), 0, delta=1)
                i += 1

    def test_get_reg_cell_code_iceland3(self):
        N = 1000
        #self.pts = []

        base_lats = np.random.randint(isl_bbox[1],isl_bbox[3],N)
        base_lons = np.random.randint(isl_bbox[0],isl_bbox[2],N)

        arr = np.random.uniform(size=(N, 2))
        arr[:, 0] += base_lats
        arr[:, 1] += base_lons
        #print("arr", arr.shape)

        self.pts = [tuple(arr[i]) for i in range(len(arr))]
        np.random.shuffle(self.pts)

        for i, (lat, lon) in enumerate(self.pts):
            code = get_reg_cell_code(lat, lon)
            rev_lat, rev_lon = get_rev_reg_cell_code(code)
            d = get_dist((lat, lon), (rev_lat, rev_lon))
            if d > 0.71:
                print(f"ERROR - get_reg_cell_code: {i:4} {code} {d:.5f} - {lat:10.6f}, {lon:10.6f} | {rev_lat:10.6f}, {rev_lon:10.6f}")
            self.assertAlmostEqual(d, 0, 
                                   msg=f"ERROR - get_reg_cell_code: {code} {d:.2f} - {lat:10.6f}, {lon:10.6f} | {rev_lat:10.6f}, {rev_lon:10.6f}",
                                   delta=0.9)
  

class TestGetRegCellNumpy(unittest.TestCase):
    def test_get_reg_cell_code_numpy(self):
        arr = np.array([[48.137154, 11.576124], [52.520008, 13.404954], [51.507351, -0.127758], 
                        [55.755825, 37.617298], [65.755825, -20.617298], [33.755825, -17.617298],
                        [38.755825, -28.617298], [27.755825, -16.617298], [16.25, -23.8], [48.856613, 2.352222]])

        lats = arr[:, 0]
        lons = arr[:, 1]

        params = vec_get_latlon_params_for_point(np.floor(lats).astype(int), np.floor(lons).astype(int))

        self.assertEqual(params[0]['x1'], 2, "ERROR - params #1")
        self.assertEqual(params[0]['x2'], 3, "ERROR - params #2")
        self.assertEqual(params[0]['x3'], 9, "ERROR - params #3")
        self.assertEqual(params[1]['y1'], 3, "ERROR - params #4")
        self.assertEqual(params[2]['y2'], 3, "ERROR - params #5")
        self.assertEqual(params[3]['y3'], 9, "ERROR - params #6")

        self.assertEqual(params[4], {'x1': 1.5, 'x2': 3, 'x3': 9, 'y1': 3, 'y2': 3, 'y3': 9, 
                                     'extent_lon': 4, 'extent_lat': 1, 
                                     'n_lttr1': 6, 'n_lttr2': 3, 'off1': 0, 'off2': 0, 
                                     'base_lat': 65, 'base_lon': -22, 'len_lttrs': 38}, "ERROR - params[4]")
        self.assertEqual(params[5], {'x1': 3, 'x2': 3, 'x3': 9, 'y1': 3, 'y2': 3, 'y3': 9, 
                                     'extent_lon': 2, 'extent_lat': 2, 
                                     'n_lttr1': 6, 'n_lttr2': 3, 'off1': 0, 'off2': 0, 
                                     'base_lat': 32, 'base_lon': -18, 'len_lttrs': 45}, "ERROR - params[5]")
        self.assertEqual(params[6], {'x1': 1, 'x2': 6, 'x3': 9, 'y1': 1.5, 'y2': 6, 'y3': 9, 
                                     'extent_lon': 8, 'extent_lat': 4, 
                                     'n_lttr1': 8, 'n_lttr2': 6, 'off1': 0, 'off2': 0, 
                                     'base_lat': 36, 'base_lon': -32, 'len_lttrs': 45}, "ERROR - params[6]")
        self.assertEqual(params[7], {'x1': 1, 'x2': 7, 'x3': 9, 'y1': 1, 'y2': 8, 'y3': 9, 
                                     'extent_lon': 7, 'extent_lat': 5, 
                                     'n_lttr1': 7, 'n_lttr2': 7, 'off1': 0, 'off2': 0, 
                                     'base_lat': 26, 'base_lon': -19, 'len_lttrs': 45}, "ERROR - params[7]")
        self.assertEqual(params[8], {'x1': np.float32(1.3333333333333333), 'x2': 6, 'x3': 9, 'y1': np.float32(1.4), 
                                     'y2': 6, 'y3': 9, 'extent_lon': 6, 'extent_lat': 5, 
                                     'n_lttr1': 8, 'n_lttr2': 6, 'off1': 0, 'off2': 0, 
                                     'base_lat': 14, 'base_lon': -27, 'len_lttrs': 45}, "ERROR - params[8]")
        self.assertEqual(params[9], {'x1': 2, 'x2': 3, 'x3': 9, 'y1': 3, 'y2': 3, 'y3': 9, 
                                     'extent_lon': 1, 'extent_lat': 1, 
                                     'n_lttr1': 2, 'n_lttr2': 3, 'off1': 1, 'off2': 1, 
                                     'base_lat': 48, 'base_lon': 2, 'len_lttrs': 38}, "ERROR - params[9]")


        codes = get_reg_cell_code_by_params_numpy(lats, lons, params)
        self.assertEqual(codes, ['CM24-353a40', 'GO36-744ESO', 'FA46-633H3G', 'Jm61-8781A8', 'T1E0-8281TQ', 
                                 '20V0-846ZPG', '30Z2-836ZNa', '109i-17EI3b', '00S1-963CD9', 'CD56-72Da0R'], 
                                 "ERROR - incorrect CODES created")

    def test_get_rev_reg_cell_code_numpy(self):
        arr = np.array([[48.137154, 11.576124], [52.520008, 13.404954], [51.507351, -0.127758], 
                        [55.755825, 37.617298], [65.755825, -20.617298], [33.755825, -17.617298],
                        [38.755825, -28.617298], [27.755825, -16.617298], [16.25, -23.8], [48.856613, 2.352222]])

        lats = arr[:, 0]
        lons = arr[:, 1]

        params = vec_get_latlon_params_for_point(np.floor(lats).astype(int), np.floor(lons).astype(int))
        codes = get_reg_cell_code_by_params_numpy(lats, lons, params)

        rev_arr = get_rev_reg_cell_code_by_params_numpy(codes)
        self.assertTrue(np.max(get_dist_numpy(arr, np.array(rev_arr))) < 0.6, 
                        "ERROR - get_rev_reg_cell_code_numpy #1")
        self.assertAlmostEqual(np.sum(np.round(get_dist_numpy(arr, np.array(rev_arr)),1)), 3.8, 5,
                               "ERROR - get_rev_reg_cell_code_numpy #2")

class TestGetRegCellNumpy2(unittest.TestCase):
    def setUp(self):
        N = 1000
        self.pts = []
        for i, bbox in enumerate([base_bbox, med_bbox, az_bbox, isl_bbox, mad_bbox, can_bbox, cpv_bbox]):
            N = 1000 if i > 0 else 2000
            base_lats = np.random.randint(bbox[1],bbox[3],N)
            base_lons = np.random.randint(bbox[0],bbox[2],N)

            arr = np.random.uniform(size=(N, 2))
            arr[:, 0] += base_lats
            arr[:, 1] += base_lons

            if i == 0:
                mask = arr[:, 1] <= -7
                mask1 = arr[:,0] >= 62
                mask2 = arr[:,0] < 40
                mask *= (mask1 + mask2)
                arr = arr[np.where(mask == 0)][:1000]

            self.pts += [tuple(arr[i]) for i in range(len(arr))]

        np.random.shuffle(self.pts)

    def test_get_rev_reg_cell_code_numpy(self):
        arr = np.array(self.pts)

        lats = arr[:, 0]
        lons = arr[:, 1]

        params = vec_get_latlon_params_for_point(np.floor(lats).astype(int), np.floor(lons).astype(int))
        codes = get_reg_cell_code_by_params_numpy(lats, lons, params)
        #print()
        #print("codes", len(codes), codes[:6])

        rev_arr = get_rev_reg_cell_code_by_params_numpy(codes)
        dists = get_dist_numpy(arr, np.array(rev_arr))

        lat, lon = self.pts[np.argmax(dists)]
        code = get_reg_cell_code(lat, lon)
        #rev_lat, rev_lon = get_rev_reg_cell_code(code)
        #d_arr = get_dist(rev_arr[np.argmax(dists)], (lat, lon))
        #d_reg = get_dist((rev_lat, rev_lon), (lat, lon))
        #d_rev = get_dist(rev_arr[np.argmax(dists)], (rev_lat, rev_lon))
        
        self.assertTrue(np.max(get_dist_numpy(arr, np.array(rev_arr))) < 0.75, 
                        "ERROR - get_rev_reg_cell_code_numpy PTS array")
        self.assertTrue(code == codes[np.argmax(dists)], 
                        "ERROR - get_rev_reg_cell_code_numpy PTS array #2 {codes} != {codes[np.argmax(dists)]}")



class TestVecGetSubstring(unittest.TestCase):
    def test_vec_get_substring(self):
        LTTRS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        LTTRS_ARR = np.array(LTTRS)
        for _ in range(3):
            idxs = np.random.choice(range(38, 46), 5)
            res = vec_get_substring(LTTRS_ARR, idxs)
            res_str = "".join([r[-1] for r in res])
            data_str = "".join([LTTRS[n-1] for n in idxs])

            self.assertTrue(res_str == data_str, 
                            f"Error - vec_get_char: {idxs} - {data_str} vs {res_str}")

class TestVecGetChar(unittest.TestCase):
    def test_vec_get_char(self):
        data_arr = np.array([['6723-172K60', *list('6723-172K60')], 
                             ['GA35-444A3G', *list('GA35-444A3G')], 
                             ['EL35-440000', *list('EL35-440000')], ])

        for idx in np.random.choice(11,5, False):
            res = vec_get_char(data_arr[:, 0], idx)
            res_str = "".join(res)
            data_str = "".join(data_arr[:, idx+1])

            self.assertTrue(data_str == res_str, 
                            f"Error - vec_get_char: {idx} - {data_str} vs {res_str}")



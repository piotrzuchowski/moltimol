import unittest

import numpy as np

from moltimol.data_curation import (
    cross_dist_feature,
    nearest_neighbor_distances,
    nn_distances_features,
    parse_psi4geom_string,
)


CO2_CO2_GEOM_1 = """\
symmetry c1
no_com
no_reorient
units angstrom
0 1
C -0.0603090150 0.1393817390 -0.0472806615
O 0.0497430972 -0.1251072525 -1.2457668039
O -0.0044670438 0.0204685834 1.2812620240
--
0 1
C 0.3615300371 6.7666654016 -3.3592356068
O -0.1545201444 7.6202122066 -2.9748971166
O 0.8970424674 5.8930966065 -4.0161820241
"""

CO2_CO2_GEOM_2 = """\
symmetry c1
no_com
no_reorient
units angstrom
0 1
C 0.1571030643 -0.0330245874 -0.1433735567
O -0.0295721766 -0.0858929995 -1.0297182215
O -0.0883705014 0.1106856940 1.1373536855
--
0 1
C 0.1914386554 -2.5486619309 6.1636122889
O 0.2872691208 -3.0557473412 7.3534655859
O 0.5813891493 -2.6257665735 4.9623427282
"""


class TestDataCuration(unittest.TestCase):
    def test_parse_and_nn_features(self):
        symA, symB, XA, XB, units = parse_psi4geom_string(CO2_CO2_GEOM_1)
        self.assertEqual(units, "angstrom")
        self.assertEqual(len(symA), 3)
        self.assertEqual(len(symB), 3)
        self.assertEqual(XA.shape, (3, 3))
        self.assertEqual(XB.shape, (3, 3))

        d = nearest_neighbor_distances(symA, symB, XA, XB)
        self.assertEqual(len(d["A_only"]), 3)
        self.assertEqual(len(d["B_only"]), 3)
        self.assertEqual(len(d["AB"]), 9)

        feat = cross_dist_feature(XA, XB, sort=True, flatten=True)
        self.assertEqual(len(feat), 9)
        self.assertTrue(np.all(np.diff(feat) >= 0.0))

    def test_nn_distances_features(self):
        symA1, symB1, XA1, XB1, _ = parse_psi4geom_string(CO2_CO2_GEOM_1)
        symA2, symB2, XA2, XB2, _ = parse_psi4geom_string(CO2_CO2_GEOM_2)

        f1 = cross_dist_feature(XA1, XB1)
        f2 = cross_dist_feature(XA2, XB2)
        F = np.vstack([f1, f2])

        d_nn, idx_nn = nn_distances_features(F)
        self.assertEqual(len(d_nn), 2)
        self.assertEqual(len(idx_nn), 2)
        self.assertTrue(np.all(np.isfinite(d_nn)))


if __name__ == "__main__":
    unittest.main()

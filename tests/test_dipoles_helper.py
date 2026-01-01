import unittest

import numpy as np

from moltimol.dipoles_helper import (
    E_charge_tt,
    T_dipole_tt,
    mbis_charges_rhf_aug_cc_pvdz,
    dimer_dipole_charges_polarizable,
    f_tt,
)
from moltimol import build_dimer_frame_COM, mass_of


def _co_coords():
    # CO geometry from CO.xyz (units arbitrary for these tests)
    return np.array([
        [0.0, 0.0, -0.644],
        [0.0, 0.0, 0.484],
    ])


class TestDipolesHelper(unittest.TestCase):
    def test_f_tt_limits(self):
        self.assertAlmostEqual(f_tt(3, 0.0), 0.0, places=14)
        self.assertGreater(f_tt(3, 50.0), 1.0 - 1e-12)

    def test_T_dipole_tt_symmetry(self):
        R = np.array([0.0, 0.0, 2.0])
        T = T_dipole_tt(R, b=1.2)
        self.assertTrue(np.allclose(T, T.T, atol=1e-12))
        self.assertAlmostEqual(T[0, 1], 0.0, places=12)
        self.assertAlmostEqual(T[0, 0], T[1, 1], places=12)

    def test_E_charge_tt_direction(self):
        R = np.array([0.0, 0.0, 3.0])
        E = E_charge_tt(R, b=0.8)
        self.assertAlmostEqual(E[0], 0.0, places=12)
        self.assertAlmostEqual(E[1], 0.0, places=12)
        self.assertGreater(E[2], 0.0)

    def test_dimer_dipole_zero_charges(self):
        XA = _co_coords()
        XB = _co_coords() + np.array([0.0, 0.0, 10.0])
        qA = np.zeros(2)
        qB = np.zeros(2)
        alphaA = np.zeros(2)
        alphaB = np.zeros(2)

        out = dimer_dipole_charges_polarizable(
            XA, XB, qA, qB, alphaA, alphaB, mutual=False
        )
        self.assertTrue(np.allclose(out["mu_lab"], 0.0, atol=1e-12))
        self.assertTrue(np.allclose(out["mu_ind"], 0.0, atol=1e-12))

    def test_dimer_dipole_perm_only(self):
        XA = _co_coords()
        XB = _co_coords() + np.array([0.0, 0.0, 10.0])
        qA = np.array([-0.5, 0.5])
        qB = np.array([-0.5, 0.5])
        alphaA = np.zeros(2)
        alphaB = np.zeros(2)

        out = dimer_dipole_charges_polarizable(
            XA, XB, qA, qB, alphaA, alphaB, mutual=False
        )
        self.assertTrue(np.allclose(out["mu_ind"], 0.0, atol=1e-12))
        self.assertTrue(np.allclose(out["mu_lab"], 0.0, atol=1e-12))

    def test_build_dimer_frame_com_coords(self):
        XA = _co_coords()
        XB = _co_coords() + np.array([0.0, 0.0, 10.0])
        coords = np.vstack([XA, XB])
        n_atoms_A = len(XA)
        massesA = np.array([mass_of("C"), mass_of("O")])
        massesB = np.array([mass_of("C"), mass_of("O")])

        origin, ex, ey, ez, B = build_dimer_frame_COM(
            XA, XB,
            massesA=massesA, massesB=massesB,
            principalA="min", principalB="min",
            origin="midpoint",
        )
        coords_body = (coords - origin) @ B

        self.assertEqual(coords_body.shape, coords.shape)
        self.assertAlmostEqual(np.linalg.norm(ex), 1.0, places=12)
        self.assertAlmostEqual(np.linalg.norm(ey), 1.0, places=12)
        self.assertAlmostEqual(np.linalg.norm(ez), 1.0, places=12)
        self.assertAlmostEqual(np.dot(ex, ey), 0.0, places=12)
        self.assertAlmostEqual(np.dot(ex, ez), 0.0, places=12)
        self.assertAlmostEqual(np.dot(ey, ez), 0.0, places=12)

    def test_mbis_charges_co(self):
        try:
            import psi4  # noqa: F401
        except Exception:
            self.skipTest("psi4 not available")

        geo = """
        0 1
        C 0.000000 0.000000 -0.644
        O 0.000000 0.000000  0.484
        """
        charges = mbis_charges_rhf_aug_cc_pvdz(geo, scf_options={"scf_type": "direct"})
        print("charges of CO molecule:", charges)
        self.assertEqual(len(charges), 2)
        self.assertAlmostEqual(float(np.sum(charges)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

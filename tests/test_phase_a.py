import math
import unittest

from pjrope.diagnostics import effective_mass, functional_energy
from pjrope.kernels import affine_component, beta_l, fj_component, phi_l
from pjrope.rotary import jordan2_complex, matmul, max_abs_diff


class PhaseATest(unittest.TestCase):
    def test_scalar_special_points(self):
        length = 128.0
        omega = 0.17
        self.assertAlmostEqual(fj_component(9, omega, 0, length), math.cos(omega * 9))
        self.assertAlmostEqual(affine_component(16, length), -16 / length)
        self.assertLess(beta_l(16, length), 1.0)
        self.assertAlmostEqual(phi_l(1.0, 10_000.0), 1.0, places=8)

    def test_diagnostics(self):
        mass = effective_mass(0.25, {0: 0.5, 1: 0.5}, {0: 1.0, 1: 3.0})
        self.assertGreater(mass[1], mass[0])
        energy = functional_energy({0: [1.0, 0.0], 1: [0.0, 1.0]})
        self.assertGreater(energy[0], 0.0)
        self.assertGreater(energy[1], 0.0)

    def test_jordan_group_law(self):
        omega = 0.03
        left = matmul(jordan2_complex(2.0, omega), jordan2_complex(4.0, omega))
        right = jordan2_complex(6.0, omega)
        self.assertLess(max_abs_diff(left, right), 1e-12)


if __name__ == "__main__":
    unittest.main()


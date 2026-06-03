import cmath
import math
import unittest


def phi(z: complex, r: int, d: int) -> complex:
    if r < 0:
        return 0.0
    return math.comb(d, r) * (z**d)


def apply_shift_minus_root(values, z: complex):
    return [values[index + 1] - z * values[index] for index in range(len(values) - 1)]


def complex_rank(matrix, tol: float = 1e-9) -> int:
    rows = [list(row) for row in matrix]
    if not rows:
        return 0
    row_count = len(rows)
    col_count = len(rows[0])
    rank = 0
    for col in range(col_count):
        pivot = None
        for row in range(rank, row_count):
            if abs(rows[row][col]) > tol:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [value / scale for value in rows[rank]]
        for row in range(row_count):
            if row == rank:
                continue
            factor = rows[row][col]
            if abs(factor) > tol:
                rows[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(rows[row], rows[rank])
                ]
        rank += 1
        if rank == row_count:
            break
    return rank


class DifferenceModuleTest(unittest.TestCase):
    def test_shift_lowers_binomial_jet_order(self):
        z = cmath.exp(0.37j)
        for r in range(4):
            for d in range(12):
                left = phi(z, r, d + 1) - z * phi(z, r, d)
                right = z * phi(z, r - 1, d)
                self.assertAlmostEqual(abs(left - right), 0.0, places=10)

    def test_repeated_root_annihilates_finite_jet(self):
        z = 0.91 * cmath.exp(0.23j)
        for r in range(4):
            values = [phi(z, r, d) for d in range(20)]
            for _ in range(r + 1):
                values = apply_shift_minus_root(values, z)
            self.assertLess(max(abs(value) for value in values), 1e-9)

    def test_repeated_unit_root_contains_affine_direction(self):
        values = [float(d) for d in range(8)]
        first = apply_shift_minus_root(values, 1.0)
        second = apply_shift_minus_root(first, 1.0)
        self.assertTrue(all(abs(value) < 1e-12 for value in second))

    def test_monomial_and_binomial_jet_spans_match(self):
        z = cmath.exp(0.19j)
        max_order = 3
        lags = list(range(12))
        monomial_columns = [
            [(d**r) * (z**d) for d in lags]
            for r in range(max_order + 1)
        ]
        binomial_columns = [
            [math.comb(d, r) * (z**d) for d in lags]
            for r in range(max_order + 1)
        ]
        combined_columns = monomial_columns + binomial_columns
        monomial_matrix = list(zip(*monomial_columns))
        binomial_matrix = list(zip(*binomial_columns))
        combined_matrix = list(zip(*combined_columns))
        expected_rank = max_order + 1
        self.assertEqual(complex_rank(monomial_matrix), expected_rank)
        self.assertEqual(complex_rank(binomial_matrix), expected_rank)
        self.assertEqual(complex_rank(combined_matrix), expected_rank)


if __name__ == "__main__":
    unittest.main()

"""Tests for nexoma_labs.costs. Standard library unittest — no dependencies."""

import unittest

from nexoma_labs.costs import CostLedger, Price, estimate


class TestPrice(unittest.TestCase):
    def test_rejects_negative_rates(self):
        with self.assertRaises(ValueError):
            Price(input_per_mtok=-1, output_per_mtok=1)


class TestCostLedger(unittest.TestCase):
    def setUp(self):
        self.price = Price(input_per_mtok=3.0, output_per_mtok=15.0, cached_input_per_mtok=0.30)
        self.ledger = CostLedger(self.price)

    def test_uncached_call(self):
        # 1000 in @ $3/Mtok = $0.003 ; 500 out @ $15/Mtok = $0.0075
        cost = self.ledger.record("a", input_tokens=1000, output_tokens=500)
        self.assertAlmostEqual(cost, 0.0105, places=9)

    def test_caching_reduces_cost(self):
        plain = estimate(self.price, 4000, 200)
        cached = estimate(self.price, 4000, 200, cached_tokens=3800)
        self.assertLess(cached, plain)
        # 200 fresh @3 + 3800 cached @0.30 + 200 out @15
        self.assertAlmostEqual(cached, (200 * 3 + 3800 * 0.30 + 200 * 15) / 1e6, places=9)

    def test_totals_accumulate(self):
        self.ledger.record("a", 100, 10)
        self.ledger.record("b", 200, 20)
        self.assertEqual(len(self.ledger.calls), 2)
        self.assertEqual(self.ledger.total_tokens, 330)
        self.assertAlmostEqual(self.ledger.total, estimate(self.price, 300, 30), places=9)

    def test_projection_uses_mean_call(self):
        self.ledger.record("a", 1000, 100)
        self.ledger.record("b", 1000, 100)
        self.assertAlmostEqual(self.ledger.project(1000), self.ledger.total / 2 * 1000, places=9)

    def test_projection_of_empty_ledger_is_zero(self):
        self.assertEqual(CostLedger(self.price).project(10_000), 0.0)

    def test_cached_cannot_exceed_input(self):
        with self.assertRaises(ValueError):
            self.ledger.record("bad", input_tokens=10, output_tokens=1, cached_tokens=11)

    def test_negative_tokens_rejected(self):
        with self.assertRaises(ValueError):
            self.ledger.record("bad", input_tokens=-1, output_tokens=0)

    def test_report_contains_total(self):
        self.ledger.record("classify", 1000, 50)
        self.assertIn("TOTAL", self.ledger.report())


if __name__ == "__main__":
    unittest.main()

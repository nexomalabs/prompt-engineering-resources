"""Tests for nexoma_labs.fixtures."""

import os
import tempfile
import unittest

from nexoma_labs.fixtures import FixtureMissing, FixtureStore, lab_mode


class TestLabMode(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("NEXOMA_LAB_MODE", None)

    def test_defaults_to_fixture(self):
        os.environ.pop("NEXOMA_LAB_MODE", None)
        self.assertEqual(lab_mode(), "fixture")

    def test_rejects_unknown_mode(self):
        os.environ["NEXOMA_LAB_MODE"] = "wibble"
        with self.assertRaises(ValueError):
            lab_mode()


class TestFixtureStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = FixtureStore(self.tmp.name)
        self.req = {"model": "test", "prompt": "hello"}
        os.environ["NEXOMA_LAB_MODE"] = "fixture"

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("NEXOMA_LAB_MODE", None)

    def test_key_is_stable_and_order_independent(self):
        a = FixtureStore.key({"a": 1, "b": 2})
        b = FixtureStore.key({"b": 2, "a": 1})
        self.assertEqual(a, b)

    def test_key_changes_with_request(self):
        self.assertNotEqual(FixtureStore.key(self.req), FixtureStore.key({"model": "test", "prompt": "hi"}))

    def test_missing_fixture_fails_loudly(self):
        with self.assertRaises(FixtureMissing):
            self.store.load(self.req)

    def test_round_trip(self):
        self.store.save(self.req, {"text": "world"})
        self.assertEqual(self.store.load(self.req), {"text": "world"})

    def test_call_replays_without_invoking_live_fn(self):
        self.store.save(self.req, {"text": "cached"})

        def live(_):
            raise AssertionError("live function must not be called in fixture mode")

        self.assertEqual(self.store.call(self.req, live), {"text": "cached"})

    def test_record_mode_saves(self):
        os.environ["NEXOMA_LAB_MODE"] = "record"
        self.store.call(self.req, lambda r: {"text": "fresh"})
        os.environ["NEXOMA_LAB_MODE"] = "fixture"
        self.assertEqual(self.store.load(self.req), {"text": "fresh"})


if __name__ == "__main__":
    unittest.main()

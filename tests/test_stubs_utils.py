import os
import shutil
import tempfile
import unittest

from utils.stubs_utils import read_stub, save_stub


class TestSaveStub(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_stub_path_none_is_a_no_op(self):
        # Regression test: save_stub used to call os.path.dirname(stub_path)
        # unconditionally, which raised TypeError when stub_path was None
        # (the exact case a "compute but don't cache" caller relies on, e.g.
        # scripts/benchmark_detection.py running old-vs-new comparisons with
        # no persisted pickle files).
        save_stub(None, {"some": "object"})  # must not raise

    def test_save_then_read_round_trips(self):
        stub_path = os.path.join(self.tmp_dir, "nested", "dir", "stub.pkl")
        payload = [{"a": 1}, {"b": 2}]

        save_stub(stub_path, payload)

        self.assertTrue(os.path.exists(stub_path))
        loaded = read_stub(read_from_stub=True, stub_path=stub_path)
        self.assertEqual(loaded, payload)

    def test_read_stub_returns_none_when_read_from_stub_false(self):
        stub_path = os.path.join(self.tmp_dir, "stub.pkl")
        save_stub(stub_path, {"x": 1})

        self.assertIsNone(read_stub(read_from_stub=False, stub_path=stub_path))

    def test_read_stub_returns_none_when_path_missing(self):
        missing_path = os.path.join(self.tmp_dir, "does_not_exist.pkl")
        self.assertIsNone(read_stub(read_from_stub=True, stub_path=missing_path))

    def test_read_stub_returns_none_when_path_is_none(self):
        self.assertIsNone(read_stub(read_from_stub=True, stub_path=None))


if __name__ == "__main__":
    unittest.main()

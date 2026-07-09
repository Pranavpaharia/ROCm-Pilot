"""
Tests for gpu_compat module.
"""

import unittest
from src.gpu_compat import (
    load_gpu_database,
    lookup_gpu,
    get_compatible_rocm,
    get_compatible_pytorch,
    get_install_command,
    check_compatibility,
    get_gpu_by_detected_name,
)


class TestGPUCompat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load from default local path (written by script)
        cls.db = load_gpu_database('data/gpu_database.json')

    def test_load(self):
        self.assertIsNotNone(self.db)
        self.assertIn("gpu_architectures", self.db)
        self.assertIn("rocm_versions", self.db)
        self.assertIn("pytorch_rocm_matrix", self.db)

    def test_lookup_gpu(self):
        # Exact gfx
        gpu = lookup_gpu(self.db, "gfx1100")
        self.assertIsNotNone(gpu)
        self.assertEqual(gpu["gfx_id"], "gfx1100")
        self.assertIn("7900", gpu["name"])

        # Prefix variants
        gpu_prefix = lookup_gpu(self.db, "1100")
        self.assertEqual(gpu_prefix["gfx_id"], "gfx1100")

        # Fuzzy match name
        gpu_fuzzy = lookup_gpu(self.db, "7900 XTX")
        self.assertIsNotNone(gpu_fuzzy)
        self.assertEqual(gpu_fuzzy["gfx_id"], "gfx1100")

    def test_get_compatible_rocm(self):
        rocm_vers = get_compatible_rocm(self.db, "gfx942")
        self.assertIn("6.2", rocm_vers)

    def test_get_compatible_pytorch(self):
        pt_vers = get_compatible_pytorch(self.db, "gfx1100", "6.2")
        self.assertIn("2.5", pt_vers)

    def test_get_install_command(self):
        cmd = get_install_command(self.db, "2.5", "6.2")
        self.assertIsNotNone(cmd)
        self.assertIn("torch==2.5.1+rocm6.2", cmd)

    def test_check_compatibility(self):
        res = check_compatibility(self.db, "gfx1100", "6.2", "2.5")
        self.assertTrue(res["compatible"])
        self.assertEqual(len(res["warnings"]), 0)

        # Legacy warning check
        res_legacy = check_compatibility(self.db, "gfx906", "6.2")
        self.assertFalse(res_legacy["compatible"])
        self.assertTrue(any("legacy" in w.lower() or "not officially supported" in w.lower() for w in res_legacy["warnings"]))

    def test_get_gpu_by_detected_name(self):
        gpu = get_gpu_by_detected_name(self.db, "AMD Instinct MI300X OAM")
        self.assertIsNotNone(gpu)
        self.assertEqual(gpu["gfx_id"], "gfx942")


if __name__ == '__main__':
    unittest.main()

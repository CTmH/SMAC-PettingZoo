"""Run rendering tests in a fresh interpreter where smacv2 cannot be imported."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class TestWithoutSMACv2(unittest.TestCase):
    def test_render_without_dependency(self):
        root = Path(__file__).resolve().parents[1]
        code = """
import importlib.abc
import sys
import unittest

class BlockSMACv2(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'smacv2' or fullname.startswith('smacv2.'):
            raise ModuleNotFoundError('smacv2 deliberately unavailable')

sys.meta_path.insert(0, BlockSMACv2())
suite = unittest.defaultTestLoader.discover('tests', pattern='test_render.py')
result = unittest.TextTestRunner().run(suite)
assert not any(name == 'smacv2' or name.startswith('smacv2.') for name in sys.modules)
sys.exit(not result.wasSuccessful())
"""
        subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=root,
                       env={**os.environ, "PYTHONPATH": str(root)}, check=True)

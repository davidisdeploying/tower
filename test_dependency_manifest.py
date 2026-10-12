import importlib
import importlib.metadata
import os
import re
import sys
import unittest


ROOT = os.path.dirname(os.path.abspath(__file__))
LOCK = os.path.join(ROOT, "requirements.lock")
DOC = os.path.join(ROOT, "DEPENDENCIES.md")

DIRECT_IMPORTS = {
    "anyio": "anyio",
    "mcp": "mcp",
    "numpy": "numpy",
    "onnxruntime": "onnxruntime",
    "jwt": "PyJWT",
    "sqlite_vec": "sqlite-vec",
    "tokenizers": "tokenizers",
    "uvicorn": "uvicorn",
}


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def locked():
    entries = {}
    with open(LOCK, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.count("==") != 1:
                raise AssertionError(f"unpinned lock entry: {line}")
            name, version = line.split("==", 1)
            key = canonical(name)
            if key in entries:
                raise AssertionError(f"duplicate lock entry: {name}")
            entries[key] = (name, version)
    return entries


class DependencyManifestTestCase(unittest.TestCase):
    def test_runtime_python_matches_reviewed_abi(self):
        self.assertEqual(sys.version_info[:2], (3, 13))

    def test_every_lock_entry_is_installed_at_exact_version(self):
        entries = locked()
        self.assertGreaterEqual(len(entries), 40)
        for _key, (name, expected) in entries.items():
            with self.subTest(distribution=name):
                self.assertEqual(importlib.metadata.version(name), expected)

    def test_all_direct_runtime_imports_are_declared_and_importable(self):
        entries = locked()
        for module, distribution in DIRECT_IMPORTS.items():
            with self.subTest(module=module):
                self.assertIn(canonical(distribution), entries)
                self.assertIsNotNone(importlib.import_module(module))

    def test_rebuild_document_names_lock_and_verification_gates(self):
        with open(DOC, encoding="utf-8") as handle:
            text = handle.read()
        for required in (
            "requirements.lock", "python3.13 -m venv", "pip check",
            "unittest discover", "restart `tower.service` as",
            "final mutation", "rollback",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()

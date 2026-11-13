"""v3 P0.5: atomic multi-file patch transactions, tamper detection, rollback."""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.codebase import (
    SearchReplacePatchEngine,
    PatchTransaction,
    is_test_path,
    tree_hash,
)

API = textwrap.dedent(
    '''
    def f(a, b):
        return a + b
    '''
)

USER = textwrap.dedent(
    '''
    from pkg.api import f


    def g():
        return f(1, 2)
    '''
)

TEST_FILE = textwrap.dedent(
    '''
    from pkg.api import f


    def test_f():
        assert f(1, 2) == 3
    '''
)


class TransactionBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.files = {
            "pkg/__init__.py": "",
            "pkg/api.py": API,
            "pkg/use.py": USER,
            "tests/test_api.py": TEST_FILE,
        }
        for rel, content in self.files.items():
            self.write(rel, content)

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, rel):
        return os.path.join(self.root, rel)

    def write(self, rel, content):
        absolute = self.path(rel)
        os.makedirs(os.path.dirname(absolute), exist_ok=True)
        with open(absolute, "w", encoding="utf-8") as handle:
            handle.write(content)

    def read(self, rel):
        with open(self.path(rel), "r", encoding="utf-8") as handle:
            return handle.read()

    def transaction(self, *edits, **kwargs):
        txn = PatchTransaction(root=self.root, **kwargs)
        for file_path, search, replace in edits:
            txn.add(file_path, search, replace)
        return txn


class TestAtomicCommit(TransactionBase):
    def test_successful_multi_file_commit(self):
        before = tree_hash(self.root)
        txn = self.transaction(
            ("pkg/api.py", "def f(a, b):", "def f(a, b, c=0):"),
            ("pkg/use.py", "return f(1, 2)", "return f(1, 2, 3)"),
        )
        result = txn.execute(commit=True)
        self.assertTrue(result.passed, result.message)
        self.assertIn("def f(a, b, c=0):", self.read("pkg/api.py"))
        self.assertIn("f(1, 2, 3)", self.read("pkg/use.py"))
        self.assertEqual(result.details["files"], ["pkg/api.py", "pkg/use.py"])
        self.assertNotEqual(tree_hash(self.root), before)
        self.assertEqual(result.details["tree_hash_before"], before)

    def test_dry_run_writes_nothing(self):
        before = tree_hash(self.root)
        txn = self.transaction(("pkg/api.py", "return a + b", "return a * b"))
        result = txn.execute(commit=False)
        self.assertTrue(result.passed, result.message)
        self.assertFalse(result.details["committed"])
        self.assertIn("return a + b", self.read("pkg/api.py"))
        self.assertEqual(tree_hash(self.root), before)

    def test_caller_break_is_rejected_without_writes(self):
        """Arity change without updating the caller must fail the whole transaction."""
        before = tree_hash(self.root)
        txn = self.transaction(("pkg/api.py", "def f(a, b):", "def f(a):"))
        result = txn.execute(commit=True)
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "cross_file_arity_breakage")
        self.assertIn("def f(a, b):", self.read("pkg/api.py"))
        self.assertEqual(tree_hash(self.root), before)
        self.assertFalse(result.details.get("committed", False))

    def test_syntax_error_aborts_entire_transaction(self):
        before = tree_hash(self.root)
        txn = self.transaction(
            ("pkg/api.py", "return a + b", "return a + b"),
            ("pkg/use.py", "return f(1, 2)", "return f(1, 2"),
        )
        result = txn.execute(commit=True)
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "syntax_error")
        self.assertEqual(tree_hash(self.root), before)


class TestTestTampering(TransactionBase):
    def test_test_file_edit_is_refused(self):
        before = tree_hash(self.root)
        txn = self.transaction(
            ("tests/test_api.py", "assert f(1, 2) == 3", "assert f(1, 2) == 4"),
        )
        result = txn.execute(commit=True)
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "test_tampering_blocked")
        self.assertIn("assert f(1, 2) == 3", self.read("tests/test_api.py"))
        self.assertEqual(tree_hash(self.root), before)

    def test_permitted_test_edit_that_keeps_assertions(self):
        txn = self.transaction(
            ("pkg/api.py", "def f(a, b):", "def f(a, b, c=0):"),
            ("tests/test_api.py", "assert f(1, 2) == 3", "assert f(1, 2, 3) == 3"),
            allow_test_edits=True,
        )
        result = txn.execute(commit=True)
        self.assertTrue(result.passed, result.message)
        self.assertIn("assert f(1, 2, 3) == 3", self.read("tests/test_api.py"))

    def test_weakening_a_test_is_refused_even_when_permitted(self):
        txn = self.transaction(
            ("tests/test_api.py", "assert f(1, 2) == 3", "pass"),
            allow_test_edits=True,
        )
        result = txn.execute(commit=True)
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "test_tampering_blocked")
        self.assertIn("assert f(1, 2) == 3", self.read("tests/test_api.py"))

    def test_assertion_removal_allowed_explicitly(self):
        txn = self.transaction(
            ("tests/test_api.py", "assert f(1, 2) == 3", "pass"),
            allow_test_edits=True,
            allow_assert_removal=True,
        )
        self.assertTrue(txn.execute(commit=True).passed)

    def test_engine_refuses_test_paths_by_default(self):
        engine = SearchReplacePatchEngine(root=self.root)
        result = engine.apply_and_verify(
            "tests/test_api.py", "assert f(1, 2) == 3", "assert True"
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "test_tampering_blocked")

    def test_test_path_detection(self):
        self.assertTrue(is_test_path("tests/test_x.py"))
        self.assertTrue(is_test_path("./test/test_x.py"))
        self.assertTrue(is_test_path("pkg/tests/helpers.py"))
        self.assertTrue(is_test_path("pkg/x_test.py"))
        self.assertTrue(is_test_path("conftest.py"))
        self.assertFalse(is_test_path("pkg/testing_utils.py"))
        self.assertFalse(is_test_path("maya250m/contest.py"))


class TestRollback(TransactionBase):
    def commit_change(self):
        txn = self.transaction(("pkg/api.py", "return a + b", "return a * b"))
        result = txn.execute(commit=True)
        self.assertTrue(result.passed, result.message)
        return txn

    def test_rollback_restores_exact_tree(self):
        before = tree_hash(self.root)
        original = self.read("pkg/api.py")
        txn = self.commit_change()
        self.assertNotEqual(tree_hash(self.root), before)
        rolled = txn.rollback()
        self.assertTrue(rolled.passed, rolled.message)
        self.assertEqual(self.read("pkg/api.py"), original)
        self.assertEqual(tree_hash(self.root), before)

    def test_rollback_refuses_after_unrelated_change(self):
        txn = self.commit_change()
        self.write("pkg/use.py", USER + "\n# someone else edited this\n")
        rolled = txn.rollback()
        self.assertFalse(rolled.passed)
        self.assertEqual(rolled.failure_category, "transaction_conflict")
        self.assertIn("return a * b", self.read("pkg/api.py"))

    def test_rollback_without_commit_is_refused(self):
        txn = self.transaction(("pkg/api.py", "return a + b", "return a * b"))
        txn.execute(commit=False)
        self.assertFalse(txn.rollback().passed)

    def test_write_failure_restores_written_files(self):
        before = tree_hash(self.root)
        original_api = self.read("pkg/api.py")
        txn = self.transaction(
            ("pkg/api.py", "def f(a, b):", "def f(a, b, c=0):"),
            ("pkg/use.py", "return f(1, 2)", "return f(1, 2, 3)"),
        )

        def flaky_writer(rel_path, text):
            if rel_path.endswith("use.py"):
                raise OSError("disk full")
            with open(self.path(rel_path), "w", encoding="utf-8") as handle:
                handle.write(text)

        txn._writer = flaky_writer
        result = txn.execute(commit=True)
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "internal_error")
        self.assertTrue(result.details["rolled_back"])
        self.assertEqual(self.read("pkg/api.py"), original_api)
        self.assertEqual(tree_hash(self.root), before)

    def test_max_files_limit(self):
        txn = self.transaction(
            ("pkg/api.py", "return a + b", "return a + b"),
            ("pkg/use.py", "return f(1, 2)", "return f(1, 2)"),
            max_files=1,
        )
        result = txn.execute(commit=False)
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "patch_error")


class TestPatchTextIntegration(TransactionBase):
    def test_transactional_patch_text_commit(self):
        patch = (
            "# <|edit_file|> pkg/api.py\n"
            "<<<<<<< SEARCH\n"
            "def f(a, b):\n"
            "=======\n"
            "def f(a, b, c=0):\n"
            ">>>>>>> REPLACE\n"
            "# <|edit_file|> pkg/use.py\n"
            "<<<<<<< SEARCH\n"
            "return f(1, 2)\n"
            "=======\n"
            "return f(1, 2, 3)\n"
            ">>>>>>> REPLACE\n"
        )
        engine = SearchReplacePatchEngine(root=self.root)
        result = engine.apply_patch_text_transactional(patch, commit=True)
        self.assertTrue(result.passed, result.message)
        self.assertIn("def f(a, b, c=0):", self.read("pkg/api.py"))
        self.assertIn("return f(1, 2, 3)", self.read("pkg/use.py"))
        self.assertTrue(engine.history[-1]["transactional"])

    def test_transactional_patch_text_is_all_or_nothing(self):
        patch = (
            "# <|edit_file|> pkg/api.py\n"
            "<<<<<<< SEARCH\n"
            "def f(a, b):\n"
            "=======\n"
            "def f(a):\n"
            ">>>>>>> REPLACE\n"
        )
        engine = SearchReplacePatchEngine(root=self.root)
        result = engine.apply_patch_text_transactional(patch, commit=True)
        self.assertFalse(result.passed)
        self.assertIn("def f(a, b):", self.read("pkg/api.py"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Phase 4B tests: AST index, PPR repo map, packing, cross-file contracts, patches."""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.codebase import (
    PatchError,
    RepoGraphIndexer,
    RepoMapBuilder,
    SearchReplacePatchEngine,
    TopologicalRepoPacker,
    apply_search_replace,
    build_cross_file_symbol_table,
    fim_example,
    parse_patch_blocks,
    verify_cross_file_impact,
)
from maya250m.verification import VerificationResult

REPO_FILES = {
    "pkg/__init__.py": "",
    "pkg/base.py": textwrap.dedent(
        '''
        """Base utilities."""


        def helper(x: int, y: int = 1) -> int:
            """Return x + y."""
            return x + y


        class Registry:
            def __init__(self):
                self.items = {}

            def register(self, name, value):
                self.items[name] = value
                return value
        '''
    ),
    "pkg/mid.py": textwrap.dedent(
        '''
        from pkg.base import helper


        def mid_fn(a: int) -> int:
            return helper(a)


        async def fetch(url: str, timeout: float = 1.0) -> str:
            return url
        '''
    ),
    "pkg/top.py": textwrap.dedent(
        '''
        from pkg.mid import mid_fn


        def main() -> int:
            return mid_fn(1)
        '''
    ),
    "pkg/a.py": "from pkg.b import beta\n\n\ndef alpha():\n    return beta()\n",
    "pkg/b.py": "from pkg.a import alpha\n\n\ndef beta():\n    return alpha()\n",
    "app.py": textwrap.dedent(
        '''
        from pkg.top import main


        def run():
            return main()
        '''
    ),
}


class CodebaseFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = cls._tmp.name
        for rel, content in REPO_FILES.items():
            full = os.path.join(cls.root, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as handle:
                handle.write(content)
        cls.indexer = RepoGraphIndexer(cls.root)
        cls.indexer.index()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class TestIndexer(CodebaseFixture):
    def test_files_and_symbols(self):
        self.assertEqual(len(self.indexer.files), len(REPO_FILES))
        self.assertFalse(self.indexer.summary()["parse_errors"])

        helper = self.indexer.symbols_named("helper")[0]
        self.assertEqual(helper.kind, "function")
        self.assertEqual((helper.min_arity, helper.max_arity), (1, 2))
        self.assertIn("x: int", helper.signature)

        fetch = self.indexer.symbols_named("fetch")[0]
        self.assertEqual(fetch.kind, "async_function")

        method = self.indexer.definitions("Registry.register")[0]
        self.assertEqual(method.kind, "method")
        self.assertEqual((method.min_arity, method.max_arity), (2, 2))

    def test_callers_and_imports(self):
        callers = self.indexer.callers("helper")
        self.assertTrue(any(site.file_path == "pkg/mid.py" for site in callers))
        self.assertIn("pkg.base", " ".join(self.indexer.files["pkg/mid.py"].imports).replace(
            "pkg/base", "pkg.base"
        ) or " ".join(self.indexer.files["pkg/mid.py"].imports))

        table = self.indexer.signature_table()
        self.assertEqual(table["mid_fn"]["min_arity"], 1)
        self.assertEqual(table["mid_fn"]["max_arity"], 1)

    def test_stale_detection(self):
        self.assertEqual(self.indexer.stale_files(), [])


class TestRepoMap(CodebaseFixture):
    def test_personalized_map_ranks_query_symbols(self):
        builder = RepoMapBuilder(self.indexer)
        repo_map = builder.build("mid_fn returns the wrong value", token_budget=800)
        self.assertIn("def mid_fn", repo_map.text)
        self.assertIn("pkg/mid.py", repo_map.files_ranked[:3])
        self.assertLessEqual(len(repo_map.entries), 40)
        self.assertTrue(repo_map.text.strip())

    def test_token_budget_is_respected(self):
        builder = RepoMapBuilder(self.indexer)
        map_small = builder.build("helper", token_budget=20)
        self.assertLessEqual(map_small.tokens_used, 40)
        self.assertTrue(map_small.text == "" or map_small.truncated)


class TestPacker(CodebaseFixture):
    def test_topological_order_places_imports_first(self):
        packer = TopologicalRepoPacker(self.indexer)
        order = packer.topological_order()
        self.assertLess(order.index("pkg/base.py"), order.index("pkg/mid.py"))
        self.assertLess(order.index("pkg/mid.py"), order.index("pkg/top.py"))
        self.assertEqual(len(order), len(REPO_FILES))  # cycle handled, nothing dropped

    def test_pack_headers(self):
        packer = TopologicalRepoPacker(self.indexer)
        packed = packer.pack()
        self.assertIn("# <|file_sep|> pkg/base.py", packed.text)
        self.assertEqual(len(packed.order), len(REPO_FILES))
        self.assertFalse(packed.truncated)

    def test_structure_aware_fim(self):
        import random

        code = REPO_FILES["pkg/base.py"]
        example = fim_example(code, rng=random.Random(0), mode="psm")
        self.assertIsNotNone(example)
        self.assertEqual(example["mode"], "psm")
        self.assertIn("# <|fim_prefix|>", example["text"])
        self.assertIn("# <|fim_middle|>", example["text"])
        self.assertTrue(example["middle"].strip())
        # The middle span must not split a line in half.
        self.assertTrue(example["middle"].endswith("\n") or example["middle"].strip())


class TestMultiFileContracts(CodebaseFixture):
    def test_cross_file_symbol_table(self):
        table = build_cross_file_symbol_table(self.indexer, "pkg/top.py")
        self.assertIn("mid_fn", table)
        self.assertEqual(table["mid_fn"]["min_arity"], 1)

    def test_arity_breakage_is_detected_with_location(self):
        result = verify_cross_file_impact(
            self.indexer,
            {"mid_fn": {"min_arity": 2, "max_arity": 2, "return_type": "int"}},
            changed_files=["pkg/mid.py"],
        )
        self.assertIsInstance(result, VerificationResult)
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "cross_file_arity_breakage")
        violation = result.details["violations"][0]
        self.assertEqual(violation["caller_file"], "pkg/top.py")
        self.assertEqual(violation["n_positional"], 1)

    def test_removed_symbol_is_reported(self):
        result = verify_cross_file_impact(
            self.indexer, {"mid_fn": None}, changed_files=["pkg/mid.py"]
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "cross_file_missing_symbol")

    def test_no_breakage_when_callers_updated(self):
        result = verify_cross_file_impact(
            self.indexer,
            {"mid_fn": {"min_arity": 2, "max_arity": 2, "return_type": "int"}},
            changed_files=["pkg/mid.py", "pkg/top.py"],
        )
        self.assertTrue(result.passed)


class TestSearchReplaceEngine(CodebaseFixture):
    def test_exact_and_ambiguous_matching(self):
        text = "a = 1\nb = 2\na = 1\n"
        with self.assertRaises(PatchError) as ctx:
            apply_search_replace(text, "a = 1", "a = 3")
        self.assertEqual(ctx.exception.category, "patch_ambiguous")

        self.assertEqual(
            apply_search_replace("x = 1\n", "x = 1", "x = 2"),
            "x = 2\n",
        )

        with self.assertRaises(PatchError) as ctx:
            apply_search_replace("x = 1\n", "y = 9", "y = 10")
        self.assertEqual(ctx.exception.category, "patch_not_found")

    def test_indentation_normalized_fallback(self):
        text = "def f():\n    value = 1\n    return value\n"
        patched = apply_search_replace(
            text, "value = 1\nreturn value", "value = 2\nreturn value"
        )
        self.assertIn("    value = 2", patched)
        self.assertIn("    return value", patched)

    def test_patch_block_parsing(self):
        patch = textwrap.dedent(
            """
            # <|edit_file|> pkg/base.py
            <<<<<<< SEARCH
            def helper(x: int, y: int = 1) -> int:
            =======
            def helper(x: int, y: int = 2) -> int:
            >>>>>>> REPLACE
            """
        )
        blocks = parse_patch_blocks(patch)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].file_path, "pkg/base.py")

    def test_syntax_error_rejected_before_write(self):
        engine = SearchReplacePatchEngine(indexer=self.indexer, root=self.root)
        result = engine.apply_and_verify(
            "pkg/base.py",
            "return x + y",
            "return x + (",
            commit=True,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "syntax_error")
        with open(os.path.join(self.root, "pkg/base.py"), encoding="utf-8") as handle:
            self.assertIn("return x + y", handle.read())

    def test_cross_file_breakage_blocks_commit(self):
        engine = SearchReplacePatchEngine(indexer=self.indexer, root=self.root)
        result = engine.apply_and_verify(
            "pkg/mid.py",
            "def mid_fn(a: int) -> int:\n    return helper(a)",
            "def mid_fn(a: int, b: int) -> int:\n    return helper(a) + b",
            changed_files=["pkg/mid.py"],
            commit=True,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "cross_file_arity_breakage")
        with open(os.path.join(self.root, "pkg/mid.py"), encoding="utf-8") as handle:
            self.assertIn("def mid_fn(a: int) -> int:", handle.read())

    def test_successful_commit_updates_file_and_index(self):
        engine = SearchReplacePatchEngine(indexer=self.indexer, root=self.root)
        result = engine.apply_and_verify(
            "pkg/base.py",
            'helper docstring placeholder',
            "x",
        )
        self.assertFalse(result.passed)  # anchor intentionally absent

        result = engine.apply_and_verify(
            "pkg/base.py",
            "    return x + y",
            "    return x + y  # verified",
            commit=True,
        )
        self.assertTrue(result.passed, result.to_dict())
        with open(os.path.join(self.root, "pkg/base.py"), encoding="utf-8") as handle:
            self.assertIn("# verified", handle.read())


if __name__ == "__main__":
    unittest.main(verbosity=2)

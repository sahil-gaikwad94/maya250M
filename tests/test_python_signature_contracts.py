"""v3 P0.4: full Python calling-convention contracts (adversarial cases)."""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.codebase import RepoGraphIndexer, verify_cross_file_impact
from maya250m.codebase.contracts import (
    KEYWORD_ONLY,
    POS_ONLY,
    POS_OR_KEYWORD,
    VAR_KEYWORD,
    VAR_POSITIONAL,
    CallableContract,
    build_contracts,
    check_call,
    contract_from_source,
    header_contract_from_text,
    signature_table_from_source,
)
from maya250m.vexx import Vexx35Controller

SOURCE = textwrap.dedent(
    '''
    """Signatures covering the awkward parts of Python's calling convention."""
    from typing import Dict, List
    from typing import overload


    def plain(a, b):
        return a + b


    def posonly(a, /, b, c=3):
        return a + b + c


    def kwonly(a, *args, b, c=1, **kwargs):
        return a


    def defaults(x: Dict[str, List[int]] = {"a": []}, y: int = 2):
        return x


    def star(a, *args):
        return args


    def double_star(a, **kwargs):
        return kwargs


    def nested(


        a: int,
        b: "Dict[str, int]" = {},
    ) -> "List[int]":
        return [a]


    class Service:
        def __init__(self, name):
            self.name = name

        def run(self, x, *, strict=False):
            return x

        @staticmethod
        def make(a, b=2):
            return a

        @classmethod
        def build(cls, a):
            return cls(a)


    class Other:
        def run(self, x, y, z):
            return x


    @overload
    def pick(value: int) -> int: ...


    @overload
    def pick(value: str) -> str: ...


    def pick(value):
        return value
    '''
)


class TestContractModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contracts = build_contracts(SOURCE, module="sample")

    def contract(self, qualname):
        return self.contracts[qualname][-1]

    def test_parameter_kinds(self):
        kinds = [(p.name, p.kind) for p in self.contract("posonly").parameters]
        self.assertEqual(
            kinds,
            [("a", POS_ONLY), ("b", POS_OR_KEYWORD), ("c", POS_OR_KEYWORD)],
        )
        kw = self.contract("kwonly")
        self.assertEqual(
            [p.kind for p in kw.parameters],
            [POS_OR_KEYWORD, VAR_POSITIONAL, KEYWORD_ONLY, KEYWORD_ONLY, VAR_KEYWORD],
        )
        self.assertEqual(kw.required_keyword_names(), ["b"])

    def test_defaults_and_nested_annotations(self):
        defaults = self.contract("defaults")
        annotations = [p.annotation for p in defaults.parameters]
        self.assertEqual(annotations[0], "Dict[str, List[int]]")
        self.assertEqual(annotations[1], "int")
        self.assertEqual(defaults.return_annotation, None)
        self.assertEqual(defaults.positional_bounds(), (0, 2))

    def test_multiline_signature_parses(self):
        nested = self.contract("nested")
        self.assertEqual([p.name for p in nested.parameters], ["a", "b"])
        self.assertEqual(nested.return_annotation, "'List[int]'")

    def test_binding_and_self_removal(self):
        self.assertEqual(self.contract("Service.run").binding, "instance")
        self.assertEqual(
            [p.name for p in self.contract("Service.run").parameters], ["x", "strict"]
        )
        self.assertEqual(self.contract("Service.make").binding, "static")
        self.assertEqual(self.contract("Service.build").binding, "class")
        self.assertEqual(
            [p.name for p in self.contract("Service.build").parameters], ["a"]
        )

    def test_overloads_are_preserved_and_implementation_selected(self):
        self.assertEqual(len(self.contracts["pick"]), 3)
        self.assertIsNotNone(self.contracts["pick"][0].overload_group)
        self.assertIsNone(self.contract("pick").overload_group)
        self.assertEqual(contract_from_source(SOURCE, "pick").return_annotation, None)

    def test_same_method_name_does_not_collapse(self):
        self.assertIn("Service.run", self.contracts)
        self.assertIn("Other.run", self.contracts)
        self.assertNotEqual(
            self.contract("Service.run").symbol_id, self.contract("Other.run").symbol_id
        )

    def test_signature_table_uses_widest_bounds(self):
        table = signature_table_from_source(SOURCE)
        self.assertEqual(table["plain"]["min_arity"], 2)
        self.assertIsNone(table["star"]["max_arity"])
        self.assertEqual(table["double_star"]["max_arity"], 1)
        self.assertIn("b", table["kwonly"]["keyword_only"])


class TestCallChecking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contracts = build_contracts(SOURCE, module="sample")

    def contract(self, qualname):
        return self.contracts[qualname][-1]

    def test_positional_only_cannot_be_keyword(self):
        contract = self.contract("posonly")
        self.assertTrue(check_call(contract, 2).passed)
        self.assertTrue(check_call(contract, 3).passed)
        self.assertEqual(check_call(contract, 1).status, "too_few_positional")
        self.assertEqual(check_call(contract, 1, ["a"]).status, "duplicate_argument")
        as_keyword = check_call(contract, 0, ["a", "b"])
        self.assertEqual(as_keyword.status, "unexpected_keyword")
        self.assertEqual(as_keyword.unexpected_keyword, ("a",))

    def test_keyword_only_requirements_are_separate_from_arity(self):
        contract = self.contract("kwonly")
        self.assertTrue(check_call(contract, 1, ["b"]).passed)
        missing = check_call(contract, 1)
        self.assertEqual(missing.status, "missing_keyword")
        self.assertEqual(missing.kind, "keyword")
        self.assertEqual(missing.missing_keyword, ("b",))
        # **kwargs in the *callee* absorbs unknown keywords, so this call is valid.
        self.assertTrue(check_call(contract, 1, ["b", "zzz"]).passed)
        strict = self.contracts["Service.run"][-1]
        extra = check_call(strict, 1, ["bogus"])
        self.assertEqual(extra.status, "unexpected_keyword")

    def test_star_args_allow_extra_positional(self):
        contract = self.contract("star")
        self.assertTrue(check_call(contract, 9).passed)
        self.assertEqual(check_call(contract, 0).status, "too_few_positional")

    def test_double_star_is_not_positional_varargs(self):
        """The v2 bug: **kwargs must not permit extra positional arguments."""
        contract = self.contract("double_star")
        self.assertTrue(contract.has_var_keyword)
        self.assertFalse(contract.has_var_positional)
        self.assertEqual(contract.positional_bounds(), (1, 1))
        self.assertEqual(check_call(contract, 3).status, "too_many_positional")
        self.assertTrue(check_call(contract, 1, ["anything", "else"]).passed)

    def test_defaults_make_positional_optional(self):
        contract = self.contract("defaults")
        self.assertTrue(check_call(contract, 0).passed)
        self.assertTrue(check_call(contract, 2).passed)
        self.assertEqual(check_call(contract, 3).status, "too_many_positional")

    def test_duplicate_argument_detected(self):
        contract = self.contract("plain")
        duplicate = check_call(contract, 2, ["a"])
        self.assertEqual(duplicate.status, "duplicate_argument")

    def test_call_site_kwargs_suppresses_false_rejection(self):
        contract = self.contract("plain")
        self.assertTrue(check_call(contract, 0, [], has_kwargs=True).passed)
        self.assertTrue(check_call(contract, 5, [], has_starargs=True).passed)

    def test_unknown_contract_is_not_a_verdict(self):
        unknown = CallableContract.unknown("m::f", "dynamic")
        result = check_call(unknown, 3)
        self.assertEqual(result.status, "unknown_dynamic_dispatch")
        self.assertFalse(result.passed)


class TestInFlightHeaders(unittest.TestCase):
    def test_complete_header_is_partial_ast(self):
        contract = header_contract_from_text(
            "def f(a: int, b: Dict[str, List[int]] = {}) -> None:\n    "
        )
        self.assertIsNotNone(contract)
        self.assertEqual(contract.symbol_id.rsplit(":", 1)[-1], "f")
        self.assertEqual(contract.confidence, "partial_ast")
        self.assertEqual([p.name for p in contract.parameters], ["a", "b"])
        self.assertEqual(contract.positional_bounds(), (1, 2))

    def test_multiline_incomplete_header(self):
        text = 'def g(\n    a: int,\n    b: "Tuple[int, ...]" = (),\n'
        contract = header_contract_from_text(text)
        self.assertIsNotNone(contract)
        self.assertEqual([p.name for p in contract.parameters], ["a", "b"])

    def test_unbalanced_header_recovers_partially_never_exactly(self):
        contract = header_contract_from_text("def broken(a, b: List[int")
        self.assertIsNotNone(contract)
        self.assertEqual(contract.confidence, "partial_ast")
        self.assertEqual([p.name for p in contract.parameters], ["a", "b"])
        self.assertEqual(contract.positional_bounds(), (2, 2))

    def test_no_signature_returns_none(self):
        self.assertIsNone(header_contract_from_text("x = 1\ny = 2\n"))

    def test_vexx_table_marks_partial_and_keeps_kwargs_semantics(self):
        table = Vexx35Controller.extract_ast_symbol_table(
            "def f(a, **kw):\n    pass\n"
        )
        self.assertIn("f", table)
        self.assertEqual(table["f"]["max_arity"], 1)
        self.assertTrue(table["f"]["has_var_keyword"])
        self.assertFalse(table["f"]["partial"])
        partial_table = Vexx35Controller.extract_ast_symbol_table("def g(a, b=1")
        self.assertTrue(partial_table["g"]["partial"])
        self.assertEqual(partial_table["g"]["max_arity"], 2)


REPO = {
    "pkg/__init__.py": "",
    "pkg/api.py": textwrap.dedent(
        '''
        def greet(name, *, punctuation="!"):
            return name + punctuation
        '''
    ),
    "pkg/caller.py": textwrap.dedent(
        '''
        from pkg.api import greet
        import pkg.api


        def call_positional():
            return greet("a", "b")


        def call_missing_kwarg_only():
            return greet("a")


        def call_valid():
            return greet("a", punctuation="?")


        def call_via_kwargs(mapping):
            return greet("a", **mapping)


        def call_module_qualified():
            return pkg.api.greet("a")
        '''
    ),
}


class TestCrossFileTypedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = cls._tmp.name
        for rel, content in REPO.items():
            path = os.path.join(cls.root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
        cls.indexer = RepoGraphIndexer(cls.root)
        cls.indexer.index()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_indexer_stores_full_contract(self):
        symbol = self.indexer.symbols_named("greet")[0]
        contract = symbol.callable_contract()
        self.assertIsNotNone(contract)
        self.assertEqual(contract.required_keyword_names(), [])
        self.assertTrue(contract.accepts_keyword("punctuation"))
        table = self.indexer.signature_table()
        self.assertIn("punctuation", table["greet"]["keyword_only"])

    def test_too_many_positional_reported_with_kind(self):
        contract = self.indexer.symbols_named("greet")[0].to_dict()["contract"]
        result = verify_cross_file_impact(
            self.indexer, {"greet": contract}, changed_files=["pkg/api.py"]
        )
        self.assertFalse(result.passed)
        kinds = {v["kind"] for v in result.details["violations"]}
        self.assertIn("positional", kinds)
        self.assertTrue(any(v["caller_file"] == "pkg/caller.py" for v in result.details["violations"]))

    def test_kwargs_call_site_is_not_falsely_rejected(self):
        contract = self.indexer.symbols_named("greet")[0].to_dict()["contract"]
        result = verify_cross_file_impact(
            self.indexer, {"greet": contract}, changed_files=["pkg/api.py"]
        )
        flagged_callers = [v["caller"] for v in result.details["violations"]]
        self.assertNotIn("call_via_kwargs", flagged_callers)

    def test_required_keyword_only_addition_is_detected(self):
        contract = self.indexer.symbols_named("greet")[0].to_dict()["contract"]
        contract["parameters"].append(
            {
                "name": "mode",
                "kind": "keyword_only",
                "required": True,
                "annotation": None,
                "default_repr": None,
            }
        )
        result = verify_cross_file_impact(
            self.indexer, {"greet": contract}, changed_files=["pkg/api.py"]
        )
        self.assertFalse(result.passed)
        statuses = {v["status"] for v in result.details["violations"]}
        self.assertIn("missing_keyword", statuses)


if __name__ == "__main__":
    unittest.main(verbosity=2)

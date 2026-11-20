"""Tests for the three-valued-logic predicate model."""

from __future__ import annotations

from fractions import Fraction

import pytest

from extcontract import (
    ContractError,
    LimitError,
    F,
    T,
    U,
    parse_contract,
    parse_predicate,
    tribool_and,
    tribool_not,
    tribool_or,
    evaluate_atom,
    evaluate_predicate,
)


def _reader(fields, version="r1"):
    return parse_contract(
        {"version": version, "fields": fields}, role="reader"
    )


def _field(field_id, type_json, nullable=False):
    return {
        "field_id": field_id,
        "name": field_id,
        "type": type_json,
        "nullable": nullable,
    }


INT_NULL = {"kind": "INT", "bits": 8}
DEC = {"kind": "DECIMAL", "precision": 5, "scale": 2}


# ---------------------------------------------------------------------------
# Kleene truth tables: unknown and false stay distinct
# ---------------------------------------------------------------------------

def test_not_truth_table():
    assert tribool_not(T) == F
    assert tribool_not(F) == T
    assert tribool_not(U) == U


def test_and_truth_table_unknown_distinct_from_false():
    assert tribool_and([T, T]) == T
    assert tribool_and([T, F]) == F
    assert tribool_and([F, U]) == F       # false dominates unknown
    assert tribool_and([T, U]) == U       # unknown propagates
    assert tribool_and([U, U]) == U


def test_or_truth_table_unknown_distinct_from_false():
    assert tribool_or([F, F]) == F
    assert tribool_or([F, T]) == T
    assert tribool_or([T, U]) == T        # true dominates unknown
    assert tribool_or([F, U]) == U
    assert tribool_or([U, U]) == U


# ---------------------------------------------------------------------------
# Atom semantics
# ---------------------------------------------------------------------------

def test_eq_atom_null_is_unknown_never_false():
    reader = _reader([_field("x", INT_NULL, nullable=True)])
    pred = parse_predicate(
        {
            "predicate_id": "p",
            "atoms": [{"atom_id": "A", "field_id": "x", "op": "eq", "constant": 1}],
            "expression": {"atom": "A"},
        },
        reader,
    )
    atom = pred.atoms[0]
    assert evaluate_atom(atom, 1) == T
    assert evaluate_atom(atom, 2) == F
    assert evaluate_atom(atom, None) == U


def test_lt_atom_exact_numeric_and_null_unknown():
    reader = _reader([_field("d", DEC, nullable=True)])
    pred = parse_predicate(
        {
            "predicate_id": "p",
            "atoms": [{"atom_id": "L", "field_id": "d", "op": "lt", "threshold": "3/2"}],
            "expression": {"atom": "L"},
        },
        reader,
    )
    atom = pred.atoms[0]
    assert atom.constant == Fraction(3, 2)
    assert evaluate_atom(atom, Fraction(1)) == T
    assert evaluate_atom(atom, Fraction(3, 2)) == F      # strict less-than
    assert evaluate_atom(atom, Fraction(2)) == F
    assert evaluate_atom(atom, None) == U


def test_is_null_atom_always_definite():
    reader = _reader([_field("x", INT_NULL, nullable=True)])
    pred = parse_predicate(
        {
            "predicate_id": "p",
            "atoms": [{"atom_id": "N", "field_id": "x", "op": "is_null"}],
            "expression": {"atom": "N"},
        },
        reader,
    )
    atom = pred.atoms[0]
    assert evaluate_atom(atom, None) == T
    assert evaluate_atom(atom, 5) == F
    assert U not in (evaluate_atom(atom, None), evaluate_atom(atom, 5))


def test_text_eq_uses_string_identity():
    reader = _reader([_field("s", {"kind": "TEXT"}, nullable=True)])
    pred = parse_predicate(
        {
            "predicate_id": "p",
            "atoms": [{"atom_id": "A", "field_id": "s", "op": "eq", "constant": "ok"}],
            "expression": {"atom": "A"},
        },
        reader,
    )
    atom = pred.atoms[0]
    assert evaluate_atom(atom, "ok") == T
    assert evaluate_atom(atom, "no") == F
    assert evaluate_atom(atom, None) == U


# ---------------------------------------------------------------------------
# NOT(A OR B) distinguishes FF (true) from UF (unknown)
# ---------------------------------------------------------------------------

def test_not_or_separates_ff_from_uf():
    reader = _reader(
        [_field("x", INT_NULL, True), _field("y", INT_NULL, True)]
    )
    pred = parse_predicate(
        {
            "predicate_id": "p",
            "atoms": [
                {"atom_id": "A", "field_id": "x", "op": "eq", "constant": 1},
                {"atom_id": "B", "field_id": "y", "op": "eq", "constant": 1},
            ],
            "expression": {"not": {"or": ["A", "B"]}},
        },
        reader,
    )
    cases = {
        (2, 2): T,        # FF
        (None, 2): U,     # UF - unknown, NOT unknown stays unknown
        (2, None): U,     # FU
        (None, None): U,  # UU
        (1, 1): F,        # TT
    }
    for (x, y), verdict in cases.items():
        _, expr = evaluate_predicate(pred, {"x": x, "y": y})
        assert expr == verdict, (x, y)


def test_only_true_is_delivered_f_and_u_both_stop():
    reader = _reader([_field("x", INT_NULL, True)])
    pred = parse_predicate(
        {
            "predicate_id": "p",
            "atoms": [{"atom_id": "A", "field_id": "x", "op": "eq", "constant": 1}],
            "expression": {"atom": "A"},
        },
        reader,
    )
    assert evaluate_predicate(pred, {"x": 1})[1] == T
    assert evaluate_predicate(pred, {"x": 2})[1] == F
    assert evaluate_predicate(pred, {"x": None})[1] == U
    # F and U are different values but neither is delivered.
    outcomes = {evaluate_predicate(pred, {"x": v})[1] for v in (1, 2, None)}
    assert outcomes == {T, F, U}


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------

def test_unknown_field_rejected():
    reader = _reader([_field("x", INT_NULL)])
    with pytest.raises(ContractError, match="not declared by the reader"):
        parse_predicate(
            {
                "predicate_id": "p",
                "atoms": [{"atom_id": "A", "field_id": "z", "op": "is_null"}],
                "expression": {"atom": "A"},
            },
            reader,
        )


def test_duplicate_atom_id_rejected():
    reader = _reader([_field("x", INT_NULL), _field("y", INT_NULL)])
    with pytest.raises(ContractError, match="duplicate atom_id"):
        parse_predicate(
            {
                "predicate_id": "p",
                "atoms": [
                    {"atom_id": "A", "field_id": "x", "op": "is_null"},
                    {"atom_id": "A", "field_id": "y", "op": "is_null"},
                ],
                "expression": {"and": ["A", "A"]},
            },
            reader,
        )


def test_two_atoms_on_one_field_rejected():
    reader = _reader([_field("x", INT_NULL)])
    with pytest.raises(ContractError, match="more than one"):
        parse_predicate(
            {
                "predicate_id": "p",
                "atoms": [
                    {"atom_id": "A", "field_id": "x", "op": "is_null"},
                    {"atom_id": "B", "field_id": "x", "op": "eq", "constant": 1},
                ],
                "expression": {"and": ["A", "B"]},
            },
            reader,
        )


def test_atom_limit_enforced():
    reader = _reader([_field(f"c{i}", INT_NULL) for i in range(9)])
    with pytest.raises(LimitError, match="at most 8 atoms"):
        parse_predicate(
            {
                "predicate_id": "p",
                "atoms": [
                    {"atom_id": f"A{i}", "field_id": f"c{i}", "op": "is_null"}
                    for i in range(9)
                ],
                "expression": {"atom": "A0"},
            },
            reader,
        )


def test_lt_on_text_rejected():
    reader = _reader([_field("s", {"kind": "TEXT"})])
    with pytest.raises(ContractError, match="numeric field"):
        parse_predicate(
            {
                "predicate_id": "p",
                "atoms": [{"atom_id": "L", "field_id": "s", "op": "lt", "threshold": 1}],
                "expression": {"atom": "L"},
            },
            reader,
        )


def test_null_constant_rejected():
    reader = _reader([_field("x", INT_NULL, True)])
    with pytest.raises(ContractError, match="constant may not be null"):
        parse_predicate(
            {
                "predicate_id": "p",
                "atoms": [{"atom_id": "A", "field_id": "x", "op": "eq", "constant": None}],
                "expression": {"atom": "A"},
            },
            reader,
        )


def test_decimal_constant_must_be_fraction_string():
    reader = _reader([_field("d", DEC)])
    with pytest.raises(ContractError, match="fraction string"):
        parse_predicate(
            {
                "predicate_id": "p",
                "atoms": [{"atom_id": "L", "field_id": "d", "op": "lt", "threshold": 1.5}],
                "expression": {"atom": "L"},
            },
            reader,
        )


def test_constant_outside_reader_domain_rejected():
    reader = _reader([_field("x", INT_NULL)])
    with pytest.raises(ContractError, match="outside INT"):
        parse_predicate(
            {
                "predicate_id": "p",
                "atoms": [{"atom_id": "A", "field_id": "x", "op": "eq", "constant": 999}],
                "expression": {"atom": "A"},
            },
            reader,
        )


def test_unknown_expression_reference_rejected():
    reader = _reader([_field("x", INT_NULL)])
    with pytest.raises(ContractError, match="unknown atom reference"):
        parse_predicate(
            {
                "predicate_id": "p",
                "atoms": [{"atom_id": "A", "field_id": "x", "op": "is_null"}],
                "expression": {"atom": "Z"},
            },
            reader,
        )


def test_bare_atom_reference_normalised():
    reader = _reader([_field("x", INT_NULL), _field("y", INT_NULL)])
    pred = parse_predicate(
        {
            "predicate_id": "p",
            "atoms": [
                {"atom_id": "A", "field_id": "x", "op": "is_null"},
                {"atom_id": "B", "field_id": "y", "op": "is_null"},
            ],
            "expression": {"and": ["A", "B"]},
        },
        reader,
    )
    assert pred.expression == {"and": [{"atom": "A"}, {"atom": "B"}]}


def test_fingerprint_stable_and_content_sensitive():
    reader = _reader([_field("x", INT_NULL, True)])
    base = {
        "predicate_id": "p",
        "atoms": [{"atom_id": "A", "field_id": "x", "op": "eq", "constant": 1}],
        "expression": {"atom": "A"},
    }
    p1 = parse_predicate(base, reader)
    p2 = parse_predicate(dict(base), reader)
    assert p1.fingerprint() == p2.fingerprint()
    changed = dict(base)
    changed["atoms"] = [
        {"atom_id": "A", "field_id": "x", "op": "eq", "constant": 2}
    ]
    assert parse_predicate(changed, reader).fingerprint() != p1.fingerprint()

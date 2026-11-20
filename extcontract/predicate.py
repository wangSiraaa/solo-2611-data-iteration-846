"""Three-valued-logic filter predicates bound to reader fields by ``field_id``.

A predicate document is deliberately tiny - this is not a SQL compiler:

    {
      "predicate_id": "p1",
      "atoms": [
        {"atom_id": "A", "field_id": "x", "op": "eq", "constant": 1},
        {"atom_id": "B", "field_id": "y", "op": "lt", "threshold": "3/2"},
        {"atom_id": "C", "field_id": "z", "op": "is_null"}
      ],
      "expression": {"and": ["A", {"or": ["B", {"not": "C"}]}]}
    }

Atom semantics (``T`` = true, ``F`` = false, ``U`` = unknown):

  * ``eq``  - value equals a same-typed constant; ``null`` gives ``U``
  * ``lt``  - numeric value is strictly below an exact numeric threshold;
              ``null`` gives ``U``
  * ``is_null`` - always gives a definite ``T`` or ``F``, never ``U``

Expressions are built only from atom references and ``and`` / ``or`` /
``not``, combined with Kleene three-valued logic.  A row is delivered iff the
whole expression evaluates to exactly ``T``; ``F`` and ``U`` are kept distinct
all the way through and both mean "not delivered", but for different reasons.

At most :data:`MAX_ATOMS` atoms per predicate (the documented work-order
envelope).  Constants are validated against the *reader* field type because
predicates run on adapted reader rows; exact fraction strings are required for
DECIMAL constants and JSON integers for INT constants, matching the type
model in :mod:`extcontract.types`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .contract import Contract
from .errors import ContractError, LimitError
from .types import (
    IntType,
    TextType,
    Type,
    is_legal_value,
    rational_string,
)

T = "T"
F = "F"
U = "U"
TRIBOOL = (T, F, U)

MAX_ATOMS = 8

_OPS = ("eq", "lt", "is_null")
_EXPR_KEYS = ("atom", "and", "or", "not")


@dataclass(frozen=True)
class Atom:
    atom_id: str
    field_id: str
    op: str
    constant: Any = None  # int / Fraction / str / None

    def to_json(self) -> dict:
        if self.op == "eq":
            return {
                "atom_id": self.atom_id,
                "field_id": self.field_id,
                "op": "eq",
                "constant": _encode_constant(self.constant),
            }
        if self.op == "lt":
            return {
                "atom_id": self.atom_id,
                "field_id": self.field_id,
                "op": "lt",
                "threshold": _encode_constant(self.constant),
            }
        return {
            "atom_id": self.atom_id,
            "field_id": self.field_id,
            "op": "is_null",
        }


@dataclass
class Predicate:
    predicate_id: str
    atoms: tuple[Atom, ...]
    expression: dict
    atom_by_id: dict[str, Atom] = field(init=False)

    def __post_init__(self) -> None:
        self.atom_by_id = {a.atom_id: a for a in self.atoms}

    @property
    def atom_ids(self) -> list[str]:
        return [a.atom_id for a in self.atoms]

    def to_json(self) -> dict:
        return {
            "predicate_id": self.predicate_id,
            "atoms": [a.to_json() for a in self.atoms],
            "expression": _copy_expression(self.expression),
        }

    def fingerprint(self) -> str:
        """Stable content hash over atoms + expression (ids included)."""
        payload = json.dumps(
            self.to_json(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------

def parse_predicate(raw: Any, reader: Contract) -> Predicate:
    """Parse and fully validate a predicate document against a reader contract."""
    if not isinstance(raw, dict):
        raise ContractError("predicate must be a JSON object")
    extra = set(raw) - {"predicate_id", "atoms", "expression"}
    if extra:
        raise ContractError(f"predicate: unexpected keys {sorted(extra)}")
    predicate_id = raw.get("predicate_id")
    if not isinstance(predicate_id, str) or not predicate_id:
        raise ContractError("predicate_id must be a non-empty string")

    atoms_raw = raw.get("atoms")
    if not isinstance(atoms_raw, list) or not atoms_raw:
        raise ContractError("predicate: atoms must be a non-empty list")
    if len(atoms_raw) > MAX_ATOMS:
        raise LimitError(
            f"predicate: at most {MAX_ATOMS} atoms, got {len(atoms_raw)}"
        )

    reader_fields = reader.field_by_id()
    atoms: list[Atom] = []
    seen_ids: set[str] = set()
    seen_fields: set[str] = set()
    for index, item in enumerate(atoms_raw):
        atom = _parse_atom(item, index, reader_fields)
        if atom.atom_id in seen_ids:
            raise ContractError(f"predicate: duplicate atom_id {atom.atom_id!r}")
        if atom.field_id in seen_fields:
            raise ContractError(
                f"predicate: field_id {atom.field_id!r} is bound by more than one "
                f"atom; each column needs at most one cold read"
            )
        seen_ids.add(atom.atom_id)
        seen_fields.add(atom.field_id)
        atoms.append(atom)

    expression = raw.get("expression")
    normalized = _validate_expression(expression, seen_ids)
    return Predicate(
        predicate_id=predicate_id,
        atoms=tuple(atoms),
        expression=normalized,
    )


def _parse_atom(item: Any, index: int, reader_fields: dict) -> Atom:
    if not isinstance(item, dict):
        raise ContractError(f"atom #{index}: must be an object")
    atom_id = item.get("atom_id")
    if not isinstance(atom_id, str) or not atom_id:
        raise ContractError(f"atom #{index}: atom_id must be a non-empty string")
    field_id = item.get("field_id")
    if not isinstance(field_id, str) or not field_id:
        raise ContractError(f"atom {atom_id!r}: field_id must be a non-empty string")
    if field_id not in reader_fields:
        raise ContractError(
            f"atom {atom_id!r}: field_id {field_id!r} is not declared by the "
            f"reader contract"
        )
    op = item.get("op")
    if op not in _OPS:
        raise ContractError(
            f"atom {atom_id!r}: op must be one of {list(_OPS)}, got {op!r}"
        )
    reader_type: Type = reader_fields[field_id].type

    if op == "is_null":
        extra = set(item) - {"atom_id", "field_id", "op"}
        if extra:
            raise ContractError(f"atom {atom_id!r}: unexpected keys {sorted(extra)}")
        return Atom(atom_id=atom_id, field_id=field_id, op="is_null")

    if op == "eq":
        extra = set(item) - {"atom_id", "field_id", "op", "constant"}
        if extra:
            raise ContractError(f"atom {atom_id!r}: unexpected keys {sorted(extra)}")
        if "constant" not in item:
            raise ContractError(f"atom {atom_id!r}: eq requires a constant")
        constant = _decode_constant(reader_type, item["constant"], where=f"atom {atom_id!r}")
        return Atom(atom_id=atom_id, field_id=field_id, op="eq", constant=constant)

    # lt: numeric only, exact threshold
    extra = set(item) - {"atom_id", "field_id", "op", "threshold"}
    if extra:
        raise ContractError(f"atom {atom_id!r}: unexpected keys {sorted(extra)}")
    if isinstance(reader_type, TextType):
        raise ContractError(
            f"atom {atom_id!r}: lt requires a numeric field; {field_id!r} is TEXT"
        )
    if "threshold" not in item:
        raise ContractError(f"atom {atom_id!r}: lt requires a threshold")
    threshold = _decode_constant(
        reader_type, item["threshold"], where=f"atom {atom_id!r}"
    )
    return Atom(atom_id=atom_id, field_id=field_id, op="lt", constant=threshold)


def _decode_constant(t: Type, raw: Any, *, where: str) -> Any:
    """Validate a predicate constant against the reader field type.

    Returns the abstract domain value (int / Fraction / str).  ``null`` is
    never a legal constant: equality with null is the unknown case, expressed
    by the data rather than by a null literal.
    """
    if raw is None:
        raise ContractError(f"{where}: constant may not be null")
    if isinstance(t, TextType):
        if type(raw) is not str:
            raise ContractError(f"{where}: TEXT constant must be a JSON string")
        return raw
    if isinstance(t, IntType):
        if type(raw) is not int:
            raise ContractError(
                f"{where}: INT constant must be a JSON integer"
            )
        if not is_legal_value(t, raw):
            raise ContractError(
                f"{where}: constant {raw} outside {t.describe()}"
            )
        return raw
    # DECIMAL: exact fraction string only - no float inference
    if not isinstance(raw, str):
        raise ContractError(
            f"{where}: DECIMAL constant must be an exact fraction string"
        )
    try:
        value = Fraction(raw)
    except (ValueError, ZeroDivisionError):
        raise ContractError(f"{where}: invalid fraction constant {raw!r}") from None
    if not is_legal_value(t, value):
        raise ContractError(
            f"{where}: constant {rational_string(value)} outside {t.describe()}"
        )
    return value


def _encode_constant(value: Any) -> Any:
    if isinstance(value, Fraction):
        return rational_string(value)
    return value


def _validate_expression(node: Any, atom_ids: set[str]) -> dict:
    # A bare atom id ("A") is accepted and normalised to the explicit
    # {"atom": "A"} node form.
    if isinstance(node, str):
        if node not in atom_ids:
            raise ContractError(f"predicate expression: unknown atom reference {node!r}")
        return {"atom": node}
    if not isinstance(node, dict):
        raise ContractError("predicate expression: every node must be an object")
    keys = [k for k in node if k in _EXPR_KEYS]
    if len(keys) != 1 or len(node) != 1:
        raise ContractError(
            "predicate expression: each node needs exactly one of "
            f"{list(_EXPR_KEYS)}"
        )
    key = keys[0]
    if key == "atom":
        ref = node[key]
        if not isinstance(ref, str) or ref not in atom_ids:
            raise ContractError(
                f"predicate expression: unknown atom reference {ref!r}"
            )
        return {"atom": ref}
    if key == "not":
        operand = node[key]
        # Accept both {"not": <node>} and the uniform {"not": [<node>]} form.
        if isinstance(operand, list):
            if len(operand) != 1:
                raise ContractError("predicate expression: not takes one operand")
            operand = operand[0]
        return {"not": _validate_expression(operand, atom_ids)}
    children = node[key]
    if not isinstance(children, list) or len(children) < 2:
        raise ContractError(
            f"predicate expression: {key!r} needs at least two operands"
        )
    return {key: [_validate_expression(child, atom_ids) for child in children]}


def _copy_expression(node: dict) -> dict:
    return json.loads(json.dumps(node, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Three-valued logic
# ---------------------------------------------------------------------------

def tribool_not(a: str) -> str:
    return {T: F, F: T, U: U}[a]


def tribool_and(values: list[str]) -> str:
    # Kleene AND: false dominates, then unknown, true only if all true.
    if any(v == F for v in values):
        return F
    if any(v == U for v in values):
        return U
    return T


def tribool_or(values: list[str]) -> str:
    # Kleene OR: true dominates, then unknown, false only if all false.
    if any(v == T for v in values):
        return T
    if any(v == U for v in values):
        return U
    return F


def evaluate_atom(atom: Atom, value: Any) -> str:
    """Evaluate one atom on an abstract reader value (None means SQL null)."""
    if atom.op == "is_null":
        return T if value is None else F
    if value is None:
        # eq/lt on null is unknown, never true or false.
        return U
    if atom.op == "eq":
        if isinstance(value, Fraction) or isinstance(atom.constant, Fraction):
            equal = Fraction(value) == Fraction(atom.constant)
        else:
            equal = value == atom.constant
        return T if equal else F
    # lt on two exact numerics
    return T if Fraction(value) < Fraction(atom.constant) else F


def evaluate_expression(node: dict, outcomes: dict[str, str]) -> str:
    key = next(iter(node))
    if key == "atom":
        return outcomes[node[key]]
    if key == "not":
        return tribool_not(evaluate_expression(node[key], outcomes))
    if key == "and":
        return tribool_and([evaluate_expression(c, outcomes) for c in node[key]])
    return tribool_or([evaluate_expression(c, outcomes) for c in node[key]])


def evaluate_predicate(predicate: Predicate, canonical_values: dict[str, Any]) -> tuple[dict[str, str], str]:
    """Evaluate every atom then the expression on one adapted reader row.

    Returns ``(atom_outcomes, expression_value)`` where both use ``T/F/U``.
    """
    outcomes: dict[str, str] = {}
    for atom in predicate.atoms:
        outcomes[atom.atom_id] = evaluate_atom(
            atom, canonical_values.get(atom.field_id)
        )
    return outcomes, evaluate_expression(predicate.expression, outcomes)


def vector_of(outcomes: dict[str, str], atom_ids: list[str]) -> str:
    """Pack atom outcomes into a fixed-order ``T/F/U`` vector string."""
    return "".join(outcomes[a] for a in atom_ids)


def is_delivered(expression_value: str) -> bool:
    """Only a definite T is delivered; F and U stay distinct but both stop."""
    return expression_value == T

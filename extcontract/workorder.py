"""Joint atom statistics and minimum-cost branch work orders.

Two artifacts live here, both bound to a sealed input file (via its canonical
fingerprint), a fixed reader version and a parsed predicate:

1. **Joint statistics** (``filter-statistics/v1``) - the *joint* distribution
   of atom truth vectors actually present in the batch: each observed
   ``T/F/U`` vector, its row count and the traceable input ``row_id`` list.
   Marginal per-column frequencies are deliberately not produced and never
   multiplied: ``U`` (unknown) and ``F`` (false) are kept distinct and their
   co-occurrence is the whole point of the joint table.

2. **Branch work order** (``filter-work-order/v1``) - a decision tree that
   requests one atom result at a time, per row, under independent positive
   per-atom retrieval fees.  A node stops as soon as every vector still
   consistent with the observed answers gives the same delivery decision
   (all ``T`` -> deliver; all non-``T`` -> reject, mixing ``F`` and ``U`` is
   fine).  Costs are weighted by the real per-vector row counts, so the
   expected total over *this* batch is minimised exactly by dynamic
   programming.  Equivalent branches share one subtree through DP memoisation.

The optimiser never sees ``row_id`` values or row order - only vectors and
frequencies.  Row identities are attached to the statistics purely for the
post-execution cross-check in :mod:`extcontract.replay`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

from .errors import ContractError, LimitError
from .predicate import (
    F,
    T,
    U,
    Predicate,
    evaluate_predicate,
    is_delivered,
    vector_of,
)
from .types import rational_string

STATISTICS_FORMAT = "filter-statistics/v1"
WORK_ORDER_FORMAT = "filter-work-order/v1"
MAX_ROWS = 5000

# Vector character -> DP digit.  Three digits suffice (T/F/U); the slot
# algebra keeps one power-of-two bit per possible outcome at each atom.
_DIGIT = {T: 0, F: 1, U: 2}
_VALUE = ("T", "F", "U")


# ---------------------------------------------------------------------------
# Sealed-file fingerprint and joint statistics
# ---------------------------------------------------------------------------

def sealed_file_fingerprint(
    rows: list[dict], *, writer_version: str, reader_version: str
) -> str:
    """Canonical, representation-independent hash of a sealed row batch.

    Rows are the adapted reader rows (``row_id`` + canonical values), in the
    fixed batch order.  Numeric values are tagged as exact rationals so
    ``1`` and ``"1"`` cannot collide with TEXT, and null is its own tag.
    """
    entries = []
    for row in rows:
        entries.append([row["row_id"], _tagged(row["values"])])
    payload = json.dumps(
        {
            "writer_version": writer_version,
            "reader_version": reader_version,
            "rows": entries,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tagged(values: dict[str, Any]) -> list:
    out = []
    for field_id in sorted(values):
        value = values[field_id]
        if value is None:
            tag = ["NULL"]
        elif isinstance(value, bool):  # pragma: no cover - never stored
            tag = ["BOOL", value]
        elif isinstance(value, Fraction):
            tag = ["NUM", rational_string(value)]
        elif isinstance(value, int):
            tag = ["NUM", str(value)]
        elif isinstance(value, str) and _looks_numeric(value):
            # Minimal rational strings on already-encoded adapted rows.
            tag = ["NUM", value]
        else:
            tag = ["TEXT", value]
        out.append([field_id, tag])
    return out


def _looks_numeric(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        from fractions import Fraction

        Fraction(value)
    except (ValueError, ZeroDivisionError):
        return False
    return True


@dataclass
class VectorStat:
    vector: str
    count: int
    row_ids: list[str]
    delivered: bool

    def to_json(self) -> dict:
        return {
            "vector": self.vector,
            "count": self.count,
            "row_ids": list(self.row_ids),
            "delivery": "DELIVER" if self.delivered else "REJECT",
        }


@dataclass
class JointStatistics:
    predicate_id: str
    predicate_fingerprint: str
    reader_version: str
    file_id: str | None
    file_fingerprint: str
    row_count: int
    atom_ids: list[str]
    vectors: list[VectorStat]

    def to_json(self) -> dict:
        return {
            "format": STATISTICS_FORMAT,
            "reader_version": self.reader_version,
            "file_id": self.file_id,
            "file_fingerprint": self.file_fingerprint,
            "predicate_id": self.predicate_id,
            "predicate_fingerprint": self.predicate_fingerprint,
            "row_count": self.row_count,
            "atom_ids": list(self.atom_ids),
            "vectors": [v.to_json() for v in self.vectors],
        }

    def frequency_table(self) -> dict[str, int]:
        """Vectors and counts only - the row-id-free view the optimiser gets."""
        return {v.vector: v.count for v in self.vectors}


def parse_statistics_document(raw: Any) -> JointStatistics:
    if not isinstance(raw, dict) or raw.get("format") != STATISTICS_FORMAT:
        raise ContractError(f"expected a {STATISTICS_FORMAT} document")
    required = {
        "reader_version",
        "file_fingerprint",
        "predicate_id",
        "predicate_fingerprint",
        "row_count",
        "atom_ids",
        "vectors",
    }
    missing = required - set(raw)
    if missing:
        raise ContractError(f"statistics document missing keys {sorted(missing)}")
    vectors = []
    total = 0
    seen: set[str] = set()
    for item in raw["vectors"]:
        vec = item["vector"]
        if vec in seen:
            raise ContractError(f"statistics: duplicate vector {vec!r}")
        seen.add(vec)
        count = item["count"]
        if type(count) is not int or count < 1:
            raise ContractError(f"statistics: vector {vec!r} has no positive count")
        row_ids = item["row_ids"]
        if not isinstance(row_ids, list) or len(row_ids) != count or any(
            not isinstance(r, str) or not r for r in row_ids
        ):
            raise ContractError(
                f"statistics: vector {vec!r} row_ids must list exactly count ids"
            )
        delivery = item.get("delivery")
        if delivery not in ("DELIVER", "REJECT"):
            raise ContractError(f"statistics: vector {vec!r} missing delivery verdict")
        vectors.append(
            VectorStat(
                vector=vec,
                count=count,
                row_ids=list(row_ids),
                delivered=delivery == "DELIVER",
            )
        )
        total += count
    if total != raw["row_count"]:
        raise ContractError("statistics: vector counts do not sum to row_count")
    atom_ids = raw["atom_ids"]
    if any(len(v.vector) != len(atom_ids) for v in vectors):
        raise ContractError("statistics: vector width does not match atom count")
    if any(ch not in _DIGIT for v in vectors for ch in v.vector):
        raise ContractError("statistics: vectors may only contain T, F, U")
    return JointStatistics(
        predicate_id=raw["predicate_id"],
        predicate_fingerprint=raw["predicate_fingerprint"],
        reader_version=raw["reader_version"],
        file_id=raw.get("file_id"),
        file_fingerprint=raw["file_fingerprint"],
        row_count=raw["row_count"],
        atom_ids=list(atom_ids),
        vectors=vectors,
    )


def build_joint_statistics(
    predicate: Predicate,
    canonical_rows: list[Any],
    *,
    reader_version: str,
    file_id: str | None,
    file_fingerprint: str,
) -> JointStatistics:
    """Compute the joint truth-vector distribution over a fixed, closed batch.

    ``canonical_rows`` are :class:`~extcontract.adapter.CanonicalRow` objects
    (anything with ``row_id`` and ``domain_values``).  Each row is evaluated
    once; rows sharing a vector share one entry and never get their column
    marginals multiplied.
    """
    if len(canonical_rows) > MAX_ROWS:
        raise LimitError(f"at most {MAX_ROWS} rows per work order, got {len(canonical_rows)}")
    atom_ids = predicate.atom_ids
    grouped: dict[str, list[str]] = {}
    for row in canonical_rows:
        outcomes, expr_value = evaluate_predicate(predicate, row.domain_values)
        vec = vector_of(outcomes, atom_ids)
        grouped.setdefault(vec, []).append(row.row_id)

    vectors: list[VectorStat] = []
    for vec in sorted(grouped):  # T < F < U lexicographic, stable output
        # Re-derive the expression value from any row of this vector; the
        # verdict is a property of the vector, not of the row identity.
        expr_value = _expression_for_vector(predicate, vec, atom_ids)
        vectors.append(
            VectorStat(
                vector=vec,
                count=len(grouped[vec]),
                row_ids=list(grouped[vec]),
                delivered=is_delivered(expr_value),
            )
        )
    return JointStatistics(
        predicate_id=predicate.predicate_id,
        predicate_fingerprint=predicate.fingerprint(),
        reader_version=reader_version,
        file_id=file_id,
        file_fingerprint=file_fingerprint,
        row_count=len(canonical_rows),
        atom_ids=atom_ids,
        vectors=vectors,
    )


def _expression_for_vector(
    predicate: Predicate, vector: str, atom_ids: list[str]
) -> str:
    outcomes = {atom_id: vector[i] for i, atom_id in enumerate(atom_ids)}
    from .predicate import evaluate_expression

    return evaluate_expression(predicate.expression, outcomes)


# ---------------------------------------------------------------------------
# Fee document
# ---------------------------------------------------------------------------

def parse_fees(raw: Any, atom_ids: list[str]) -> dict[str, int]:
    """Validate a fee document: every atom has an independent positive integer fee."""
    if isinstance(raw, dict) and "fees" in raw:
        raw = raw["fees"]
    if not isinstance(raw, dict):
        raise ContractError("fees must be an object {atom_id: positive integer}")
    if set(raw) != set(atom_ids):
        raise ContractError(
            f"fees must name exactly the predicate atoms {atom_ids}, "
            f"got {sorted(raw)}"
        )
    fees: dict[str, int] = {}
    for atom_id in atom_ids:
        fee = raw[atom_id]
        if type(fee) is not int or fee < 1:
            raise ContractError(
                f"fee for atom {atom_id!r} must be a positive integer, got {fee!r}"
            )
        fees[atom_id] = fee
    return fees


# ---------------------------------------------------------------------------
# Exact minimum-cost decision tree (dynamic program over partial assignments)
# ---------------------------------------------------------------------------

@dataclass
class WorkOrderNode:
    node_id: int
    kind: str  # "test" | "leaf"
    atom_id: str | None = None
    decision: str | None = None  # "DELIVER" | "REJECT"
    children: dict[str, "WorkOrderNode"] = field(default_factory=dict)

    def to_json(self) -> dict:
        if self.kind == "leaf":
            return {"node": self.node_id, "type": "leaf", "decision": self.decision}
        return {
            "node": self.node_id,
            "type": "test",
            "atom_id": self.atom_id,
            "branches": [
                {"outcome": outcome, "node": child.node_id}
                for outcome, child in sorted(self.children.items())
            ],
        }


@dataclass
class WorkOrder:
    predicate_id: str
    predicate_fingerprint: str
    reader_version: str
    file_id: str | None
    file_fingerprint: str
    atom_ids: list[str]
    fees: dict[str, int]
    total_cost: int
    expected_cost_per_row: float
    row_count: int
    root: WorkOrderNode
    node_count: int
    observed_vectors: list[str]

    def to_json(self) -> dict:
        nodes = []
        _flatten(self.root, nodes)
        return {
            "format": WORK_ORDER_FORMAT,
            "reader_version": self.reader_version,
            "file_id": self.file_id,
            "file_fingerprint": self.file_fingerprint,
            "predicate_id": self.predicate_id,
            "predicate_fingerprint": self.predicate_fingerprint,
            "atom_ids": list(self.atom_ids),
            "fees": {a: self.fees[a] for a in self.atom_ids},
            "row_count": self.row_count,
            "total_cost": self.total_cost,
            "expected_cost_per_row": self.expected_cost_per_row,
            "observed_vectors": list(self.observed_vectors),
            "root_node": self.root.node_id,
            "nodes": nodes,
        }


def _flatten(node: WorkOrderNode, out: list[dict]) -> None:
    out.append(node.to_json())
    for child in sorted(node.children.values(), key=lambda n: n.node_id):
        _flatten(child, out)


@dataclass
class _DP:
    atom_ids: list[str]
    fees: dict[str, int]
    weights: list[int]          # weight per observed vector
    vector_digits: list[tuple]  # digit tuple per observed vector
    deliver: list[bool]         # delivery verdict per observed vector
    memo: dict = field(default_factory=dict)

    def weight_of_slot(self, slot: int) -> int:
        """Total row weight of every observed vector still inside the slot."""
        total = 0
        for i, digits in enumerate(self.vector_digits):
            if all(slot & (1 << (j * 3 + digit)) for j, digit in enumerate(digits)):
                total += self.weights[i]
        return total


def build_work_order(
    statistics: JointStatistics,
    predicate: Predicate,
    fees: dict[str, int],
) -> WorkOrder:
    """Build the exact minimum expected-cost branch work order.

    Only the vector frequency table (never ``row_id`` or row order) enters the
    optimiser.  Ties between atoms are broken deterministically by declared
    atom order; any minimum-cost order is acceptable, this one is stable.
    """
    if predicate.fingerprint() != statistics.predicate_fingerprint:
        raise ContractError(
            "work order: statistics were built for a different predicate"
        )
    if predicate.atom_ids != statistics.atom_ids:
        raise ContractError("work order: atom order mismatch with statistics")
    fees = parse_fees(fees, statistics.atom_ids)

    atom_ids = statistics.atom_ids
    k = len(atom_ids)
    vectors = sorted(v.vector for v in statistics.vectors)
    weights = [
        next(s.count for s in statistics.vectors if s.vector == vec)
        for vec in vectors
    ]
    deliver = [
        next(s.delivered for s in statistics.vectors if s.vector == vec)
        for vec in vectors
    ]
    vector_digits = [tuple(_DIGIT[ch] for ch in vec) for vec in vectors]

    dp = _DP(
        atom_ids=atom_ids,
        fees=fees,
        weights=weights,
        vector_digits=vector_digits,
        deliver=deliver,
    )

    root_slot = _full_slot(k)
    total_cost, root_builder = _solve(root_slot, dp)

    # Materialise the shared-DAG solution into a tree with preorder node ids.
    counter = {"id": 0}
    root_node = root_builder(counter)
    node_count = counter["id"]

    row_count = statistics.row_count
    return WorkOrder(
        predicate_id=statistics.predicate_id,
        predicate_fingerprint=statistics.predicate_fingerprint,
        reader_version=statistics.reader_version,
        file_id=statistics.file_id,
        file_fingerprint=statistics.file_fingerprint,
        atom_ids=list(atom_ids),
        fees=dict(fees),
        total_cost=total_cost,
        expected_cost_per_row=total_cost / row_count,
        row_count=row_count,
        root=root_node,
        node_count=node_count,
        observed_vectors=vectors,
    )


def _full_slot(k: int) -> int:
    """Slot whose every atom position carries all three outcome digits."""
    slot = 0
    for j in range(k):
        slot |= 0b0111 << (j * 3)
    return slot


def _solve(slot: int, dp: _DP) -> tuple[int, Callable]:
    """Return (minimum total cost over the batch, tree builder) for a slot.

    The slot is the canonical partial assignment: per atom position, a bit
    set of still-possible digits (T=0, F=1, U=2).  A state is a leaf when its
    surviving vectors are uniformly delivered or uniformly rejected.
    """
    cached = dp.memo.get(slot)
    if cached is not None:
        return cached

    k = len(dp.atom_ids)
    state_weight = dp.weight_of_slot(slot)
    if state_weight == 0:
        # No observed vector can reach this slot; the parent never materialises
        # the branch, so any zero-cost subtree will do.  Memoised and shared.
        def empty_builder(counter):
            counter["id"] += 1
            return WorkOrderNode(
                node_id=counter["id"], kind="leaf", decision="REJECT"
            )

        dp.memo[slot] = (0, empty_builder)
        return dp.memo[slot]

    # A state is a leaf when every vector still consistent with the observed
    # answers gives the same delivery decision.  F and U may share a REJECT
    # leaf freely; only the T/non-T boundary matters for delivery.
    surviving = [
        i for i in range(len(dp.weights)) if _row_in_slot(i, slot, k, dp)
    ]
    verdicts = {dp.deliver[i] for i in surviving}
    if len(verdicts) == 1:
        decision = "DELIVER" if True in verdicts else "REJECT"

        def leaf_builder(counter, _decision=decision):
            counter["id"] += 1
            return WorkOrderNode(
                node_id=counter["id"], kind="leaf", decision=_decision
            )

        dp.memo[slot] = (0, leaf_builder)
        return dp.memo[slot]

    best: tuple[int, int, Callable] | None = None
    for j in range(k):
        position = slot >> (j * 3) & 0b0111
        if position in (0, 0b0001, 0b0010, 0b0100):
            continue  # absent, or already requested and fixed on this path
        fee = dp.fees[dp.atom_ids[j]]
        child_costs: list[tuple[str, int, Callable, int]] = []
        for digit, outcome in enumerate(_VALUE):
            bit = 1 << (j * 3 + digit)
            if not (slot & bit):
                continue
            child_slot = (slot & ~(0b0111 << (j * 3))) | bit
            child_cost, child_builder = _solve(child_slot, dp)
            child_weight = dp.weight_of_slot(child_slot)
            if child_weight == 0:
                continue  # outcome cannot occur on any observed vector
            child_costs.append((outcome, child_cost, child_builder, child_weight))
        # Every surviving row pays this atom's fee exactly once.
        candidate_cost = state_weight * fee + sum(
            c for _, c, _, _ in child_costs
        )
        # Deterministic tie break: declared atom order (j ascending).
        if best is None or candidate_cost < best[0]:
            atom_id = dp.atom_ids[j]

            def make_builder(
                counter,
                _atom_id=atom_id,
                _children=child_costs,
            ):
                counter["id"] += 1
                node = WorkOrderNode(
                    node_id=counter["id"], kind="test", atom_id=_atom_id
                )
                for outcome, _cost, builder, _w in _children:
                    node.children[outcome] = builder(counter)
                return node

            best = (candidate_cost, j, make_builder)

    if best is None:  # pragma: no cover - a non-uniform state always has a test
        raise ContractError("internal: no atom can split an ambiguous state")
    dp.memo[slot] = (best[0], best[2])
    return dp.memo[slot]


def _row_in_slot(row: int, slot: int, k: int, dp: _DP) -> bool:
    digits = dp.vector_digits[row]
    for j in range(k):
        if not (slot & (1 << (j * 3 + digits[j]))):
            return False
    return True


# ---------------------------------------------------------------------------
# Work-order document parsing (for replay)
# ---------------------------------------------------------------------------

def parse_work_order_document(raw: Any) -> tuple[WorkOrder, dict[int, dict]]:
    """Parse a ``filter-work-order/v1`` document back into navigable nodes."""
    if not isinstance(raw, dict) or raw.get("format") != WORK_ORDER_FORMAT:
        raise ContractError(f"expected a {WORK_ORDER_FORMAT} document")
    required = {
        "reader_version",
        "file_fingerprint",
        "predicate_id",
        "predicate_fingerprint",
        "atom_ids",
        "fees",
        "row_count",
        "total_cost",
        "root_node",
        "nodes",
    }
    missing = required - set(raw)
    if missing:
        raise ContractError(f"work order missing keys {sorted(missing)}")
    atom_ids = raw["atom_ids"]
    fees = parse_fees(raw["fees"], atom_ids)

    nodes_raw: dict[int, dict] = {}
    for item in raw["nodes"]:
        node_id = item.get("node")
        if type(node_id) is not int or node_id in nodes_raw:
            raise ContractError("work order: each node needs a unique integer id")
        nodes_raw[node_id] = item
    if raw["root_node"] not in nodes_raw:
        raise ContractError("work order: root_node not found in nodes")

    def build(node_id: int, seen: frozenset) -> WorkOrderNode:
        if node_id in seen:
            raise ContractError(f"work order: cycle at node {node_id}")
        item = nodes_raw[node_id]
        kind = item.get("type")
        if kind == "leaf":
            decision = item.get("decision")
            if decision not in ("DELIVER", "REJECT"):
                raise ContractError(f"work order node {node_id}: bad leaf decision")
            return WorkOrderNode(node_id=node_id, kind="leaf", decision=decision)
        if kind != "test":
            raise ContractError(f"work order node {node_id}: unknown type {kind!r}")
        atom_id = item.get("atom_id")
        if atom_id not in atom_ids:
            raise ContractError(f"work order node {node_id}: unknown atom {atom_id!r}")
        branches = item.get("branches")
        if not isinstance(branches, list) or not branches:
            raise ContractError(f"work order node {node_id}: needs branches")
        node = WorkOrderNode(node_id=node_id, kind="test", atom_id=atom_id)
        outcomes: set[str] = set()
        for branch in branches:
            outcome = branch.get("outcome")
            if outcome not in _VALUE or outcome in outcomes:
                raise ContractError(
                    f"work order node {node_id}: bad/duplicate branch {outcome!r}"
                )
            outcomes.add(outcome)
            node.children[outcome] = build(branch["node"], seen | {node_id})
        return node

    root = build(raw["root_node"], frozenset())
    work_order = WorkOrder(
        predicate_id=raw["predicate_id"],
        predicate_fingerprint=raw["predicate_fingerprint"],
        reader_version=raw["reader_version"],
        file_id=raw.get("file_id"),
        file_fingerprint=raw["file_fingerprint"],
        atom_ids=list(atom_ids),
        fees=fees,
        total_cost=raw["total_cost"],
        expected_cost_per_row=raw.get("expected_cost_per_row", 0.0),
        row_count=raw["row_count"],
        root=root,
        node_count=len(nodes_raw),
        observed_vectors=list(raw.get("observed_vectors", [])),
    )
    return work_order, nodes_raw

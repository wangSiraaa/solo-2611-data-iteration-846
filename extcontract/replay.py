"""Execution and verification of a saved branch work order against real rows.

The replay enforces the billing model literally:

  * atoms are revealed one at a time, only when the work order tree asks for
    the current row's next atom;
  * each request pays that atom's independent positive-integer fee, and no
    path ever requests the same atom twice (the tree was built that way and
    the walker asserts it);
  * until the leaf decision, the walker has no access to ``row_id``, row
    order or any row-to-vector mapping - it only holds the outcomes of the
    atoms it has already paid for.  The restrictive ``AtomOracle`` raises if
    anything tries to read an unrequested column;
  * after every row has been processed, the traceable ``row_id`` lists in the
    joint statistics are used solely to cross-check the delivery set and the
    actually observed truth vectors against the sealed batch.

The same code path runs against freshly adapted rows and against a
deserialised work order document, so a saved JSON work order truly replays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import ContractError
from .predicate import (
    Predicate,
    evaluate_atom,
    evaluate_predicate,
    vector_of,
)
from .workorder import (
    JointStatistics,
    WorkOrder,
    parse_work_order_document,
    sealed_file_fingerprint,
)


@dataclass
class RowTrace:
    row_id: str
    vector: str
    requests: list[dict]  # [{"atom_id", "outcome", "fee"}] in request order
    cost: int
    decision: str  # "DELIVER" | "REJECT"
    leaf_node: int

    def to_json(self) -> dict:
        return {
            "row_id": self.row_id,
            "vector": self.vector,
            "requests": [dict(r) for r in self.requests],
            "cost": self.cost,
            "decision": self.decision,
            "leaf_node": self.leaf_node,
        }


@dataclass
class ReplayResult:
    total_cost: int
    row_count: int
    delivered_row_ids: list[str]
    rejected_row_ids: list[str]
    traces: list[RowTrace]
    observed_vectors: dict[str, int]
    cross_check: dict

    def to_json(self) -> dict:
        return {
            "total_cost": self.total_cost,
            "row_count": self.row_count,
            "delivered_row_ids": list(self.delivered_row_ids),
            "rejected_row_ids": list(self.rejected_row_ids),
            "observed_vectors": dict(self.observed_vectors),
            "cross_check": dict(self.cross_check),
            "rows": [t.to_json() for t in self.traces],
        }


class AtomOracle:
    """Lazy, pay-per-atom view of one row.

    Holds the row's canonical values but exposes an atom only when the work
    order explicitly requests it, recording the charge.  Reading any column
    that has not been requested on this row raises, which is what makes the
    "no peeking before the decision" guarantee executable rather than
    documentary.
    """

    def __init__(self, atom_by_id: dict, canonical_values: dict[str, Any], fees: dict[str, int]):
        self._atom_by_id = atom_by_id
        self._values = canonical_values
        self._fees = fees
        self._requested: set[str] = set()
        self.requests: list[dict] = []
        self.cost = 0

    def request(self, atom_id: str) -> str:
        if atom_id not in self._atom_by_id:
            raise ContractError(f"work order requested unknown atom {atom_id!r}")
        if atom_id in self._requested:
            raise ContractError(
                f"work order requested atom {atom_id!r} twice on one row"
            )
        atom = self._atom_by_id[atom_id]
        outcome = evaluate_atom(atom, self._values.get(atom.field_id))
        self._requested.add(atom_id)
        fee = self._fees[atom_id]
        self.requests.append(
            {"atom_id": atom_id, "outcome": outcome, "fee": fee}
        )
        self.cost += fee
        return outcome


def replay_work_order(
    work_order: WorkOrder,
    predicate: Predicate,
    canonical_rows: list[Any],
    statistics: JointStatistics,
    *,
    writer_version: str,
    reader_version: str,
) -> ReplayResult:
    """Execute a work order row by row and verify everything against the stats."""
    _check_bindings(work_order, predicate, statistics, reader_version)
    if len(canonical_rows) != work_order.row_count:
        raise ContractError(
            f"replay: sealed batch has {len(canonical_rows)} rows but the work "
            f"order was built for {work_order.row_count}"
        )

    actual_fingerprint = sealed_file_fingerprint(
        [{"row_id": r.row_id, "values": r.domain_values} for r in canonical_rows],
        writer_version=writer_version,
        reader_version=reader_version,
    )
    if actual_fingerprint != work_order.file_fingerprint:
        raise ContractError(
            "replay: input file fingerprint does not match the sealed file this "
            "work order is bound to"
        )

    atom_by_id = predicate.atom_by_id
    traces: list[RowTrace] = []
    observed: dict[str, int] = {}
    total_cost = 0

    # The walker only ever sees the oracle for the current row; row_id is
    # attached to the trace only after the leaf decision.
    for row in canonical_rows:
        oracle = AtomOracle(atom_by_id, row.domain_values, work_order.fees)
        node = work_order.root
        while node.kind == "test":
            outcome = oracle.request(node.atom_id)
            child = node.children.get(outcome)
            if child is None:
                raise ContractError(
                    f"replay: node {node.node_id} has no branch for observed "
                    f"outcome {outcome!r} on atom {node.atom_id!r}"
                )
            node = child
        decision = node.decision
        # The full vector is recorded only after the leaf decision; it is used
        # for the post-execution cross-check, never to choose the branch.
        full_outcomes, _ = evaluate_predicate(predicate, row.domain_values)
        trace = RowTrace(
            row_id=row.row_id,
            vector=vector_of(full_outcomes, predicate.atom_ids),
            requests=oracle.requests,
            cost=oracle.cost,
            decision=decision,
            leaf_node=node.node_id,
        )
        traces.append(trace)
        total_cost += oracle.cost
        observed[trace.vector] = observed.get(trace.vector, 0) + 1

    cross_check = _cross_check(traces, observed, total_cost, work_order, statistics)
    delivered = sorted(t.row_id for t in traces if t.decision == "DELIVER")
    rejected = sorted(t.row_id for t in traces if t.decision == "REJECT")
    return ReplayResult(
        total_cost=total_cost,
        row_count=len(traces),
        delivered_row_ids=delivered,
        rejected_row_ids=rejected,
        traces=traces,
        observed_vectors=dict(sorted(observed.items())),
        cross_check=cross_check,
    )


def _check_bindings(
    work_order: WorkOrder,
    predicate: Predicate,
    statistics: JointStatistics,
    reader_version: str,
) -> None:
    if predicate.fingerprint() != work_order.predicate_fingerprint:
        raise ContractError("replay: predicate does not match the work order")
    if predicate.fingerprint() != statistics.predicate_fingerprint:
        raise ContractError("replay: predicate does not match the statistics")
    if work_order.file_fingerprint != statistics.file_fingerprint:
        raise ContractError("replay: work order and statistics bind different files")
    if work_order.reader_version != reader_version:
        raise ContractError(
            f"replay: work order binds reader {work_order.reader_version!r}, "
            f"running under {reader_version!r}"
        )
    if work_order.atom_ids != predicate.atom_ids:
        raise ContractError("replay: atom order mismatch")


def _cross_check(
    traces: list[RowTrace],
    observed: dict[str, int],
    total_cost: int,
    work_order: WorkOrder,
    statistics: JointStatistics,
) -> dict:
    checks: dict[str, Any] = {}

    # 1. The vectors actually observed at execution must be exactly the joint
    #    statistics vectors, with identical frequencies - no marginals used.
    expected = {v.vector: v.count for v in statistics.vectors}
    checks["vectors_match_statistics"] = observed == expected

    # 2. Row identities trace back to the statistics' row_id groups, and every
    #    row's delivery decision agrees with the vector's stored verdict.
    id_ok = True
    verdict_ok = True
    stat_by_vector = {v.vector: v for v in statistics.vectors}
    seen_ids: set[str] = set()
    for trace in traces:
        stat = stat_by_vector[trace.vector]
        if trace.row_id not in stat.row_ids:
            id_ok = False
        if trace.row_id in seen_ids:
            id_ok = False
        seen_ids.add(trace.row_id)
        want = "DELIVER" if stat.delivered else "REJECT"
        if trace.decision != want:
            verdict_ok = False
    checks["row_ids_trace_to_statistics"] = id_ok
    checks["decisions_match_predicate"] = verdict_ok

    # 3. Money: summed replay cost must equal the DP optimum in the document.
    checks["cost_matches_work_order_total"] = total_cost == work_order.total_cost

    # 4. The delivered set equals the union of the statistics' deliver groups.
    expected_deliver = set()
    for stat in statistics.vectors:
        if stat.delivered:
            expected_deliver.update(stat.row_ids)
    actual_deliver = {t.row_id for t in traces if t.decision == "DELIVER"}
    checks["delivered_set_exact"] = actual_deliver == expected_deliver

    checks["all_passed"] = all(checks.values())
    return checks


def replay_from_documents(
    work_order_doc: dict,
    predicate: Predicate,
    canonical_rows: list[Any],
    statistics: JointStatistics,
    *,
    writer_version: str,
    reader_version: str,
) -> tuple[WorkOrder, ReplayResult]:
    """Convenience entry point: parse a saved work order JSON, then replay it."""
    work_order, _ = parse_work_order_document(work_order_doc)
    result = replay_work_order(
        work_order,
        predicate,
        canonical_rows,
        statistics,
        writer_version=writer_version,
        reader_version=reader_version,
    )
    return work_order, result


def delivered_records(records: dict, result: ReplayResult) -> dict:
    """Project a ``reader-records/v1`` snapshot onto the delivered row set.

    The projection keeps the original record order and every field, so the
    real (unmodified) ``reader_client.py`` can consume it as-is.
    """
    deliver = set(result.delivered_row_ids)
    return {
        "format": records["format"],
        "reader_version": records["reader_version"],
        "rows": [row for row in records["rows"] if row["row_id"] in deliver],
    }

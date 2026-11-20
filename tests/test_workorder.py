"""Tests for joint statistics, minimum-cost work orders and replay.

The dynamic program is checked against an independent brute-force enumeration
of every possible decision tree on small instances, so optimality is proved
rather than asserted.  Replay tests additionally enforce the billing model:
one atom at a time, no atom twice per path, and no row identity available to
the decision walk.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from extcontract import (
    ContractError,
    LimitError,
    AtomOracle,
    build_joint_statistics,
    build_work_order,
    delivered_records,
    parse_contract,
    parse_fees,
    parse_predicate,
    parse_statistics_document,
    parse_work_order_document,
    replay_from_documents,
    replay_work_order,
    sealed_file_fingerprint,
    adapt_rows,
)
from extcontract.predicate import T, F, U

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "fixtures" / "filterwork"
CLIENT = ROOT / "materials" / "745" / "reader_client.py"


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

def _contracts(nullable: bool = False):
    writer = parse_contract(
        {
            "version": "fw-w1",
            "fields": [
                {"field_id": "x", "name": "x", "type": {"kind": "INT", "bits": 8},
                 "nullable": nullable},
                {"field_id": "y", "name": "y", "type": {"kind": "INT", "bits": 8},
                 "nullable": nullable},
            ],
        },
        role="writer",
    )
    reader = parse_contract(
        {
            "version": "fw-r1",
            "fields": [
                {"field_id": "x", "name": "x", "type": {"kind": "INT", "bits": 16},
                 "nullable": nullable},
                {"field_id": "y", "name": "y", "type": {"kind": "INT", "bits": 16},
                 "nullable": nullable},
            ],
        },
        role="reader",
    )
    return writer, reader


def _and_predicate(reader):
    return parse_predicate(
        {
            "predicate_id": "p-and",
            "atoms": [
                {"atom_id": "A", "field_id": "x", "op": "eq", "constant": 1},
                {"atom_id": "B", "field_id": "y", "op": "eq", "constant": 1},
            ],
            "expression": {"and": ["A", "B"]},
        },
        reader,
    )


def _stats(predicate, adapted, file_id="f"):
    fingerprint = sealed_file_fingerprint(
        [{"row_id": r.row_id, "values": r.domain_values} for r in adapted.canonical_rows],
        writer_version="fw-w1",
        reader_version="fw-r1",
    )
    return build_joint_statistics(
        predicate,
        adapted.canonical_rows,
        reader_version="fw-r1",
        file_id=file_id,
        file_fingerprint=fingerprint,
    )


def _adapt(writer, reader, rows):
    result = adapt_rows(writer, reader, rows)
    assert result.status == "ADAPTED", result.errors
    return result


FOUR_ROWS = [
    {"row_id": "r1", "values": {"x": 1, "y": 1}},
    {"row_id": "r2", "values": {"x": 1, "y": 2}},
    {"row_id": "r3", "values": {"x": 2, "y": 1}},
    {"row_id": "r4", "values": {"x": 2, "y": 2}},
]


# ---------------------------------------------------------------------------
# Mandated numeric scenarios
# ---------------------------------------------------------------------------

def test_mandated_four_vectors_both_fee_directions():
    writer, reader = _contracts()
    predicate = _and_predicate(reader)
    adapted = _adapt(writer, reader, FOUR_ROWS)
    stats = _stats(predicate, adapted)
    assert stats.frequency_table() == {"FF": 1, "FT": 1, "TF": 1, "TT": 1}

    cheap_a = build_work_order(stats, predicate, {"A": 1, "B": 5})
    assert cheap_a.total_cost == 14
    assert cheap_a.root.atom_id == "A"

    cheap_b = build_work_order(stats, predicate, {"A": 5, "B": 1})
    assert cheap_b.total_cost == 14
    assert cheap_b.root.atom_id == "B"

    for order in (cheap_a, cheap_b):
        replay = replay_work_order(
            order, predicate, adapted.canonical_rows, stats,
            writer_version="fw-w1", reader_version="fw-r1",
        )
        assert replay.total_cost == 14
        assert replay.delivered_row_ids == ["r1"]
        assert replay.cross_check["all_passed"]


def test_mandated_revised_two_two_single_read():
    writer, reader = _contracts()
    predicate = _and_predicate(reader)
    rows = [
        {"row_id": "s1", "values": {"x": 1, "y": 1}},
        {"row_id": "s2", "values": {"x": 1, "y": 1}},
        {"row_id": "s3", "values": {"x": 2, "y": 2}},
        {"row_id": "s4", "values": {"x": 2, "y": 2}},
    ]
    adapted = _adapt(writer, reader, rows)
    stats = _stats(predicate, adapted)
    assert stats.frequency_table() == {"FF": 2, "TT": 2}

    order = build_work_order(stats, predicate, {"A": 1, "B": 5})
    assert order.total_cost == 4
    replay = replay_work_order(
        order, predicate, adapted.canonical_rows, stats,
        writer_version="fw-w1", reader_version="fw-r1",
    )
    assert replay.total_cost == 4
    assert replay.delivered_row_ids == ["s1", "s2"]
    assert all(
        [q["atom_id"] for q in trace.requests] == ["A"] for trace in replay.traces
    )
    # B is never paid for on any path.
    assert all(
        "B" not in {q["atom_id"] for q in trace.requests} for trace in replay.traces
    )


def test_mandated_null_not_or_distinguishes_ff_from_uf():
    writer, reader = _contracts(nullable=True)
    predicate = parse_predicate(
        {
            "predicate_id": "pnull",
            "atoms": [
                {"atom_id": "A", "field_id": "x", "op": "eq", "constant": 1},
                {"atom_id": "B", "field_id": "y", "op": "eq", "constant": 1},
            ],
            "expression": {"not": {"or": ["A", "B"]}},
        },
        reader,
    )
    rows = [
        {"row_id": "ff", "values": {"x": 2, "y": 2}},
        {"row_id": "uf", "values": {"x": None, "y": 2}},
        {"row_id": "fu", "values": {"x": 2, "y": None}},
        {"row_id": "uu", "values": {"x": None, "y": None}},
        {"row_id": "tt", "values": {"x": 1, "y": 1}},
    ]
    adapted = _adapt(writer, reader, rows)
    stats = _stats(predicate, adapted)
    table = {v.vector: v.delivered for v in stats.vectors}
    assert table == {"FF": True, "FU": False, "TT": False, "UF": False, "UU": False}

    order = build_work_order(stats, predicate, {"A": 3, "B": 1})
    replay = replay_work_order(
        order, predicate, adapted.canonical_rows, stats,
        writer_version="fw-w1", reader_version="fw-r1",
    )
    assert replay.delivered_row_ids == ["ff"]
    assert set(replay.rejected_row_ids) == {"fu", "tt", "uf", "uu"}
    assert replay.cross_check["all_passed"]


# ---------------------------------------------------------------------------
# Joint statistics: joint, never marginal products
# ---------------------------------------------------------------------------

def test_statistics_are_joint_not_marginal_products():
    writer, reader = _contracts()
    predicate = _and_predicate(reader)
    # Only TF and FT occur: marginals would predict TT and FF as well.
    rows = [
        {"row_id": "a", "values": {"x": 1, "y": 2}},
        {"row_id": "b", "values": {"x": 1, "y": 2}},
        {"row_id": "c", "values": {"x": 2, "y": 1}},
    ]
    adapted = _adapt(writer, reader, rows)
    stats = _stats(predicate, adapted)
    assert stats.frequency_table() == {"TF": 2, "FT": 1}
    assert "TT" not in stats.frequency_table()
    assert "FF" not in stats.frequency_table()
    doc = stats.to_json()
    assert sum(v["count"] for v in doc["vectors"]) == stats.row_count


def test_statistics_document_round_trip():
    writer, reader = _contracts()
    predicate = _and_predicate(reader)
    adapted = _adapt(writer, reader, FOUR_ROWS)
    stats = _stats(predicate, adapted)
    reparsed = parse_statistics_document(json.loads(json.dumps(stats.to_json())))
    assert reparsed.frequency_table() == stats.frequency_table()
    assert reparsed.predicate_fingerprint == stats.predicate_fingerprint


def test_statistics_reject_count_mismatch():
    writer, reader = _contracts()
    predicate = _and_predicate(reader)
    adapted = _adapt(writer, reader, FOUR_ROWS)
    stats = _stats(predicate, adapted)
    doc = stats.to_json()
    doc["row_count"] = 99
    with pytest.raises(ContractError, match="do not sum"):
        parse_statistics_document(doc)


# ---------------------------------------------------------------------------
# Exact optimality: brute-force every decision tree on small instances
# ---------------------------------------------------------------------------

def brute_force_optimum(frequency: dict[str, int], fees: dict[str, list]) -> int:
    """Enumerate all sequential testing trees; return the minimum batch cost.

    A state is the set of still-possible vectors.  It is a leaf when the
    surviving vectors agree on delivery.  Otherwise every unused atom is a
    candidate split with three outcomes.
    """
    atom_ids = list(fees)
    deliver = {vec: vec.count("F") + vec.count("U") == 0 for vec in frequency}

    def state_weight(vectors):
        return sum(frequency[v] for v in vectors)

    def solve(vectors, used):
        verdicts = {deliver[v] for v in vectors}
        if len(verdicts) == 1:
            return 0
        best = None
        for j, atom_id in enumerate(atom_ids):
            if j in used:
                continue
            cost = state_weight(vectors) * fees[atom_id]
            for outcome in (T, F, U):
                child = tuple(v for v in vectors if v[j] == outcome)
                if not child:
                    continue
                cost += solve(child, used | {j})
            best = cost if best is None else min(best, cost)
        return best

    return solve(tuple(sorted(frequency)), frozenset())


@pytest.mark.parametrize(
    "frequency,fees",
    [
        ({"TT": 1, "TF": 1, "FT": 1, "FF": 1}, {"A": 1, "B": 5}),
        ({"TT": 1, "TF": 1, "FT": 1, "FF": 1}, {"A": 5, "B": 1}),
        ({"TT": 2, "FF": 2}, {"A": 1, "B": 5}),
        ({"TF": 3, "FT": 2}, {"A": 4, "B": 4}),
        ({"TT": 1, "FF": 1, "UF": 1, "FU": 1, "UU": 1}, {"A": 3, "B": 1}),
        ({"TT": 2, "TF": 1, "UU": 4}, {"A": 2, "B": 9}),
    ],
)
def test_dp_matches_brute_force(frequency, fees):
    writer, reader = _contracts(nullable=True)
    predicate = _and_predicate(reader)

    # Materialise one row per required vector.
    value_for = {T: 1, F: 2, U: None}
    rows = []
    seq = 0
    for vec, count in frequency.items():
        for _ in range(count):
            seq += 1
            rows.append(
                {
                    "row_id": f"r{seq}",
                    "values": {
                        "x": value_for[vec[0]],
                        "y": value_for[vec[1]],
                    },
                }
            )
    adapted = _adapt(writer, reader, rows)
    stats = _stats(predicate, adapted)
    order = build_work_order(stats, predicate, fees)
    assert order.total_cost == brute_force_optimum(frequency, fees)


def test_three_atom_dp_matches_brute_force():
    writer = parse_contract(
        {
            "version": "w",
            "fields": [
                {"field_id": f"c{i}", "name": f"c{i}",
                 "type": {"kind": "INT", "bits": 8}, "nullable": True}
                for i in range(3)
            ],
        },
        role="writer",
    )
    reader = parse_contract(
        {
            "version": "r",
            "fields": [
                {"field_id": f"c{i}", "name": f"c{i}",
                 "type": {"kind": "INT", "bits": 16}, "nullable": True}
                for i in range(3)
            ],
        },
        role="reader",
    )
    predicate = parse_predicate(
        {
            "predicate_id": "p3",
            "atoms": [
                {"atom_id": f"A{i}", "field_id": f"c{i}", "op": "eq", "constant": 1}
                for i in range(3)
            ],
            "expression": {"and": ["A0", "A1", "A2"]},
        },
        reader,
    )
    vectors = ["TTT", "TFF", "FTU", "UFT", "FFF", "TUF"]
    value_for = {T: 1, F: 2, U: None}
    rows = []
    for seq, vec in enumerate(vectors):
        rows.append(
            {
                "row_id": f"r{seq}",
                "values": {
                    f"c{i}": value_for[vec[i]] for i in range(3)
                },
            }
        )
    adapted = _adapt(writer, reader, rows)
    fingerprint = sealed_file_fingerprint(
        [{"row_id": r.row_id, "values": r.domain_values} for r in adapted.canonical_rows],
        writer_version="w",
        reader_version="r",
    )
    stats = build_joint_statistics(
        predicate, adapted.canonical_rows,
        reader_version="r", file_id="f", file_fingerprint=fingerprint,
    )
    fees = {"A0": 2, "A1": 7, "A2": 3}
    order = build_work_order(stats, predicate, fees)
    # Independent brute force over the three-atom decision tree.
    atom_ids = ["A0", "A1", "A2"]
    deliver = {v: v == "TTT" for v in vectors}

    def weight(vs):
        return sum(1 for _ in vs)

    def solve(vs, used):
        if len({deliver[v] for v in vs}) == 1:
            return 0
        best = None
        for j, atom_id in enumerate(atom_ids):
            if j in used:
                continue
            cost = weight(vs) * fees[atom_id]
            for outcome in (T, F, U):
                child = tuple(v for v in vs if v[j] == outcome)
                if child:
                    cost += solve(child, used | {j})
            best = cost if best is None else min(best, cost)
        return best

    assert order.total_cost == solve(tuple(vectors), frozenset())


# ---------------------------------------------------------------------------
# Decision-tree structure and the no-repeat / stop-early rules
# ---------------------------------------------------------------------------

def test_no_path_requests_an_atom_twice_and_leaves_are_uniform():
    writer, reader = _contracts()
    predicate = _and_predicate(reader)
    adapted = _adapt(writer, reader, FOUR_ROWS)
    stats = _stats(predicate, adapted)
    order = build_work_order(stats, predicate, {"A": 1, "B": 5})

    def walk(node, seen):
        if node.kind == "leaf":
            return
        assert node.atom_id not in seen
        for child in node.children.values():
            walk(child, seen | {node.atom_id})

    walk(order.root, frozenset())


def test_uniform_batch_builds_a_single_leaf_and_costs_zero():
    writer, reader = _contracts()
    predicate = parse_predicate(
        {
            "predicate_id": "p",
            "atoms": [
                {"atom_id": "L", "field_id": "x", "op": "lt", "threshold": 10},
                {"atom_id": "N", "field_id": "y", "op": "is_null"},
            ],
            "expression": {"or": ["L", {"not": "N"}]},
        },
        reader,
    )
    adapted = _adapt(writer, reader, FOUR_ROWS)
    stats = _stats(predicate, adapted)
    assert stats.frequency_table() == {"TF": 4}
    order = build_work_order(stats, predicate, {"L": 2, "N": 7})
    assert order.root.kind == "leaf"
    assert order.root.decision == "DELIVER"
    assert order.total_cost == 0
    replay = replay_work_order(
        order, predicate, adapted.canonical_rows, stats,
        writer_version="fw-w1", reader_version="fw-r1",
    )
    assert all(trace.requests == [] for trace in replay.traces)
    assert replay.total_cost == 0


def test_leaf_may_mix_false_and_unknown_when_both_reject():
    writer, reader = _contracts(nullable=True)
    predicate = _and_predicate(reader)
    rows = [
        {"row_id": "a", "values": {"x": 2, "y": 2}},      # FF
        {"row_id": "b", "values": {"x": None, "y": 2}},   # UF
    ]
    adapted = _adapt(writer, reader, rows)
    stats = _stats(predicate, adapted)
    order = build_work_order(stats, predicate, {"A": 1, "B": 1})
    # Both vectors reject for AND; no reads are needed at all.
    assert order.root.kind == "leaf"
    assert order.root.decision == "REJECT"


# ---------------------------------------------------------------------------
# Replay mechanics: fees, no peeking, cross-checks, round trip
# ---------------------------------------------------------------------------

def test_replay_charge_matches_independent_fees():
    writer, reader = _contracts()
    predicate = _and_predicate(reader)
    adapted = _adapt(writer, reader, FOUR_ROWS)
    stats = _stats(predicate, adapted)
    order = build_work_order(stats, predicate, {"A": 3, "B": 4})
    replay = replay_work_order(
        order, predicate, adapted.canonical_rows, stats,
        writer_version="fw-w1", reader_version="fw-r1",
    )
    for trace in replay.traces:
        assert trace.cost == sum(q["fee"] for q in trace.requests)
    assert replay.total_cost == sum(t.cost for t in replay.traces)


def test_oracle_refuses_unrequested_and_repeated_reads():
    writer, reader = _contracts(nullable=True)
    predicate = _and_predicate(reader)
    adapted = _adapt(writer, reader, FOUR_ROWS)
    oracle = AtomOracle(
        predicate.atom_by_id, adapted.canonical_rows[0].domain_values,
        {"A": 1, "B": 5},
    )
    assert oracle.request("A") == T
    with pytest.raises(ContractError, match="twice on one row"):
        oracle.request("A")
    with pytest.raises(ContractError, match="unknown atom"):
        oracle.request("Z")


def test_full_document_round_trip_reproduces_every_trace():
    writer, reader = _contracts(nullable=True)
    predicate = _and_predicate(reader)
    rows = FOUR_ROWS + [
        {"row_id": "n1", "values": {"x": None, "y": 1}},
        {"row_id": "n2", "values": {"x": 1, "y": None}},
    ]
    adapted = _adapt(writer, reader, rows)
    stats = _stats(predicate, adapted)
    order = build_work_order(stats, predicate, {"A": 2, "B": 6})
    doc = json.loads(json.dumps(order.to_json()))
    _, replay = replay_from_documents(
        doc, predicate, adapted.canonical_rows, stats,
        writer_version="fw-w1", reader_version="fw-r1",
    )
    assert replay.cross_check["all_passed"]
    # Every observed vector in the replay is one of the statistics vectors.
    assert set(replay.observed_vectors) == set(stats.frequency_table())


def test_replay_refuses_foreign_file_fingerprint():
    writer, reader = _contracts()
    predicate = _and_predicate(reader)
    adapted = _adapt(writer, reader, FOUR_ROWS)
    stats = _stats(predicate, adapted)
    order = build_work_order(stats, predicate, {"A": 1, "B": 5})

    other = _adapt(
        writer,
        reader,
        [
            {"row_id": "z1", "values": {"x": 1, "y": 1}},
            {"row_id": "z2", "values": {"x": 2, "y": 2}},
        ],
    )
    other_stats = _stats(predicate, other)
    # Reusing a work order with another batch's statistics is a binding error.
    with pytest.raises(ContractError, match="bind different files"):
        replay_work_order(
            order, predicate, other.canonical_rows, other_stats,
            writer_version="fw-w1", reader_version="fw-r1",
        )

    # Same statistics document but a tampered row batch trips the live
    # fingerprint recomputation inside replay.
    tampered = _adapt(writer, reader, [dict(r, values=dict(r["values"], x=9))
                                       for r in FOUR_ROWS])
    with pytest.raises(ContractError, match="fingerprint does not match"):
        replay_work_order(
            order, predicate, tampered.canonical_rows, stats,
            writer_version="fw-w1", reader_version="fw-r1",
        )


def test_replay_refuses_predicate_mismatch():
    writer, reader = _contracts()
    predicate = _and_predicate(reader)
    adapted = _adapt(writer, reader, FOUR_ROWS)
    stats = _stats(predicate, adapted)
    order = build_work_order(stats, predicate, {"A": 1, "B": 5})

    changed = parse_predicate(
        {
            "predicate_id": "p-and",
            "atoms": [
                {"atom_id": "A", "field_id": "x", "op": "eq", "constant": 2},
                {"atom_id": "B", "field_id": "y", "op": "eq", "constant": 1},
            ],
            "expression": {"and": ["A", "B"]},
        },
        reader,
    )
    with pytest.raises(ContractError, match="predicate does not match"):
        replay_work_order(
            order, changed, adapted.canonical_rows, stats,
            writer_version="fw-w1", reader_version="fw-r1",
        )


def test_fee_validation_requires_positive_integer_for_every_atom():
    writer, reader = _contracts()
    predicate = _and_predicate(reader)
    adapted = _adapt(writer, reader, FOUR_ROWS)
    stats = _stats(predicate, adapted)
    for bad in ({"A": 0, "B": 5}, {"A": -1, "B": 5}, {"A": 1, "B": 2.5}, {"A": 1}):
        with pytest.raises(ContractError):
            build_work_order(stats, predicate, bad)
    with pytest.raises(ContractError, match="exactly the predicate atoms"):
        parse_fees({"A": 1, "B": 2, "C": 3}, ["A", "B"])


def test_row_limit_enforced():
    writer, reader = _contracts()
    predicate = _and_predicate(reader)
    rows = [
        {"row_id": f"r{i}", "values": {"x": 1, "y": 1}} for i in range(5001)
    ]
    adapted = _adapt(writer, reader, rows)
    fingerprint = sealed_file_fingerprint(
        [{"row_id": r.row_id, "values": r.domain_values} for r in adapted.canonical_rows],
        writer_version="fw-w1",
        reader_version="fw-r1",
    )
    with pytest.raises(LimitError, match="at most 5000 rows"):
        build_joint_statistics(
            predicate, adapted.canonical_rows,
            reader_version="fw-r1", file_id="big", file_fingerprint=fingerprint,
        )


# ---------------------------------------------------------------------------
# Delivered projection is consumable by the real reader_client
# ---------------------------------------------------------------------------

def test_delivered_records_consumed_by_real_client(tmp_path):
    writer, reader = _contracts()
    predicate = _and_predicate(reader)
    adapted = _adapt(writer, reader, FOUR_ROWS)
    stats = _stats(predicate, adapted)
    order = build_work_order(stats, predicate, {"A": 1, "B": 5})
    replay = replay_work_order(
        order, predicate, adapted.canonical_rows, stats,
        writer_version="fw-w1", reader_version="fw-r1",
    )
    delivered = delivered_records(adapted.records, replay)
    out = tmp_path / "delivered.json"
    out.write_text(json.dumps(delivered, ensure_ascii=False, indent=2))
    contract = tmp_path / "reader.json"
    contract.write_text(
        json.dumps(reader.canonical_json(), ensure_ascii=False), encoding="utf-8"
    )
    proc = subprocess.run(
        [sys.executable, str(CLIENT), str(contract), str(out)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {
        "numeric_totals": {"x": "1", "y": "1"},
        "row_count": 1,
    }


def test_fixture_documents_load_and_solve():
    reader = parse_contract(
        json.loads((FW / "reader-contract.json").read_text()), role="reader"
    )
    writer = parse_contract(
        json.loads((FW / "writer-contract.json").read_text()), role="writer"
    )
    predicate = parse_predicate(
        json.loads((FW / "predicate-and-eq.json").read_text()), reader
    )
    batch = json.loads((FW / "batch-four-vectors.json").read_text())
    adapted = _adapt(writer, reader, batch["rows"])
    fingerprint = sealed_file_fingerprint(
        [{"row_id": r.row_id, "values": r.domain_values} for r in adapted.canonical_rows],
        writer_version=writer.version,
        reader_version=reader.version,
    )
    stats = build_joint_statistics(
        predicate, adapted.canonical_rows,
        reader_version=reader.version, file_id=batch["file_id"],
        file_fingerprint=fingerprint,
    )
    fees = parse_fees(
        json.loads((FW / "fees-a1-b5.json").read_text()), stats.atom_ids
    )
    order = build_work_order(stats, predicate, fees)
    assert order.total_cost == 14
    assert order.root.atom_id == "A"


# ---------------------------------------------------------------------------
# Saved work order document validation
# ---------------------------------------------------------------------------

def test_work_order_document_rejects_cycle_and_bad_branch():
    writer, reader = _contracts()
    predicate = _and_predicate(reader)
    adapted = _adapt(writer, reader, FOUR_ROWS)
    stats = _stats(predicate, adapted)
    order = build_work_order(stats, predicate, {"A": 1, "B": 5})
    doc = order.to_json()

    # Point a branch back at its ancestor -> cycle.
    root_id = doc["root_node"]
    for node in doc["nodes"]:
        if node["type"] == "test":
            node["branches"][0]["node"] = root_id
    with pytest.raises(ContractError, match="cycle"):
        parse_work_order_document(doc)


def test_work_order_document_rejects_unknown_outcome():
    writer, reader = _contracts()
    predicate = _and_predicate(reader)
    adapted = _adapt(writer, reader, FOUR_ROWS)
    stats = _stats(predicate, adapted)
    doc = build_work_order(stats, predicate, {"A": 1, "B": 5}).to_json()
    for node in doc["nodes"]:
        if node["type"] == "test":
            node["branches"][0]["outcome"] = "X"
    with pytest.raises(ContractError, match="bad/duplicate branch"):
        parse_work_order_document(doc)


def test_work_order_document_rejects_wrong_format():
    with pytest.raises(ContractError, match="filter-work-order/v1"):
        parse_work_order_document({"format": "something-else"})


def test_statistics_document_rejects_wrong_format():
    with pytest.raises(ContractError, match="filter-statistics/v1"):
        parse_statistics_document({"format": "nope"})

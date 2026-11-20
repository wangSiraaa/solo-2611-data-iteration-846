"""Scale envelope for filter work orders: 8 atoms, 5000 rows.

The DP state space is bounded by the partial assignments reachable from the
observed vectors (at most ``4**k`` slots), never by row count; per-row replay
and statistics stay linear.  These assertions mirror the documented envelope
"at most eight atoms, five thousand rows per contract".
"""

from __future__ import annotations

import time

from extcontract import (
    adapt_rows,
    build_joint_statistics,
    build_work_order,
    parse_contract,
    parse_predicate,
    replay_work_order,
    sealed_file_fingerprint,
)

ATOMS = 8
ROWS = 5000


def _build():
    writer = parse_contract(
        {
            "version": "w",
            "fields": [
                {
                    "field_id": f"c{i}",
                    "name": f"c{i}",
                    "type": {"kind": "INT", "bits": 16},
                    "nullable": True,
                }
                for i in range(ATOMS)
            ],
        },
        role="writer",
    )
    reader = parse_contract(
        {
            "version": "r",
            "fields": [
                {
                    "field_id": f"c{i}",
                    "name": f"c{i}",
                    "type": {"kind": "DECIMAL", "precision": 6, "scale": 0},
                    "nullable": True,
                }
                for i in range(ATOMS)
            ],
        },
        role="reader",
    )
    predicate = parse_predicate(
        {
            "predicate_id": "p8",
            "atoms": [
                {
                    "atom_id": f"A{i}",
                    "field_id": f"c{i}",
                    "op": "eq",
                    "constant": "1",
                }
                for i in range(ATOMS)
            ],
            "expression": {"and": [f"A{i}" for i in range(ATOMS)]},
        },
        reader,
    )
    return writer, reader, predicate


def _rows():
    # Spread rows across many distinct joint T/F/U patterns so the 8-atom DP
    # has to split real ambiguous states rather than stop at a root leaf.
    # Every 256th row is all-true, guaranteeing a non-empty DELIVER class for
    # the AND expression.
    rows = []
    for i in range(ROWS):
        values = {}
        for j in range(ATOMS):
            if i % 256 == 0:
                values[f"c{j}"] = 1
                continue
            pick = (i * 7 + j * 13) % 5
            if pick == 4:
                values[f"c{j}"] = None             # -> U on the eq atom
            elif pick in (0, 1):
                values[f"c{j}"] = 1                # -> T
            else:
                values[f"c{j}"] = 2 + (pick % 2)   # -> F
        rows.append({"row_id": f"r{i}", "revision": 1, "values": values})
    return rows


def test_full_envelope_8_atoms_5000_rows():
    writer, reader, predicate = _build()
    adapted = adapt_rows(writer, reader, _rows())
    assert adapted.status == "ADAPTED", adapted.errors
    assert len(adapted.canonical_rows) == ROWS

    start = time.perf_counter()
    fingerprint = sealed_file_fingerprint(
        [
            {"row_id": r.row_id, "values": r.domain_values}
            for r in adapted.canonical_rows
        ],
        writer_version="w",
        reader_version="r",
    )
    stats = build_joint_statistics(
        predicate,
        adapted.canonical_rows,
        reader_version="r",
        file_id="big",
        file_fingerprint=fingerprint,
    )
    stats_ms = (time.perf_counter() - start) * 1000
    assert stats.row_count == ROWS
    # Only 3**8 = 6561 joint vectors are even possible; far fewer are observed.
    assert len(stats.vectors) <= 3**ATOMS

    fees = {f"A{i}": (i + 1) for i in range(ATOMS)}
    start = time.perf_counter()
    order = build_work_order(stats, predicate, fees)
    dp_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    replay = replay_work_order(
        order,
        predicate,
        adapted.canonical_rows,
        stats,
        writer_version="w",
        reader_version="r",
    )
    replay_ms = (time.perf_counter() - start) * 1000

    assert replay.cross_check["all_passed"]
    assert replay.total_cost == order.total_cost
    assert replay.total_cost > 0  # real splits were needed
    # AND over 8 atoms: a row delivers only when every atom is true.
    assert all(trace.vector == "T" * ATOMS for trace in replay.traces
               if trace.decision == "DELIVER")
    assert any(trace.decision == "DELIVER" for trace in replay.traces)
    # No path ever repeats an atom.
    for trace in replay.traces:
        requested = [q["atom_id"] for q in trace.requests]
        assert len(requested) == len(set(requested))
        assert len(requested) <= ATOMS

    print(
        f"vectors={len(stats.vectors)} stats={stats_ms:.1f}ms "
        f"dp={dp_ms:.1f}ms replay={replay_ms:.1f}ms "
        f"total_cost={order.total_cost} nodes={order.node_count}"
    )
    # Generous ceilings: the DP is slot-count bound, replay is linear.
    assert dp_ms < 10000
    assert replay_ms < 10000

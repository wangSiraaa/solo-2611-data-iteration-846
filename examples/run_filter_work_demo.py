"""End-to-end demonstration of per-column billed filter work orders.

Run with:

    PYTHONPATH=. python3.11 examples/run_filter_work_demo.py

The archive charges a cold-column read fee per atom retrieval.  This demo
exercises the full flow on real files under ``fixtures/filterwork``:

  1. joint three-valued statistics for A AND B over TT/TF/FT/FF;
  2. fees A=1,B=5 -> optimum 14, A first; fees reversed -> still 14, B first;
  3. revised source data TT/TT/FF/FF -> optimum 4, a single A read suffices;
  4. nullable NOT(A OR B): FF delivers but UF stays unknown and is rejected;
  5. every saved work order is replayed from its JSON document, per-row
     requests/costs/delivery are reproduced, and the delivered reader rows
     are re-checked with the real (unmodified) materials/745/reader_client.py;
  6. the original adapt / reader-records/v1 entry point is unchanged and the
     same client still reads the whole sealed batch.

All work orders bind exactly one (reader version, sealed file, predicate)
triple via content fingerprints; reusing one against another file raises.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extcontract import (  # noqa: E402
    build_joint_statistics,
    build_work_order,
    delivered_records,
    load_contract,
    parse_predicate,
    parse_work_order_document,
    replay_work_order,
    sealed_file_fingerprint,
    adapt_file_document,
    write_records,
)

FW = ROOT / "fixtures" / "filterwork"
CLIENT = ROOT / "materials" / "745" / "reader_client.py"
OUT = ROOT / "examples" / "out" / "filterwork"


def banner(n: str, title: str) -> None:
    print(f"\n=== {n}. {title} ===")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def client(contract: Path, records: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(CLIENT), str(contract), str(records)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"reader_client failed: {proc.stderr}")
    return json.loads(proc.stdout)


def adapt_and_stats(predicate_doc: dict, batch_doc: dict):
    writer = load_contract(FW / "writer-contract.json", role="writer")
    reader = load_contract(FW / "reader-contract.json", role="reader")
    predicate = parse_predicate(predicate_doc, reader)
    result = adapt_file_document(writer, reader, batch_doc)
    assert result.status == "ADAPTED", result.errors
    fingerprint = sealed_file_fingerprint(
        [
            {"row_id": row.row_id, "values": row.domain_values}
            for row in result.canonical_rows
        ],
        writer_version=writer.version,
        reader_version=reader.version,
    )
    statistics = build_joint_statistics(
        predicate,
        result.canonical_rows,
        reader_version=reader.version,
        file_id=batch_doc["file_id"],
        file_fingerprint=fingerprint,
    )
    return writer, reader, predicate, result, statistics


def show_replay(replay) -> None:
    for trace in replay.traces:
        requests = ", ".join(
            f"{q['atom_id']}={q['outcome']}({q['fee']})" for q in trace.requests
        ) or "(no reads)"
        print(
            f"  {trace.row_id:>3} {trace.vector} cost={trace.cost:>2} "
            f"{trace.decision:<7} [{requests}]"
        )
    print(
        f"  total={replay.total_cost} delivered={replay.delivered_row_ids} "
        f"cross_check={replay.cross_check['all_passed']}"
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reader_contract_path = FW / "reader-contract.json"

    and_pred = load(FW / "predicate-and-eq.json")
    not_or_pred = load(FW / "predicate-not-or.json")
    four = load(FW / "batch-four-vectors.json")
    two_two = load(FW / "batch-two-two.json")
    null_batch = load(FW / "batch-null.json")

    # 1. Joint statistics over TT/TF/FT/FF ----------------------------------
    banner("1", "joint statistics for A AND B over TT, TF, FT, FF")
    writer, reader, predicate, adapted, stats = adapt_and_stats(and_pred, four)
    assert stats.frequency_table() == {"FF": 1, "FT": 1, "TF": 1, "TT": 1}
    for vec in stats.to_json()["vectors"]:
        print(f"  {vec['vector']} count={vec['count']} rows={vec['row_ids']} "
              f"-> {vec['delivery']}")
    write_records(stats.to_json(), OUT / "stats-four.json")

    # 2. Both fee directions give 14; first atom follows the cheap column ---
    banner("2", "A=1,B=5 and A=5,B=1: both total 14, first atom flips")
    for label, fees, expected_root in (
        ("A=1,B=5", {"A": 1, "B": 5}, "A"),
        ("A=5,B=1", {"A": 5, "B": 1}, "B"),
    ):
        work_order = build_work_order(stats, predicate, fees)
        assert work_order.total_cost == 14
        assert work_order.root.atom_id == expected_root
        print(f"fees {label}: total={work_order.total_cost} first={work_order.root.atom_id}")
        # Save and replay from the persisted document.
        doc = work_order.to_json()
        (OUT / f"workorder-{label.replace('=', '').replace(',', '-')}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        reparsed, _ = parse_work_order_document(json.loads(json.dumps(doc)))
        replay = replay_work_order(
            reparsed,
            predicate,
            adapted.canonical_rows,
            stats,
            writer_version=writer.version,
            reader_version=reader.version,
        )
        assert replay.total_cost == 14
        assert replay.delivered_row_ids == ["r1"]
        show_replay(replay)

        # The delivered subset is itself a valid reader-records/v1 file.
        delivered_path = OUT / f"delivered-{label.replace('=', '').replace(',', '-')}.json"
        write_records(delivered_records(adapted.records, replay), delivered_path)
        client_payload = client(reader_contract_path, delivered_path)
        print(f"  reader_client on delivered subset: {client_payload}")
        assert client_payload == {"numeric_totals": {"x": "1", "y": "1"}, "row_count": 1}

    # 3. Revised source data: TT,TT,FF,FF -> single A read, total 4 ---------
    banner("3", "revised data TT,TT,FF,FF: one A read per row, total 4")
    w2, r2, p2, adapted2, stats2 = adapt_and_stats(and_pred, two_two)
    assert stats2.frequency_table() == {"FF": 2, "TT": 2}
    work_order2 = build_work_order(stats2, p2, {"A": 1, "B": 5})
    assert work_order2.total_cost == 4
    replay2 = replay_work_order(
        work_order2,
        p2,
        adapted2.canonical_rows,
        stats2,
        writer_version=w2.version,
        reader_version=r2.version,
    )
    assert replay2.total_cost == 4
    assert all(len(t.requests) == 1 and t.requests[0]["atom_id"] == "A"
               for t in replay2.traces)
    show_replay(replay2)
    print("  B is never billed: every row is decided from A alone.")

    # 4. Nullable NOT(A OR B): FF is true, UF stays unknown -----------------
    banner("4", "NOT(A OR B) with nulls: FF delivers, UF/FU/UU unknown")
    w3, r3, p3, adapted3, stats3 = adapt_and_stats(not_or_pred, null_batch)
    table = {v.vector: v.delivered for v in stats3.vectors}
    assert table == {
        "FF": True, "FU": False, "TT": False, "UF": False, "UU": False
    }
    for vec in stats3.to_json()["vectors"]:
        print(f"  {vec['vector']} count={vec['count']} rows={vec['row_ids']} "
              f"-> {vec['delivery']}")
    work_order3 = build_work_order(stats3, p3, {"A": 3, "B": 1})
    replay3 = replay_work_order(
        work_order3,
        p3,
        adapted3.canonical_rows,
        stats3,
        writer_version=w3.version,
        reader_version=r3.version,
    )
    show_replay(replay3)
    assert replay3.delivered_row_ids == ["ff"]
    # A naive 2-valued engine treating null as false would wrongly deliver the
    # unknown rows; the 3-valued cross-check keeps them out.
    assert set(replay3.rejected_row_ids) == {"fu", "tt", "uf", "uu"}
    delivered3 = OUT / "delivered-null.json"
    write_records(delivered_records(adapted3.records, replay3), delivered3)
    null_client = client(reader_contract_path, delivered3)
    print(f"  reader_client on null-safe delivered subset: {null_client}")
    assert null_client == {"numeric_totals": {"x": "2", "y": "2"}, "row_count": 1}

    # 5. Binding: a work order refuses any other sealed file ----------------
    banner("5", "work orders are bound to one sealed file and predicate")
    try:
        replay_work_order(
            work_order2,
            predicate,
            adapted.canonical_rows,
            stats,
            writer_version=writer.version,
            reader_version=reader.version,
        )
    except Exception as exc:  # ContractError on fingerprint mismatch
        print(f"  cross-file replay refused: {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("cross-file replay must be refused")

    # 6. Old entry point preserved: the whole batch still adapts and reads ---
    banner("6", "legacy adapt entry unchanged: full batch through reader_client")
    full_path = OUT / "full-batch-records.json"
    write_records(adapted3.records, full_path)
    full_payload = client(reader_contract_path, full_path)
    print(f"  reader_client on the complete sealed batch: {full_payload}")
    assert full_payload["row_count"] == 5

    print("\nAll filter work order demo assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

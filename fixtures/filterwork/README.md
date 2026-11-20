# filterwork — per-column billed filter fixtures

Sealed, append-free batches for the three-valued filter work order flow. The
writer and reader contracts pair two nullable INT8 columns (`x`, `y`) onto
wider nullable INT16 reader columns by `field_id`, so the existing adaptation
entry point produces the rows the filter runs on.

## Predicates

| Document | Atoms | Expression |
|----------|-------|------------|
| `predicate-and-eq.json` | `A: x == 1`, `B: y == 1` | `A AND B` |
| `predicate-not-or.json` | same | `NOT(A OR B)` (null-sensitive) |
| `predicate-lt-isnull.json` | `L: x < 10`, `N: y IS NULL` | `L OR NOT N` |

Atoms follow the contracted grammar: `eq` / `lt` yield `U` on null, `is_null`
always yields a definite `T`/`F`.

## Batches

| Document | Rows (in order) | Purpose |
|----------|-----------------|---------|
| `batch-four-vectors.json` | `TT TF FT FF` | both fee directions total 14; first atom flips |
| `batch-two-two.json` | `TT TT FF FF` | one `A` read per row, total 4 |
| `batch-null.json` | `FF UF FU UU TT` | `NOT(A OR B)` delivers only `FF`; `UF` stays unknown |

## Fees

`fees-a1-b5.json` (`A=1,B=5`) and `fees-a5-b1.json` (`A=5,B=1`); every fee is a
positive integer.

Every work order built from these binds the reader version, the sealed batch's
content fingerprint and the predicate fingerprint, so a work order cannot be
replayed against any other file. See `examples/run_filter_work_demo.py` for the
end-to-end run and `tests/test_workorder.py` for the asserted totals.

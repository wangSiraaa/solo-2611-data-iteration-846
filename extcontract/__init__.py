"""External contract toolkit: exact finite-type compatibility, adaptation
planning, file adaptation, and revision-aware atomic ingestion.

Public entry points (see README for the full walkthrough):

    load_contract / parse_contract        contract documents
    check_compatibility                   row-free whole-domain planning
    replace_field_type                    contract revisions
    adapt_rows / adapt_file_document      file -> reader-records/v1
    IngestionLibrary                      stateful, revision-aware ingestion

The supplied ``materials/745/reader_client.py`` is never imported or
modified; its CLI consumes the ``reader-records/v1`` documents this package
produces.
"""

from .types import (
    DecimalType,
    IntType,
    TextType,
    decode_writer_value,
    encode_reader_value,
    parse_type,
    rational_string,
    type_to_json,
)
from .contract import Contract, Field, load_contract, parse_contract
from .compat import CompatibilityReport, ColumnPlan, check_compatibility, replace_field_type
from .adapter import (
    AdaptationResult,
    CanonicalRow,
    RECORDS_FORMAT,
    RowError,
    adapt_file_document,
    adapt_rows,
    write_records,
)
from .ingest import IngestionLibrary, IngestResult, Receipt, RowState
from .predicate import (
    F,
    T,
    U,
    Atom,
    Predicate,
    evaluate_atom,
    evaluate_expression,
    evaluate_predicate,
    parse_predicate,
    tribool_and,
    tribool_not,
    tribool_or,
)
from .workorder import (
    STATISTICS_FORMAT,
    WORK_ORDER_FORMAT,
    JointStatistics,
    VectorStat,
    WorkOrder,
    WorkOrderNode,
    build_joint_statistics,
    build_work_order,
    parse_fees,
    parse_statistics_document,
    parse_work_order_document,
    sealed_file_fingerprint,
)
from .replay import (
    AtomOracle,
    ReplayResult,
    RowTrace,
    delivered_records,
    replay_from_documents,
    replay_work_order,
)
from .errors import (
    ContractError,
    ConversionError,
    DomainError,
    ExtContractError,
    LimitError,
)

__all__ = [
    "DecimalType",
    "IntType",
    "TextType",
    "decode_writer_value",
    "encode_reader_value",
    "parse_type",
    "rational_string",
    "type_to_json",
    "Contract",
    "Field",
    "load_contract",
    "parse_contract",
    "CompatibilityReport",
    "ColumnPlan",
    "check_compatibility",
    "replace_field_type",
    "AdaptationResult",
    "CanonicalRow",
    "RECORDS_FORMAT",
    "RowError",
    "adapt_file_document",
    "adapt_rows",
    "write_records",
    "IngestionLibrary",
    "IngestResult",
    "Receipt",
    "RowState",
    "Atom",
    "Predicate",
    "T",
    "F",
    "U",
    "parse_predicate",
    "evaluate_atom",
    "evaluate_expression",
    "evaluate_predicate",
    "tribool_not",
    "tribool_and",
    "tribool_or",
    "JointStatistics",
    "VectorStat",
    "WorkOrder",
    "WorkOrderNode",
    "STATISTICS_FORMAT",
    "WORK_ORDER_FORMAT",
    "build_joint_statistics",
    "build_work_order",
    "parse_fees",
    "parse_statistics_document",
    "parse_work_order_document",
    "sealed_file_fingerprint",
    "AtomOracle",
    "RowTrace",
    "ReplayResult",
    "replay_work_order",
    "replay_from_documents",
    "delivered_records",
    "ContractError",
    "ConversionError",
    "DomainError",
    "ExtContractError",
    "LimitError",
]

__version__ = "1.0.0"

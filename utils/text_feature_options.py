"""Shared, explicit controls for the versioned merchant-aware experiments."""

from utils.proposed_features import ProposedFeatureBuilder


def add_transaction_text_arguments(parser):
    parser.add_argument(
        "--transaction-text-columns",
        nargs="+",
        default=None,
        metavar="COLUMN",
        help=(
            "Genuine labelled text columns, normally merchant and description. "
            "Omit to preserve the original v1 feature contract."
        ),
    )
    parser.add_argument(
        "--max-transaction-text-features",
        type=int,
        default=0,
        help="Training-only TF-IDF vocabulary cap; must be positive when text columns are enabled.",
    )
    return parser


def feature_builder_from_config(config):
    return ProposedFeatureBuilder(
        transaction_text_columns=getattr(config, "transaction_text_columns", None),
        max_transaction_text_features=getattr(
            config, "max_transaction_text_features", 0
        ),
    )


def text_feature_manifest(builder):
    enabled = bool(builder.transaction_text_columns)
    return {
        "enabled": enabled,
        "columns": builder.transaction_text_columns,
        "vocabulary_size": (
            len(builder.transaction_text_vectorizer.get_feature_names_out())
            if enabled
            else 0
        ),
        "fit_scope": "training split only" if enabled else None,
    }


from pathlib import Path
from datetime import datetime
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V2 = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset/budgetwise_finance_dataset.csv"

groups = {
    "Bonus": "bonus",
    "Education": "edu education educaton",
    "Entertainment": "entertain entertainment entrtnmnt",
    "Food": "fod food foodd foods",
    "Freelance": "freelance",
    "Health": "health helth",
    "Investment": "investment",
    "Other": "misc other others",
    "Rent": "rent rentt rnt",
    "Salary": "salary",
    "Savings": "saving savings",
    "Travel": "traval travel travl",
    "Utilities": "utilities utility utilties utlities",
}

ALIASES = {
    spelling: category
    for category, variants in groups.items()
    for spelling in variants.split()
}

AMOUNT_RE = re.compile(
    r"^\s*(?:(₹|\$|Rs\.?|INR)\s*)?"
    r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))"
    r"\s*(INR|USD)?\s*$",
    re.IGNORECASE,
)


def parse_amount(value):
    match = AMOUNT_RE.fullmatch(str(value))
    if not match:
        return None, None

    prefix = (match.group(1) or "").upper()
    suffix = (match.group(3) or "").upper()

    if (prefix == "$" and suffix == "INR") or (
        prefix not in ("", "$") and suffix == "USD"
    ):
        return None, None

    currency = "UNMARKED"
    if prefix == "$" or suffix == "USD":
        currency = "USD"
    elif prefix in ("₹", "RS", "RS.", "INR") or suffix == "INR":
        currency = "INR"

    return float(match.group(2)), currency


def parse_date(value):
    value = str(value).strip()

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        fmt, parts, ambiguous = "%Y-%m-%d", None, False
    elif re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", value):
        fmt = "%m/%d/%Y"
        parts = value.split("/")
        ambiguous = int(parts[0]) <= 12 and int(parts[1]) <= 12
    elif re.fullmatch(r"\d{1,2}-\d{1,2}-(\d{2}|\d{4})", value):
        parts = value.split("-")
        fmt = "%d-%m-%y" if len(parts[2]) == 2 else "%d-%m-%Y"
        ambiguous = int(parts[0]) <= 12 and int(parts[1]) <= 12
    else:
        return None, False

    try:
        return datetime.strptime(value, fmt).strftime("%Y-%m-%d"), ambiguous
    except ValueError:
        return None, False


def main():
    raw = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    source_columns = list(raw.columns)

    raw.insert(0, "source_row_number", range(2, len(raw) + 2))
    original_count = len(raw)

    raw = raw.drop_duplicates(subset=source_columns).copy()
    duplicates_removed = original_count - len(raw)

    raw["category_raw"] = raw["category"]
    raw["amount_raw"] = raw["amount"]
    raw["date_raw"] = raw["date"]

    raw["category"] = (
        raw["category"].str.strip().str.lower().map(ALIASES)
    )

    amounts = raw["amount_raw"].map(parse_amount)
    raw[["amount", "currency_marker"]] = pd.DataFrame(
        amounts.tolist(), index=raw.index
    )

    dates = raw["date_raw"].map(parse_date)
    raw[["date", "ambiguous_date"]] = pd.DataFrame(
        dates.tolist(), index=raw.index
    )

    counts = {}
    counts["missing_category"] = int(raw["category"].isna().sum())
    raw = raw.dropna(subset=["category"])

    counts["invalid_amount"] = int(raw["amount"].isna().sum())
    raw = raw.dropna(subset=["amount"])

    counts["invalid_date"] = int(raw["date"].isna().sum())
    raw = raw.dropna(subset=["date"])

    raw["note_quality_flag"] = raw["notes"].apply(
        lambda value: "low_information"
        if str(value).strip().lower()
        in {"", "n/a", "unknown", "asdfgh", "xyz123",
            "!!!", "...", "misc", "test", "null"}
        else "unverified_content"
    )

    (V2 / "data").mkdir(parents=True, exist_ok=True)
    (V2 / "reports").mkdir(parents=True, exist_ok=True)

    output = V2 / "data/clean_budgetwise_v2.csv"
    raw.to_csv(output, index=False)

    report = {
        "original_rows": original_count,
        "exact_duplicates_removed": duplicates_removed,
        "removal_reasons": counts,
        "v2_rows": len(raw),
        "categories": raw["category"].nunique(),
        "currency_markers": raw["currency_marker"].value_counts().to_dict(),
        "ambiguous_date_count": int(raw["ambiguous_date"].sum()),
        "low_information_note_rows": int(
            (raw["note_quality_flag"] == "low_information").sum()
        ),
        "warnings": [
            "Currency markers are preserved; no FX conversion performed.",
            "Slash dates assumed MM/DD/YYYY.",
            "Dash dates assumed DD-MM-YY or DD-MM-YYYY.",
            "Category labels and note meanings were not manually corrected.",
            "V2 is exploratory and has not been trained or evaluated."
        ],
    }

    (V2 / "reports/dataset_v2_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

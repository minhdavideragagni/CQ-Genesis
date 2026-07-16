from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .models import ColumnProfile, DatasetProfileResponse


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".json",
    ".xml",
}


def read_dataset(
    filename: str,
    file_bytes: bytes,
) -> pd.DataFrame:
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported dataset format. "
            "Use CSV, TSV, XLSX, JSON, or XML."
        )

    if extension == ".csv":
        last_error: Exception | None = None

        for encoding in (
            "utf-8",
            "utf-8-sig",
            "latin-1",
        ):
            try:
                return pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding=encoding,
                )
            except Exception as exc:
                last_error = exc

        raise ValueError(
            f"The CSV file could not be read: {last_error}"
        )

    if extension == ".tsv":
        return pd.read_csv(
            io.BytesIO(file_bytes),
            sep="\t",
        )

    if extension == ".xlsx":
        return pd.read_excel(
            io.BytesIO(file_bytes)
        )
    
    if extension == ".xml":
        try:
            return pd.read_xml(
                io.BytesIO(file_bytes)
            )
        except Exception as exc:
            raise ValueError(
                "The XML file could not be read as a "
                "structured tabular dataset. "
                "CQ-Genesis expects repeated XML elements "
                "representing records or observations. "
                f"Original error: {exc}"
            ) from exc
    
    payload = json.loads(
        file_bytes.decode("utf-8-sig")
    )

    if isinstance(payload, list):
        return pd.json_normalize(payload)

    if isinstance(payload, dict):
        list_values = [
            value
            for value in payload.values()
            if isinstance(value, list)
        ]

        if len(list_values) == 1:
            return pd.json_normalize(
                list_values[0]
            )

        return pd.json_normalize(payload)

    raise ValueError(
        "The JSON file must contain an object "
        "or a list of objects."
    )


def shorten(
    value: Any,
    max_length: int = 90,
) -> str:
    text = str(value).replace(
        "\n",
        " ",
    ).strip()

    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def looks_like_datetime(
    series: pd.Series,
) -> bool:
    values = (
        series
        .dropna()
        .astype(str)
        .head(30)
    )

    if values.empty:
        return False

    parsed = pd.to_datetime(
        values,
        errors="coerce",
    )

    return bool(
        parsed.notna().mean() >= 0.7
    )


def suggest_structural_role(
    column_name: str,
    series: pd.Series,
) -> str:
    """
    Simple deterministic heuristic.

    The result is only a suggestion and is not treated
    as a semantic or ontological classification.
    """

    name = column_name.lower().strip()

    non_missing = int(
        series.notna().sum()
    )

    unique = int(
        series.nunique(
            dropna=True
        )
    )

    temporal_tokens = (
        "date",
        "time",
        "year",
        "month",
        "day",
        "period",
    )

    spatial_tokens = (
        "country",
        "region",
        "city",
        "place",
        "location",
        "latitude",
        "longitude",
        "territory",
    )

    identifier_tokens = (
        "id",
        "code",
        "identifier",
        "uuid",
    )

    if any(
        token in name
        for token in temporal_tokens
    ):
        return "temporal dimension"

    if any(
        token in name
        for token in spatial_tokens
    ):
        return "spatial dimension"

    if (
        any(
            token in name
            for token in identifier_tokens
        )
        and unique >= max(
            int(non_missing * 0.8),
            1,
        )
    ):
        return "identifier"

    if looks_like_datetime(series):
        return "temporal dimension"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric measure"

    if 0 < unique <= 50:
        return "categorical dimension"

    return "textual attribute"


def build_dataset_profile(
    filename: str,
    dataframe: pd.DataFrame,
    sample_rows: int = 10,
) -> DatasetProfileResponse:
    column_profiles: list[ColumnProfile] = []
    textual_lines: list[str] = []

    rows, columns = dataframe.shape

    textual_lines.append(
        f"Dataset size: {rows} rows and {columns} columns."
    )

    for column in dataframe.columns:
        series = dataframe[column]

        non_missing = int(
            series.notna().sum()
        )

        missing_percentage = (
            1 - non_missing / max(rows, 1)
        ) * 100

        unique_values = int(
            series.nunique(
                dropna=True
            )
        )

        suggested_role = suggest_structural_role(
            str(column),
            series,
        )

        examples = [
            shorten(value)
            for value in (
                series
                .dropna()
                .head(3)
                .tolist()
            )
        ]

        profile = ColumnProfile(
            name=str(column),
            data_type=str(series.dtype),
            suggested_role=suggested_role,
            non_missing_values=non_missing,
            missing_percentage=round(
                missing_percentage,
                2,
            ),
            unique_values=unique_values,
            example_values=examples,
        )

        column_profiles.append(profile)

        textual_lines.extend(
            [
                "",
                f"Field: {column}",
                f"- Data type: {series.dtype}",
                (
                    "- Suggested structural role: "
                    f"{suggested_role}"
                ),
                (
                    "- Non-missing values: "
                    f"{non_missing}"
                ),
                (
                    "- Missing values: "
                    f"{missing_percentage:.2f}%"
                ),
                (
                    "- Unique non-empty values: "
                    f"{unique_values}"
                ),
            ]
        )

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            if numeric.notna().any():
                textual_lines.append(
                    "- Numeric range: "
                    f"{numeric.min()} to {numeric.max()}"
                )

        if examples:
            textual_lines.append(
                f"- Example values: {examples}"
            )

    if sample_rows > 0 and not dataframe.empty:
        effective_rows = min(
            sample_rows,
            len(dataframe),
        )

        if len(dataframe) <= effective_rows:
            sampled = dataframe.copy()
        else:
            sampled = dataframe.sample(
                n=effective_rows,
            )

        sampled = sampled.fillna("")

        for column in sampled.columns:
            sampled[column] = sampled[column].map(
                shorten
            )

        sample = sampled.to_dict(
            orient="records"
        )

    else:
        sample = []

    return DatasetProfileResponse(
        filename=filename,
        rows=int(rows),
        columns=int(columns),
        missing_cells=int(
            dataframe.isna().sum().sum()
        ),
        column_profiles=column_profiles,
        textual_profile="\n".join(
            textual_lines
        ),
        sample=sample,
    )

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


InputMode = Literal[
    "dataset_only",
    "user_stories_only",
    "multi_source",
]

CountMode = Literal[
    "automatic",
    "fixed",
]


class GenerationSettings(BaseModel):
    provider: str = "openai"
    model: str

    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
    )

    max_output_tokens: int = Field(
        default=5000,
        ge=500,
        le=16000,
    )

    language: str = "same_as_input"

    count_mode: CountMode = "automatic"

    fixed_number: int | None = Field(
        default=None,
        ge=1,
        le=100,
    )

    sample_rows: int = Field(
        default=10,
        ge=0,
        le=100,
    )


class CompetencyQuestion(BaseModel):
    id: str
    question: str

    pattern: str | None = None

    source: Literal[
        "dataset",
        "user_stories",
        "both",
    ]

    notes: str | None = None


class GenerationMetadata(BaseModel):
    provider: str
    model: str

    prompt_version: str
    prompting_strategy: str

    input_mode: InputMode

    temperature: float
    max_output_tokens: int
    language: str

    count_mode: CountMode
    fixed_number: int | None = None

    sample_rows: int

    generation_time_seconds: float

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    validation_notes: list[str] = []


class GenerationResponse(BaseModel):
    questions: list[CompetencyQuestion]
    metadata: GenerationMetadata

    raw_output: str | None = None


class ColumnProfile(BaseModel):
    name: str
    data_type: str

    suggested_role: str

    non_missing_values: int
    missing_percentage: float
    unique_values: int

    example_values: list[str] = []


class DatasetProfileResponse(BaseModel):
    filename: str

    rows: int
    columns: int
    missing_cells: int

    column_profiles: list[ColumnProfile]

    textual_profile: str
    sample: list[dict[str, Any]]

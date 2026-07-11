from __future__ import annotations

import json

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware

from backend.cq_genesis.generation import (
    generate_competency_questions,
)
from backend.cq_genesis.models import (
    DatasetProfileResponse,
    GenerationResponse,
    GenerationSettings,
    InputMode,
)
from backend.cq_genesis.profiling import (
    build_dataset_profile,
    read_dataset,
)


app = FastAPI(
    title="CQ-Genesis API",
    description=(
        "Backend API for generating Competency Questions "
        "from structured datasets and user stories."
    ),
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "CQ-Genesis API",
    }


@app.post(
    "/profile",
    response_model=DatasetProfileResponse,
)
async def profile_dataset(
    dataset: UploadFile = File(...),
    sample_rows: int = Form(10),
) -> DatasetProfileResponse:
    try:
        file_bytes = await dataset.read()

        dataframe = read_dataset(
            filename=dataset.filename or "dataset.csv",
            file_bytes=file_bytes,
        )

        return build_dataset_profile(
            filename=dataset.filename or "dataset.csv",
            dataframe=dataframe,
            sample_rows=sample_rows,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post(
    "/generate",
    response_model=GenerationResponse,
)
async def generate(
    input_mode: InputMode = Form(...),

    provider: str = Form("openai"),
    model: str = Form(...),

    api_key: str = Form(...),
    base_url: str | None = Form(None),

    temperature: float = Form(0.2),
    max_output_tokens: int = Form(5000),

    language: str = Form("same_as_input"),

    count_mode: str = Form("automatic"),
    fixed_number: int | None = Form(None),

    sample_rows: int = Form(10),

    dataset_context: str = Form(""),
    user_stories: str = Form(""),

    dataset: UploadFile | None = File(None),
) -> GenerationResponse:
    try:
        if (
            input_mode in {
                "dataset_only",
                "multi_source",
            }
            and dataset is None
        ):
            raise ValueError(
                "A structured dataset is required "
                "for the selected input mode."
            )

        if (
            input_mode in {
                "user_stories_only",
                "multi_source",
            }
            and not user_stories.strip()
        ):
            raise ValueError(
                "User stories are required "
                "for the selected input mode."
            )

        if (
            count_mode == "fixed"
            and fixed_number is None
        ):
            raise ValueError(
                "A fixed number of CQs is required "
                "when count mode is fixed."
            )

        dataset_profile = ""
        dataset_sample: list[dict] = []

        if dataset is not None:
            file_bytes = await dataset.read()

            dataframe = read_dataset(
                filename=(
                    dataset.filename
                    or "dataset.csv"
                ),
                file_bytes=file_bytes,
            )

            profile = build_dataset_profile(
                filename=(
                    dataset.filename
                    or "dataset.csv"
                ),
                dataframe=dataframe,
                sample_rows=sample_rows,
            )

            dataset_profile = (
                profile.textual_profile
            )

            dataset_sample = (
                profile.sample
            )

        settings = GenerationSettings(
            provider=provider,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            language=language,
            count_mode=count_mode,
            fixed_number=fixed_number,
            sample_rows=sample_rows,
        )

        return generate_competency_questions(
            input_mode=input_mode,
            settings=settings,
            api_key=api_key,
            base_url=base_url,
            dataset_context=dataset_context,
            user_stories=user_stories,
            dataset_profile=dataset_profile,
            dataset_sample=dataset_sample,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

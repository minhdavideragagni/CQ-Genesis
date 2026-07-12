from __future__ import annotations

import json
import re
from typing import Any

from .models import CompetencyQuestion, InputMode


VALID_SOURCES = {
    "dataset",
    "user_stories",
    "both",
}


def parse_json_output(
    raw_output: str,
) -> dict[str, Any]:
    cleaned = raw_output.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        parsed = json.loads(cleaned)

    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start < 0 or end <= start:
            raise ValueError(
                "The model did not return valid JSON."
            )

        parsed = json.loads(
            cleaned[start : end + 1]
        )

    if not isinstance(parsed, dict):
        raise ValueError(
            "The model output is not a JSON object."
        )

    return parsed


def question_fingerprint(
    question: str,
) -> str:
    return re.sub(
        r"\W+",
        " ",
        question.lower(),
    ).strip()


def validate_questions(
    parsed_output: dict[str, Any],
    input_mode: InputMode,
) -> tuple[
    list[CompetencyQuestion],
    list[str],
]:
    raw_questions = parsed_output.get(
        "questions"
    )

    if not isinstance(
        raw_questions,
        list,
    ):
        raise ValueError(
            "The JSON object does not contain "
            "a valid 'questions' list."
        )

    questions: list[CompetencyQuestion] = []
    validation_notes: list[str] = []

    seen: set[str] = set()

    for raw_question in raw_questions:
        if not isinstance(
            raw_question,
            dict,
        ):
            validation_notes.append(
                "A non-object item was removed."
            )
            continue

        question = str(
            raw_question.get(
                "question",
                "",
            )
        ).strip()

        if not question:
            validation_notes.append(
                "An empty question was removed."
            )
            continue

        if not question.endswith("?"):
            question += "?"

            validation_notes.append(
                "A final question mark was added "
                f"to: {question}"
            )

        fingerprint = question_fingerprint(
            question
        )

        if fingerprint in seen:
            validation_notes.append(
                f"Duplicate removed: {question}"
            )
            continue

        seen.add(fingerprint)

        source = str(
            raw_question.get(
                "source",
                "both",
            )
        ).strip().lower()

        if input_mode == "dataset_only":
            source = "dataset"

        elif input_mode == "user_stories_only":
            source = "user_stories"

        elif source not in VALID_SOURCES:
            source = "both"

            validation_notes.append(
                "An invalid source label was "
                f"corrected for: {question}"
            )

        pattern = str(
            raw_question.get(
                "pattern",
                "Free-form",
            )
        ).strip()

        notes = str(
            raw_question.get(
                "notes",
                "",
            )
        ).strip()

        questions.append(
            CompetencyQuestion(
                id=f"CQ{len(questions) + 1}",
                question=question,
                pattern=pattern or "Free-form",
                source=source,
                notes=notes or None,
            )
        )

    if not questions:
        raise ValueError(
            "No valid Competency Questions "
            "were found in the model output."
        )

    return (
        questions,
        validation_notes,
    )

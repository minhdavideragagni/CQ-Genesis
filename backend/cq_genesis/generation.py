from __future__ import annotations

import time
from typing import Any

from anthropic import Anthropic
from google import genai
from google.genai import types
from openai import OpenAI

from .models import (
    GenerationMetadata,
    GenerationResponse,
    GenerationSettings,
    InputMode,
)
from .prompting import (
    PROMPT_VERSION,
    PROMPTING_STRATEGY,
    build_messages,
)
from .validation import (
    parse_json_output,
    validate_questions,
)


def create_llm_client(
    *,
    provider: str,
    api_key: str,
    base_url: str | None = None,
) -> Any:
    if not api_key.strip():
        raise ValueError(
            "An API key is required."
        )

    clean_key = api_key.strip()

    if provider == "openai":
        return OpenAI(
            api_key=clean_key
        )

    if provider == "anthropic":
        return Anthropic(
            api_key=clean_key
        )

    if provider == "gemini":
        return genai.Client(
            api_key=clean_key
        )

    if provider == "huggingface":
        if not base_url:
            raise ValueError(
                "A base URL is required for Hugging Face."
            )

        return OpenAI(
            api_key=clean_key,
            base_url=base_url.rstrip("/"),
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )

def call_gemini(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_output_tokens: int,
) -> tuple[
    str,
    dict[str, int | None],
    list[str],
]:
    system_instruction = ""
    user_contents: list[str] = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "system":
            system_instruction = content
        elif role == "user":
            user_contents.append(content)

    response = client.models.generate_content(
        model=model,
        contents="\n\n".join(user_contents),
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        ),
    )

    raw_output = (
        response.text
        or ""
    ).strip()

    usage = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }

    usage_metadata = getattr(
        response,
        "usage_metadata",
        None,
    )

    if usage_metadata is not None:
        usage = {
            "prompt_tokens": getattr(
                usage_metadata,
                "prompt_token_count",
                None,
            ),
            "completion_tokens": getattr(
                usage_metadata,
                "candidates_token_count",
                None,
            ),
            "total_tokens": getattr(
                usage_metadata,
                "total_token_count",
                None,
            ),
        }

    return (
        raw_output,
        usage,
        [],
    )

def call_anthropic(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_output_tokens: int,
) -> tuple[
    str,
    dict[str, int | None],
    list[str],
]:
    system_instruction = ""
    anthropic_messages: list[dict[str, str]] = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "system":
            system_instruction = content

        elif role in {
            "user",
            "assistant",
        }:
            anthropic_messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    response = client.messages.create(
        model=model,
        system=system_instruction,
        messages=anthropic_messages,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )

    output_parts: list[str] = []

    for block in response.content:
        if getattr(
            block,
            "type",
            None,
        ) == "text":
            output_parts.append(
                getattr(
                    block,
                    "text",
                    "",
                )
            )

    raw_output = "\n".join(
        output_parts
    ).strip()

    usage = {
        "prompt_tokens": getattr(
            response.usage,
            "input_tokens",
            None,
        ),
        "completion_tokens": getattr(
            response.usage,
            "output_tokens",
            None,
        ),
        "total_tokens": None,
    }

    if (
        usage["prompt_tokens"] is not None
        and usage["completion_tokens"] is not None
    ):
        usage["total_tokens"] = (
            usage["prompt_tokens"]
            + usage["completion_tokens"]
        )

    return (
        raw_output,
        usage,
        [
            (
                "Claude was instructed to return JSON through "
                "the prompt specification."
            )
        ],
    )

def call_openai_compatible(
    *,
    client: Any,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_output_tokens: int,
) -> tuple[
    str,
    dict[str, int | None],
    list[str],
]:
    notes: list[str] = []

    base_arguments: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {
            "type": "json_object"
        },
    }

    if provider == "openai":
        base_arguments["max_completion_tokens"] = (
            max_output_tokens
        )
    else:
        base_arguments["max_tokens"] = (
            max_output_tokens
        )

    attempts: list[dict[str, Any]] = [
        base_arguments.copy()
    ]

    without_json_mode = base_arguments.copy()
    without_json_mode.pop(
        "response_format",
        None,
    )

    attempts.append(
        without_json_mode
    )

    without_temperature = (
        without_json_mode.copy()
    )

    without_temperature.pop(
        "temperature",
        None,
    )

    attempts.append(
        without_temperature
    )

    last_error: Exception | None = None

    for attempt_index, arguments in enumerate(
        attempts
    ):
        try:
            response = (
                client
                .chat
                .completions
                .create(**arguments)
            )

            if attempt_index == 1:
                notes.append(
                    "The selected provider did not accept "
                    "JSON mode. Prompt-only JSON instructions "
                    "were used."
                )

            if attempt_index == 2:
                notes.append(
                    "The selected provider did not accept "
                    "JSON mode or temperature. Both optional "
                    "parameters were omitted."
                )

            raw_output = (
                response
                .choices[0]
                .message
                .content
                or ""
            ).strip()

            usage = {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }

            if getattr(
                response,
                "usage",
                None,
            ):
                usage = {
                    "prompt_tokens": getattr(
                        response.usage,
                        "prompt_tokens",
                        None,
                    ),
                    "completion_tokens": getattr(
                        response.usage,
                        "completion_tokens",
                        None,
                    ),
                    "total_tokens": getattr(
                        response.usage,
                        "total_tokens",
                        None,
                    ),
                }

            return (
                raw_output,
                usage,
                notes,
            )

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "The LLM request failed after "
        f"compatibility retries: {last_error}"
    )

def call_llm(
    *,
    client: Any,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_output_tokens: int,
) -> tuple[
    str,
    dict[str, int | None],
    list[str],
]:
    if provider == "gemini":
        return call_gemini(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    if provider == "anthropic":
        return call_anthropic(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    if provider in {
        "openai",
        "huggingface",
    }:
        return call_openai_compatible(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )

def generate_competency_questions(
    *,
    input_mode: InputMode,
    settings: GenerationSettings,
    api_key: str,
    base_url: str | None,
    dataset_context: str,
    user_stories: str,
    dataset_profile: str,
    dataset_sample: list[dict],
) -> GenerationResponse:
    messages = build_messages(
        input_mode=input_mode,
        settings=settings,
        dataset_context=dataset_context,
        user_stories=user_stories,
        dataset_profile=dataset_profile,
        dataset_sample=dataset_sample,
    )

    client = create_llm_client(
        provider=settings.provider,
        api_key=api_key,
        base_url=base_url,
    )

    started_at = time.perf_counter()

    (
        raw_output,
        usage,
        provider_notes,
    ) = call_llm(
        client=client,
        provider=settings.provider,
        model=settings.model,
        messages=messages,
        temperature=settings.temperature,
        max_output_tokens=(
            settings.max_output_tokens
        ),
    )

    parsed_output = parse_json_output(
        raw_output
    )

    (
        questions,
        validation_notes,
    ) = validate_questions(
        parsed_output,
        input_mode,
    )

    generation_time = (
        time.perf_counter()
        - started_at
    )

    metadata = GenerationMetadata(
        provider=settings.provider,
        model=settings.model,
        prompt_version=PROMPT_VERSION,
        prompting_strategy=PROMPTING_STRATEGY,
        input_mode=input_mode,
        temperature=settings.temperature,
        max_output_tokens=(
            settings.max_output_tokens
        ),
        language=settings.language,
        count_mode=settings.count_mode,
        fixed_number=settings.fixed_number,
        sample_rows=settings.sample_rows,
        generation_time_seconds=round(
            generation_time,
            3,
        ),
        prompt_tokens=usage.get(
            "prompt_tokens"
        ),
        completion_tokens=usage.get(
            "completion_tokens"
        ),
        total_tokens=usage.get(
            "total_tokens"
        ),
        validation_notes=(
            provider_notes
            + validation_notes
        ),
    )

    return GenerationResponse(
        questions=questions,
        metadata=metadata,
        raw_output=raw_output,
    )

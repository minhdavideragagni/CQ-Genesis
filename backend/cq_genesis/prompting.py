from __future__ import annotations

import json

from .models import GenerationSettings, InputMode


PROMPT_VERSION = "1.0"

PROMPTING_STRATEGY = (
    "Pattern-guided few-shot prompting"
)


CQ_PATTERNS = [
    {
        "pattern": (
            "Which [class expression 1] "
            "[object property expression] "
            "[class expression 2]?"
        ),
        "example": "Which pizzas contain pork?",
    },
    {
        "pattern": (
            "How much does [class expression] "
            "[datatype property]?"
        ),
        "example": "How much does a pizza weigh?",
    },
    {
        "pattern": (
            "What type of [class expression] "
            "is [individual]?"
        ),
        "example": "What type of software is it?",
    },
    {
        "pattern": (
            "Is the [class expression 1] "
            "[class expression 2]?"
        ),
        "example": "Is the software open source?",
    },
    {
        "pattern": (
            "What [class expression] has the "
            "[numeric modifier] [datatype property]?"
        ),
        "example": "What pizza has the lowest price?",
    },
    {
        "pattern": "Which are [class expressions]?",
        "example": "Which are gluten-free bases?",
    },
]


QUALITY_EXAMPLES = [
    {
        "principle": "Avoid dataset-specific terminology.",
        "incorrect": (
            "How many COVID_19_CASES are recorded "
            "in IT_2021?"
        ),
        "correct": (
            "How many cases of a disease were recorded "
            "in a given territory during a given period?"
        ),
    },
    {
        "principle": "Keep each question atomic.",
        "incorrect": (
            "What is the creator and creation date "
            "of an artwork?"
        ),
        "correct": [
            "Who created a given artwork?",
            "When was a given artwork created?",
        ],
    },
    {
        "principle": (
            "Abstract specific instances "
            "into domain concepts."
        ),
        "incorrect": (
            "Who is the author of Harry Potter?"
        ),
        "correct": (
            "Who is the author of a given book?"
        ),
    },
    {
        "principle": (
            "Avoid paraphrased or superficial duplicates."
        ),
        "incorrect": [
            "How many people are affected?",
            "What is the number of affected people?",
        ],
        "correct": (
            "What is the number of people affected "
            "by a given condition?"
        ),
    },
]


def input_mode_instruction(
    input_mode: InputMode,
) -> str:
    if input_mode == "dataset_only":
        return """
Use the structured dataset as evidence of the domain and of the
information represented by the available fields and observations.

Infer domain-level concepts, relationships, measures, temporal
dimensions, spatial dimensions, and classifications when supported.

Do not copy raw field names, identifiers, codes, abbreviations,
or accidental sample values into the final CQs.

Every generated CQ must use "dataset" as its source.
""".strip()

    if input_mode == "user_stories_only":
        return """
Use the user stories to identify stakeholder goals and information
needs.

Generalise from specific scenarios and examples while preserving
the original intent. Do not introduce unrelated requirements.

Every generated CQ must use "user_stories" as its source.
""".strip()

    return """
Use the user stories as evidence of stakeholder goals and the
structured dataset as evidence of the information currently
represented by the available data.

Prefer questions supported by both sources.

A question may also be grounded mainly in the dataset or mainly
in the user stories when this is useful and clearly indicated.

Use:
- "both" when both sources support the question;
- "dataset" when the question is suggested mainly by the data;
- "user_stories" when the question expresses a stakeholder need
  that is not clearly represented in the dataset.

Do not assume that every stakeholder need is already supported by
the available data.
""".strip()


def language_instruction(
    language: str,
) -> str:
    if language == "same_as_input":
        return (
            "Generate the CQs in the predominant language "
            "of the provided input."
        )

    return (
        f"Generate all CQs in {language}."
    )


def number_instruction(
    settings: GenerationSettings,
) -> str:
    if (
        settings.count_mode == "fixed"
        and settings.fixed_number is not None
    ):
        return (
            f"Generate exactly {settings.fixed_number} "
            "distinct Competency Questions."
        )

    return """
Identify all distinct information needs that can be meaningfully
derived from the provided sources.

Generate as many CQs as are needed to cover the supported concepts,
relationships, attributes, classifications, measures, temporal
dimensions, spatial dimensions, comparisons, and aggregations.

Do not stop after producing only a small representative sample.

Stop only when an additional question would be redundant,
overly generic, or unsupported by the input.
""".strip()


def build_messages(
    *,
    input_mode: InputMode,
    settings: GenerationSettings,
    dataset_context: str,
    user_stories: str,
    dataset_profile: str,
    dataset_sample: list[dict],
) -> list[dict[str, str]]:
    system_prompt = f"""
You are an ontology engineer specialised in the elicitation and
formulation of Competency Questions.

A Competency Question is a natural-language question expressing
an information requirement that an ontology should be able to
answer.

PROMPTING APPROACH

Follow the CQ-Genesis Prompt Schema {PROMPT_SCHEMA_VERSION}.

The approach combines:

1. explicit task instructions;
2. reusable Competency Question patterns;
3. positive and negative formulation examples;
4. mode-specific interpretation of the input sources.

You may analyse the input internally, but return only the requested
JSON output. Do not expose private reasoning or chain-of-thought.

QUALITY REQUIREMENTS

- Relevance:
  Every CQ must be supported by at least one input source.

- Appropriate abstraction:
  Use domain-level concepts rather than raw field names,
  identifiers, codes, or accidental sample values.

- Atomicity:
  Each CQ should express one principal information need.

- Clarity:
  Use concise, grammatical, and unambiguous questions.

- Terminological consistency:
  Use the same expression for the same domain concept.

- Conceptual diversity:
  Avoid duplicates, paraphrases, and superficial variations.

- Source faithfulness:
  Do not introduce unsupported concepts merely to increase
  the number of generated questions.

- Conceptual coverage:
  Cover the meaningful informational dimensions exposed by
  the input sources.

MODE-SPECIFIC INSTRUCTIONS

{input_mode_instruction(input_mode)}

LANGUAGE

{language_instruction(settings.language)}

NUMBER OF QUESTIONS

{number_instruction(settings)}

COMPETENCY QUESTION PATTERNS

The following patterns are methodological guidance rather than
mandatory templates. Use them when appropriate, but allow a
well-formed free-form question when none applies.

{json.dumps(CQ_PATTERNS, indent=2, ensure_ascii=False)}

POSITIVE AND NEGATIVE EXAMPLES

{json.dumps(QUALITY_EXAMPLES, indent=2, ensure_ascii=False)}

CLUSTERING

Assign each CQ to a concise thematic cluster representing a
coherent conceptual or functional area.

OUTPUT FORMAT

Return valid JSON only.

Use exactly this top-level structure:

{{
  "questions": [
    {{
      "id": "CQ1",
      "question": "A natural-language question ending with ?",
      "pattern": "Matched CQ pattern or Free-form",
      "source": "dataset | user_stories | both",
      "cluster": "Short thematic cluster",
      "notes": "Short grounding note"
    }}
  ]
}}

Do not return markdown, code fences, explanations, or additional
top-level fields.
""".strip()

    context_parts: list[str] = [
        f"INPUT MODE: {input_mode}"
    ]

    if dataset_context.strip():
        context_parts.append(
            "DATASET CONTEXT AND DOCUMENTATION:\n"
            + dataset_context.strip()
        )

    if user_stories.strip():
        context_parts.append(
            "USER STORIES:\n"
            + user_stories.strip()
        )

    if dataset_profile.strip():
        context_parts.append(
            "STRUCTURED DATASET PROFILE:\n"
            + dataset_profile.strip()
        )

    if dataset_sample:
        context_parts.append(
            "REPRESENTATIVE DATASET SAMPLE:\n"
            + json.dumps(
                dataset_sample,
                indent=2,
                ensure_ascii=False,
            )
        )

    context_parts.append(
        "Generate the Competency Questions now."
    )

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": "\n\n".join(
                context_parts
            ),
        },
    ]

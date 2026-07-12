from __future__ import annotations

import json

from .models import GenerationSettings, InputMode


PROMPT_VERSION = "CGPS v1.1"

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
        "principle": (
            "Generalise dataset instances into reusable "
            "domain-level expressions."
        ),
        "incorrect": (
            "How many cases of tuberculosis were reported in 2018?"
        ),
        "correct": (
            "How many cases of a given disease were reported "
            "during a given period?"
        ),
    },
    {
        "principle": (
            "Do not reproduce dataset-specific field names, "
            "codes, or identifiers."
        ),
        "incorrect": (
            "How many COVID_19_CASES are recorded in IT_2021?"
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
            "Abstract individual entities into domain concepts."
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
Use the structured dataset as evidence of:

- the domain being represented;
- candidate domain concepts;
- relationships between concepts;
- attributes and measures;
- temporal, spatial, and categorical dimensions.

The dataset profile and sample are evidence for conceptual
elicitation, not content to be copied literally.

Do not include raw field names, codes, identifiers, abbreviations,
or specific values observed in the sample in the final CQs.

Every generated CQ must use "dataset" as its source.
""".strip()

    if input_mode == "user_stories_only":
        return """
Use the user stories to identify stakeholder goals, intended uses,
and information needs.

If the stories use optional headings such as Persona, Goal, or
Scenario, interpret those headings as structural guidance.

If the stories are written in free form, infer the same information
without requiring a predefined format.

Generalise from specific scenarios while preserving stakeholder
intent.

Every generated CQ must use "user_stories" as its source.
""".strip()

    return """
Interpret the two sources as complementary:

- User stories provide stakeholder goals, intended uses, and
  information needs.
- Dataset context provides semantic framing.
- The dataset profile and sample provide structural evidence about
  the information currently available.

Use "both" when a CQ is supported by the stakeholder need and by
the structured data.

Use "dataset" when the CQ is primarily suggested by the information
represented in the dataset.

Use "user_stories" when the CQ expresses a stakeholder requirement
that is not clearly supported by the available dataset.

Do not assume that every stakeholder need is already represented
in the dataset.
""".strip()


def language_instruction(
    language: str,
) -> str:
    if language == "same_as_input":
        return (
            "Generate the CQs in the predominant language "
            "of the provided input."
        )

    return f"Generate all CQs in {language}."


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
Generate as many CQs as are required to cover the distinct,
meaningful information needs supported by the provided sources.

Cover relevant concepts, relationships, attributes, measures,
classifications, temporal and spatial dimensions, comparisons,
and aggregations when they are supported.

Do not stop after producing only a small representative set.

Do not increase the number by adding paraphrases, unsupported
questions, excessively generic questions, or questions based only
on individual sample records.

Stop when any additional CQ would be redundant, unsupported,
or conceptually insignificant.
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

CQ-Genesis assists the knowledge engineer. It does not replace
their expertise or impose modelling decisions.

PROMPTING STRATEGY

Use a pattern-guided few-shot prompting strategy combining:

1. explicit task instructions;
2. positive and negative formulation examples;
3. optional CQ patterns as linguistic support;
4. mode-specific interpretation of the input sources.

You may perform internal analysis, but return only the requested
JSON output. Do not expose private reasoning or chain-of-thought.

INTERNAL CONCEPTUAL ANALYSIS

Before formulating the CQs, internally identify:

- candidate domain classes;
- candidate relationships;
- candidate attributes and measures;
- temporal and spatial dimensions;
- stakeholder goals and information needs;
- terminology that should be used consistently.

Do not output this analysis.

Use it only to formulate abstract and ontology-oriented CQs.

SOURCE INTERPRETATION

{input_mode_instruction(input_mode)}

OPTIONAL INPUT STRUCTURES

Dataset descriptions may optionally contain headings such as:

- Domain
- Purpose
- Unit of observation
- Provenance
- Additional notes

User stories may optionally contain headings such as:

- Persona
- Goal
- Scenario

These structures are recommendations only.

Do not require them, and do not penalise free-form input.
When headings are absent, infer the relevant information from the
available description.

QUALITY REQUIREMENTS

- Relevance:
  Every CQ must be supported by at least one provided input source.

- Appropriate abstraction:
  Treat dataset profiles and samples as evidence for discovering
  domain-level concepts and information needs.

  Do not include raw column names, identifiers, codes, abbreviations,
  or specific observed values in the final CQs.

  In particular, do not mention specific persons, organisations,
  diseases, years, age groups, territories, products, or other
  observed instances unless the user explicitly asks for
  instance-level questions.

  Generalise observed values into reusable expressions such as
  "a given disease", "a given period", "a given age group",
  "a given territory", or another suitable domain-level term.

- Generality test:
  Before returning a CQ, verify that replacing an observed sample
  value with another value of the same kind would not invalidate
  the question.

- Atomicity:
  Each CQ must express one principal information need.

- Clarity:
  Use concise, grammatical, and unambiguous formulations.

- Terminological consistency:
  Use the same expression for the same domain concept throughout
  the output.

- Conceptual diversity:
  Avoid duplicates, paraphrases, and superficial variations.

- Source faithfulness:
  Do not introduce concepts that are unsupported by the sources.

- Semantic caution:
  Do not assume that every dataset field directly represents a
  meaningful domain concept.

  If a field or value remains ambiguous after considering its
  documentation and surrounding context, do not generate a CQ
  from it.

- Conceptual coverage:
  Cover meaningful information needs even when they cannot be
  expressed through one of the supplied CQ patterns.

LANGUAGE

{language_instruction(settings.language)}

NUMBER OF QUESTIONS

{number_instruction(settings)}

POSITIVE AND NEGATIVE EXAMPLES

{json.dumps(
    QUALITY_EXAMPLES,
    indent=2,
    ensure_ascii=False,
)}

COMPETENCY QUESTION PATTERNS

The following patterns are optional linguistic scaffolds, not
mandatory templates and not an exhaustive inventory of valid CQs.

First identify the supported information need and formulate the CQ
without consulting the pattern inventory.

Pattern assignment is a separate post-generation annotation task.

The default value of the pattern field is "Free-form".

Replace "Free-form" with one of the listed patterns only when the
CQ already exhibits a clear, direct, and unambiguous syntactic and
semantic match with that pattern.

Do not rewrite, simplify, distort, or omit a CQ in order to obtain
a pattern match.

A shared interrogative form or a partial lexical resemblance is not
sufficient evidence of a match.

If the correspondence is approximate, partial, uncertain, or requires
reinterpretation, retain "Free-form".

Pattern coverage is not a quality objective. The number or proportion
of pattern-matched CQs must not influence which information needs are
selected or how they are formulated.

A valid output may contain any proportion of pattern-matched and
free-form CQs.

{json.dumps(
    CQ_PATTERNS,
    indent=2,
    ensure_ascii=False,
)}

OUTPUT FORMAT

Return valid JSON only.

Use exactly this top-level structure:

{{
  "questions": [
    {{
      "id": "CQ1",
      "question": "A natural-language question ending with ?",
      "pattern": "Free-form by default; otherwise one clearly matched CQ pattern",
      "source": "dataset | user_stories | both",
      "notes": "A short justification explaining how the CQ is grounded in the input and why the pattern is either a clear match or Free-form"
    }}
  ]
}}

Do not return markdown, code fences, explanations, clusters,
CQ categories, or additional top-level fields.
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
            "USER STORIES OR REQUIREMENT DESCRIPTION:\n"
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

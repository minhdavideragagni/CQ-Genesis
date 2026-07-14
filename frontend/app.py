from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


# =============================================================================
# Application configuration
# =============================================================================

APP_NAME = "CQ-Genesis"
APP_SUBTITLE = (
    "LLM-Assisted and Human-Guided Competency Question Generation from Structured Data and User Stories"
)

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000",
).rstrip("/")

st.set_page_config(
    page_title=APP_NAME,
    page_icon="❓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Visual style
# =============================================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1450px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }

        .main-subtitle {
            font-size: 1.05rem;
            opacity: 0.75;
            margin-top: -0.35rem;
            margin-bottom: 1.5rem;
        }

        .feature-card {
            height: 100%;
            padding: 1.1rem 1.15rem;
            border: 1px solid rgba(120, 120, 120, 0.22);
            border-radius: 14px;
            background: rgba(120, 120, 120, 0.025);
        }

        .feature-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .feature-text {
            font-size: 0.9rem;
            line-height: 1.45;
            opacity: 0.78;
        }

        .cq-card {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(120, 120, 120, 0.22);
            border-radius: 12px;
            margin-bottom: 0.75rem;
        }

        .small-muted {
            font-size: 0.85rem;
            opacity: 0.7;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(120, 120, 120, 0.18);
            border-radius: 12px;
            padding: 0.65rem 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Session state
# =============================================================================

def initialise_session_state() -> None:
    defaults = {
        "dataset_profile": None,
        "generation_response": None,
        "last_error": None,
        "dataset_context_text": "",
        "user_stories_text": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialise_session_state()

DATASET_CONTEXT_TEMPLATE = """Domain:

Purpose:

Unit of observation:

Provenance:

Additional notes:
"""


USER_STORY_TEMPLATE = """Persona:

Goal:

Scenario:
"""


# =============================================================================
# Utility functions
# =============================================================================

def backend_health() -> tuple[bool, str]:
    try:
        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=5,
        )

        if response.ok:
            return True, "Backend connected"

        return False, f"Backend error: HTTP {response.status_code}"

    except requests.RequestException:
        return (
            False,
            "CQ-Genesis services are starting. "
            "Please wait up to one minute and reload the page.",
        )


def response_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()

        if isinstance(payload, dict):
            detail = payload.get("detail")

            if detail:
                return str(detail)

        return json.dumps(
            payload,
            ensure_ascii=False,
        )

    except Exception:
        return response.text or (
            f"Request failed with HTTP {response.status_code}"
        )


def profile_dataset(
    uploaded_file: Any,
    sample_rows: int,
) -> dict[str, Any]:
    files = {
        "dataset": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }

    data = {
        "sample_rows": str(sample_rows),
    }

    response = requests.post(
        f"{BACKEND_URL}/profile",
        files=files,
        data=data,
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            response_error_message(response)
        )

    return response.json()


def generate_cqs(
    *,
    uploaded_file: Any | None,
    input_mode: str,
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_output_tokens: int,
    language: str,
    count_mode: str,
    fixed_number: int | None,
    sample_rows: int,
    dataset_context: str,
    user_stories: str,
) -> dict[str, Any]:
    data = {
        "input_mode": input_mode,
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "temperature": str(temperature),
        "max_output_tokens": str(max_output_tokens),
        "language": language,
        "count_mode": count_mode,
        "sample_rows": str(sample_rows),
        "dataset_context": dataset_context,
        "user_stories": user_stories,
    }

    if base_url.strip():
        data["base_url"] = base_url.strip()

    if fixed_number is not None:
        data["fixed_number"] = str(fixed_number)

    files = None

    if uploaded_file is not None:
        files = {
            "dataset": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        }

    response = requests.post(
        f"{BACKEND_URL}/generate",
        data=data,
        files=files,
        timeout=600,
    )

    if not response.ok:
        raise RuntimeError(
            response_error_message(response)
        )

    return response.json()


def questions_to_dataframe(
    questions: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    for item in questions:
        rows.append(
            {
                "include": True,
                "id": item.get("id", ""),
                "question": item.get("question", ""),
                "pattern": item.get("pattern", ""),
                "source": item.get("source", ""),
                "notes": item.get("notes", ""),
            }
        )

    return pd.DataFrame(rows)


def edited_questions_from_dataframe(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    questions = []

    for _, row in dataframe.iterrows():
        if not bool(row.get("include", True)):
            continue

        question = str(
            row.get("question", "")
        ).strip()

        if not question:
            continue

        if not question.endswith("?"):
            question += "?"

        questions.append(
            {
                "id": f"CQ{len(questions) + 1}",
                "question": question,
                "pattern": str(
                    row.get("pattern", "")
                ).strip(),
                "source": str(
                    row.get("source", "")
                ).strip(),
                "notes": str(
                    row.get("notes", "")
                ).strip(),
            }
        )

    return questions


# =============================================================================
# Header
# =============================================================================

st.title("CQ-Genesis 🌱")

st.caption(
    "LLM-Assisted and Human-Guided Competency Question Generation from Structured Data and User Stories"
)

st.markdown(
    """
CQ-Genesis supports knowledge engineers in generating and refining Competency Questions from structured datasets, user stories, or their combination, while keeping humans in control of input formulation, model configuration, generation, and review.
"""
)


feature_1, feature_2, feature_3 = st.columns(
    3,
    gap="large",
)

with feature_1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-title">Multiple input configurations</div>
            <div class="feature-text">
                Generate Competency Questions from a structured dataset,
                user stories, or the combination of both sources.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feature_2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-title">Structured instruction prompting</div>
            <div class="feature-text">
                CQ patterns and formulation examples guide the model toward
                abstract, atomic, consistent, and non-redundant questions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feature_3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-title">Knowledge engineer control</div>
            <div class="feature-text">
                Generated questions remain editable candidates. The knowledge
                engineer can configure, inspect, revise, exclude, and export
                them according to the goals of the ontology project.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Sidebar configuration
# =============================================================================

with st.sidebar:
    st.header("Generation configuration")

    backend_ok, backend_message = backend_health()

    if backend_ok:
        st.success(
            backend_message,
            icon="✅",
        )
    else:
        st.warning(
            backend_message,
            icon="⏳",
        )

    st.divider()

    input_mode_label = st.radio(
        "Input configuration",
        [
            "Structured Dataset",
            "User Stories",
            "Structured Dataset + User Stories",
        ],
        index=2,
        help=(
            "Choose which requirement sources should be used "
            "to generate the Competency Questions."
        ),
    )

    input_mode_mapping = {
        "Structured Dataset": "dataset_only",
        "User Stories": "user_stories_only",
        "Structured Dataset + User Stories": "multi_source",
    }

    input_mode = input_mode_mapping[
        input_mode_label
    ]

    st.divider()

    provider_label = st.selectbox(
        "LLM provider",
        [
            "OpenAI",
            "Anthropic",
            "Google",
            "Hugging Face",
        ],
    )

    if provider_label == "OpenAI":
        provider = "openai"
    
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
        )
    
        model = st.text_input(
            "Model",
            value="gpt-4o-mini",
        )
    
        base_url = ""

    elif provider_label == "Anthropic":
        provider = "anthropic"
    
        api_key = st.text_input(
            "Anthropic API key",
            type="password",
        )
    
        model = st.text_input(
            "Model",
            value="claude-haiku-4-5",
            help=(
                "Claude Haiku is suggested as a fast and "
                "cost-conscious default. The knowledge engineer "
                "can enter another model available to their "
                "Anthropic account."
            ),
        )
    
        base_url = ""
    
    elif provider_label == "Google":
        provider = "gemini"
    
        api_key = st.text_input(
            "Google AI Studio API key",
            type="password",
        )
    
        model = st.text_input(
            "Model",
            value="gemini-3.1-flash-lite",
            help=(
                "Gemini 3.1 Flash Lite is used as the default Google model. "
                "The knowledge engineer can enter another compatible "
                "model identifier."
            ),
        )
    
        base_url = ""
    
    else:
        provider = "huggingface"
    
        api_key = st.text_input(
            "Hugging Face User Access Token",
            type="password",
        )
    
        model = st.text_input(
            "Open-weight model",
            value="",
            placeholder="Example: openai/gpt-oss-20b",
        )
    
        base_url = (
            "https://router.huggingface.co/v1"
        )


    st.divider()

    count_mode_label = st.radio(
        "Number of CQs",
        [
            "Automatic conceptual coverage",
            "Fixed number",
        ],
        help=(
            "Automatic mode asks the model to cover all distinct "
            "information needs supported by the sources. "
            "Fixed mode is useful for controlled experiments."
        ),
    )

    if count_mode_label == "Fixed number":
        count_mode = "fixed"

        fixed_number = int(
            st.number_input(
                "Exact number of CQs",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
            )
        )

    else:
        count_mode = "automatic"
        fixed_number = None

    language_choice = st.selectbox(
        "Output language",
        [
            "Same language as the input",
            "English",
            "Italian",
            "Spanish",
            "French",
            "German",
            "Portuguese",
            "Custom",
        ],
        help=(
            "Language availability depends on the selected LLM. "
            "Technical support does not imply equal multilingual quality."
        ),
    )

    if language_choice == "Custom":
        language = st.text_input(
            "Custom language",
            placeholder="Example: Dutch",
        ).strip()

    elif language_choice == "Same language as the input":
        language = "same_as_input"

    else:
        language = language_choice

    with st.expander(
        "Knowledge Engineer Preferences"
    ):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help=(
                "A low default value promotes stable and "
                "instruction-consistent outputs while retaining "
                "limited linguistic flexibility."
            ),
        )

        max_output_tokens = st.slider(
            "Maximum output tokens",
            min_value=1000,
            max_value=16000,
            value=5000,
            step=500,
            help=(
                "This limits the maximum response length. "
                "It does not directly specify how many CQs "
                "the model should generate."
            ),
        )

        sample_rows = st.slider(
            "Representative Dataset Rows",
            min_value=0,
            max_value=50,
            value=10,
            step=1,
            help=(
                "A deterministic sample complements the structural "
                "profile without sending the entire dataset. "
                "This reduces context size, cost, latency, "
                "instance-level overfitting, and privacy exposure."
            ),
        )


# =============================================================================
# Tabs
# =============================================================================

(
    tab_sources,
    tab_profile,
    tab_generate,
    tab_results,
    tab_record,
) = st.tabs(
    [
        "1. Sources",
        "2. Data Review",
        "3. Generate",
        "4. Results",
        "5. Generation Record",
    ]
)


uploaded_file = None
dataset_context = ""
user_stories = ""


# =============================================================================
# Sources tab
# =============================================================================

with tab_sources:
    st.header("Requirement Sources")

    source_left, source_right = st.columns(
        2,
        gap="large",
    )

    with source_left:
        st.subheader("Structured Dataset")

        if input_mode in {
            "dataset_only",
            "multi_source",
        }:
            uploaded_file = st.file_uploader(
                "Upload a structured dataset",
                type=[
                    "csv",
                    "tsv",
                    "xlsx",
                    "json",
                    "xml",
                ],
                help=(
                    "Supported formats: CSV, TSV, XLSX, JSON and XML."
                ),
            )

            dataset_template_col, dataset_clear_col = st.columns(2)
            
            with dataset_template_col:
                if st.button(
                    "Use suggested dataset template",
                    use_container_width=True,
                ):
                    st.session_state.dataset_context_text = (
                        DATASET_CONTEXT_TEMPLATE
                    )
            
            with dataset_clear_col:
                if st.button(
                    "Clear dataset description",
                    use_container_width=True,
                ):
                    st.session_state.dataset_context_text = ""
            
            dataset_context = st.text_area(
                "Dataset context and documentation",
                key="dataset_context_text",
                placeholder=(
                    "Example: The dataset describes observations of "
                    "infectious diseases across territories and periods. "
                    "Each row represents a reported observation."
                ),
                height=220,
                help=(
                    "You may use the suggested structure or provide a "
                    "completely free-form description."
                ),
            )
            
            st.caption(
                "The suggested structure is optional. It helps provide "
                "semantic context but does not constrain how the dataset "
                "must be described."
            )

        else:
            st.info(
                "Dataset input is not required in "
                "User stories mode."
            )

    with source_right:
        st.subheader("User Story")

        if input_mode in {
            "user_stories_only",
            "multi_source",
        }:
            story_template_col, story_clear_col = st.columns(2)

            with story_template_col:
                if st.button(
                    "Use suggested user-story template",
                    use_container_width=True,
                ):
                    st.session_state.user_stories_text = (
                        USER_STORY_TEMPLATE
                    )
            
            with story_clear_col:
                if st.button(
                    "Clear user stories",
                    use_container_width=True,
                ):
                    st.session_state.user_stories_text = ""
            
            user_stories = st.text_area(
                "Enter user stories or requirement statements",
                key="user_stories_text",
                placeholder=(
                    "Example: A researcher needs to compare observations "
                    "across locations and periods in order to identify "
                    "relevant trends."
                ),
                height=220,
                help=(
                    "You may use Persona, Goal, and Scenario as optional "
                    "headings, or provide a free-form requirement description."
                ),
            )
            
            st.caption(
                "The suggested structure is optional. CQ-Genesis also "
                "accepts free-form user stories and requirement statements."
            )

        else:
            st.info(
                "User Stories are not required in "
                "Structured Dataset mode."
            )

    if input_mode == "multi_source":
        st.info(
            "In multi-source mode, user stories represent stakeholder "
            "goals, while the dataset represents the information "
            "currently available. The two sources are complementary."
        )


# =============================================================================
# Dataset profiling
# =============================================================================

if (
    uploaded_file is not None
    and input_mode in {
        "dataset_only",
        "multi_source",
    }
):
    profile_signature = (
        uploaded_file.name,
        len(uploaded_file.getvalue()),
        sample_rows,
    )

    previous_signature = st.session_state.get(
        "profile_signature"
    )

    if previous_signature != profile_signature:
        try:
            with st.spinner(
                "Analysing the dataset..."
            ):
                st.session_state.dataset_profile = (
                    profile_dataset(
                        uploaded_file,
                        sample_rows,
                    )
                )

            st.session_state.profile_signature = (
                profile_signature
            )

        except Exception as exc:
            st.session_state.dataset_profile = None
            st.session_state.last_error = str(exc)


# =============================================================================
# Data review tab
# =============================================================================

with tab_profile:
    st.header("Dataset Review")

    if input_mode == "user_stories_only":
        st.info(
            "Dataset review is not required for the selected "
            "input configuration."
        )

    elif uploaded_file is None:
        st.warning(
            "Upload a dataset in the Sources tab."
        )

    elif st.session_state.dataset_profile is None:
        if st.session_state.last_error:
            st.error(
                st.session_state.last_error
            )
        else:
            st.info(
                "The dataset profile is being prepared."
            )

    else:
        profile = (
            st.session_state.dataset_profile
        )
    
        st.subheader("Dataset Preview")
    
        st.caption(
            "A selection of dataset rows is shown without additional "
            "interpretation, allowing users to verify that the uploaded "
            "file has been read correctly."
        )
    
        if profile.get("sample"):
            st.dataframe(
                pd.DataFrame(
                    profile["sample"]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No preview rows are available. Increase the number of "
                "representative dataset rows under Knowledge Engineer preferences."
            )
    
        st.subheader("Dataset summary")
    
        metric_1, metric_2, metric_3 = st.columns(3)

        metric_1.metric(
            "Rows",
            f"{profile['rows']:,}",
            help=(
                "Number of observations available in the dataset."
            ),
        )

        metric_2.metric(
            "Columns",
            f"{profile['columns']:,}",
            help=(
                "Number of fields used to describe each observation."
            ),
        )

        metric_3.metric(
            "Missing cells",
            f"{profile['missing_cells']:,}",
            help=(
                "Total number of cells without a recorded value."
            ),
        )

        st.subheader("Column Profile")

        profile_dataframe = pd.DataFrame(
            profile["column_profiles"]
        )

        profile_dataframe = profile_dataframe.rename(
            columns={
                "name": "Field",
                "data_type": "Data type",
                "suggested_role": "Suggested structural role",
                "non_missing_values": "Non-missing values",
                "missing_percentage": "Missing values (%)",
                "unique_values": "Unique values",
                "example_values": "Example values",
            }
        )

        if "Example values" in profile_dataframe.columns:
            profile_dataframe["Example values"] = (
                profile_dataframe["Example values"].apply(
                    lambda values: " | ".join(values)
                    if isinstance(values, list)
                    else str(values)
                )
            )

        st.dataframe(
            profile_dataframe,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Suggested structural role": (
                    st.column_config.TextColumn(
                        help=(
                            "A simple suggestion based on column names, "
                            "data types, and cardinality. It is not "
                            "treated as an ontological classification."
                        )
                    )
                ),
                "Missing values (%)": (
                    st.column_config.NumberColumn(
                        help=(
                            "Percentage of rows with no recorded value "
                            "for the field."
                        ),
                        format="%.2f",
                    )
                ),
                "Unique values": (
                    st.column_config.NumberColumn(
                        help=(
                            "Number of different non-empty values "
                            "observed in the field."
                        )
                    )
                ),
            },
        )

        with st.expander(
            "Why does CQ-Genesis use a profile and a sample?"
        ):
            st.markdown(
                """
                Sending the entire dataset is often unnecessary or
                technically impractical. CQ generation primarily requires
                evidence about the dataset structure, its dimensions,
                measures, categories, and representative values.

                CQ-Genesis therefore combines:

                - a **structural profile**, describing fields, data types,
                  missingness, cardinality, and numeric ranges;
                - a **deterministic representative sample**, helping the
                  model interpret ambiguous field names and values.

                This strategy reduces token usage, latency, cost, privacy
                exposure, and the risk that the model formulates questions
                around accidental individual records instead of domain-level
                concepts.
                """
            )

        with st.expander(
            "Textual Profile sent to the model"
        ):
            st.text(
                profile["textual_profile"]
            )

        with st.expander(
            "Representative sample sent to the model"
        ):
            if profile["sample"]:
                st.dataframe(
                    pd.DataFrame(
                        profile["sample"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "No sample rows are currently included."
                )


# =============================================================================
# Generate tab
# =============================================================================

with tab_generate:
    st.header("Generate Competency Questions")

    card_1, card_2, card_3 = st.columns(
        3,
        gap="large",
    )

    with card_1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">
                    Structured instruction prompting
                </div>
                <div class="feature-text">
                    The prompt combines explicit task instructions, conceptual guidance, optional CQ patterns, quality constraints, formulation examples, and structured output specifications.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card_2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">
                    CQ quality guidance
                </div>
                <div class="feature-text">
                    The model is instructed to preserve source relevance,
                    appropriate abstraction, atomicity, clarity,
                    terminological consistency, conceptual diversity,
                    and coverage.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card_3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">
                    Human-in-the-loop control
                </div>
                <div class="feature-text">
                    The output is treated as a set of candidate
                    ontology requirements. The user remains responsible
                    for reviewing, editing, excluding, and exporting CQs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    readiness_errors = []

    if not backend_ok:
        readiness_errors.append(
            "Start the CQ-Genesis backend."
        )

    if not model.strip():
        readiness_errors.append(
            "Enter a model identifier."
        )

    if not api_key.strip():
        readiness_errors.append(
            "Enter the provider API key."
        )

    if (
        input_mode in {
            "dataset_only",
            "multi_source",
        }
        and uploaded_file is None
    ):
        readiness_errors.append(
            "Upload a structured dataset."
        )

    if (
        input_mode in {
            "user_stories_only",
            "multi_source",
        }
        and not user_stories.strip()
    ):
        readiness_errors.append(
            "Enter at least one user story "
            "or requirement statement."
        )

    if (
        language_choice == "Custom"
        and not language.strip()
    ):
        readiness_errors.append(
            "Enter the custom output language."
        )

    if readiness_errors:
        st.warning(
            "Before generating:\n\n- "
            + "\n- ".join(
                readiness_errors
            )
        )

    generate_clicked = st.button(
        "Generate Competency Questions",
        type="primary",
        use_container_width=True,
        disabled=bool(readiness_errors),
    )

    if generate_clicked:
        try:
            with st.spinner(
                "Generating Competency Questions..."
            ):
                generation_response = generate_cqs(
                    uploaded_file=uploaded_file,
                    input_mode=input_mode,
                    provider=provider,
                    model=model.strip(),
                    api_key=api_key.strip(),
                    base_url=base_url,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    language=language,
                    count_mode=count_mode,
                    fixed_number=fixed_number,
                    sample_rows=sample_rows,
                    dataset_context=dataset_context,
                    user_stories=user_stories,
                )

            st.session_state.generation_response = (
                generation_response
            )

            st.session_state.pop(
                "cq_editor",
                None,
            )

            number_of_questions = len(
                generation_response.get(
                    "questions",
                    [],
                )
            )

            st.success(
                f"{number_of_questions} Competency Questions "
                "were generated. Open the Results tab to review them."
            )

        except Exception as exc:
            st.error(
                str(exc)
            )


# =============================================================================
# Results tab
# =============================================================================

with tab_results:
    st.header("Generated Competency Questions")

    generation_response = (
        st.session_state.generation_response
    )

    if not generation_response:
        st.warning(
            "No generation results are available yet."
        )

    else:
        original_questions = (
            generation_response.get(
                "questions",
                [],
            )
        )

        questions_dataframe = (
            questions_to_dataframe(
                original_questions
            )
        )

        metric_1, metric_2 = st.columns(2)

        metric_1.metric(
            "Generated CQs",
            len(questions_dataframe),
        )

        metric_2.metric(
            "Input-source labels",
            (
                questions_dataframe["source"]
                .replace("", pd.NA)
                .dropna()
                .nunique()
            ),
        )

        st.subheader("Review and Edit")

        st.caption(
            "The JSON returned by the model is used internally. "
            "The primary user-facing output is the editable list below."
        )

        edited_dataframe = st.data_editor(
            questions_dataframe,
            key="cq_editor",
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_order=[
                "include",
                "id",
                "question",
                "pattern",
                "source",
                "notes",
            ],
            column_config={
                "include": (
                    st.column_config.CheckboxColumn(
                        "Include",
                        default=True,
                        width="small",
                    )
                ),
                "id": (
                    st.column_config.TextColumn(
                        "ID",
                        disabled=True,
                        width="small",
                    )
                ),
                "question": (
                    st.column_config.TextColumn(
                        "Competency Question",
                        required=True,
                        width="large",
                    )
                ),
                "pattern": (
                    st.column_config.TextColumn(
                        "CQ pattern",
                        width="large",
                        help=(
                            "The linguistic CQ pattern used as guidance, "
                            "or Free-form when no listed pattern applies."
                        ),
                    )
                ),
                "source": (
                    st.column_config.SelectboxColumn(
                        "Source",
                        options=[
                            "dataset",
                            "user_stories",
                            "both",
                        ],
                        width="medium",
                    )
                ),
                "notes": (
                    st.column_config.TextColumn(
                        "Explanation",
                        width="large",
                    )
                ),
            },
        )

        final_questions = (
            edited_questions_from_dataframe(
                edited_dataframe
            )
        )

        st.subheader("Final CQ list")

        if not final_questions:
            st.info(
                "No Competency Questions are currently selected."
            )

        else:
            for item in final_questions:
                with st.container(
                    border=True
                ):
                    st.markdown(
                        f"**{item['id']}. "
                        f"{item['question']}**"
                    )

                    secondary_details = []

                    if item["source"]:
                        secondary_details.append(
                            f"Source: {item['source']}"
                        )

                    if secondary_details:
                        st.caption(
                            " · ".join(
                                secondary_details
                            )
                        )

        metadata = generation_response.get(
            "metadata",
            {},
        )

        export_payload = {
            "metadata": metadata,
            "questions": final_questions,
        }

        txt_output = "\n".join(
            f"{item['id']}. {item['question']}"
            for item in final_questions
        )

        csv_output = pd.DataFrame(
            final_questions
        ).to_csv(
            index=False
        )

        json_output = json.dumps(
            export_payload,
            indent=2,
            ensure_ascii=False,
        )

        download_1, download_2, download_3 = st.columns(3)

        download_1.download_button(
            "Download CQ list",
            data=txt_output,
            file_name="cq-genesis-cqs.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=not final_questions,
        )

        download_2.download_button(
            "Download CSV",
            data=csv_output,
            file_name="cq-genesis-cqs.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=not final_questions,
        )

        download_3.download_button(
            "Download complete JSON",
            data=json_output,
            file_name="cq-genesis-generation.json",
            mime="application/json",
            use_container_width=True,
            disabled=not final_questions,
        )


# =============================================================================
# Generation record tab
# =============================================================================

with tab_record:
    st.header("Generation Record")

    st.markdown(
        """
        The Generation Record makes explicit the process through which
        the current Competency Questions were created.

        It documents both the technical configuration and the choices
        made by the knowledge engineer, including the selected requirement
        sources, LLM provider and model, generation parameters, output
        language, requested coverage, and representative dataset sample.

        These choices may reflect the knowledge engineer's experience,
        intentions, priorities, and interpretation of the knowledge
        engineering task. Recording them therefore supports:

        - transparency of the CQ creation process;
        - explicit documentation of the knowledge engineer's decisions;
        - reproducibility and comparison across generation configurations;
        - critical review of how human choices influenced the resulting CQs.
        """
    )

    generation_response = (
        st.session_state.generation_response
    )

    if not generation_response:
        st.info(
            "The Generation Record will appear after the first successful "
            "run and will document the configuration choices that shaped "
            "the creation of the Competency Questions."
        )

    else:
        metadata = generation_response.get(
            "metadata",
            {},
        )

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)

        metric_1.metric(
            "Prompt specification",
            metadata.get(
                "prompt_version",
                "",
            ),
        )

        metric_2.metric(
            "Provider",
            metadata.get(
                "provider",
                "",
            ),
        )

        metric_3.metric(
            "Model",
            metadata.get(
                "model",
                "",
            ),
        )

        metric_4.metric(
            "Generation time",
            (
                f"{metadata.get('generation_time_seconds', 0)} s"
            ),
        )

        st.info(
            "Prompting strategy: "
            + metadata.get(
                "prompting_strategy",
                "Structured instruction prompting",
            )
        )

        st.subheader("Knowledge Engineer Configuration")

        st.caption(
            "This configuration records the explicit choices that shaped "
            "the generation process. It should be interpreted as part of "
            "the provenance of the resulting Competency Questions, rather "
            "than as a purely technical execution log."
        )

        st.json(
            metadata
        )

        validation_notes = metadata.get(
            "validation_notes",
            [],
        )

        if validation_notes:
            with st.expander(
                f"Validation notes ({len(validation_notes)})"
            ):
                for note in validation_notes:
                    st.write(
                        f"- {note}"
                    )

        with st.expander(
            "Raw Model Output"
        ):
            st.code(
                generation_response.get(
                    "raw_output",
                    "",
                ),
                language="json",
            )

        record_json = json.dumps(
            generation_response,
            indent=2,
            ensure_ascii=False,
        )

        st.download_button(
            "Download Generation Record",
            data=record_json,
            file_name="cq-genesis-generation-record.json",
            mime="application/json",
        )

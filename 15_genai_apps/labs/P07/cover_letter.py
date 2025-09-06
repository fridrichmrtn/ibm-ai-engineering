"""Gradio app to generate tailored cover letters using OpenAI chat completions."""

import os
import logging
from functools import lru_cache
from typing import List, Optional, Literal, TypedDict
from dataclasses import dataclass

import gradio as gr
from openai import OpenAI
from openai import (
    APIError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
)

# -------------------- Types --------------------


class ChatMessage(TypedDict):
    """Minimal chat message structure for OpenAI chat completions."""
    role: Literal["user", "assistant", "system"]
    content: str


# -------------------- Logger --------------------

LOGGER = logging.getLogger("cover_letter_app")


# -------------------- Validators --------------------


def _validate_temperature(t: float) -> float:
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 0.4
    return max(0.0, min(2.0, t))


def _validate_top_p(tp: float) -> float:
    try:
        tp = float(tp)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, tp))


def _validate_max_tokens(m: Optional[int]) -> Optional[int]:
    if m is None:
        return None
    try:
        m = int(m)
    except (TypeError, ValueError):
        return 1024
    return max(1, min(4096, m))


def _build_messages(system_prompt: str, user_prompt: str) -> List[ChatMessage]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# -------------------- OpenAI client management --------------------


def _ensure_api_key() -> bool:
    """
    Ensure OPENAI_API_KEY is present.
    If missing, try reading from local 'openai_token' file.
    """
    if os.getenv("OPENAI_API_KEY"):
        return True

    token_path = "openai_token"
    if not os.path.exists(token_path):
        LOGGER.debug(
            "OPENAI_API_KEY not set and '%s' file not found.",
            token_path,
        )
        return False

    try:
        with open(token_path, "r", encoding="utf-8") as f:
            api_token = f.read().strip()
        if api_token:
            os.environ["OPENAI_API_KEY"] = api_token
            return True
        LOGGER.warning("Token file '%s' is empty.", token_path)
        return False
    except (OSError, UnicodeError) as exc:
        LOGGER.exception("Failed reading '%s': %s", token_path, exc)
        return False


@lru_cache(maxsize=1)
def _get_client() -> Optional[OpenAI]:
    if not _ensure_api_key():
        LOGGER.error("Missing OpenAI API key.")
        return None

    timeout = float(os.getenv("OPENAI_TIMEOUT", "30"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    base_url = os.getenv("OPENAI_BASE_URL")  # supports Azure/proxies
    organization = os.getenv("OPENAI_ORGANIZATION")
    project = os.getenv("OPENAI_PROJECT")

    kwargs = {
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if organization:
        kwargs["organization"] = organization
    if project:
        kwargs["project"] = project

    try:
        client = OpenAI(**kwargs)
        return client
    except (TypeError, ValueError) as exc:
        LOGGER.exception("Failed to initialize OpenAI client: %s", exc)
        return None


def _get_model_choices() -> List[str]:
    """
    Populate model choices dynamically if possible; otherwise fallback to a
    sane static list.
    """
    client = _get_client()
    fallback = [
        os.getenv("OPENAI_MODEL", "gpt-4o"),
        "gpt-4o-mini",
        "o4-mini",
        "gpt-4.1-mini",
    ]
    dedup_fallback = list(dict.fromkeys([m for m in fallback if m]))
    if client is None:
        return dedup_fallback

    try:
        resp = client.models.list()
        ids = sorted(
            {m.id for m in getattr(resp, "data", []) if getattr(m, "id", "")}
        )
        preferred_prefixes = ("gpt-4o", "o4", "gpt-4.1")
        preferred = [m for m in ids if m.startswith(preferred_prefixes)]
        others = [m for m in ids if m not in preferred]
        ordered = preferred + others
        return ordered[:50] or dedup_fallback
    except (
        AuthenticationError,
        RateLimitError,
        APIConnectionError,
        BadRequestError,
        APIError,
        AttributeError,
        TypeError,
    ) as exc:
        LOGGER.warning("Could not fetch model list, using fallback. Reason: %s", exc)
        return dedup_fallback


# -------------------- Model invocation helpers --------------------


@dataclass(frozen=True)
class ChatConfig:
    """Configuration options for a chat completion request."""
    model: str
    temperature: float
    top_p: float
    max_tokens: Optional[int]
    seed: Optional[int]


def _load_chat_config(model: str) -> ChatConfig:
    """Load chat configuration from environment with validation."""
    temperature = _validate_temperature(
        float(os.getenv("OPENAI_TEMPERATURE", "0.4"))
    )
    top_p = _validate_top_p(float(os.getenv("OPENAI_TOP_P", "1.0")))
    max_tokens_env = os.getenv("OPENAI_MAX_TOKENS", "1024").strip()
    max_tokens = (
        None
        if max_tokens_env.lower() in ("", "none", "auto")
        else _validate_max_tokens(int(max_tokens_env))
    )
    seed_env = os.getenv("OPENAI_SEED", "").strip()
    seed = int(seed_env) if seed_env.isdigit() else None
    effective_model = (model or os.getenv("OPENAI_MODEL", "gpt-4o")).strip()
    return ChatConfig(
        model=effective_model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )


# -------------------- Model invocation --------------------


def _chat_completion(
    messages: List[ChatMessage],
    model: str,
) -> str:
    """
    Call OpenAI chat completions using environment-tuned parameters.
    """
    client = _get_client()
    if client is None:
        return (
            "Missing OpenAI credentials. Set OPENAI_API_KEY or create a file named "
            "'openai_token' containing your key."
        )

    error_msg: Optional[str] = None
    content: Optional[str] = None

    try:
        cfg = _load_chat_config(model)
        kwargs = {
            "model": cfg.model,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "messages": messages,
        }
        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.seed is not None:
            kwargs["seed"] = cfg.seed

        completion = client.chat.completions.create(**kwargs)
        if completion and completion.choices and completion.choices[0].message:
            content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("Empty response from model.")

        LOGGER.debug("Completion id: %s", getattr(completion, "id", "n/a"))

    except AuthenticationError:
        error_msg = "Authentication failed. Check OPENAI_API_KEY."
    except RateLimitError:
        error_msg = "Rate limit exceeded. Try again later or lower the request rate."
    except APIConnectionError:
        error_msg = "Network error reaching OpenAI. Check connectivity or BASE_URL."
    except BadRequestError as exc:
        error_msg = f"Bad request: {exc}"
    except APIError as exc:
        error_msg = f"API error: {exc}"
    except (RuntimeError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        LOGGER.exception("Unexpected error from OpenAI: %s", exc)
        error_msg = f"Unexpected error: {exc}"

    return content if content else (error_msg or "Unknown error occurred.")


# -------------------- App logic --------------------


def generate_cover_letter(
    company_name: str,
    position_name: str,
    job_description: str,
    resume_content: str,
    model: str,
) -> str:
    """
    Generate a customized cover letter aligned to the job description
    while grounded strictly in the candidate's resume content.
    """
    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        (
            "You are an expert cover-letter writer. Produce a concise, professional "
            "letter tailored to the target role and company. Maintain factual accuracy—"
            "do not invent experiences or credentials not present in the resume. "
            "Emphasize relevant achievements, quantified impact, and cultural/role fit. "
            "Avoid sensitive PII (e.g., full address, SSN, birthdate). Keep it 250–400 words."
        ),
    )

    # Build a clear user prompt with grounding + structure
    user_prompt = (
        f"Company: {company_name}\n"
        f"Position: {position_name}\n\n"
        f"Job description:\n\"\"\"\n{job_description}\n\"\"\"\n\n"
        "Resume content (authoritative source—do not add facts not in this section):\n"
        f"\"\"\"\n{resume_content}\n\"\"\"\n\n"
        "Write a tailored cover letter that:\n"
        "1) Directly aligns resume evidence to the job's top requirements,\n"
        "2) Highlights 2–3 quantified, relevant achievements,\n"
        "3) Uses crisp, confident, natural language (no hype or clichés),\n"
        "4) Includes a brief closing call-to-action,\n"
        "5) Stays within 250–400 words.\n"
        "Return only the letter text (no headings beyond standard letter formatting)."
    )

    messages = _build_messages(system_prompt, user_prompt)
    response = _chat_completion(messages=messages, model=model)
    return response.strip()


# -------------------- UI --------------------


def _build_interface() -> gr.Interface:
    model_choices = _get_model_choices()
    default_model = os.getenv(
        "OPENAI_MODEL",
        model_choices[0] if model_choices else "gpt-4o",
    )
    description = (
        "Generate a tailored cover letter by entering the company, position, job "
        "description, and your resume content. The model will not invent experience "
        "not present in your resume."
    )

    iface = gr.Interface(
        fn=generate_cover_letter,
        flagging_mode="never",
        inputs=[
            gr.Textbox(label="Company Name", placeholder="Enter the company name..."),
            gr.Textbox(label="Position Name", placeholder="Enter the position title..."),
            gr.Textbox(
                label="Job Description",
                placeholder="Paste the job description here...",
                lines=10,
            ),
            gr.Textbox(
                label="Resume Content",
                placeholder="Paste your resume content here...",
                lines=14,
            ),
            gr.Dropdown(
                label="Model",
                choices=model_choices or [default_model],
                value=default_model,
                allow_custom_value=True,
            ),
        ],
        outputs=gr.Textbox(label="Customized Cover Letter"),
        title="Customized Cover Letter Generator",
        description=description,
        examples=[
            [
                "NVIDIA",
                "Senior Data Scientist",
                "We are seeking a data scientist with experience in ML pipelines, "
                "model deployment, and stakeholder communication...",
                "Built end-to-end ML pipelines in Python; deployed models to production; "
                "led attribution modeling project with measurable revenue uplift...",
                default_model,
            ],
            [
                "Databricks",
                "AI Engineer",
                "Looking for engineers with experience in RAG systems, LLM evaluation, "
                "prompt ops, and secure deployment of AI services...",
                "Implemented RAG pipeline with vector store; built evaluation harness; "
                "containerized and deployed via CI/CD; collaborated with platform team...",
                default_model,
            ],
        ],
    )
    return iface


# -------------------- Main --------------------

if __name__ == "__main__":
    # .env (optional)
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError as exc:
        print(f"[DEBUG] python-dotenv not available: {exc}")
    else:
        try:
            load_dotenv()
        except OSError as exc:
            print(f"[DEBUG] .env load failed: {exc}")

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    LOGGER.setLevel(LOG_LEVEL)

    # Ensure client early (logs helpful errors but UI will still load)
    _get_client()

    # Server config
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "8000"))
    share = os.getenv("GRADIO_SHARE", "false").lower() in ("1", "true", "yes")

    app = _build_interface().queue()
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )

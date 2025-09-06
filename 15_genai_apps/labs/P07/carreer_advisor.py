"""Gradio app to generate actionable career advice using OpenAI chat completions."""

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

LOGGER = logging.getLogger("career_advice_app")


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
    except (
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
    ) as exc:
        LOGGER.exception("Unexpected error from OpenAI: %s", exc)
        error_msg = f"Unexpected error: {exc}"

    return content if content else (error_msg or "Unknown error occurred.")


# -------------------- App logic --------------------


def generate_career_advice(
    position_applied: str,
    job_description: str,
    resume_content: str,
    model: str,
) -> str:
    """
    Provide targeted resume enhancement advice for a specific position.
    Outputs prioritized gaps, concrete fixes, and sample bullet rewrites
    grounded in the resume.
    """
    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        (
            "You are a meticulous career advisor and resume coach. "
            "Analyze a job description vs. a candidate's resume and provide: "
            "1) prioritized gap analysis (skills, tools, outcomes), "
            "2) specific, ATS-friendly fixes (keywords, phrasing), "
            "3) 2–4 sample bullet rewrites using STAR patterns and quantified "
            "impact. Never invent experience or credentials that are not present "
            "in the resume. Avoid sensitive PII. Keep the response concise and "
            "actionable."
        ),
    )

    user_prompt = (
        f"Target position: {position_applied}\n\n"
        f"Job description:\n\"\"\"\n{job_description}\n\"\"\"\n\n"
        "Resume content (authoritative—do not add facts not present here):\n"
        f"\"\"\"\n{resume_content}\n\"\"\"\n\n"
        "Deliver:\n"
        "A) Top 5 Gaps (ordered, cite JD phrases where relevant)\n"
        "B) Quick Wins (ATS keywords/phrases to add naturally to existing bullets)\n"
        "C) Bullet Rewrites (2–4) using STAR, with metrics if present in resume\n"
        "D) Optional Section Tweaks (e.g., skills section ordering, summary tune-up)\n"
        "Keep total output under ~350 words."
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
        "Paste the job description and your resume. Get prioritized gaps, "
        "ATS keyword suggestions, and concrete bullet rewrites aligned to "
        "a specific role."
    )

    iface = gr.Interface(
        fn=generate_career_advice,
        flagging_mode="never",
        inputs=[
            gr.Textbox(
                label="Position Applied For",
                placeholder="e.g., Senior Data Scientist",
            ),
            gr.Textbox(
                label="Job Description",
                placeholder="Paste the JD here...",
                lines=10,
            ),
            gr.Textbox(
                label="Your Resume Content",
                placeholder="Paste your resume here...",
                lines=14,
            ),
            gr.Dropdown(
                label="Model",
                choices=model_choices or [default_model],
                value=default_model,
                allow_custom_value=True,
            ),
        ],
        outputs=gr.Textbox(label="Career Advice"),
        title="Career Advisor",
        description=description,
        examples=[
            [
                "AI Engineer",
                (
                    "Seeking experience in RAG, LLMOps, evaluation frameworks, "
                    "and secure API deployment."
                ),
                (
                    "Built RAG with vector store; added eval harness for factuality; "
                    "deployed via Docker + CI/CD; Python, FastAPI, Snowflake, "
                    "basic Terraform."
                ),
                default_model,
            ],
            [
                "Data Analyst",
                (
                    "SQL, dashboards, stakeholder storytelling, "
                    "experimentation a plus."
                ),
                (
                    "Advanced SQL; built KPI dashboards; ran simple A/B tests; "
                    "presented findings to marketing."
                ),
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

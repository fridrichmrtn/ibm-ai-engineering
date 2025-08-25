"""Speech analyzer: Whisper ASR → Llama-3 summarization with a Gradio UI."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, Optional

import gradio as gr
import torch
from langchain.chains.llm import LLMChain  # Keep this import to satisfy pylint in your env
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# -----------------------------------------------------------------------------
# Configuration (via env; sane defaults)
# -----------------------------------------------------------------------------
ASR_MODEL_ID: str = os.getenv("ASR_MODEL_ID", "openai/whisper-large-v3")
LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
LLM_MAX_NEW_TOKENS: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_REPETITION_PENALTY: float = float(os.getenv("LLM_REPETITION_PENALTY", "1.05"))
ASR_CHUNK_SEC: int = int(os.getenv("ASR_CHUNK_SEC", "30"))
WHISPER_LANGUAGE: Optional[str] = os.getenv("WHISPER_LANGUAGE")  # e.g., "en", "cs", or None for auto
SEED: int = int(os.getenv("SEED", "42"))
SERVER_PORT: int = int(os.getenv("SERVER_PORT", "7860"))
SERVER_NAME: str = os.getenv("SERVER_NAME", "0.0.0.0")

HF_TOKEN_PATH: str = os.getenv("HF_TOKEN_PATH", "hf_token")

# Logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("speech_analyzer")

# Make HF token available if present (without failing on missing file)
if os.path.exists(HF_TOKEN_PATH):
    with open(HF_TOKEN_PATH, "r", encoding="utf-8") as _f:
        token = _f.read().strip()
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)
            LOGGER.info("Loaded Hugging Face token from %s", HF_TOKEN_PATH)

# -----------------------------------------------------------------------------
# Hardware-aware settings
# -----------------------------------------------------------------------------
_IS_CUDA: bool = torch.cuda.is_available()
_ASR_DTYPE = torch.float16 if _IS_CUDA else torch.float32
_LLM_DTYPE = torch.float16 if _IS_CUDA else torch.float32
_DEVICE: int = 0 if _IS_CUDA else -1

# For reproducibility on generation
torch.manual_seed(SEED)
if _IS_CUDA:
    torch.cuda.manual_seed_all(SEED)

# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------
SUMMARY_PROMPT_TEMPLATE: str = (
    "<s><<SYS>>\n"
    "Extract concise bullet-point key takeaways from the context. "
    "Prefer specifics (who/what/when) over generic statements.\n"
    "<</SYS>>\n"
    "[INST]\n"
    "Context:\n{context}\n"
    "[/INST]\n"
)

PROMPT = PromptTemplate(template=SUMMARY_PROMPT_TEMPLATE, input_variables=["context"])

# -----------------------------------------------------------------------------
# Pipelines (cached singletons)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _asr_pipeline():
    """Create a Whisper ASR pipeline once."""
    kwargs: Dict[str, Any] = {
        "task": "automatic-speech-recognition",
        "model": ASR_MODEL_ID,
        "device": _DEVICE,
        "torch_dtype": _ASR_DTYPE,
        "chunk_length_s": ASR_CHUNK_SEC,
    }
    # Optional language hint can improve accuracy/latency
    if WHISPER_LANGUAGE:
        kwargs["generate_kwargs"] = {"language": WHISPER_LANGUAGE}
    asr = pipeline(**kwargs)
    return asr


@lru_cache(maxsize=1)
def _hf_textgen_pipeline():
    """Create a text-generation pipeline (Llama-3) once."""
    return pipeline(
        task="text-generation",
        model=LLM_MODEL_ID,
        device=_DEVICE,
        torch_dtype=_LLM_DTYPE,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        do_sample=LLM_TEMPERATURE > 0.0,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        repetition_penalty=LLM_REPETITION_PENALTY,
        pad_token_id=128001,  # safe fallback for Llama-3 pad id
    )


@lru_cache(maxsize=1)
def _langchain_llm():
    """Wrap HF pipeline for LangChain."""
    return HuggingFacePipeline(pipeline=_hf_textgen_pipeline())


@lru_cache(maxsize=1)
def _summary_chain():
    """LLMChain that summarizes transcriptions into bullet points."""
    return LLMChain(llm=_langchain_llm(), prompt=PROMPT)


# -----------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------
def _run_asr(audio_file: str) -> str:
    """Run ASR on a file path and return transcription text."""
    try:
        result: Dict[str, Any] = _asr_pipeline()(audio_file)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("ASR failed: %s", exc)
        return ""
    return str(result.get("text", "")).strip()


def _summarize(text: str) -> str:
    """Summarize text with the LLM chain."""
    try:
        summary: str = _summary_chain().run(context=text).strip()
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("LLM summarization failed: %s", exc)
        return ""
    return summary


def transcribe_and_summarize(audio_file: Optional[str]) -> str:
    """Gradio callback: transcribe an uploaded audio file and summarize it."""
    if not audio_file:
        return "No audio file provided."
    transcription = _run_asr(audio_file)
    if not transcription:
        return "Transcription was empty or failed."
    summary = _summarize(transcription)
    return summary or "Summary was empty."


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def build_interface() -> gr.Interface:
    """Create the Gradio interface."""
    audio_input = gr.Audio(
        sources="upload",
        type="filepath",
        label="Upload audio",
        waveform_options={"waveform_color": None},  # relies on default theme
    )
    output_text = gr.Textbox(label="Summary", lines=14, show_copy_button=True)
    return gr.Interface(
        fn=transcribe_and_summarize,
        inputs=audio_input,
        outputs=output_text,
        title="Speech Analyzer (Whisper → Llama-3)",
        description=(
            "Upload an audio file. The app transcribes it with Whisper and summarizes key points with Llama-3."
        ),
        allow_flagging="never",
        analytics_enabled=False,
    )


if __name__ == "__main__":
    build_interface().launch(server_name=SERVER_NAME, server_port=SERVER_PORT)

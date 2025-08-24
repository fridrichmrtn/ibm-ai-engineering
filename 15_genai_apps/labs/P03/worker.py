"""Unified ASR/TTS worker using Hugging Face pipelines and an OpenAI chat wrapper.
- ASR: Whisper (fp16 on CUDA, fp32 on CPU)
- TTS: MMS-TTS (VITS; forced fp32 for numerical stability)
- OpenAI: simple chat helper
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Tuple, Union
import io
import os
import wave

import numpy as np
import torch
from transformers import pipeline
from openai import OpenAI  # required by the rest of the file

# ============================================================
# Device / precision
# ============================================================
_IS_CUDA = torch.cuda.is_available()
_DEVICE = torch.device("cuda:0" if _IS_CUDA else "cpu")
_ASR_DTYPE = torch.float16 if _IS_CUDA else torch.float32   # Whisper fast path
_TTS_DTYPE = torch.float32                                   # VITS stability

# ============================================================
# Hugging Face model IDs
# ============================================================
_ASR_MODEL_ID = "openai/whisper-large-v3"
_TTS_MODEL_ID = "facebook/mms-tts-eng"  # single-speaker VITS

# ============================================================
# Pipelines (cached singletons; no globals)
# ============================================================

@lru_cache(maxsize=1)
def _asr_pipeline():
    """Build (once) an ASR pipeline using Whisper."""
    return pipeline(
        task="automatic-speech-recognition",
        model=_ASR_MODEL_ID,
        device=0 if _IS_CUDA else -1,
        torch_dtype=_ASR_DTYPE,
    )


@lru_cache(maxsize=1)
def _tts_pipeline():
    """Build (once) a TTS pipeline using MMS-TTS (VITS, FP32)."""
    # Prefer "text-to-speech". If unavailable in this transformers version,
    # fall back to "text-to-audio" catching only likely constructor errors.
    try:
        return pipeline(
            task="text-to-speech",
            model=_TTS_MODEL_ID,
            device=0 if _IS_CUDA else -1,
            torch_dtype=_TTS_DTYPE,
        )
    except (TypeError, ValueError, OSError, RuntimeError):
        return pipeline(
            task="text-to-audio",
            model=_TTS_MODEL_ID,
            device=0 if _IS_CUDA else -1,
            torch_dtype=_TTS_DTYPE,
        )


# ============================================================
# Public API: ASR
# ============================================================

def speech_to_text(audio_binary: Any) -> str:
    """
    Transcribe audio to text via Whisper pipeline.
    Accepts bytes/np array/torch tensor/filename per HF pipeline semantics.
    """
    result = _asr_pipeline()(audio_binary)
    if isinstance(result, dict) and "text" in result:
        return result["text"]
    if isinstance(result, list):
        return " ".join(seg.get("text", "") for seg in result)
    return str(result)


# ============================================================
# WAV helpers
# ============================================================

def _float32_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """
    Convert mono float32 [-1, 1] waveform to in-memory 16-bit PCM WAV bytes.
    Uses stdlib wave + BytesIO (no external deps).
    """
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    wf = wave.open(buf, "wb")  # pylint: disable=consider-using-with
    try:
        getattr(wf, "setnchannels")(1)
        getattr(wf, "setsampwidth")(2)     # 16-bit PCM
        getattr(wf, "setframerate")(int(sr))
        getattr(wf, "writeframes")(pcm.tobytes())
    finally:
        wf.close()
    return buf.getvalue()


# ============================================================
# Public API: TTS
# ============================================================

def text_to_speech(
    text: str,
    voice: str = "",
    *,
    as_wav: bool = False,
    return_sr: bool = False,
) -> Union[np.ndarray, bytes, Tuple[np.ndarray, int], Tuple[bytes, int]]:
    """
    Synthesize speech via MMS-TTS pipeline.

    Default: returns mono float32 NumPy array (backward compatible).
    If `as_wav=True`: returns in-memory WAV bytes (16-bit PCM).
    `return_sr=True` returns a tuple (..., sampling_rate).
    Note: MMS-TTS has a single built-in voice; `voice` is unused.
    """
    # Intentionally unused parameter retained for API compatibility.
    _ = voice  # prevents pylint unused-argument warning

    if not text or not text.strip():
        empty = np.zeros(0, dtype=np.float32)
        if as_wav:
            # Use a conventional SR for empty output; consumers typically ignore it.
            sr_empty = 22050
            wav = _float32_to_wav_bytes(empty, sr_empty)
            return (wav, sr_empty) if return_sr else wav
        return (empty, 22050) if return_sr else empty

    out = _tts_pipeline()(text)  # {'audio': np.ndarray, 'sampling_rate': int}
    audio = out["audio"]
    sr = int(out["sampling_rate"])

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if as_wav:
        wav_bytes = _float32_to_wav_bytes(audio, sr)
        return (wav_bytes, sr) if return_sr else wav_bytes
    return (audio, sr) if return_sr else audio


# ============================================================
# OpenAI chat wrapper (kept as-is for Flask integration)
# ============================================================

_openai_api_key = os.getenv("OPENAI_API_KEY")
if _openai_api_key is None and os.path.exists("openai_apikey"):
    with open("openai_apikey", "r", encoding="utf-8") as f:
        _openai_api_key = f.readline().strip()

openai_client = OpenAI(api_key=_openai_api_key) if _openai_api_key else None


def openai_process_message(user_message: str) -> str:
    """
    Send a user message to OpenAI Chat and return the assistant text.
    """
    system_prompt = (
        "Act like a personal assistant. You can respond to questions, "
        "translate sentences, summarize news, and give recommendations."
    )

    if openai_client is None:
        return "OpenAI API key not configured."

    openai_response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    # choices[0].message.content can be None; coerce to empty string.
    return openai_response.choices[0].message.content or ""


import asyncio
from functools import lru_cache
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from src.config import settings
from src.logging_config import setup_app_logging

logger = setup_app_logging(name="ScholarGraph-llm")


class LLMUnavailable(Exception):
    """Every model in the chain failed."""


def allowed_models() -> list[str]:
    models = [m.strip() for m in settings.GEMINI_MODELS.split(",") if m.strip()]
    if settings.DEFAULT_MODEL not in models:
        models.insert(0, settings.DEFAULT_MODEL)
    return models


def normalize_model_name(model_name: str | None) -> str:
    """Accept the frontend's choice only if it is on the allowed list."""
    if model_name and model_name.strip() in allowed_models():
        return model_name.strip()
    return settings.DEFAULT_MODEL


@lru_cache(maxsize=32)
def _gemini(model: str, temperature: float):
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


@lru_cache(maxsize=4)
def _groq(temperature: float):
    return ChatGroq(model=settings.FALLBACK_GROQ_MODEL, temperature=temperature,api_key=os.getenv("GROQ_API_KEY"))


def get_chat_model(model_name: str | None = None, *, tools=None, structured=None, temperature: float = 0.2):
    """Return a chat model that falls back automatically if the first one fails.

    tools=[...]       -> the model may call these tools
    structured=Schema -> the model must answer in this Pydantic shape
    """
    selected = normalize_model_name(model_name)
    names = [selected] + [m for m in allowed_models() if m != selected][:1]
    raw = [_gemini(name, temperature) for name in names] + [_groq(temperature)]

    prepared = []
    for llm in raw:
        model = llm
        if tools:
            model = model.bind_tools(tools)
        if structured is not None:
            model = model.with_structured_output(structured)
        prepared.append(model)

    class _FallbackModel:
        """Tries each model in order. Falls through to the next on ANY exception,
        so a Google rate-limit on gemini-2.5-flash reaches the Groq model."""

        async def ainvoke(self, input, config=None, **kwargs):
            last_exc = None
            for model in prepared:
                try:
                    return await model.ainvoke(input, config=config, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning("Model failed, trying next: %s", exc)
                    continue
            raise last_exc

        def invoke(self, input, config=None, **kwargs):
            last_exc = None
            for model in prepared:
                try:
                    return model.invoke(input, config=config, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning("Model failed, trying next: %s", exc)
                    continue
            raise last_exc

    return _FallbackModel()


def content_to_text(content) -> str:
    """Model replies can be a string or a list of parts (text, thinking, ...). Keep the text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in (None, "text") and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content or "")

          
async def llm_text(prompt, model_name: str | None = None, temperature: float = 0.2) -> str:
    """Ask the model for plain text. Raises LLMUnavailable instead of returning error text,
    so an error message can never be shown to a student as if it were an answer."""
    try:
        response = await get_chat_model(model_name, temperature=temperature).ainvoke(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("All models failed: %s: %s", type(exc).__name__, exc)
        raise LLMUnavailable(str(exc)) from exc
    return content_to_text(getattr(response, "content", response)).strip()


def safe_llm_invoke(prompt, model_name: str | None = None) -> str:
    """Sync version kept for older callers. Now raises LLMUnavailable instead of returning
    strings that start with an error emoji."""
    try:
        response = get_chat_model(model_name).invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        raise LLMUnavailable(str(exc)) from exc
    return content_to_text(getattr(response, "content", response)).strip()


async def check_models(timeout: float = 20.0) -> dict[str, str]:
    """Called once at startup. Logs which configured Gemini models actually answer, so a
    retired model name is noticed on deploy day instead of by a student."""
    results: dict[str, str] = {}
    for name in allowed_models():
        try:
            await asyncio.wait_for(_gemini(name, 0.0).ainvoke("Reply with the word OK."), timeout)
            results[name] = "ok"
            logger.info("Model check: %s is available.", name)
        except Exception as exc:  # noqa: BLE001
            results[name] = f"FAILED ({type(exc).__name__})"
            logger.warning("Model check: %s is NOT available: %s", name, str(exc)[:200])
    if not any(v == "ok" for v in results.values()):
        logger.error("No configured Gemini model responded. The app will fall back to Groq only.")
    return results

"""Pin the settings shipped to a fresh install.

Defaults aren't arbitrary — they're a UX contract. Out of the
box the engine should:

  * Render images via the embedded SDXL backend (no separate
    ComfyUI install required for first run).
  * Talk to a sensible default LLM via OpenRouter — DeepSeek
    V3.2 currently, picked for cost / quality balance.
  * Have music gen OFF (ACE-Step is heavyweight, off by default).
  * Treat first-time-setup as not-yet-done so the welcome
    wizard fires on a fresh install.
"""

from __future__ import annotations

from lucidium.domain.settings import ImageBackend, Settings


def test_default_image_backend_embedded() -> None:
    """Embedded SDXL is the out-of-box choice — a fresh install
    should render without forcing the player to stand up
    ComfyUI separately."""
    s = Settings()
    assert s.image.backend == ImageBackend.embedded


def test_default_llm_model_deepseek() -> None:
    """DeepSeek V3.2 is the default storyteller. Pinned so an
    upstream cost/quality regression on the model is a
    deliberate decision, not a silent drift."""
    s = Settings()
    assert s.llm.model.startswith("deepseek/"), f"expected DeepSeek default; got {s.llm.model!r}"


def test_default_llm_base_url_openrouter() -> None:
    """OpenRouter is the default route — keeps the wizard's
    OpenRouter sign-up flow consistent with what the renderer
    actually points at."""
    s = Settings()
    assert "openrouter" in s.llm.base_url.lower()


def test_default_music_disabled() -> None:
    """Music gen requires a separate ACE-Step server, so it ships
    OFF — flipping it on is an explicit player choice."""
    s = Settings()
    assert s.music.enabled is False


def test_first_time_setup_incomplete_by_default() -> None:
    """A fresh install has not yet run the welcome wizard. The
    renderer routes to the wizard while this flag is false."""
    s = Settings()
    assert s.first_time_setup_complete is False

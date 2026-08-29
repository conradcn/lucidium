"""Settings entity: per-installation configuration."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    SerializationInfo,
    field_serializer,
)

from ..config import (
    DEFAULT_IMAGE_BACKGROUND_WORKFLOW,
    DEFAULT_IMAGE_BASE_URL,
    DEFAULT_IMAGE_MAX_IN_FLIGHT,
    DEFAULT_IMAGE_PORTRAIT_WORKFLOW,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MAX_IN_FLIGHT,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_PROMPT_HISTORY_CLAMP_CHARS,
    DEFAULT_TYPEWRITER_CHARS_PER_SEC,
)


class ImageBackend(StrEnum):
    """Where image generation runs.

    ``embedded`` (the default, see ``ImageSettings.backend``) runs
    diffusers + an SDXL-family checkpoint inside the engine
    process; no external server is involved. ``comfyui`` instead
    speaks the ComfyUI HTTP API at ``ImageSettings.base_url`` —
    the player runs ComfyUI as a separate process, possibly on
    another machine. For the embedded backend the checkpoint is
    loaded from
    ``ImageSettings.embedded_models_dir`` and the player picks
    which file to use via ``embedded_model_name``.
    """

    comfyui = "comfyui"
    embedded = "embedded"


# Serialization context flag that unmasks ``LlmSettings.api_key`` in
# JSON output. EXACTLY ONE caller sets it: ``persistence.settings_store``
# writing ``settings.json``, which is where the real key has to live for
# the engine to authenticate at all. Everything else — the ``s2c/state/*``
# messages, the ``/settings`` patch echo, any log or debug dump that
# reaches for ``model_dump_json`` — gets the mask, so the key never
# leaves the process by accident.
REVEAL_SECRETS_CONTEXT_KEY: Final[str] = "reveal_secrets"


class LlmSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = DEFAULT_LLM_BASE_URL
    model: str = DEFAULT_LLM_MODEL
    # ``SecretStr`` so the key can't land in a repr, an f-string, or a
    # log line by accident; reading it is an explicit
    # ``.get_secret_value()`` call. Accepts a plain ``str`` on input, so
    # ``settings.json`` and ``c2s/settings/update`` are unchanged.
    api_key: SecretStr = SecretStr("")
    temperature: float = DEFAULT_LLM_TEMPERATURE
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS

    @field_serializer("api_key", when_used="json")
    def _serialize_api_key(self, value: SecretStr, info: SerializationInfo) -> str:
        """Mask the key in every JSON dump unless explicitly unmasked.

        ``when_used="json"`` — python-mode ``model_dump()`` keeps the
        live ``SecretStr``, which is what makes the read-modify-write in
        ``settings_update_handler`` round-trip without wiping the key.
        """
        context = info.context or {}
        if context.get(REVEAL_SECRETS_CONTEXT_KEY):
            return value.get_secret_value()
        return ""


class ImageSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Default to embedded so a fresh install renders out of the
    # box (auto-downloads SDXL Turbo into the embedded models
    # directory on first call). ComfyUI users flip this in
    # Settings if they prefer to point at their existing
    # workflow server.
    backend: ImageBackend = ImageBackend.embedded
    base_url: str = DEFAULT_IMAGE_BASE_URL
    portrait_workflow: str = DEFAULT_IMAGE_PORTRAIT_WORKFLOW
    background_workflow: str = DEFAULT_IMAGE_BACKGROUND_WORKFLOW
    # Where ``EmbeddedImageClient`` looks for safetensors checkpoints.
    # Empty string means "use the bundled default" — resolved at
    # use-site to a per-user app-data path so the engine can write a
    # downloaded SDXL Turbo there without prompting the player.
    embedded_models_dir: str = ""
    # Filename (within ``embedded_models_dir``) of the checkpoint
    # the embedded backend should load. Empty string means "pick the
    # first .safetensors found"; the renderer surfaces the live list
    # via ``c2s/embedded/list_models``. This is the LEGACY single-
    # model field — kept as a fallback for the per-workflow fields
    # below so existing settings files keep working.
    embedded_model_name: str = ""
    # Per-workflow model overrides. When non-empty, the embedded
    # backend loads THIS checkpoint for the matching workflow type
    # rather than the legacy single ``embedded_model_name``. Two
    # different filenames here means two diffusers pipelines live
    # in memory simultaneously (subject to VRAM headroom — the
    # client drops the older pipeline if loading the second one
    # OOMs). Same filename means a single shared pipeline. Empty
    # string falls back to ``embedded_model_name``, which itself
    # falls back to "first .safetensors in the directory".
    embedded_character_model_name: str = ""
    embedded_environment_model_name: str = ""
    # Run a second SDXL pass over the centered face region with the
    # ``face_prompt`` as its only prompt, then composite it back onto
    # the body render with a feathered mask. The embedded equivalent
    # of ComfyUI's FaceDetailer node — without this pass, the
    # ``face_prompt`` only contributes as appended text on the body
    # prompt and competes with the framing tokens at the head of the
    # prompt. The pass roughly doubles per-render time on the
    # character workflow; off by default so existing players don't
    # see a silent latency regression.
    embedded_face_detail: bool = False
    # Automatically provision the GPU-correct torch overlay on startup.
    # When True (default), the backend detects the GPU on launch and — if
    # a faster torch flavor than the active one is available — downloads +
    # activates it itself, and self-heals a broken/mismatched active GPU
    # overlay. The download is multi-hundred-MB to a few GB and takes
    # effect on the NEXT launch (torch is already imported in the running
    # process), so this never blocks the current session and never
    # downgrades a working overlay. Set False to keep the legacy behaviour
    # where the player must click "Download" in Settings for a GPU build;
    # the seeded CPU overlay still runs out of the box either way.
    auto_gpu_provision: bool = True
    # Wall-clock seconds the torch-overlay download may go with ZERO
    # bytes arriving before it's aborted. This is a safety bound, not a
    # speed knob: the download runs inside the install-in-flight gate
    # that pauses image generation, so a connection that black-holes
    # (captive portal, dead CDN edge) would otherwise pin that gate for
    # the process lifetime and refuse every render with
    # ``torch_installing``. Raise it on very bursty links; it can never
    # be disabled.
    torch_download_stall_timeout_s: float = 120.0


class MusicSettings(BaseModel):
    """ACE-Step background-music generation settings.

    The engine talks to a configurable ACE-Step server (HTTP) —
    typically the official Gradio-served instance the user runs
    locally on port 7860. When ``enabled``, every NEW GAME (any
    path: New Game, Surprise Me, save load that produces a fresh
    track) generates an instrumental looping background track
    using a prompt the world_init LLM emits. Mid-game music
    changes are also supported via the storyteller's beat schema
    (``music_change`` field).

    All fields are advisory — the actual ACE-Step pipeline
    decides what they map to. Keeping the request shape generic
    lets the player run any compatible inference server (the
    official Gradio app, a custom shim, a hosted endpoint).
    """

    model_config = ConfigDict(extra="forbid")

    # Master toggle. Default OFF: generating a music track adds
    # ~30-90 s to the new-game flow, and not every player wants
    # generated music under their game.
    enabled: bool = False
    # ACE-Step server endpoint. The engine POSTs JSON to
    # ``{base_url}/api/v1/generate`` and reads audio bytes back
    # from the response. The default targets ``127.0.0.1:8001``
    # — pick a port that doesn't collide with the Gradio default
    # (7860) the user might already be running for SDXL or
    # ComfyUI. Redirect to a different host/port to use a remote
    # server.
    base_url: str = "http://127.0.0.1:8001"
    # ACE-Step DiT model name passed through to the server's
    # ``/v1/init`` slot loader. The default matches the upstream
    # FastAPI server's stock checkpoint; smaller / fine-tuned
    # variants land here when the player picks them from the
    # settings dropdown (which is populated live from
    # ``GET /v1/model_inventory`` when the server is reachable).
    model_name: str = "acestep-v15-turbo"
    # Length of each generated clip in seconds. Longer clips cost
    # more inference time but loop less obviously. ACE-Step has
    # internal limits — the player can lower this if their
    # hardware can't sustain the default.
    clip_seconds: float = Field(default=60.0, ge=15.0, le=240.0)
    # Number of inference steps the ACE-Step pipeline runs.
    # Higher = better quality, lower = faster. 27 is the
    # recommended default; the field exists so users on slower
    # GPUs can drop to 15-20 for snappier mid-game music swaps.
    inference_steps: int = Field(default=27, ge=10, le=100)
    # Optional CFG / guidance value passed through to ACE-Step.
    # Pinned at the model's recommended default unless the user
    # has a reason to override.
    guidance_scale: float = Field(default=15.0, ge=1.0, le=30.0)
    # When True (default), the engine treats ACE-Step as
    # competing for the SAME local GPU as the embedded SDXL
    # pipeline and serialises their inference paths through a
    # shared lock — only one runs at a time. Before each music
    # call the engine evicts the SDXL pipelines to CPU; the
    # next image render brings them back. Disable this when
    # ACE-Step runs on a different machine / different GPU
    # (the engine still serialises consecutive music calls via
    # the audio client's internal lock, but doesn't shuttle
    # SDXL weights off the GPU on every music swap).
    local_gpu_coordination: bool = True


class ConcurrencyLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_max_in_flight: int = Field(default=DEFAULT_LLM_MAX_IN_FLIGHT, ge=1, le=64)
    image_max_in_flight: int = Field(default=DEFAULT_IMAGE_MAX_IN_FLIGHT, ge=1, le=64)


class UserProfile(BaseModel):
    """Cross-save persistent profile of the player's tastes.

    Lives on ``Settings`` (not per-save) so a player who starts a
    new game keeps the preferences they've expressed across past
    sessions. Two parallel sets of buckets, separated by PROVENANCE
    rather than by who is allowed to edit them:

      * ``likes`` / ``dislikes`` / ``notes`` — originated from the
        PLAYER through the Settings screen.
      * ``summarizer_likes`` / ``summarizer_dislikes`` /
        ``summarizer_notes`` — inferred by the summarizer LLM from
        the player's actions across past sessions.

    Both buckets are editable by both parties:

      * The player can add / edit / delete entries in EITHER bucket
        through the Settings screen — including pruning summarizer
        inferences that don't actually match their tastes.
      * The summarizer writes new inferences into the
        ``summarizer_*`` bucket (its own provenance) and runs the
        consolidation pass on it when any bucket hits 10 entries.
        Player-bucket entries pass through summarizer writes
        untouched (the summarizer doesn't add to that bucket — only
        the player does — but it can read from it for dedup).

    Both halves feed every storyteller / summarizer prompt — the LLM
    sees a deduplicated combined list so its inferences don't repeat
    something already in either bucket.
    """

    model_config = ConfigDict(extra="forbid")

    likes: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    summarizer_likes: list[str] = Field(default_factory=list)
    summarizer_dislikes: list[str] = Field(default_factory=list)
    summarizer_notes: list[str] = Field(default_factory=list)
    # Per-tag STRENGTH for the inferred buckets — keyed by the
    # case-folded tag, value is a 0–``TRAIT_STRENGTH_CAP`` float.
    # New tags from the summarizer LLM start at
    # ``INITIAL_TRAIT_STRENGTH`` (0.1); each subsequent re-
    # inference adds ``TRAIT_STRENGTH_INCREMENT`` (0.1), capped
    # at ``TRAIT_STRENGTH_CAP`` (2.0). The summarizer's
    # consolidation pass also sums strengths when it merges
    # near-duplicate tags, so multiple weak observations
    # contribute to a single trait reaching surface.
    #
    # SURFACE GATE: only tags whose strength meets
    # ``TRAIT_SURFACE_THRESHOLD`` (1.0) are visible to the
    # storyteller, the summarizer, and the Settings UI. Below
    # the gate, the engine still TRACKS the tag (so further
    # reinforcement can promote it) but treats it as
    # speculative — never feeds it into prompts, never displays
    # it. This stops a single noisy inference from contaminating
    # the player profile until enough independent evidence has
    # accumulated. Old saves whose scores were integer-typed
    # (the prior scheme used hit counts 1–5) round-trip cleanly:
    # pydantic coerces int→float and an old "score=2" tag
    # remains surfaced under the new threshold.
    summarizer_likes_scores: dict[str, float] = Field(default_factory=dict)
    summarizer_dislikes_scores: dict[str, float] = Field(default_factory=dict)
    summarizer_notes_scores: dict[str, float] = Field(default_factory=dict)
    # Player-dismissed inferences. When the player deletes an entry
    # from a ``summarizer_*`` bucket via the Settings UI, the
    # ``settings_update`` handler appends it here so the next
    # summarizer pass sees it in the dedup ``seen`` set and refuses
    # to re-add it. Without this list a deleted entry would come
    # back the moment the model re-noticed the underlying pattern,
    # making the player feel like their edits don't stick.
    dismissed_likes: list[str] = Field(default_factory=list)
    dismissed_dislikes: list[str] = Field(default_factory=list)
    dismissed_notes: list[str] = Field(default_factory=list)

    def merged_likes(self) -> list[str]:
        """Combined player + summarizer likes, ALL strengths.

        For storyteller / summarizer prompts that drive
        narrative decisions, prefer ``surfaced_likes`` —
        below-threshold tags are speculative and shouldn't
        steer scene selection. ``merged_*`` is kept for the
        Settings UI dedup path and any caller that genuinely
        needs the full set."""
        return _merge_unique(self.likes, self.summarizer_likes)

    def merged_dislikes(self) -> list[str]:
        return _merge_unique(self.dislikes, self.summarizer_dislikes)

    def merged_notes(self) -> list[str]:
        return _merge_unique(self.notes, self.summarizer_notes)

    def surfaced_likes(self) -> list[str]:
        """Player-typed entries (always trusted) plus
        summarizer-inferred entries whose strength meets
        ``TRAIT_SURFACE_THRESHOLD``. THIS is the list the
        storyteller and summarizer prompts should read —
        below-threshold tags are tracked-but-speculative
        and excluding them prevents a single noisy inference
        from contaminating downstream narrative decisions."""
        return _merge_unique(
            self.likes,
            _surfaced(self.summarizer_likes, self.summarizer_likes_scores),
        )

    def surfaced_dislikes(self) -> list[str]:
        return _merge_unique(
            self.dislikes,
            _surfaced(self.summarizer_dislikes, self.summarizer_dislikes_scores),
        )

    def surfaced_notes(self) -> list[str]:
        return _merge_unique(
            self.notes,
            _surfaced(self.summarizer_notes, self.summarizer_notes_scores),
        )


# Trait-strength bookkeeping for summarizer-inferred profile
# entries. The user reported that the prior scheme — which
# surfaced any tag with two reinforcements — let one-off LLM
# observations contaminate the player profile within a session.
# The new scheme requires a tag to ACCUMULATE strength across
# many independent inferences before it's allowed to influence
# narration.
#
# Lifecycle of a strength:
#   * First inference adds the tag at INITIAL_TRAIT_STRENGTH (0.1).
#   * Each subsequent re-inference of the same tag adds
#     TRAIT_STRENGTH_INCREMENT (0.1).
#   * The summarizer's consolidation pass merges near-duplicate
#     tags and SUMS their strengths, so multiple weak observations
#     about a similar trait converge on a single visible entry.
#   * Strength is clamped to TRAIT_STRENGTH_CAP (2.0).
#   * A tag is "surfaced" — visible to storyteller / summarizer
#     prompts AND to the Settings UI — only once strength reaches
#     TRAIT_SURFACE_THRESHOLD (1.0).
INITIAL_TRAIT_STRENGTH: float = 0.1
TRAIT_STRENGTH_INCREMENT: float = 0.1
TRAIT_STRENGTH_CAP: float = 2.0
TRAIT_SURFACE_THRESHOLD: float = 1.0


def tag_strength(tag: str, scores: dict[str, float]) -> float:
    """Look up a tag's accumulated strength.

    Tags missing from ``scores`` default to
    ``TRAIT_SURFACE_THRESHOLD`` so legacy entries — populated
    before the strength mechanism existed — stay visible
    instead of silently disappearing on first load. Newly
    inferred tags arrive at ``INITIAL_TRAIT_STRENGTH`` (set by
    the merge path) and must be reinforced to cross the
    surface gate.
    """
    key = (tag or "").strip().casefold()
    if not key:
        return 0.0
    return float(scores.get(key, TRAIT_SURFACE_THRESHOLD))


def is_tag_surfaced(tag: str, scores: dict[str, float]) -> bool:
    """Predicate: would this tag be visible to LLM prompts /
    Settings UI under the current strength scheme?

    The epsilon tolerance handles IEEE-754 accumulation drift —
    summing 0.1 ten times lands at 0.9999999999999999 in
    floating point, which would fail a strict >= 1.0 comparison
    even though semantically the player has just earned ten
    increments. The tolerance is a nudge wider than the
    representation error so genuine 0.99 strengths (one
    increment short of surface) stay correctly hidden.
    """
    return tag_strength(tag, scores) >= TRAIT_SURFACE_THRESHOLD - 1e-9


def _surfaced(
    entries: list[str],
    scores: dict[str, float],
) -> list[str]:
    """Filter ``entries`` down to tags whose strength meets
    the surface threshold. Order is preserved."""
    return [tag for tag in entries if is_tag_surfaced(tag, scores)]


# ---------- Backwards-compat shims ------------------------------------------
# Earlier code paths read these constants by their old names.
# Keep aliases so a stray import elsewhere keeps working — they
# now reference the new strength-based equivalents.
SUMMARIZER_CERTAINTY_THRESHOLD: float = TRAIT_SURFACE_THRESHOLD
SUMMARIZER_CERTAINTY_CAP: float = TRAIT_STRENGTH_CAP


# Deprecated alias for ``tag_strength``. Kept by reference (not
# a wrapper function) so an ``is`` / ``==`` comparison treats
# them as the same callable.
tag_certainty = tag_strength


def _merge_unique(*lists: list[str]) -> list[str]:
    """Concatenate ``lists`` while dropping case-insensitive
    duplicates. Player entries appear first so the storyteller /
    summarizer prompts read the player's own words at the top of
    each bucket."""
    seen: set[str] = set()
    out: list[str] = []
    for items in lists:
        for tag in items:
            stripped = (tag or "").strip()
            if not stripped:
                continue
            key = stripped.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(stripped)
    return out


class AudioSettings(BaseModel):
    """Global audio mixing controls. Applied renderer-side to
    every audio element the engine drives (background music,
    main-menu theme, future ambient / SFX layers). The renderer
    multiplies ``master_volume`` and the per-channel volume into
    the underlying ``HTMLAudioElement.volume`` — so a
    ``master_volume`` of 0 silences everything regardless of the
    per-channel sliders, and each channel's slider only affects
    that channel.

    All sliders are 0..1 floats so the UI can map them straight
    to a percent without rescaling. Defaults are quiet enough
    that music doesn't fight the storyteller's prose on first
    launch — players who want it louder bump the slider rather
    than us shipping a default that masks dialogue."""

    model_config = ConfigDict(extra="forbid")
    master_volume: float = Field(default=0.7, ge=0.0, le=1.0)
    music_volume: float = Field(default=0.4, ge=0.0, le=1.0)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LlmSettings = Field(default_factory=LlmSettings)
    image: ImageSettings = Field(default_factory=ImageSettings)
    music: MusicSettings = Field(default_factory=MusicSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    typewriter_speed_chars_per_sec: int = Field(
        default=DEFAULT_TYPEWRITER_CHARS_PER_SEC, ge=1, le=1000
    )
    prompt_history_clamp_chars: int = DEFAULT_PROMPT_HISTORY_CLAMP_CHARS
    concurrency: ConcurrencyLimits = Field(default_factory=ConcurrencyLimits)
    # Mature-content mode (FR-032a). When true, an explicit allowance
    # directive is appended to the system prompt for every text-gen,
    # world-init, and world-refresh call. Default: disabled.
    mature_content: bool = False
    # Narrator linguistic style. Free-form string fed verbatim
    # into the storyteller's system prompt as a stylistic
    # constraint on the prose. Defaults to ``conversational`` —
    # plain modern English, the safest baseline. Players who
    # want a stronger voice (``literary``, ``hard-boiled noir``,
    # ``pirate``, ``fairy tale``, ``gothic``, ``terse and
    # sparse``, ``lyrical``) type their own here. Doesn't
    # affect choice options or character / location / outfit
    # values — only the narrative ``text`` of each beat.
    narrator_style: str = "conversational"
    # Cross-save player profile. Editable from the Settings screen
    # AND mutated by the summarizer when it learns something about
    # what the player likes / dislikes from their actions.
    user_profile: UserProfile = Field(default_factory=UserProfile)
    # Master toggle for the summarizer's profile-inference step.
    # When True (default), the summarizer proposes new
    # ``summarizer_*`` tags from the player's choices each turn.
    # When False, the engine still runs the summarizer (facts /
    # stage selection / drift check stay important) but DISCARDS
    # any user_profile additions the LLM emits — none of the
    # ``summarizer_*`` buckets grow, the cross-save profile stops
    # learning. Player-typed entries in the ``likes`` / ``dislikes``
    # / ``notes`` buckets remain editable from Settings either way.
    learn_user_profile: bool = True
    # Renderer-side gate for the first-run wizard. Stays false on a
    # fresh install so the welcome / OpenRouter-key / image-backend
    # walkthrough fires before the start screen; the wizard flips
    # this to true on its final step. Re-running the wizard is a
    # matter of clearing the flag from the Settings screen — the
    # engine never auto-resets it.
    first_time_setup_complete: bool = False

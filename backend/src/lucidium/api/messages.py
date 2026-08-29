"""WebSocket IPC messages — single source of truth for the renderer.

Constitution V (DRY): every wire-format type lives in this module.
``scripts/codegen/export-schemas.py`` walks this file and emits JSON
Schema; the renderer's ``json-schema-to-typescript`` step consumes the
output. The renderer never re-declares these shapes.
"""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..config import PROTOCOL_VERSION
from ..domain.character import Character
from ..domain.dialog import DialogNode
from ..domain.game import Game
from ..domain.settings import Settings
from .errors import ErrorCode

# ---------- Envelope ----------


class MessageType(StrEnum):
    # Client -> Server
    c2s_hello = "c2s/hello"
    c2s_saves_list = "c2s/saves/list"
    c2s_saves_continue = "c2s/saves/continue"
    c2s_saves_load = "c2s/saves/load"
    c2s_saves_rename = "c2s/saves/rename"
    c2s_saves_delete = "c2s/saves/delete"
    c2s_new_game_start = "c2s/new_game/start"
    c2s_new_game_answer = "c2s/new_game/answer"
    c2s_new_game_add_side_character = "c2s/new_game/add_side_character"
    c2s_new_game_edit_side_character = "c2s/new_game/edit_side_character"
    c2s_new_game_delete_side_character = "c2s/new_game/delete_side_character"
    c2s_new_game_edit_review = "c2s/new_game/edit_review"
    c2s_new_game_confirm = "c2s/new_game/confirm"
    c2s_new_game_go_back = "c2s/new_game/go_back"
    c2s_new_game_surprise_me = "c2s/new_game/surprise_me"
    c2s_play_advance = "c2s/play/advance"
    c2s_play_free_text = "c2s/play/free_text"
    c2s_play_undo = "c2s/play/undo"
    c2s_play_cancel = "c2s/play/cancel"
    c2s_edit_world = "c2s/edit/world"
    c2s_edit_character = "c2s/edit/character"
    c2s_edit_character_dismiss = "c2s/edit/character/dismiss"
    c2s_edit_character_show = "c2s/edit/character/show"
    c2s_edit_character_rerender = "c2s/edit/character/rerender"
    c2s_edit_history = "c2s/edit/history"
    c2s_edit_history_delete = "c2s/edit/history/delete"
    c2s_edit_history_retcon = "c2s/edit/history/retcon"
    c2s_edit_environment = "c2s/edit/environment"
    c2s_edit_environment_rerender = "c2s/edit/environment/rerender"
    c2s_edit_environment_apply = "c2s/edit/environment/apply"
    c2s_settings_get = "c2s/settings/get"
    c2s_settings_update = "c2s/settings/update"
    c2s_settings_validate_api_key = "c2s/settings/validate_api_key"
    c2s_embedded_list_models = "c2s/embedded/list_models"
    c2s_embedded_recommend_model = "c2s/embedded/recommend_model"
    c2s_embedded_download_model = "c2s/embedded/download_model"
    c2s_torch_overlay_status = "c2s/torch_overlay/status"
    c2s_torch_overlay_install = "c2s/torch_overlay/install"
    c2s_music_inventory = "c2s/music/inventory"
    c2s_music_regenerate = "c2s/music/regenerate"
    c2s_app_exit = "c2s/app/exit"

    # Server -> Client
    s2c_hello = "s2c/hello"
    s2c_saves_list = "s2c/saves/list"
    s2c_state_full = "s2c/state/full"
    s2c_state_patch = "s2c/state/patch"
    s2c_text_streaming = "s2c/text/streaming"
    s2c_text_complete = "s2c/text/complete"
    s2c_image_ready = "s2c/image/ready"
    s2c_music_ready = "s2c/music/ready"
    s2c_error = "s2c/error"
    s2c_notice = "s2c/notice"
    s2c_cost = "s2c/cost"
    s2c_embedded_models = "s2c/embedded/models"
    s2c_embedded_recommended_model = "s2c/embedded/recommended_model"
    s2c_embedded_download_progress = "s2c/embedded/download_progress"
    s2c_settings_api_key_validation = "s2c/settings/api_key_validation"
    s2c_torch_overlay_status = "s2c/torch_overlay/status"
    s2c_torch_overlay_progress = "s2c/torch_overlay/progress"
    s2c_music_inventory = "s2c/music/inventory"
    s2c_play_cancelled = "s2c/play/cancelled"


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: MessageType
    payload: dict[str, Any] = Field(default_factory=dict)
    protocol_version: int = PROTOCOL_VERSION


# ---------- Client -> Server payloads ----------


class C2SHello(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol_version: int


class C2SSavesList(BaseModel):
    model_config = ConfigDict(extra="forbid")


class C2SSavesContinue(BaseModel):
    model_config = ConfigDict(extra="forbid")


# A save id is a single on-disk directory NAME under the saves root —
# never a path. Constraining it at the wire boundary is what stops
# ``saves/<id>`` from escaping that root: ``Path.__truediv__`` silently
# DISCARDS the left operand when the right one is absolute, so an
# unconstrained ``save_id`` of ``C:/Users/<u>/Documents`` would resolve
# to Documents itself and ``saves_delete`` would rmtree it. The pattern
# admits exactly the ids the engine mints (uuid4 hex + dashes) and
# excludes ``/``, ``\``, ``:``, ``.`` and the empty string, so neither
# an absolute path nor a ``..`` traversal can be expressed.
_SAVE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class C2SSavesLoad(BaseModel):
    model_config = ConfigDict(extra="forbid")
    save_id: str = Field(pattern=_SAVE_ID_PATTERN)


class C2SSavesRename(BaseModel):
    model_config = ConfigDict(extra="forbid")
    save_id: str = Field(pattern=_SAVE_ID_PATTERN)
    new_name: str


class C2SSavesDelete(BaseModel):
    model_config = ConfigDict(extra="forbid")
    save_id: str = Field(pattern=_SAVE_ID_PATTERN)


class C2SNewGameStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The renderer's currently-active menu carousel pair (e.g.
    # "harbor-noir"). The backend uses this to look up a character
    # body+face description and re-render the same dream guide in
    # whatever visual style the player picks at step 3.
    preview_character_slug: str = ""


# NOT converted to ``StrEnum``: ``handlers.py`` interpolates a bare
# member into an error message (``f"unhandled interview step: {payload.step}"``),
# and ``StrEnum.__str__`` returns the bare value where ``(str, Enum)``
# returns ``"InterviewStep.<member>"``. Converting would silently change
# that user-visible string.
class InterviewStep(str, Enum):  # noqa: UP042
    setting = "setting"
    genre = "genre"
    visual_style = "visual_style"
    character_description = "character_description"
    name = "name"


class C2SNewGameAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: InterviewStep
    answer: str
    is_free_text: bool = False
    # Only meaningful on the Identity step (``InterviewStep.name``).
    # The combined Identity screen submits pronouns alongside the
    # player's chosen name; for every other step this stays empty
    # and the handler ignores it.
    pronouns: str = ""


class C2SNewGameAddSideCharacter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str


class C2SNewGameEditSideCharacter(BaseModel):
    """Rename a side-character stub the player previously added.

    The ``description`` replaces both ``name`` and ``description`` on
    the stub — at this stage the LLM hasn't expanded the stub yet, so
    the two fields are still in sync. Confirm-time expansion re-fills
    them from the new description.
    """

    model_config = ConfigDict(extra="forbid")
    character_id: str
    description: str


class C2SNewGameDeleteSideCharacter(BaseModel):
    """Remove a side-character stub the player previously added."""

    model_config = ConfigDict(extra="forbid")
    character_id: str


class C2SNewGameEditReview(BaseModel):
    """Edit a previously-answered interview field from the Review
    step. Updates the named field in-place AND cancels + re-fires
    the world_init prefetch so the (now-corrected) values feed
    the next ``new_game/confirm`` call.

    ``field`` must be one of the answered-step keys: ``setting``,
    ``visual_style``, ``genre``, ``character_description``,
    ``character_name``. The handler rejects anything else so a
    stray patch can't write arbitrary fields into the interview
    state."""

    model_config = ConfigDict(extra="forbid")
    field: str
    value: str


class C2SNewGameConfirm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overrides: dict[str, Any] = Field(default_factory=dict)


class C2SNewGameGoBack(BaseModel):
    """Step the interview one screen backwards. Sent from the Review
    screen's ``Go back`` button so the player can return to the Name
    step (the previous question) without losing their answers — every
    answered field stays in ``session.interview`` and the step pointer
    is the only thing that moves.

    Cancels the in-flight world_init prefetch (which was launched
    against the answers we're now letting the player change). The
    next time the player completes Name and lands on Review, a fresh
    prefetch fires automatically.
    """

    model_config = ConfigDict(extra="forbid")


class C2SNewGameSurpriseMe(BaseModel):
    """Skip the interview entirely. The backend asks the LLM to
    construct a scenario tailored to the player's cross-save
    profile (likes / dislikes / notes) and re-uses the most recent
    save's visual style. When no save exists yet, the visual style
    falls back to the first hardcoded option.
    """

    model_config = ConfigDict(extra="forbid")


class C2SPlayAdvance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    option_id: str | None = None


class C2SPlayFreeText(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class C2SPlayUndo(BaseModel):
    """Pop the most recent advance off the session's undo stack and
    restore the prior Game state (including character.removed flags
    that were set during the undone turn)."""

    model_config = ConfigDict(extra="forbid")


class C2SPlayCancel(BaseModel):
    """Stop the generation the player is currently waiting on.

    Cancels the foreground LLM stream for the in-flight turn (and the
    background task draining the rest of its beat chain). Speculative
    branches, summarizer passes and queued renders are deliberately
    left alone — those are not what the player is staring at, and
    killing them would only make the next turn slower."""

    model_config = ConfigDict(extra="forbid")


class S2CPlayCancelled(BaseModel):
    """Ack for ``c2s/play/cancel``.

    ``cancelled`` is ``False`` when there was nothing in flight to
    stop — the renderer uses it to decide whether to keep waiting for
    the generation's messages or drop the spinner immediately."""

    model_config = ConfigDict(extra="forbid")
    cancelled: bool


# One scalar attribute value for the ``c2s/edit/*`` messages.
#
# ``Any`` would be marginally more permissive, but it exports as an
# EMPTY JSON Schema, which json-schema-to-typescript renders as an
# index-signature object — a shape no renderer control ever produces,
# so every call site had to cast around it. Every editable field
# (``_EDITABLE_WORLD_FIELDS`` and friends in ``handlers.py``) is a
# str / int / enum, so the union below is what the wire actually
# carries; anything else was already rejected downstream by
# ``_validated_update``'s ``model_validate`` round trip.
EditValue = str | int | float | bool | None


class C2SEditWorld(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    value: EditValue


class C2SEditCharacter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    character_id: str
    field: str
    value: EditValue


class C2SEditCharacterDismiss(BaseModel):
    """Manually remove a character from the live story. They drop
    from on_stage and stop appearing in storyteller / summarizer
    prompts. Reversible: an undo that crosses this turn restores
    them; pressing Show in the characters panel does the same."""

    model_config = ConfigDict(extra="forbid")
    character_id: str
    reason: str = "dismissed"


class C2SEditCharacterShow(BaseModel):
    """Clear ``removed`` and place the character back on stage at
    the current node. Inverse of dismiss; also useful when the
    storyteller has lost track of a character the player still
    wants in the scene."""

    model_config = ConfigDict(extra="forbid")
    character_id: str


class C2SEditCharacterRerender(BaseModel):
    """Re-trigger the portrait pipeline for a single character.
    Useful when the rendered portrait drifted from the description
    or just looks wrong. The new portrait replaces the prior one."""

    model_config = ConfigDict(extra="forbid")
    character_id: str


class C2SEditHistory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str
    new_text: str


class C2SEditHistoryDelete(BaseModel):
    """Delete a single committed beat from the history. The chain
    splices: the deleted node is removed from ``committed_path`` and
    its successor's ``parent_id`` is rewritten to point at the
    deleted node's parent. Speculative descendants of the deleted
    node (and of beats after it on the live path) are invalidated."""

    model_config = ConfigDict(extra="forbid")
    node_id: str


class C2SEditHistoryRetcon(BaseModel):
    """Apply a global retcon to the entire committed history. The LLM
    rewrites every committed beat AND every character's mutable
    state (outfit, pose, expression, description, facts) in a single
    pass so the timeline reads consistently with the new reality.

    ``instructions`` is the player's natural-language description of
    what should be true (e.g., "the password was actually 'corvid'
    all along — fix every mention" or "Mira was never a witch, she
    was a librarian; rewrite scenes accordingly")."""

    model_config = ConfigDict(extra="forbid")
    instructions: str


class C2SEditEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    environment_id: str
    field: str
    value: EditValue


class C2SEditEnvironmentRerender(BaseModel):
    """Re-trigger the background pipeline for a single environment.
    The handler bumps the environment's seed to force a different
    draw (same prompt → different image) and runs the background
    workflow against the new seed. The rendered file replaces
    ``environment.image_path``."""

    model_config = ConfigDict(extra="forbid")
    environment_id: str


class C2SEditEnvironmentApply(BaseModel):
    """Swap the current scene's backdrop to a different environment.
    Mutates the current dialog node's ``location_id`` to point at
    the supplied environment id, so the renderer's location-walk
    (StoryPanel ``activeEnvironmentId``) lands on this env's
    pre-rendered image. Useful when several environments have been
    rendered up-front and the player wants to drop the current scene
    onto one of them without re-walking dialog or asking the LLM to
    write a new beat."""

    model_config = ConfigDict(extra="forbid")
    environment_id: str


class C2SSettingsGet(BaseModel):
    model_config = ConfigDict(extra="forbid")


class C2SSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patch: dict[str, Any]


class C2SSettingsValidateApiKey(BaseModel):
    """Ask the backend to verify that ``api_key`` actually authenticates
    against ``base_url`` BEFORE saving it. The handler hits the
    OpenAI-compatible ``GET /v1/models`` endpoint with the key as a
    Bearer token; a 2xx says the key is good, a 401/403 says it isn't,
    anything else is reported as an unreachable error.

    Settings + FirstTimeSetup call this on blur of the key input so the
    player gets an immediate signal instead of waiting for the first
    real prompt round-trip to fail with an opaque "LLM provider failed"
    banner mid-game.

    Both fields are passed explicitly (not pulled from server-side
    settings) so the user can validate a key BEFORE they commit it to
    the autosaving Settings store."""

    model_config = ConfigDict(extra="forbid")
    base_url: str
    api_key: str


class S2CSettingsApiKeyValidation(BaseModel):
    """Result of a ``c2s/settings/validate_api_key`` probe.

    ``ok`` is True only on a 2xx response from ``/v1/models``. ``status``
    is a short machine-friendly tag the UI maps to a label: ``valid``,
    ``unauthorized`` (key rejected), ``unreachable`` (network /
    timeout / bad URL), or ``invalid_input`` (empty key or URL).
    ``message`` is a human-readable explanation suitable for inline
    display next to the input.
    """

    model_config = ConfigDict(extra="forbid")
    ok: bool
    status: str
    message: str


class C2SEmbeddedListModels(BaseModel):
    """Request the list of safetensors / ckpt files in the configured
    embedded-backend models directory. Settings UI populates the
    "model" dropdown from the reply. Empty ``models_dir`` means
    "use the default" — same fallback rule as
    ``ImageSettings.embedded_models_dir``."""

    model_config = ConfigDict(extra="forbid")
    models_dir: str = ""


class C2SEmbeddedRecommendModel(BaseModel):
    """Ask the backend which base model it would download for THIS
    machine (GPU vendor + VRAM heuristic). Cheap + network-free; the
    first-run wizard calls it to fill the ``<highlight>`` in its copy
    and to know which key the download button should request. Empty
    ``models_dir`` means "use the default" — same fallback as
    ``ImageSettings.embedded_models_dir`` — and is only used to report
    whether the directory already holds checkpoints."""

    model_config = ConfigDict(extra="forbid")
    models_dir: str = ""


class C2SEmbeddedDownloadModel(BaseModel):
    """One-click download of a base checkpoint into the embedded models
    directory. ``key`` selects a ``MODEL_CATALOG`` entry (``sdxl`` /
    ``sdxl-turbo`` / ``z-image-turbo`` / ``qwen-image``); empty means "use
    the hardware-recommended one". Streams ``s2c/embedded/download_progress``
    while running, then a terminal ``s2c/embedded/models`` with the
    refreshed inventory so the dropdown updates in place."""

    model_config = ConfigDict(extra="forbid")
    key: str = ""
    models_dir: str = ""


class C2STorchOverlayStatus(BaseModel):
    """Ask the backend which torch runtime overlay it recommends for
    this machine, which flavors are already installed, and which is
    active. Cheap + network-free; the Settings / first-run screen calls
    it to decide whether to offer a download and which button to
    highlight. Replies with ``s2c/torch_overlay/status``."""

    model_config = ConfigDict(extra="forbid")


class C2STorchOverlayInstall(BaseModel):
    """Download + install (and by default activate) a torch runtime
    overlay flavor. The frozen build ships WITHOUT torch; this is how
    the player fetches the GPU-correct build (CUDA / ROCm / DirectML /
    XPU / CPU) after the GPU probe. Mirrors the SDXL "Download" button:
    a deliberate user click triggers a multi-hundred-MB fetch, streamed
    back as ``s2c/torch_overlay/progress`` events and terminated by a
    final ``s2c/torch_overlay/status`` (success) or ``s2c/error``.

    ``flavor`` must be one of ``cuda`` / ``rocm`` / ``directml`` /
    ``xpu`` / ``cpu``. ``activate`` (default true) flips the
    active-overlay pointer so the NEXT launch loads this flavor — the
    switch can't take effect in the running process because torch is
    already imported."""

    model_config = ConfigDict(extra="forbid")
    flavor: str
    activate: bool = True


class C2SMusicInventory(BaseModel):
    """Probe an ACE-Step server for its model inventory and
    connection status. Settings UI uses this both as a "Test
    connection" handshake and to populate the model dropdown.

    The optional ``base_url`` lets the renderer probe a URL the
    user is editing live BEFORE committing it via
    ``c2s/settings/update`` — same pattern as the embedded image
    backend's ``models_dir`` override. Empty falls through to
    ``MusicSettings.base_url``.
    """

    model_config = ConfigDict(extra="forbid")
    base_url: str = ""


class C2SMusicRegenerate(BaseModel):
    """Re-render the live game's background music. Supplying
    ``prompt`` overrides ``World.music_prompt`` for this render and
    updates the stored prompt; an empty string keeps the current
    prompt and just re-fires the pipeline (useful when a render
    failed silently and the player wants to retry). Requires
    ``MusicSettings.enabled=True``; the handler raises a schema
    error if music gen is disabled."""

    model_config = ConfigDict(extra="forbid")
    prompt: str = ""


class C2SAppExit(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Server -> Client payloads ----------


class S2CHello(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol_version: int = PROTOCOL_VERSION
    has_save: bool


class SaveSummaryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    last_played_at: str
    created_at: str
    schema_version: int
    summary: str
    # The save's ``meta.json`` is unreadable: the display fields are
    # placeholders and the only useful action is Delete.
    corrupt: bool = False


class S2CSavesList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    saves: list[SaveSummaryPayload] = Field(default_factory=list)


class S2CStateFull(BaseModel):
    model_config = ConfigDict(extra="forbid")
    game: Game
    settings: Settings


class PatchOp(BaseModel):
    """RFC 6902-style operation. ``op`` is ``replace``/``add``/``remove``."""

    model_config = ConfigDict(extra="forbid")
    op: str
    path: str
    value: Any | None = None


class S2CStatePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ops: list[PatchOp]


class S2CTextStreaming(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str
    delta: str


class S2CTextComplete(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str
    node: DialogNode | None = None


class ImageKind(StrEnum):
    portrait = "portrait"
    background = "background"


class S2CImageReady(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: ImageKind
    target_id: str
    image_path: str
    image_id: str


class S2CMusicReady(BaseModel):
    """A new background-music track is ready for playback. The
    renderer's BackgroundMusic component swaps to ``audio_path``
    on receipt; the prior track fades out, the new one fades in.
    ``music_id`` is the prompt hash so the renderer can dedup
    duplicate ready events without re-trigger-fading the same
    track."""

    model_config = ConfigDict(extra="forbid")
    audio_path: str
    music_id: str
    prompt: str


class S2CError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: ErrorCode
    message: str
    recoverable: bool = True


class NoticeKind(StrEnum):
    info = "info"
    warning = "warning"


class S2CNotice(BaseModel):
    """One-off pop-up for the renderer to display in a modal window.

    Used for safety / policy events the player needs to see (e.g. an
    automatic age-correction retcon when an under-18 character was
    described in a way that would have produced an unsafe SDXL
    render). NOT a substitute for s2c/error — errors are operational
    failures; notices are informational.
    """

    model_config = ConfigDict(extra="forbid")
    title: str
    body: str
    kind: NoticeKind = NoticeKind.info


class CostDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tokens_in: int = 0
    tokens_out: int = 0
    image_calls: int = 0
    llm_calls: int = 0
    latency_ms: int = 0
    dollar_estimate: float = 0.0


class S2CCost(BaseModel):
    model_config = ConfigDict(extra="forbid")
    delta: CostDelta


class S2CMusicInventory(BaseModel):
    """Reply to ``c2s/music/inventory``. ``ok`` is true when the
    HTTP probe succeeded; on failure ``error`` carries a short,
    user-facing string the Settings screen can show next to the
    Test Connection button. ``models`` is a flat list of model
    IDs the server reported via ``GET /v1/model_inventory``;
    populating that endpoint as a flat list — even when the
    upstream shape is richer — keeps the renderer simple. The
    renderer is also free to fall back to a free-text input when
    ``ok=false`` so the player can hand-type a model name."""

    model_config = ConfigDict(extra="forbid")
    base_url: str
    ok: bool
    models: list[str] = Field(default_factory=list)
    error: str = ""


class S2CEmbeddedModels(BaseModel):
    """Response to ``c2s/embedded/list_models``. ``models_dir`` is
    the resolved absolute path the renderer can show as a tooltip
    (so the user knows where to drop additional checkpoints).
    ``models`` is the alphabetised list of filenames found there."""

    model_config = ConfigDict(extra="forbid")
    models_dir: str
    models: list[str] = Field(default_factory=list)


class S2CEmbeddedRecommendedModel(BaseModel):
    """Reply to ``c2s/embedded/recommend_model``. ``key`` /
    ``display_name`` identify the catalog entry recommended for this
    machine (``display_name`` fills the wizard's ``<highlight>``);
    ``reason`` is a short human-readable WHY for a tooltip. ``models_dir``
    is the resolved target path and ``has_models`` is True when it already
    holds a checkpoint — the wizard hides the one-click offer in that
    case. ``approx_bytes`` lets the UI show "~N GB" before the download's
    real Content-Length arrives; it is the TOTAL for the model (the
    checkpoint plus the text encoder / VAE some families fetch at first
    render), so the quoted figure is what the player actually pays for on
    a metered connection."""

    model_config = ConfigDict(extra="forbid")
    key: str
    display_name: str
    reason: str = ""
    models_dir: str = ""
    has_models: bool = False
    approx_bytes: int = 0


class S2CEmbeddedDownloadProgress(BaseModel):
    """Streamed progress during a ``c2s/embedded/download_model``.

    ``key`` / ``display_name`` name the model being fetched. ``stage`` is
    a short phase label (``resolving`` / ``downloading`` / ``saving``) so
    the UI can caption the bar without parsing byte counts. ``bytes_done``
    is the cumulative bytes written; ``bytes_total`` is the server's
    Content-Length when known, else ``None`` (indeterminate bar)."""

    model_config = ConfigDict(extra="forbid")
    key: str
    display_name: str
    stage: str
    bytes_done: int = 0
    bytes_total: int | None = None


class S2CTorchOverlayStatus(BaseModel):
    """Snapshot of the torch runtime overlay state.

    Sent in reply to ``c2s/torch_overlay/status`` AND as the terminal
    success message after a ``c2s/torch_overlay/install`` so the UI can
    refresh its "installed / active" state from one shape.

    ``recommended`` is the flavor the GPU probe suggests for this
    machine; ``installed`` lists the flavors with an overlay on disk;
    ``active`` is the flavor the entry shim will load on next launch
    (``None`` on a fresh install before any flavor is activated).
    ``runtime_dir`` is the resolved overlay root for diagnostics /
    tooltips. ``activated`` is set on the post-install reply to tell the
    UI whether a relaunch is needed to pick up the new flavor."""

    model_config = ConfigDict(extra="forbid")
    recommended: str
    installed: list[str] = Field(default_factory=list)
    active: str | None = None
    runtime_dir: str = ""
    activated: bool = False


class S2CTorchOverlayProgress(BaseModel):
    """Streamed download progress during a ``c2s/torch_overlay/install``.

    ``flavor`` names the flavor being installed. ``bytes_done`` is the
    cumulative bytes fetched across all of the flavor's wheels;
    ``bytes_total`` is the known total when the index/server exposed it,
    else ``None`` (the UI shows an indeterminate bar). ``stage`` is a
    short human-readable phase (``resolving`` / ``downloading`` /
    ``unpacking`` / ``installed``) so the UI can label the bar without
    parsing byte counts."""

    model_config = ConfigDict(extra="forbid")
    flavor: str
    stage: str
    bytes_done: int = 0
    bytes_total: int | None = None


# ---------- Side character expansion (used by US5) ----------


class SideCharacterExpansion(BaseModel):
    """Result of a one-line description fleshed into a full character."""

    model_config = ConfigDict(extra="forbid")
    character: Character


# ---------- Dispatch table ----------

C2S_PAYLOAD_BY_TYPE: dict[MessageType, type[BaseModel]] = {
    MessageType.c2s_hello: C2SHello,
    MessageType.c2s_saves_list: C2SSavesList,
    MessageType.c2s_saves_continue: C2SSavesContinue,
    MessageType.c2s_saves_load: C2SSavesLoad,
    MessageType.c2s_saves_rename: C2SSavesRename,
    MessageType.c2s_saves_delete: C2SSavesDelete,
    MessageType.c2s_new_game_start: C2SNewGameStart,
    MessageType.c2s_new_game_answer: C2SNewGameAnswer,
    MessageType.c2s_new_game_add_side_character: C2SNewGameAddSideCharacter,
    MessageType.c2s_new_game_edit_side_character: C2SNewGameEditSideCharacter,
    MessageType.c2s_new_game_delete_side_character: C2SNewGameDeleteSideCharacter,
    MessageType.c2s_new_game_edit_review: C2SNewGameEditReview,
    MessageType.c2s_new_game_confirm: C2SNewGameConfirm,
    MessageType.c2s_new_game_go_back: C2SNewGameGoBack,
    MessageType.c2s_new_game_surprise_me: C2SNewGameSurpriseMe,
    MessageType.c2s_play_advance: C2SPlayAdvance,
    MessageType.c2s_play_free_text: C2SPlayFreeText,
    MessageType.c2s_play_undo: C2SPlayUndo,
    MessageType.c2s_play_cancel: C2SPlayCancel,
    MessageType.c2s_edit_world: C2SEditWorld,
    MessageType.c2s_edit_character: C2SEditCharacter,
    MessageType.c2s_edit_character_dismiss: C2SEditCharacterDismiss,
    MessageType.c2s_edit_character_show: C2SEditCharacterShow,
    MessageType.c2s_edit_character_rerender: C2SEditCharacterRerender,
    MessageType.c2s_edit_history: C2SEditHistory,
    MessageType.c2s_edit_history_delete: C2SEditHistoryDelete,
    MessageType.c2s_edit_history_retcon: C2SEditHistoryRetcon,
    MessageType.c2s_edit_environment: C2SEditEnvironment,
    MessageType.c2s_edit_environment_rerender: C2SEditEnvironmentRerender,
    MessageType.c2s_edit_environment_apply: C2SEditEnvironmentApply,
    MessageType.c2s_settings_get: C2SSettingsGet,
    MessageType.c2s_settings_update: C2SSettingsUpdate,
    MessageType.c2s_settings_validate_api_key: C2SSettingsValidateApiKey,
    MessageType.c2s_embedded_list_models: C2SEmbeddedListModels,
    MessageType.c2s_embedded_recommend_model: C2SEmbeddedRecommendModel,
    MessageType.c2s_embedded_download_model: C2SEmbeddedDownloadModel,
    MessageType.c2s_torch_overlay_status: C2STorchOverlayStatus,
    MessageType.c2s_torch_overlay_install: C2STorchOverlayInstall,
    MessageType.c2s_music_inventory: C2SMusicInventory,
    MessageType.c2s_music_regenerate: C2SMusicRegenerate,
    MessageType.c2s_app_exit: C2SAppExit,
}

S2C_PAYLOAD_BY_TYPE: dict[MessageType, type[BaseModel]] = {
    MessageType.s2c_hello: S2CHello,
    MessageType.s2c_saves_list: S2CSavesList,
    MessageType.s2c_state_full: S2CStateFull,
    MessageType.s2c_state_patch: S2CStatePatch,
    MessageType.s2c_text_streaming: S2CTextStreaming,
    MessageType.s2c_text_complete: S2CTextComplete,
    MessageType.s2c_image_ready: S2CImageReady,
    MessageType.s2c_music_ready: S2CMusicReady,
    MessageType.s2c_error: S2CError,
    MessageType.s2c_notice: S2CNotice,
    MessageType.s2c_cost: S2CCost,
    MessageType.s2c_embedded_models: S2CEmbeddedModels,
    MessageType.s2c_embedded_recommended_model: S2CEmbeddedRecommendedModel,
    MessageType.s2c_embedded_download_progress: S2CEmbeddedDownloadProgress,
    MessageType.s2c_settings_api_key_validation: S2CSettingsApiKeyValidation,
    MessageType.s2c_torch_overlay_status: S2CTorchOverlayStatus,
    MessageType.s2c_torch_overlay_progress: S2CTorchOverlayProgress,
    MessageType.s2c_music_inventory: S2CMusicInventory,
    MessageType.s2c_play_cancelled: S2CPlayCancelled,
}

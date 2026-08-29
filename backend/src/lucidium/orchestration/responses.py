"""Internal Pydantic schemas for parsed LLM responses.

These types are *not* IPC types — they are the validated shapes we extract
from raw model output before they are allowed to reach the player. Per
Constitution I, every LLM output passes through one of these models;
unparseable output triggers retry, never an unchecked render.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..api.errors import ProviderValidationError
from ..domain.character import (
    DEFAULT_AGE,
    DEFAULT_EXPRESSION,
    DEFAULT_OUTFIT,
    DEFAULT_POSE,
    Fact,
)
from ..domain.dialog import CharacterChange, DialogOption, NewCharacterDescriptor
from ..domain.world import PlotStage, PlotThread

_log = logging.getLogger(__name__)


class LlmCharacterPayload(BaseModel):
    """Full character attributes (no seed, no images)."""

    # LLM output is best-effort; extra fields the model invents
    # ("notes", "internal_thoughts", whatever) are silently dropped
    # rather than failing the whole call.
    model_config = ConfigDict(extra="ignore")

    name: str
    description: str
    # Anatomy / staging fields all default so a single omission in the
    # LLM's character block doesn't fail the whole call and force a
    # retry (missing fields self-heal to neutral values — see
    # character.py). Empty anatomy strings are filtered out of the
    # portrait prompt downstream; only ``name`` / ``description``
    # stay required as the character's irreducible identity.
    gender: str = ""
    # Optional narrative pronouns the storyteller uses to address
    # this character. Empty → narrator derives from gender. Free
    # text so non-binary forms work.
    pronouns: str = ""
    age: int = Field(default=DEFAULT_AGE, ge=0)
    ethnicity: str = ""
    skin: str = ""
    hair_color: str = ""
    hairstyle: str = ""
    eye_color: str = ""
    build: str = ""
    bust: str = ""
    outfit: str = DEFAULT_OUTFIT
    pose: str = DEFAULT_POSE
    expression: str = DEFAULT_EXPRESSION
    # Visible physical effects (cuts, bruises, soot, soaked clothes,
    # restraints). Optional — LLMs that predate this field default
    # to empty. Stays separate from outfit so wardrobe doesn't
    # conflate with injuries.
    effects: str = ""
    # Permanent skin markings (tattoos, scars, birthmarks, brands).
    # Optional; persists across outfit / pose / expression changes.
    decals: str = ""


class LlmBeat(BaseModel):
    """One narrative beat. Each beat becomes its own dialog node so
    metadata (speaker, character pose/expression/outfit, location) can
    change between beats and only one beat is presented to the user
    at a time (FR-010a)."""

    # LLM output is best-effort; extra fields the model invents
    # ("notes", "internal_thoughts", whatever) are silently dropped
    # rather than failing the whole call.
    model_config = ConfigDict(extra="ignore")

    text: str  # single line, no embedded newlines
    speaker_id: str | None = None
    entering_character_ids: list[str] = Field(default_factory=list)
    leaving_character_ids: list[str] = Field(default_factory=list)
    # Full descriptors for any character this beat introduces. Their
    # ids must also appear in ``entering_character_ids`` — the handler
    # creates Characters from these before computing the on-stage list.
    new_characters: list[NewCharacterDescriptor] = Field(default_factory=list)
    location_id: str | None = None
    location_prompt: str | None = None
    location_lighting: str = ""
    character_changes: list[CharacterChange] = Field(default_factory=list)
    # Optional music swap. When the storyteller wants the
    # background score to change (a tonal shift, entering a new
    # scene type, the narrative tipping into action) it emits a
    # short text prompt here. The handler queues an ACE-Step
    # render against this prompt and the renderer crossfades to
    # the new track. Empty / None means "don't change the music".
    # Ignored entirely when ``MusicSettings.enabled`` is False —
    # the storyteller is told this in the prompt rules so it
    # doesn't waste tokens on music_change emissions the engine
    # will drop anyway.
    music_change: str | None = None


class LlmDialogPayload(BaseModel):
    """A sequence of beats and the options that follow the last one.

    The handler creates one ``DialogNode`` per beat, chained by
    parent_id, all marked ``state=committed``. The user walks them
    one at a time; only the LAST beat carries options (the rest
    show a single ``Continue`` affordance).
    """

    # LLM output is best-effort; extra fields the model invents
    # ("notes", "internal_thoughts", whatever) are silently dropped
    # rather than failing the whole call.
    model_config = ConfigDict(extra="ignore")

    beats: list[LlmBeat]
    options: list[DialogOption] = Field(default_factory=list)


class LlmOptionList(BaseModel):
    # LLM output is best-effort; extra fields the model invents
    # ("notes", "internal_thoughts", whatever) are silently dropped
    # rather than failing the whole call.
    model_config = ConfigDict(extra="ignore")

    options: list[str]


class LlmWorldInit(BaseModel):
    # LLM output is best-effort; extra fields the model invents
    # ("notes", "internal_thoughts", whatever) are silently dropped
    # rather than failing the whole call.
    model_config = ConfigDict(extra="ignore")

    game_name: str
    # The structured multi-stage outline (preferred). Empty for
    # legacy responses that only emit ``overall_plot_direction``;
    # the handler synthesises a single-stage outline from that
    # string in the legacy case.
    plot_outline: list[PlotStage] = Field(default_factory=list)
    # Legacy single-string field. Retained so old recorded
    # fixtures and slow-to-update LLM prompts keep parsing.
    overall_plot_direction: str = ""
    active_plot_threads: list[PlotThread] = Field(default_factory=list)
    opening_node: LlmDialogPayload
    player_character: LlmCharacterPayload | None = None
    # Initial background-music prompt. ACE-Step renders this into
    # a looping instrumental track played under the whole game
    # until a beat-emitted ``music_change`` swaps it. Empty when
    # the LLM didn't propose a music direction (or when music
    # gen is disabled — the prompt only asks for this when the
    # setting is on, so it's typically populated).
    initial_music_prompt: str = ""


class LlmRetconBeatRewrite(BaseModel):
    """One rewritten history beat from a retcon pass. ``node_id``
    must match an existing committed dialog node — unknown ids are
    silently dropped by the handler."""

    model_config = ConfigDict(extra="ignore")

    node_id: str
    text: str
    speaker_id: str | None = None


class LlmRetconCharacterUpdate(BaseModel):
    """A single character-attribute change the retcon implies. The
    handler rejects updates whose ``field`` isn't a known
    ``CharacterAttributeField`` rather than failing the whole
    response."""

    model_config = ConfigDict(extra="ignore")

    character_id: str
    field: str
    new_value: str


class LlmRetconResult(BaseModel):
    """Aggregate result of a retcon LLM call. The handler walks the
    rewrites in order, dropping any that reference unknown nodes or
    characters; ``world_updates`` may carry replacement values for
    free-text world fields like ``summarizer_assessment`` or
    ``overall_plot_direction``."""

    model_config = ConfigDict(extra="ignore")

    rewritten_beats: list[LlmRetconBeatRewrite] = Field(default_factory=list)
    character_updates: list[LlmRetconCharacterUpdate] = Field(default_factory=list)
    world_updates: dict[str, str] = Field(default_factory=dict)


class LlmRetconBeatBatch(BaseModel):
    """One batch's worth of beat rewrites. The batched retcon flow
    sends N small LLM calls (one per ~4-beat slice) instead of one
    giant call that truncates against non-frontier models'
    short-output windows. Each batch returns ONLY the
    ``rewritten_beats`` for ids in its slice."""

    model_config = ConfigDict(extra="ignore")

    rewritten_beats: list[LlmRetconBeatRewrite] = Field(default_factory=list)


class LlmRetconCharacterUpdates(BaseModel):
    """Standalone character-updates pass for the batched retcon
    flow. Output is short regardless of history length, so the
    default max_tokens cap is plenty."""

    model_config = ConfigDict(extra="ignore")

    character_updates: list[LlmRetconCharacterUpdate] = Field(default_factory=list)


class LlmSurpriseMeScenario(BaseModel):
    """The scenario synthesised by the SURPRISE ME path. Fields map
    one-to-one onto the InterviewState slots the regular onboarding
    flow fills in across multiple steps; the surprise-me handler
    populates them all at once and then fans straight into the
    same world_init pipeline."""

    model_config = ConfigDict(extra="ignore")

    setting: str
    genre: str
    character_description: str
    name: str


class LlmRetconWorldUpdates(BaseModel):
    """Standalone world-updates pass — at most two short string
    replacements (``summarizer_assessment`` / ``overall_plot_direction``)."""

    model_config = ConfigDict(extra="ignore")

    world_updates: dict[str, str] = Field(default_factory=dict)


class LlmFactEntry(BaseModel):
    """One consolidated fact returned by the facts cleanup pass."""

    model_config = ConfigDict(extra="ignore")

    text: str
    confidence: str = "inferred"


class LlmFactsConsolidation(BaseModel):
    """Output of the per-character facts cleanup job. Keyed by
    character id; each list is the LLM's merged / dedup'd version
    of that character's accumulated facts. The handler walks the
    map and applies updates only to the characters it actually
    asked about (silently drops hallucinated ids)."""

    model_config = ConfigDict(extra="ignore")

    facts_by_character: dict[str, list[LlmFactEntry]] = Field(default_factory=dict)


class LlmProfileConsolidation(BaseModel):
    """Replacement values for the cross-save user profile after a
    consolidation pass. The LLM is asked to merge overlapping tags
    and drop anything tied to a specific storyline (character names,
    plot beats, save-specific stakes) so the profile stays a stable
    cross-save signal of player taste."""

    model_config = ConfigDict(extra="ignore")

    likes: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class LlmUserProfileAdditions(BaseModel):
    """Inferences the summarizer wants to append to the cross-save
    user profile. Empty lists in the typical case — the summarizer
    only adds entries when it has fresh evidence over multiple turns.
    """

    model_config = ConfigDict(extra="ignore")

    likes: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class LlmSummaryResult(BaseModel):
    # LLM output is best-effort; extra fields the model invents
    # ("notes", "internal_thoughts", whatever) are silently dropped
    # rather than failing the whole call.
    model_config = ConfigDict(extra="ignore")

    summarizer_assessment: str
    direction_signal: str = "none"
    new_facts_by_character: dict[str, list[Fact]] = Field(default_factory=dict)
    pruned_fact_ids: list[str] = Field(default_factory=list)
    # Pointer into ``WorldState.plot_outline``. The summarizer picks
    # which stage is currently active based on what the actual story
    # is doing — usually the same id as before, sometimes the next
    # stage (the player progressed past the current one).
    current_stage_id: str | None = None
    # If the actual story has drifted enough that the planned
    # outline no longer fits, the summarizer emits a replacement.
    # Set to None means "outline is still good, leave it alone."
    revised_outline: list[PlotStage] | None = None
    # New tags to append to the cross-save user profile. Most calls
    # should be empty — only populate when there's fresh evidence.
    user_profile_additions: LlmUserProfileAdditions = Field(default_factory=LlmUserProfileAdditions)
    # Character ids to remove from ``Game.on_stage``. The summarizer
    # uses this to declutter the stage when a character has been
    # idle on screen for many beats — they haven't spoken, entered
    # or left, taken any character_change, etc. Without this, a
    # one-line walk-on at beat 3 stays visibly on the stage for the
    # rest of the run, eating CLIP tokens and visually crowding
    # the actually-active speakers. Empty in most calls.
    characters_to_offstage: list[str] = Field(default_factory=list)


T = TypeVar("T", bound=BaseModel)


def _coerce_legacy_dialog_shape(data: object) -> object:
    """Tolerate LLM responses that emit the pre-FR-010a single-text
    shape: lift ``{text, options, ...}`` into ``{beats: [...], options}``
    so the model parses without retry storms.

    Also handles the LLM emitting ``text`` with embedded newlines —
    each newline-separated line becomes a beat.
    """
    if not isinstance(data, dict):
        return data
    if "beats" in data:
        return data
    if "text" not in data:
        return data
    text = data.get("text") or ""
    options = data.get("options", [])
    lines = [line.strip() for line in str(text).split("\n") if line.strip()]
    if not lines:
        return data
    beats = [
        {
            "text": line,
            "speaker_id": data.get("speaker_id"),
            "entering_character_ids": data.get("entering_character_ids", []) if i == 0 else [],
            "leaving_character_ids": data.get("leaving_character_ids", [])
            if i == len(lines) - 1
            else [],
            "new_characters": data.get("new_characters", []) if i == 0 else [],
            "location_id": data.get("location_id") if i == 0 else None,
            "location_prompt": data.get("location_prompt") if i == 0 else None,
            "character_changes": data.get("character_changes", []) if i == 0 else [],
        }
        for i, line in enumerate(lines)
    ]
    return {"beats": beats, "options": options}


def _strip_json_preamble(raw: str) -> str:
    """Strip any leading prose / Markdown fence from an LLM JSON
    response so ``json.loads`` sees a clean payload.

    Models often narrate their reasoning before emitting JSON
    ("I'll start with the perspective bucket since that defines the
    narrative voice.\\n\\n```json\\n{...}\\n```"). We tolerate that by:

      1. Stripping a leading ```` ``` ```` or ```` ```json ```` fence
         when the response is fence-wrapped.
      2. Otherwise, finding the first ``{`` or ``[`` and slicing
         from there — anything before is discarded as commentary.

    Trailing prose / fence-close (```` ``` ```` after the JSON) is
    handled later by ``JSONDecoder.raw_decode`` in ``parse_json_object``.

    The fall-through case is "no fence, no brace, no bracket" — we
    return the input unchanged and let json.loads fail the same way
    it would have before. We never wrap or mutate JSON content; the
    helper only removes leading non-JSON characters.
    """
    candidate = raw.strip()
    # Fence-wrapped: ```json ... ``` or ``` ... ```. Strip the
    # opening fence + optional ``json`` language tag in one pass.
    # Doesn't matter if there's no closing fence — raw_decode below
    # ignores trailing content past the parsed object.
    if candidate.startswith("```"):
        # Slice past the opening line: ```json\n or ```\n
        nl = candidate.find("\n")
        if nl != -1:
            candidate = candidate[nl + 1 :].lstrip()
        else:
            # Single-line fence like ```{...}``` — drop the backticks
            # and any leading "json" tag.
            inner = candidate.strip("`").strip()
            if inner.startswith("json"):
                inner = inner[4:].lstrip()
            candidate = inner
        # If we still have a leading ``` after stripping the opening,
        # it was the closing one — drop it too.
        if candidate.startswith("```"):
            candidate = candidate[3:].lstrip()
    # Conversational preamble: "I'll start with… \n\n{ ... }". Slice
    # from the first JSON-start character. We pick the EARLIER of
    # ``{`` and ``[`` so the helper works for object and array
    # payloads alike; raw_decode in the caller stops parsing at the
    # matching close brace so trailing prose is harmless.
    first_obj = candidate.find("{")
    first_arr = candidate.find("[")
    candidates = [pos for pos in (first_obj, first_arr) if pos != -1]
    if candidates:
        start = min(candidates)
        if start > 0:
            candidate = candidate[start:]
    return candidate


def _coerce_legacy_world_init(data: object) -> object:
    if not isinstance(data, dict):
        return data
    opening = data.get("opening_node")
    if isinstance(opening, dict) and "beats" not in opening:
        data = {**data, "opening_node": _coerce_legacy_dialog_shape(opening)}
    return data


def parse_json_object(raw: str, into: type[T]) -> T:
    """Strip an optional Markdown fence and validate ``raw`` against ``into``.

    Models often wrap JSON in ```json ... ``` fences. We tolerate that
    one specific decoration; anything else is a validation failure that
    propagates as ``ProviderValidationError`` for the caller's retry
    loop to handle.

    Three layers of recovery, applied in order:

      1. **Truncation repair.** When ``json.loads`` fails AND the
         model appears to have been cut off mid-output (LLM hit
         ``max_tokens``, finished a partial string), we attempt a
         best-effort repair — close the open string, drop the
         partial trailing pair, and balance unclosed braces /
         brackets — then validate whatever made it through.
         Pydantic drops missing optional fields; the caller gets a
         partial-but-usable payload instead of losing the whole
         pass.

      2. **Legacy shape coercion.** Old prompt shapes (single-text
         dialog payloads) get lifted into the modern beats array.

      3. **Null-typed-field coercion.** When validation fails with
         "null where string/list/dict expected", walk the errors
         and patch the data with the right empty value (``""`` /
         ``[]`` / ``{}``), then re-validate. Catches the most
         common LLM-mistake-that-isn't-actually-broken: emitting
         ``"effects": null`` for a character with no visible
         injuries instead of either omitting the field or sending
         ``""``.
    """
    candidate = _strip_json_preamble(raw)
    try:
        # ``raw_decode`` parses one JSON value and ignores anything
        # past the matching close brace, so trailing prose ("...and
        # that's the breakdown.") or fence-close (```` ``` ````)
        # after the payload doesn't break the parse.
        data, _end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError as initial_exc:
        # First fallback: tolerate ASCII control characters inside
        # string values. The LLM occasionally emits a raw ``\n`` /
        # ``\t`` / etc. inside a string instead of escaping it,
        # which json.loads with the default strict=True rejects
        # at parse time. Real failure shape captured: a
        # surprise_me scenario with a literal newline inside the
        # ``setting`` value bricked the entire new-game flow.
        try:
            data, _end = json.JSONDecoder(strict=False).raw_decode(candidate)
        except json.JSONDecodeError:
            data = None  # let the truncation-repair branch run
        if data is not None:
            _log.info(
                "LLM payload had raw control characters in string values "
                "for %s; parsed via strict=False",
                into.__name__,
            )
        else:
            repaired = _repair_truncated_json(candidate)
            if repaired is not None and repaired != candidate:
                try:
                    # Use strict=False here too — the truncation
                    # repair fixes structural breakage but leaves
                    # in-string control chars alone.
                    data = json.loads(repaired, strict=False)
                except json.JSONDecodeError:
                    _log_parse_failure(into, raw, initial_exc)
                    raise ProviderValidationError(
                        f"LLM did not return JSON: {initial_exc}"
                    ) from initial_exc
                else:
                    _log.warning(
                        "LLM truncated JSON for %s; salvaged %d chars via "
                        "repair (raw was %d chars). Last 200 chars of raw: %r",
                        into.__name__,
                        len(repaired),
                        len(candidate),
                        candidate[-200:],
                    )
            else:
                _log_parse_failure(into, raw, initial_exc)
                raise ProviderValidationError(
                    f"LLM did not return JSON: {initial_exc}"
                ) from initial_exc
    # Single-element-array wrapper unwrap. Real failure shape:
    # the Surprise Me LLM call returned ``[{...the scenario...}]``
    # — one object wrapped in an array — and Pydantic rejected
    # with ``Input should be a valid dictionary``. The model
    # wraps in arrays unpredictably; auto-unwrap one-element
    # arrays when the target schema is an object model so we
    # don't waste a full retry on a trivially-recoverable shape
    # mistake. Multi-element arrays are NOT unwrapped (we'd be
    # picking which element to keep — let validation fail loudly).
    if (
        isinstance(data, list)
        and len(data) == 1
        and isinstance(data[0], dict)
        and isinstance(into, type)
        and issubclass(into, BaseModel)
    ):
        _log.info(
            "LLM payload was a single-element array for %s; unwrapping",
            into.__name__,
        )
        data = data[0]
    # Best-effort coercion for legacy shapes — saves a retry round
    # trip when the model emits a single-text dialog payload.
    if into is LlmDialogPayload:
        data = _coerce_legacy_dialog_shape(data)
    elif into is LlmWorldInit:
        data = _coerce_legacy_world_init(data)
    try:
        return into.model_validate(data)
    except ValidationError as exc:
        # Layer 3: null-coerce the specific paths that errored
        # and try once more. Caps at one retry — if the second
        # validate still fails, raise the ORIGINAL error so the
        # log message accurately describes what the LLM emitted
        # (a coercion-failure cascade would obscure the root
        # cause).
        coerced = _coerce_null_typed_fields(data, exc)
        if coerced is not None:
            try:
                _log.info(
                    "LLM payload had null-typed-field errors for %s; "
                    "coerced %d path(s) to defaults and retrying",
                    into.__name__,
                    _count_coercion_paths(exc),
                )
                return into.model_validate(coerced)
            except ValidationError:
                pass
        _log.warning(
            "LLM payload failed %s schema; first 400 chars of raw: %r; data keys: %s",
            into.__name__,
            raw[:400],
            list(data.keys()) if isinstance(data, dict) else type(data).__name__,
        )
        raise ProviderValidationError(f"LLM payload failed schema {into.__name__}: {exc}") from exc


# Pydantic v2 error-type strings we know how to coerce. Each maps
# the expected type to the empty-value sentinel the LLM should
# have sent. ``string_type`` fires when the LLM put ``null``
# under a ``str`` field; ``list_type`` and ``dict_type`` are the
# corresponding errors for ``list[X]`` and ``dict[K, V]``.
_NULL_COERCION_DEFAULTS: dict[str, object] = {
    "string_type": "",
    "list_type": [],
    "dict_type": {},
}


def _coerce_null_typed_fields(
    data: object,
    exc: ValidationError,
) -> object | None:
    """Walk the Pydantic ValidationError and patch null values
    that should have been empty strings / lists / dicts.

    Returns a deep-copied + patched data structure when at least
    one error was fixable, or ``None`` when none of the errors
    matched the recoverable patterns. Idempotent: a second call
    on the same data is a no-op.

    Why error-driven rather than schema-driven: the schema-walk
    approach (introspect ``into.model_fields``, find every str
    field, replace nulls) is cleaner in principle but bogs down
    in nested generics (``dict[str, list[Fact]]``, etc.) and
    discriminated unions. Letting Pydantic tell us EXACTLY which
    paths errored — via ``exc.errors()`` — sidesteps the
    schema-walk entirely.
    """
    if not isinstance(data, (dict, list)):
        return None
    fixable_errors = [
        err
        for err in exc.errors()
        if err.get("type") in _NULL_COERCION_DEFAULTS and err.get("input") is None
    ]
    if not fixable_errors:
        return None
    # Deep copy so we don't mutate the caller's data even on
    # failure paths — original is preserved for the log message.
    patched = json.loads(json.dumps(data))
    fixed_any = False
    for err in fixable_errors:
        loc = err.get("loc") or ()
        default = _NULL_COERCION_DEFAULTS[err["type"]]
        if _set_at_path(patched, loc, default):
            fixed_any = True
    return patched if fixed_any else None


def _count_coercion_paths(exc: ValidationError) -> int:
    return sum(
        1
        for err in exc.errors()
        if err.get("type") in _NULL_COERCION_DEFAULTS and err.get("input") is None
    )


async def call_llm_json_with_retry(
    session: Any,
    prompt: list[dict[str, str]],
    *,
    parse_into: type[T],
    max_attempts: int = 3,
    max_tokens: int | None = None,
) -> T:
    """Call the LLM and parse the response into ``parse_into``,
    retrying on validation failure with corrective context.

    Each retry appends the failed response and the validation
    error to the conversation, then asks the LLM to re-emit the
    JSON with the errors fixed. Bounded by ``max_attempts``
    (default 3) so a malformed prompt doesn't burn the whole
    retry budget on a single bad call. The first attempt uses
    the original prompt verbatim; subsequent attempts use a
    growing conversation that includes the corrective hints.

    Recoverable failures (the null-typed-field case) are handled
    inside ``parse_json_object`` without burning a retry. This
    helper exists for the LARGER class of validation failures —
    missing required fields, wrong types Pydantic can't coerce
    automatically, structural mismatches — that need the LLM
    to actually fix its output.

    Returns the parsed model. Raises the LAST
    ``ProviderValidationError`` if all attempts fail.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    current_prompt: list[dict[str, str]] = list(prompt)
    last_exc: ProviderValidationError | None = None
    for attempt in range(max_attempts):
        raw, _chunks = await session.llm_text(
            current_prompt,
            max_tokens=max_tokens,
        )
        try:
            return parse_json_object(raw, parse_into)
        except ProviderValidationError as exc:
            last_exc = exc
            if attempt + 1 >= max_attempts:
                break
            _log.warning(
                "LLM JSON validation failed on attempt %d/%d for %s; "
                "re-prompting with corrective hint",
                attempt + 1,
                max_attempts,
                parse_into.__name__,
            )
            # Build the corrective continuation: assistant's bad
            # response, then a user message that quotes the
            # validation error and asks for a fix. Keeping the bad
            # response in the conversation lets the LLM see what
            # it produced and edit it, instead of starting over
            # cold (which often re-emits the same mistake).
            current_prompt = [
                *prompt,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Your previous response failed to parse. "
                        "Validation error:\n\n"
                        f"{exc}\n\n"
                        "Re-emit the JSON with these errors fixed. "
                        "Return ONLY the corrected JSON, no commentary. "
                        "Common cause: emitting `null` for a field that "
                        'must be a string, list, or dict — use "", '
                        "[], or {} instead. Another common cause: "
                        "omitting a required field — fill it in with "
                        "a sensible default."
                    ),
                },
            ]
    assert last_exc is not None  # max_attempts >= 1 guarantees this
    raise last_exc


def _set_at_path(
    container: object,
    loc: tuple[object, ...],
    value: object,
) -> bool:
    """Walk ``loc`` (Pydantic's error path: a tuple of dict keys
    and list indices) into ``container`` and set the leaf to
    ``value``. Returns True iff the path resolved cleanly.

    Pydantic's ``loc`` mixes strings (dict keys) and ints (list
    indices), so each step inspects the current node type and
    indexes accordingly. Out-of-range or wrong-type steps are
    treated as unfixable rather than raised — this is a recovery
    path; safety beats completeness.
    """
    if not loc:
        return False
    cur: Any = container
    for step in loc[:-1]:
        try:
            if isinstance(cur, list) and isinstance(step, int):
                cur = cur[step]
            elif isinstance(cur, dict):
                cur = cur[step]
            else:
                return False
        except (KeyError, IndexError, TypeError):
            return False
    last = loc[-1]
    try:
        if isinstance(cur, list) and isinstance(last, int):
            cur[last] = value
        elif isinstance(cur, dict):
            cur[last] = value
        else:
            return False
    except (KeyError, IndexError, TypeError):
        return False
    return True


def _log_parse_failure(into: type, raw: str, exc: Exception) -> None:
    """Common log shape for ``parse_json_object`` failures. Includes
    the LAST 200 chars in addition to the first 400, since a
    truncation cuts at the tail and the failure cause sits there
    rather than at the front."""
    _log.warning(
        "LLM did not return JSON for %s (%s). first 400 chars: %r; "
        "last 200 chars: %r; total len: %d",
        into.__name__,
        exc,
        raw[:400],
        raw[-200:],
        len(raw),
    )


def _wrap_missing_outer_brackets(text: str) -> str:
    """Prepend missing outer ``{`` / ``[`` when the text contains
    closers without matching openers — i.e. the LLM dropped the
    opening brace of the top-level object (or array) but kept the
    body and the closer.

    Real failure shape: the summarizer LLM emitted
    ```
    ```
      "summarizer_assessment": ...,
      ...
    }
    ```
    ```
    After fence-strip the candidate ends with ``}`` but starts with
    a key — ``json.loads`` fails with ``Extra data`` at the first
    string. Walking the text shows one ``}`` with no matching ``{``;
    we prepend the missing opener.

    Idempotent on well-formed JSON: a balanced text has zero
    unmatched closers and the prefix is empty. Multiple missing
    closers (rare; nested object body) get the right count of
    opening braces.
    """
    if not text:
        return text
    open_obj = 0
    open_arr = 0
    leading_close_obj = 0
    leading_close_arr = 0
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            open_obj += 1
        elif ch == "[":
            open_arr += 1
        elif ch == "}":
            if open_obj > 0:
                open_obj -= 1
            else:
                leading_close_obj += 1
        elif ch == "]":
            if open_arr > 0:
                open_arr -= 1
            else:
                leading_close_arr += 1
    if leading_close_obj == 0 and leading_close_arr == 0:
        return text
    prefix = "{" * leading_close_obj + "[" * leading_close_arr
    return prefix + text


def _repair_missing_key_colon(text: str) -> str:
    """Insert missing ``":`` between a key and its bracket-typed
    value when the LLM dropped the close-quote and colon.

    Real failure shape: the summarizer LLM emitted
    ``"dislikes [],`` instead of ``"dislikes": [],``. The brace
    structure stays balanced (the unclosed string just absorbs the
    next chunk of text up to the next ``"``), so the truncation
    repair can't find a useful safe boundary — but the actual fix
    is mechanical: an open-quote-then-identifier-then-bracket
    sequence in OBJECT-KEY position with no close-quote in between
    is almost certainly a key whose ``":`` was dropped.

    The "object-key position" guard is what keeps legitimate
    string values like ``"items [a, b]"`` (which sits in
    array-element or object-value position) from being mangled.
    We track the bracket stack and a ``expect_key`` flag —
    only at the start of a fresh object slot does the heuristic
    fire.
    """
    if not text:
        return text
    out: list[str] = []
    in_string = False
    escape = False
    # Stack of currently-open container types: "{" or "[".
    container_stack: list[str] = []
    # True iff the next ``"`` we see (outside a string) is a key
    # rather than a value — i.e. we're at the start of an object
    # slot. Flipped on ``{`` and on ``,`` while inside an object;
    # cleared on ``:`` (we just bound the key, the next ``"`` is
    # the value).
    expect_key = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == "{":
            container_stack.append("{")
            expect_key = True
            out.append(ch)
            i += 1
            continue
        if ch == "[":
            container_stack.append("[")
            expect_key = False
            out.append(ch)
            i += 1
            continue
        if ch in "}]":
            if container_stack:
                container_stack.pop()
            expect_key = False
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            # Comma between elements: at the next slot we expect
            # a key when the immediate container is an object.
            expect_key = bool(container_stack) and container_stack[-1] == "{"
            out.append(ch)
            i += 1
            continue
        if ch == ":":
            # Bound a key; next ``"`` will be a value.
            expect_key = False
            out.append(ch)
            i += 1
            continue
        if ch != '"':
            out.append(ch)
            i += 1
            continue
        # Outside string, on an opening quote. Only inspect for
        # the malformed-key shape when we're in object-key
        # position; otherwise treat as a normal string opening.
        if expect_key:
            j = i + 1
            if j < n and (text[j].isalpha() or text[j] == "_"):
                key_start = j
                while j < n and (text[j].isalnum() or text[j] in ("_", "-")):
                    j += 1
                if j > key_start:
                    k = j
                    while k < n and text[k] in (" ", "\t"):
                        k += 1
                    if k < n and text[k] in ("[", "{"):
                        # Confirmed: ``"id<ws>+[`` / ``"id<ws>+{``
                        # in key position with no close-quote in
                        # between. Rewrite as ``"id":<bracket>``.
                        out.append('"')
                        out.append(text[key_start:j])
                        out.append('":')
                        if k > j:
                            out.append(" ")
                        # Don't consume the bracket — let the main
                        # loop process it normally so the stack /
                        # expect_key flags update.
                        expect_key = False
                        i = k
                        continue
        # Normal string opening (key in well-formed JSON or value).
        out.append(ch)
        in_string = True
        # If this was a well-formed key, expect_key stays True
        # until the close-quote arrives and the colon follows;
        # the close-quote keeps in_string updates only — the colon
        # branch above clears expect_key.
        i += 1
    return "".join(out)


def _repair_truncated_json(text: str) -> str | None:
    """Best-effort repair of LLM-truncated JSON.

    Salvage path for ``parse_json_object`` when the LLM hit
    max_tokens mid-output. Targets five classic truncation
    artefacts:

      * unterminated string — the cut landed inside a string
        value (or key) and there's no closing ``"``
      * trailing key without value — ``..., "foo":`` or
        ``..., "foo": `` cut before the value started
      * trailing comma — ``..., "foo": "bar",`` cut before
        the next key
      * unclosed braces / brackets — outer ``{`` / ``[`` never
        balanced because the cut landed inside
      * missing ``":`` between a key and a bracket-typed value —
        ``"dislikes [],`` instead of ``"dislikes": [],``. Fixed
        by ``_repair_missing_key_colon`` as a pre-pass before the
        truncation walk.

    Walks the text once tracking string / escape / bracket-depth
    state, then rewinds back to the last well-formed boundary
    and re-balances. Returns the repaired string, or ``None`` if
    the input is too damaged to repair (we don't want to feed
    junk into ``json.loads`` and amplify the parse error).

    Pure-stdlib implementation — adding ``json-repair`` as a
    runtime dep felt heavyweight for one defensive code path.
    """
    if not text:
        return None
    # Pre-pass 1: wrap the top-level when the LLM dropped the
    # opening ``{`` (or ``[``) but kept the body and the closer.
    # Idempotent on balanced input.
    text = _wrap_missing_outer_brackets(text)
    # Pre-pass 2: rewrite ``"<id><ws>[`` / ``"<id><ws>{`` malformed
    # keys before the truncation walk. If the input was actually
    # well-formed, the pre-pass is a no-op.
    text = _repair_missing_key_colon(text)
    stack: list[str] = []
    in_string = False
    escape = False
    # Track the index of the last well-formed boundary OUTSIDE a
    # string. A "boundary" is a position right after a comma, a
    # colon-value pair completion, or an opening brace/bracket —
    # i.e., a point where we could legitimately stop and balance
    # the structure. Initialised to -1 ("no boundary seen yet").
    safe_end = -1
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
            safe_end = i + 1
            continue
        if ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            safe_end = i + 1
            continue
        if ch == ",":
            # Comma between elements — anything UP TO here is
            # complete. Mark this position as a safe rewind point
            # (we'll trim the trailing comma when rebalancing).
            safe_end = i
            continue
        # Whitespace / digits / null / true / false / colon — none
        # of these advance the safe_end. Specifically, ``:`` is
        # NOT a safe boundary because the value after it might be
        # incomplete; ``[1, 2,`` is repairable but ``{"a":`` is
        # not (we'd need to drop the dangling key).
    # If nothing was ever closed and we never saw a comma, refuse —
    # there's no structure to salvage.
    if safe_end < 0:
        return None
    repaired = text[:safe_end]
    # If we trimmed mid-string (cut at a comma INSIDE a string
    # somehow — rare but possible if the comma char was the last
    # one we saw before the truncation point), the tail logic
    # above wouldn't apply because we wouldn't have set safe_end
    # inside the string. So no extra string-close needed here.
    repaired = repaired.rstrip(", \t\r\n")
    # Rebuild the bracket-stack against the trimmed prefix to
    # know how many closers we owe. The original ``stack``
    # reflects the FULL text; we need it for the trimmed prefix.
    sub_stack: list[str] = []
    in_string = False
    escape = False
    for ch in repaired:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            sub_stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if sub_stack and sub_stack[-1] == ch:
                sub_stack.pop()
    # If we're still inside a string after walking the prefix,
    # something pathological happened (a string straddled our
    # boundary). Bail rather than mangle.
    if in_string:
        return None
    while sub_stack:
        repaired += sub_stack.pop()
    # Sanity: result must start the same as the input (we only
    # ever trim from the right). If we somehow returned an empty
    # string, that's not useful.
    if not repaired:
        return None
    return repaired

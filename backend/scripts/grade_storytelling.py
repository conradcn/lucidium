"""Storytelling-quality grading harness.

Drives the orchestration sidecar through a fixed scenario (interview +
several play turns) using the configured LLM, captures the committed
dialog history, and asks a separate LLM as judge to grade visual-novel
tone and continuity consistency.

Output: writes commentary + the captured story to stderr; prints a
single integer 0-100 grade to stdout (the last line). Designed for the
``autoresearch`` verify pipeline:

    python backend/scripts/grade_storytelling.py 2>backend/scripts/grade_storytelling.log | tail -1

Determinism: the playthrough uses the configured (temperature, model)
so each run will produce a different opening — the loop signal must
come from grade trends across iterations, not single-run differences.
The judge runs at temperature 0 so its rubric is reproducible given a
fixed story.

Cost: ~7-9 LLM calls per run (visual_style, char_desc, name,
world_init, 5 advances). Plus 1 judge call. At deepseek-v3.2 prices,
this is well under $0.05 per run; 20 iterations totals < $1.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Allow running as a script: add the backend src to sys.path.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend" / "src"))

from lucidium.api.handlers import HandlerContext, build_default_registry  # noqa: E402
from lucidium.api.messages import (  # noqa: E402
    Envelope,
    InterviewStep,
    MessageType,
)
from lucidium.config import settings_path  # noqa: E402
from lucidium.domain.settings import Settings  # noqa: E402
from lucidium.orchestration.session import Session  # noqa: E402
from lucidium.providers.llm_client import OpenAiCompatibleLlmClient  # noqa: E402

# ----- Fixed scenario --------------------------------------------------------

SCENARIO_SETTING = "A stone harbor at dawn"
SCENARIO_GENRE = "Mystery"
# 10 turns: enough that named characters appear, leave, and (should)
# return; enough that early facts must be remembered; enough that
# the player has made several branching choices the engine must
# honor. Continuity errors compound at this length — short
# playthroughs (3-5 turns) hide drift the engine would expose later.
TURNS_TO_PLAY = 10


# ----- Helpers ---------------------------------------------------------------


def log(msg: str) -> None:
    """Diagnostic output — goes to stderr so the verify pipe can grab
    only the final integer from stdout."""
    print(msg, file=sys.stderr, flush=True)


async def _drain(handler_result):
    out = []
    async for envelope in handler_result:
        out.append(envelope)
    return out


def _load_settings() -> Settings:
    raw = settings_path().read_text(encoding="utf-8")
    return Settings.model_validate_json(raw)


# ----- Playthrough -----------------------------------------------------------


async def _answer(ctx: HandlerContext, registry, step: InterviewStep, value: str) -> None:
    await _drain(
        registry.dispatch(
            Envelope(
                type=MessageType.c2s_new_game_answer,
                payload={"step": step.value, "answer": value, "is_free_text": False},
            ),
            ctx,
        )
    )


async def _pick_first_option_for(ctx: HandlerContext, registry, field: str) -> str:
    """Read the latest interview state and return the first option for ``field``.

    Used to wire the playthrough's answers to LLM-generated options
    (visual_style, character_description, name) — picking option [0]
    is deterministic given the LLM output for that step.
    """
    options = getattr(ctx.session.interview, field) or []
    if not options:
        raise RuntimeError(f"interview state has no options for {field}")
    return options[0]


async def play_through(settings: Settings) -> tuple[Session, list[dict]]:
    """Run the fixed scenario end-to-end and return (session, committed beats).

    Each beat is a dict carrying the speaker, text, location label, and
    on-stage characters at that moment — enough for the judge to assess
    tone and continuity without seeing the raw DialogNode shape.

    Determinism: uses temperature 0 throughout so the same code +
    prompts produce the same story on every run. This pins variance
    to "what the prompts actually changed" instead of model creativity.
    """
    deterministic_llm = settings.llm.model_copy(update={"temperature": 0.0})
    settings = settings.model_copy(update={"llm": deterministic_llm})
    llm = OpenAiCompatibleLlmClient(deterministic_llm)

    class _NullImage:
        async def generate(self, *_a, **_kw) -> bytes:
            return b""

    session = Session(
        llm_client=llm,
        image_client=_NullImage(),
        settings=settings,
    )
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    log("[harness] new_game/start")
    await _drain(registry.dispatch(Envelope(type=MessageType.c2s_new_game_start, payload={}), ctx))

    # New step order (post-FR-027b): setting -> visual_style -> genre
    # -> character_description -> name. Both setting and visual_style
    # are hardcoded at runtime; genre is hardcoded; char_desc and name
    # are LLM-driven. The char_desc options are prefetched after
    # setting and surfaced after genre; name options are emitted via
    # state/patch after character_description, so we wait briefly
    # before reading them.
    log("[harness] answer setting (hardcoded)")
    await _answer(ctx, registry, InterviewStep.setting, SCENARIO_SETTING)

    log("[harness] answer visual_style (hardcoded; surfaced after setting)")
    visual_style = await _pick_first_option_for(ctx, registry, "visual_style_options")
    await _answer(ctx, registry, InterviewStep.visual_style, visual_style)

    log("[harness] answer genre (hardcoded)")
    await _answer(ctx, registry, InterviewStep.genre, SCENARIO_GENRE)

    log("[harness] wait for char_desc prefetch to surface")
    for _ in range(1800):
        if ctx.session.interview.character_description_options:
            break
        await asyncio.sleep(0.1)
    char_desc = await _pick_first_option_for(ctx, registry, "character_description_options")
    await _answer(ctx, registry, InterviewStep.character_description, char_desc)

    log("[harness] wait for name options (async fire-and-emit)")
    for _ in range(1800):
        if ctx.session.interview.name_options:
            break
        await asyncio.sleep(0.1)
    name = await _pick_first_option_for(ctx, registry, "name_options")
    await _answer(ctx, registry, InterviewStep.name, name)

    log(f"[harness] confirm — visual_style={visual_style!r} char={char_desc!r} name={name!r}")
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )

    log(f"[harness] play through {TURNS_TO_PLAY} turns picking first option each time")
    # Track which option text the player chose at each branch — the
    # judge needs this to assess choice integrity. Without it the
    # transcript looks like a static narration with decorative choice
    # blocks, and the judge correctly flags "choices never executed."
    chosen_history: dict[str, str] = {}
    for i in range(TURNS_TO_PLAY):
        if session.game is None or session.game.current_node_id is None:
            break
        node = session.game.dialog_tree.nodes[session.game.current_node_id]
        if node.options:
            chosen_option = node.options[0]
            option_id = chosen_option.id
            chosen_history[node.id] = chosen_option.text
        else:
            option_id = None
        log(f"[harness]  turn {i + 1}/{TURNS_TO_PLAY} option_id={option_id}")
        await _drain(
            registry.dispatch(
                Envelope(
                    type=MessageType.c2s_play_advance,
                    payload={"option_id": option_id},
                ),
                ctx,
            )
        )

    # Collect committed beats — what the judge will read.
    beats = []
    if session.game is not None:
        for nid in session.game.dialog_tree.committed_path:
            node = session.game.dialog_tree.nodes[nid]
            speaker_name = None
            if node.speaker_id and node.speaker_id in session.game.characters:
                speaker_name = session.game.characters[node.speaker_id].name
            beats.append(
                {
                    "speaker": speaker_name,
                    "text": node.text,
                    "location_id": node.location_id,
                    "options_offered": [o.text for o in node.options],
                    "player_picked": chosen_history.get(node.id),
                }
            )
    return session, beats


# ----- Judge ----------------------------------------------------------------

JUDGE_RUBRIC = """You grade an AI-driven visual novel playthrough as the showrunner reviewing the rough cut. The questions you ask are NOT "is this prose pretty" but "does this WORK as a visual novel" — does it remember itself, do its characters stay coherent, do choices matter, would a player play more?

Anchor scale (calibrate to lived VN experience — current "competent" AI output is 60-72, NOT 80+):
  - 90-100: vintage VN. Player would re-read just to catch the foreshadowing. Characters recur with their voice intact, world facts compound, every choice changes what comes next. Astonishingly rare.
  - 75-89: a polished commercial VN. Continuity holds across many turns; characters distinguishable in dialogue; choices have real weight.
  - 60-74: competent AI generation. Most facts stay; voices roughly consistent; choices nominally matter. This is the band a strong current AI output should occupy.
  - 45-59: visible seams. A character changes hair color or accent; a location resets; a choice is forgotten next turn.
  - 25-44: amateur — multiple drifts, generic dialogue, decorative choices.
  - 0-24: incoherent.

VN-SPECIFIC FAILURE MODES (each costs serious points; cite by beat number):
  - Character drift: any named character's described attribute (age, hair, build, outfit, expression) silently changes between mentions → -8 pts to character_continuity.
  - Voice drift: a named character's lines could swap with another character's without the reader noticing → -6 pts.
  - Forgotten character: a character introduced mid-game, then never referenced after they should logically still be present (or absent without explanation) → -5 pts.
  - Plot-thread vanish: an active plot thread named at start, then never advanced over 4+ turns → -8 pts to plot_continuity.
  - World-fact reset: the time of day, weather, location geometry, or established prop unexpectedly changes without a beat naming the change → -6 pts.
  - Choice retconned: the player picks option A, the next beat narrates as if option B was picked, or option A's effect is missing two turns later → -10 pts to choice_integrity.
  - Re-narrated choice: the post-choice beat just repeats the chosen action verbatim → -5 pts to choice_integrity.
  - Stock VN tropes: a shadow falls, silence stretches, salt on the wind, the air feels heavy → -2 pts each (cap -6) to vn_craft.
  - Decorative choice: two or three options that lead to indistinguishable next beats → -5 pts to choice_integrity.

Rubric (sum to 100):

1. **Character continuity (30 pts)** — Every named character's description, age band, voice, posture, and current expression match across all their mentions. Player character's described traits never drift. New characters are fully introduced before being referenced; they don't appear out of thin air. CEILING 24 unless every named character appears at least twice with consistent voice.

2. **Plot/world continuity (25 pts)** — Active plot threads progress visibly turn-over-turn — at least one thread shows a small advancement every 2-3 turns. Locations stay where they are; props reappear or are explicitly removed; time of day moves forward (or holds, but never reverses). The world remembers itself.

3. **Choice integrity (20 pts)** — Each player choice changes the next 2+ turns in ways another choice wouldn't have. Post-choice beats open on the consequence (what shifts), not the gesture (what was just done). Options diverge in emotional valence and outcome, not just wording. Options must NOT all push toward the same plot beat (railroading) — at least one option per turn must allow the player to wait, observe, refuse, or sidestep instead of committing to action.

4. **Pacing (15 pts)** — Scene rhythm matches stakes, the chain doesn't drag or skip key beats, and the storyteller doesn't sabotage its own moments.
   - **Railroading penalty** (-5 each, cap -10): a turn where every option leads to the same NEXT BEAT regardless of which is picked, OR where every option is an action verb (no observe/wait/ask alternative).
   - **Gentle direction** (+0/+5): when the player has been ambiguous (multiple turns without a clear goal-pursuit) the storyteller surfaces a soft hint via an option or in-narration cue (a person to seek, a place that is mentioned by name, an object that draws attention) — without forcing it. Award +5 only if the hint reads as natural, not as the narrator interrupting to give instructions.
   - **Intensity preservation** (-5 each, cap -10): an intense beat (confrontation, reveal, time-pressured choice, character in danger) is followed by a beat that pivots to unrelated atmospheric description, comic relief, or scene-setting that drops the tension. Intense moments may be followed by reaction, escalation, or resolution — not by tonal whiplash.

5. **VN craft (10 pts)** — Beat-per-line pacing, sensory anchor on every beat, distinct character voices in dialogue, no fourth-wall break, no stock transitional imagery.

Return ONLY a JSON object on a single line, no Markdown fence, exactly:
{"character_continuity": int 0-30, "plot_continuity": int 0-25, "choice_integrity": int 0-20, "pacing": int 0-15, "vn_craft": int 0-10, "grade": int 0-100, "drift_examples": ["<beat#: what drifted>", ...], "comment": "<one sentence diagnosis focused on the most expensive failure mode>"}
"""


def _format_story_for_judge(scenario_summary: str, beats: list[dict]) -> str:
    """Render the playthrough so the judge sees a clean transcript.

    The location_id is carried forward from the most recent beat that
    set one — beats that don't change location have ``location_id ==
    None`` in the DialogNode (NOT a location reset). Tagging these as
    "—" mistakenly read like world-fact resets to the judge; instead,
    propagate the last-set location forward and only mark a TRANSITION
    when the location_id actually changes.
    """
    lines = [
        f"SCENARIO: {scenario_summary}",
        f"PLAYTHROUGH ({len(beats)} committed beats):",
    ]
    current_location = "(unset)"
    for i, b in enumerate(beats):
        speaker = b.get("speaker") or "(narrator)"
        loc = b.get("location_id")
        if loc and loc != current_location:
            current_location = loc
            location_tag = f"@ {current_location} (location set)"
        else:
            location_tag = f"@ {current_location}"
        lines.append(f"  Beat {i + 1} [{speaker}] {location_tag}:")
        lines.append(f"    {b.get('text') or '(empty)'}")
        if b.get("options_offered"):
            picked = b.get("player_picked")
            for opt in b["options_offered"]:
                marker = " ← PLAYER PICKED" if opt == picked else ""
                lines.append(f"    Option: {opt}{marker}")
    return "\n".join(lines)


async def grade_with_judge(settings: Settings, story_text: str) -> dict:
    """Send the story to a separate LLM call (temperature 0) and parse
    the rubric JSON it returns.
    """
    judge_settings = settings.llm.model_copy(update={"temperature": 0.0})
    judge_client = OpenAiCompatibleLlmClient(judge_settings)

    log("[judge] requesting rubric grade…")
    prompt = [
        {"role": "system", "content": JUDGE_RUBRIC},
        {"role": "user", "content": story_text},
    ]
    chunks: list[str] = []
    stream = await judge_client.complete(
        prompt,
        model=judge_settings.model,
        temperature=judge_settings.temperature,
        max_tokens=judge_settings.max_tokens,
        stream=False,
    )
    async for chunk in stream:
        chunks.append(chunk)
    raw = "".join(chunks).strip()

    # Tolerate an optional Markdown fence the same way responses.py does.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log(f"[judge] ERROR: judge did not return JSON: {exc}; first 400 chars: {raw[:400]!r}")
        raise

    grade = int(data.get("grade", 0))
    log(
        f"[judge] character={data.get('character_continuity')} "
        f"plot={data.get('plot_continuity')} "
        f"choice={data.get('choice_integrity')} "
        f"pacing={data.get('pacing')} "
        f"craft={data.get('vn_craft')} grade={grade}"
    )
    drift = data.get("drift_examples") or []
    if isinstance(drift, list):
        for entry in drift[:6]:
            log(f"[judge] drift: {entry}")
    log(f"[judge] comment: {data.get('comment')}")
    return data


# ----- Main ------------------------------------------------------------------


async def _run_once(settings: Settings) -> int:
    """Single playthrough + judge call. Returns the grade or 0 on failure."""
    try:
        _session, beats = await play_through(settings)
    except Exception as exc:
        log(f"[harness] FATAL during playthrough: {exc!r}")
        return 0
    if not beats:
        log("[harness] FATAL: no beats committed")
        return 0
    scenario = (
        f"Setting: {SCENARIO_SETTING}; Genre: {SCENARIO_GENRE}; "
        f"Player picks first option each turn for {TURNS_TO_PLAY} turns."
    )
    story = _format_story_for_judge(scenario, beats)
    log("=" * 60)
    log("[harness] STORY")
    log("=" * 60)
    log(story)
    log("=" * 60)
    try:
        result = await grade_with_judge(settings, story)
    except Exception as exc:
        log(f"[judge] FATAL: {exc!r}")
        return 0
    return int(result.get("grade", 0))


async def main() -> int:
    """Run the playthrough+judge cycle 3 times and report the median.

    Single runs varied by ±15 pts even at temperature 0 (OpenRouter /
    upstream provider isn't strictly deterministic). 3-run median is
    robust to one outlier and gives an interpretable signal across
    autoresearch iterations.
    """
    log("[harness] starting storytelling-grade run (median of 3)")
    settings = _load_settings()
    log(f"[harness] LLM model={settings.llm.model} temp={settings.llm.temperature}")

    grades: list[int] = []
    for i in range(3):
        log(f"[harness] === sample {i + 1}/3 ===")
        grade = await _run_once(settings)
        log(f"[harness] sample {i + 1} grade = {grade}")
        grades.append(grade)

    grades.sort()
    median = grades[1]
    log(f"[harness] sorted grades: {grades}; median = {median}")
    print(median)
    return median


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) >= 0 else 1)

"""Voice-quality grading harness — "human writing, not abstraction".

Sister script to ``grade_storytelling.py``. That harness optimises for
CONTINUITY (characters / plot / choices stay coherent across many
turns). This one optimises for VOICE: do the storyteller's beats read
like writing a human would put on a visual-novel screen, or like
mid-2020s AI tics — abstract noun-of-noun constructions, decorative
metaphor, philosophising about thresholds and frequencies and
boundaries-between-states?

The two scripts share infrastructure but diverge on:

  * **Walk strategy** — grade_storytelling picks ``options[0]`` for
    determinism. grade_voice picks RANDOMLY (seeded per run, varied
    across runs) so the grade reflects voice across the option space
    rather than one preferred branch. Random walks also surface tonal
    failures the first-option walk hides — a storyteller who only
    sounds human on the obvious choice path is still failing the
    long tail.
  * **Rubric** — grade_storytelling's rubric is built around
    continuity failures (character drift, plot-thread vanish, choice
    retconned). This rubric grades concreteness, sensory anchoring,
    natural cadence, and absence of stock-VN / philosophising tics.
  * **Sample count / temperature** — kept identical (median of 3,
    temperature 0.0 on judge) so grades are comparable round-to-round.
    Playthrough temperature stays at the user's configured value so
    we measure voice against what the player actually sees.

Output: writes commentary + the captured story to stderr; prints a
single integer 0-100 grade to stdout (the last line). Designed for
the ``autoresearch`` verify pipeline:

    python backend/scripts/grade_voice.py 2>backend/scripts/grade_voice.log | tail -1

Cost: ~7-9 LLM calls per playthrough (same as grade_storytelling) +
1 judge call, × 3 samples = ~30 LLM calls per run. At
deepseek-v3.2 prices, a 6-iteration loop totals well under $1.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
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

# Different scenario from grade_storytelling on purpose: voice fails
# differently across genres. A noir setup pushes the storyteller
# toward stock atmospherics ("salt on the wind", "shadow lengthened")
# more aggressively than the harbor-mystery scenario, so it's a
# stricter test of whether the prompt's anti-cliché rules survive.
SCENARIO_SETTING = "A neon-lit night market on the rim of a port city"
SCENARIO_GENRE = "Noir thriller"
TURNS_TO_PLAY = 8


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
    """Pick option [0] for INTERVIEW questions (visual_style, char
    desc, name). Interview answers are setup, not gameplay — picking
    the same setup each run keeps voice comparable across iterations.
    The random walk applies only to PLAY-time options."""
    options = getattr(ctx.session.interview, field) or []
    if not options:
        raise RuntimeError(f"interview state has no options for {field}")
    return options[0]


async def play_through(settings: Settings, *, walk_seed: int) -> tuple[Session, list[dict]]:
    """Run the fixed scenario end-to-end with a RANDOM walk through
    play options, returning (session, committed beats).

    ``walk_seed`` pins the walk so iterations can compare the SAME
    sequence of choices across prompt variations — different walk
    seeds across the median samples expose different parts of the
    option space.

    Playthrough temperature is left at whatever the user configured
    (typically 0.7-0.9). Voice is creative-writing output; grading
    against temperature 0 would over-fit the prompts to a degenerate
    sampling regime the player never actually sees.
    """
    llm = OpenAiCompatibleLlmClient(settings.llm)
    rng = random.Random(walk_seed)

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

    log("[harness] answer setting (hardcoded)")
    await _answer(ctx, registry, InterviewStep.setting, SCENARIO_SETTING)

    log("[harness] answer visual_style (first available)")
    visual_style = await _pick_first_option_for(ctx, registry, "visual_style_options")
    await _answer(ctx, registry, InterviewStep.visual_style, visual_style)

    log("[harness] answer genre (hardcoded)")
    await _answer(ctx, registry, InterviewStep.genre, SCENARIO_GENRE)

    log("[harness] wait for char_desc prefetch")
    for _ in range(1800):
        if ctx.session.interview.character_description_options:
            break
        await asyncio.sleep(0.1)
    char_desc = await _pick_first_option_for(ctx, registry, "character_description_options")
    await _answer(ctx, registry, InterviewStep.character_description, char_desc)

    log("[harness] wait for name options")
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

    log(f"[harness] play through {TURNS_TO_PLAY} turns (random walk, seed={walk_seed})")
    chosen_history: dict[str, str] = {}
    for i in range(TURNS_TO_PLAY):
        if session.game is None or session.game.current_node_id is None:
            break
        node = session.game.dialog_tree.nodes[session.game.current_node_id]
        if node.options:
            chosen_option = rng.choice(node.options)
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

JUDGE_RUBRIC = """You grade an AI-driven visual novel playthrough as a senior editor at a VN publisher reviewing a script for whether it sounds like writing a human would print on screen — or like 2020s AI tics.

This rubric is NOT about plot or continuity (a sister rubric handles that). It is ONLY about VOICE: does the prose feel grounded, observed, written — or does it drift into abstract noun-of-noun constructions, philosophising, and stock atmosphere?

Anchor scale (calibrate harshly — current AI prose for visual novels lands 50-65 by default; 80+ is reserved for prose a player would highlight):
  - 90-100: prose a player would screenshot. Each beat lands on a specific observed thing or said line; characters speak with distinct mouths; the writing has rhythm. Vanishingly rare for AI output.
  - 75-89: a polished commercial VN script. The story is told through what is on screen and what is said; metaphor is rare and earns its place; no stock atmospherics; cadence varies.
  - 60-74: the band a strong AI prompt should occupy. Concrete most of the time; a few clichés slip through; one or two abstract sentences per chain.
  - 45-59: visible AI tics. Multiple "the silence between heartbeats", "the boundary where memory becomes anticipation", noun-of-noun stacks, narrator philosophising at the player.
  - 25-44: heavily abstract. Beats summarise feelings instead of showing actions. Half the sentences are "the X of Y" or "a frequency / threshold / tessellation of Z".
  - 0-24: unreadable AI slop. Pure abstraction. No grounded scene.

VOICE-SPECIFIC FAILURE MODES (cite by beat number; each costs serious points):

  * **Abstract-noun subject** — the subject of a sentence is an abstraction ("the silence", "the moment", "the boundary", "the air", "the frequency", "the weight of his gaze") doing or being something. -3 pts each occurrence to grounding (cap -12).
  * **Noun-of-noun stack** — "the [noun] of [noun]" or "[adjective] [noun] of [noun]" patterns ("the geometry of regret", "the cartography of his patience", "the frequency at which her smile registered"). -2 pts each (cap -8).
  * **Stock atmospheric filler** — "shadows lengthened", "silence stretched", "the air felt heavy", "salt on the wind", "wind whispered", "boots clicked on stone", "a chill ran down". -2 pts each (cap -8).
  * **Philosophising narrator** — narrator pauses the scene to muse about thresholds, distance vs closeness, what-is-said-vs-what-is-meant, memory vs present, the space between two things. -4 pts each (cap -12).
  * **Telling not showing** — narrator names the emotion or meaning instead of letting the action / dialogue carry it ("she felt the weight of years", "the room held him in its judgement"). -2 pts each (cap -8).
  * **Decorative-only beat** — a beat that is pure mood with no observed concrete particular (a thing seen, said, smelled, picked up, moved). -3 pts each (cap -9).
  * **Diction mismatch** — academic / literary diction that doesn't fit the VN's chosen genre and visual style (e.g., "the tessellation of his refusal" in a noir thriller). -2 pts each (cap -6).
  * **Choice text reads abstract** — option text is a mood/intent description ("Embrace the unfolding", "Honour the silence") instead of a concrete action / line of dialogue / observation a person would do or say. -2 pts each option (cap -8) to choice_voice.

POSITIVE PATTERNS (note explicitly when present; they preserve points):

  * Beats that LAND ON a specific physical thing: a coin tipped over, lamp guttering low, somebody's knuckles whitening on a railing. The reader can SEE it.
  * Dialogue that distinguishes one character from another by mouth (vocabulary, rhythm, contraction patterns) — not just by content.
  * Sentence-rhythm variety: short punches mixed with longer breaths; not every sentence the same shape.
  * Choice text written as natural action verbs + concrete object/person ("Push the door", "Ask Mira about the ledger", "Step back into the alley") — not "Embrace the moment".

Rubric (sum to 100):

1. **Grounded particulars (30 pts)** — Beats anchor in named, observable things. The reader can picture what's on screen. Penalise abstract subjects, noun-of-noun stacks, decorative-only beats. CEILING 24 unless every beat lands on at least one specific observable.

2. **Show don't tell (20 pts)** — Action, gesture, dialogue carry meaning. Narrator does not summarise emotional states or explain what the scene means. CEILING 15 if more than 2 beats name an emotion the action should have shown.

3. **Natural cadence (15 pts)** — Sentence rhythm varies. Dialogue and action interleave. No two consecutive beats with the same syntactic shape.

4. **Genre voice match (15 pts)** — Diction fits the VN's chosen genre and visual style. A noir thriller reads noir; a high-fantasy beat reads high-fantasy. Penalise generic literary diction that floats free of any genre.

5. **Choice voice (10 pts)** — Options read as something a person would say or do — concrete action verb + concrete object, or a quoted line. Not abstract gestures or stance descriptions.

6. **Restraint with metaphor (10 pts)** — Metaphor is sparing and stays inside the scene's frame of reference. CEILING 7 if any beat contains philosophising about thresholds / boundaries / frequencies / tessellations / cartographies / geometries of feeling.

Return ONLY a JSON object on a single line, no Markdown fence, exactly:
{"grounded": int 0-30, "show": int 0-20, "cadence": int 0-15, "diction": int 0-15, "choice_voice": int 0-10, "metaphor": int 0-10, "grade": int 0-100, "tics": ["<beat#: <abstract-noun|noun-of-noun|stock-filler|philosophising|tell|decorative|diction|abstract-choice>: \"<verbatim phrase>\">", ...], "comment": "<one sentence diagnosing the most damaging voice failure>"}
"""


def _format_story_for_judge(scenario_summary: str, beats: list[dict]) -> str:
    """Same shape as grade_storytelling — a structured beat list with
    speaker / location / options. Carrying location forward (rather
    than emitting an "—" each time the location_id is null) keeps
    the judge from reading null locations as resets, which would
    bleed into the rubric inappropriately."""
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
    the rubric JSON it returns. Judge uses the SAME model as the
    storyteller — the alternative (a separate "smarter" judge)
    introduces a model-vs-model bias unrelated to prompt quality."""
    judge_settings = settings.llm.model_copy(update={"temperature": 0.0})
    judge_client = OpenAiCompatibleLlmClient(judge_settings)

    log("[judge] requesting voice-rubric grade…")
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
        f"[judge] grounded={data.get('grounded')} show={data.get('show')} "
        f"cadence={data.get('cadence')} diction={data.get('diction')} "
        f"choice={data.get('choice_voice')} metaphor={data.get('metaphor')} "
        f"grade={grade}"
    )
    tics = data.get("tics") or []
    if isinstance(tics, list):
        for entry in tics[:8]:
            log(f"[judge] tic: {entry}")
    log(f"[judge] comment: {data.get('comment')}")
    return data


# ----- Main ------------------------------------------------------------------


async def _run_once(settings: Settings, *, walk_seed: int) -> int:
    try:
        _session, beats = await play_through(settings, walk_seed=walk_seed)
    except Exception as exc:
        log(f"[harness] FATAL during playthrough: {exc!r}")
        return 0
    if not beats:
        log("[harness] FATAL: no beats committed")
        return 0
    scenario = (
        f"Setting: {SCENARIO_SETTING}; Genre: {SCENARIO_GENRE}; "
        f"Player random-walks options for {TURNS_TO_PLAY} turns "
        f"(walk seed {walk_seed})."
    )
    story = _format_story_for_judge(scenario, beats)
    log("=" * 60)
    log(f"[harness] STORY (walk seed {walk_seed})")
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
    """Run 3 random-walk samples and report the median.

    Each sample uses a different walk seed so the grade reflects voice
    across the option space rather than a single branch. Median (not
    mean) so one outlier playthrough doesn't drown the signal."""
    log("[harness] starting voice-grade run (median of 3 random walks)")
    settings = _load_settings()
    log(f"[harness] LLM model={settings.llm.model} temp={settings.llm.temperature}")

    # Walk seeds drawn from the wall clock + sample index — different
    # every iteration, so the loop sees the full option space across
    # iterations rather than re-grading the same path.
    base_seed = int(time.time()) & 0xFFFF
    grades: list[int] = []
    for i in range(3):
        walk_seed = base_seed + i
        log(f"[harness] === sample {i + 1}/3 (walk seed {walk_seed}) ===")
        grade = await _run_once(settings, walk_seed=walk_seed)
        log(f"[harness] sample {i + 1} grade = {grade}")
        grades.append(grade)

    grades.sort()
    median = grades[1]
    log(f"[harness] sorted grades: {grades}; median = {median}")
    print(median)
    return median


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) >= 0 else 1)

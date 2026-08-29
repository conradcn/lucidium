"""Character-pruning prompt grader.

Builds a battery of fixed scenarios for the world_refresh prompt's
job 5 (``characters_to_offstage``) and grades the live LLM's
response against expected outcomes. The scenario set spans five
case types so a single iteration's score isn't dominated by any
one trap pattern:

  Type A — explicit-leave: a named character says goodbye / is
    dismissed / walks out. Distractors include silent listeners
    who must stay, and characters who SAY they should leave but
    do not actually depart during the scene.
  Type B — player-leaves-behind: the player moves to a different
    location and characters who don't accompany must be offstaged.
    Distractors include accompanying characters (must stay), and
    multi-character left-behind groups (silent + speaking).
  Type C — negative control: no one should be offstaged. Quiet
    but still-present characters; long scenes where no one moves.
    Tests against an old "idle for N beats → prune" heuristic.
  Type D — step-out-and-return: a character physically leaves
    mid-scene then comes back before the history ends. Must stay
    on stage at the final beat.
  Type E — mixed: a single scene combines multiple triggers
    (explicit leaver + player-move + accompanying character).

Scoring per scenario:
  - Perfect match (offstage set == expected) → full credit
  - Partial (got some expected, no false positives on must-stay)
    → 40 % credit
  - Mixed (got some expected but also offstaged a must-stay or
    hallucinated an id) → 40 % credit
  - Wrong / nothing predicted when expected non-empty → 0
  - Empty prediction when expected empty (negative control) →
    full credit

Total weight per scenario is 100 / N so adding scenarios doesn't
require rebalancing. Output: diagnostic to stderr, single integer
0-100 to stdout. Designed for the autoresearch verify pipeline:

    python backend/scripts/grade_character_pruning.py \\
      2>backend/scripts/grade_character_pruning.log | tail -1

Determinism: temperature 0 for the live LLM call so prompt
changes are the only signal.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend" / "src"))

from lucidium.config import settings_path  # noqa: E402
from lucidium.domain.dialog import DialogNode, DialogNodeState  # noqa: E402
from lucidium.domain.settings import Settings  # noqa: E402
from lucidium.domain.world import WorldState  # noqa: E402
from lucidium.orchestration.prompts import world_refresh  # noqa: E402
from lucidium.orchestration.responses import (  # noqa: E402
    LlmSummaryResult,
    parse_json_object,
)
from lucidium.providers.llm_client import OpenAiCompatibleLlmClient  # noqa: E402


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _node(text: str, speaker_id: str | None = None, location_id: str | None = None) -> DialogNode:
    return DialogNode(
        text=text,
        speaker_id=speaker_id,
        location_id=location_id,
        state=DialogNodeState.committed,
        premise_hash="x" * 64,
    )


def _world(setting: str = "A coastal town", genre: str = "Mystery") -> WorldState:
    return WorldState(
        game_name="prune-grader",
        setting=setting,
        genre=genre,
        visual_style="ink",
    )


# ---------- Scenarios -------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    world: WorldState
    history: list[DialogNode]
    on_stage_ids: list[str]
    player_id: str
    character_names: dict[str, str]
    expected_offstage: set[str]
    must_stay: set[str]


# ---------- Type A: explicit-leave ------------------------------------------


def _scenario_a1() -> Scenario:
    """A1: simple explicit-leave with quiet-listener and step-out-and-return.

    Hale walks out for good (offstage). Maud is actively talking
    (stay). Pell steps out mid-scene to fetch something then comes
    back (stay). Tests that "Pell ducks out for a moment" + "Pell
    comes back" doesn't trigger a wrong offstaging.
    """
    history = [
        _node(
            "The lighthouse keeper's parlor smells of pipe-smoke and salt. Pell sits opposite you with a cup in their hands.",
            location_id="parlor",
        ),
        _node(
            "I made tea while you were upstairs. Maud had to chase the cat off again.",
            speaker_id="hale",
            location_id="parlor",
        ),
        _node("Maud sets down a chipped cup and looks across at you.", location_id="parlor"),
        _node(
            "Whatever you came to ask, ask it. We don't have all evening.",
            speaker_id="maud",
            location_id="parlor",
        ),
        _node(
            "Hold on — I left the chart in the hall. I'll grab it.",
            speaker_id="pell",
            location_id="parlor",
        ),
        _node("Pell stands and ducks out into the corridor for a moment.", location_id="parlor"),
        _node(
            "You set the tin on the desk and tell Maud what the constable said.",
            speaker_id="player",
            location_id="parlor",
        ),
        _node(
            "That doesn't square with the ledger. Have you spoken to Bran?",
            speaker_id="maud",
            location_id="parlor",
        ),
        _node(
            "Pell comes back in with a rolled chart under one arm and drops it on the desk.",
            location_id="parlor",
        ),
        _node(
            "Here. The harbour map from the year of the Carrick wreck.",
            speaker_id="pell",
            location_id="parlor",
        ),
        _node(
            "You unroll it across the desk; Maud weights one corner with her cup.",
            location_id="parlor",
        ),
        _node("Hale glances at the clock above the door, then stands.", location_id="parlor"),
        _node(
            "That's me — I'm late for the harbour shift. Maud, lock up after, will you? Goodnight.",
            speaker_id="hale",
            location_id="parlor",
        ),
        _node(
            "Hale shrugs on a coat and steps out into the wind. The door bangs shut behind him.",
            location_id="parlor",
        ),
        _node(
            "Just us, then. Pell, show them the eastern channel.",
            speaker_id="maud",
            location_id="parlor",
        ),
        _node(
            "Pell traces a finger along an ink line on the chart.",
            speaker_id="pell",
            location_id="parlor",
        ),
    ]
    return Scenario(
        name="A1-explicit-leave",
        description="Hale walks out; Maud talks; Pell step-out-and-return; nobody else leaves.",
        world=_world(),
        history=history,
        on_stage_ids=["hale", "maud", "pell"],
        player_id="player",
        character_names={"hale": "Hale", "maud": "Maud", "pell": "Pell", "player": "You"},
        expected_offstage={"hale"},
        must_stay={"maud", "pell"},
    )


def _scenario_a2() -> Scenario:
    """A2: explicit-leave with said-but-didn't-leave distractor.

    Cully is dismissed and physically leaves (offstage). Bran
    SAYS he should head home but stays talking (stay). Quill
    works silently at the far desk the whole scene (stay).
    """
    history = [
        _node(
            "The customs office smells of damp ledgers and pipe smoke. Quill is bent over a tally book at the far desk.",
            location_id="customs",
        ),
        _node(
            "Cully — the tally sheets from last Tuesday. Where are they?",
            speaker_id="bran",
            location_id="customs",
        ),
        _node(
            "Filed in the cabinet, sir. Should I fetch them?",
            speaker_id="cully",
            location_id="customs",
        ),
        _node(
            "Yes. And then go home. Whatever this stranger needs to discuss, it's not for clerk's ears.",
            speaker_id="bran",
            location_id="customs",
        ),
        _node(
            "Cully sets the ledger down, nods stiffly, and leaves through the back door.",
            location_id="customs",
        ),
        _node(
            "I should be home myself before the rain starts. But — the Carrick wreck. Tell me what you know.",
            speaker_id="bran",
            location_id="customs",
        ),
        _node(
            "You spread the constable's notes on the desk between you.",
            speaker_id="player",
            location_id="customs",
        ),
        _node(
            "Three names on the manifest that aren't on the body list. That's where I'd start.",
            speaker_id="bran",
            location_id="customs",
        ),
        _node(
            "Bran taps a finger on the desk and pulls a fresh sheet of paper toward him.",
            location_id="customs",
        ),
        _node(
            "Quill keeps his head down, but you catch him glance up at the names.",
            location_id="customs",
        ),
        _node(
            "I can pull the original manifest from the vault. It will take an hour.",
            speaker_id="bran",
            location_id="customs",
        ),
        _node("Take whatever you need. I'll wait.", speaker_id="player", location_id="customs"),
    ]
    return Scenario(
        name="A2-explicit-leave",
        description="Cully dismissed and exits; Bran says he should leave but stays; Quill silent.",
        world=_world(),
        history=history,
        on_stage_ids=["bran", "cully", "quill"],
        player_id="player",
        character_names={"bran": "Bran", "cully": "Cully", "quill": "Quill", "player": "You"},
        expected_offstage={"cully"},
        must_stay={"bran", "quill"},
    )


def _scenario_a3() -> Scenario:
    """A3: multi-leave — two characters depart in one scene.

    War-room briefing. Senna and Yorick are dismissed and leave
    one after the other. Marshal Drun stays. Captain Iv stays.
    Tests that the prompt picks up TWO departures in one pass.
    """
    history = [
        _node(
            "The war-room is hot under three lanterns. A map of the river crossings is pinned to the long table.",
            location_id="war_room",
        ),
        _node(
            "Senna, Yorick — your reports first. Then I'll hear the constable.",
            speaker_id="drun",
            location_id="war_room",
        ),
        _node(
            "South ford is breached. We lost two pickets last night. I've moved the rest to the second line.",
            speaker_id="senna",
            location_id="war_room",
        ),
        _node(
            "Yorick nods grimly and lays a sketch of the bridge on the map.", location_id="war_room"
        ),
        _node(
            "North bridge is intact but the supports are cracked. Three more days of rain and it goes.",
            speaker_id="yorick",
            location_id="war_room",
        ),
        _node(
            "Captain Iv leans over the map, tracing a line with one finger.", location_id="war_room"
        ),
        _node(
            "If we lose the north bridge, we lose the road to the salt-fields.",
            speaker_id="iv",
            location_id="war_room",
        ),
        _node(
            "Senna, Yorick — back to your posts. Send word the moment anything changes.",
            speaker_id="drun",
            location_id="war_room",
        ),
        _node(
            "Senna salutes, gathers her cloak, and leaves through the side door. Yorick rolls up his sketch and follows her out.",
            location_id="war_room",
        ),
        _node(
            "Now. Tell me what the constable said about the bodies.",
            speaker_id="drun",
            location_id="war_room",
        ),
        _node(
            "You spread the notes across the map and Iv leans in to read them.",
            speaker_id="player",
            location_id="war_room",
        ),
        _node(
            "These names match three of mine that went missing on patrol two months ago.",
            speaker_id="iv",
            location_id="war_room",
        ),
    ]
    return Scenario(
        name="A3-multi-leave",
        description="Senna and Yorick are both dismissed and leave; Drun and Iv stay to talk to the player.",
        world=_world(),
        history=history,
        on_stage_ids=["drun", "senna", "yorick", "iv"],
        player_id="player",
        character_names={
            "drun": "Marshal Drun",
            "senna": "Senna",
            "yorick": "Yorick",
            "iv": "Captain Iv",
            "player": "You",
        },
        expected_offstage={"senna", "yorick"},
        must_stay={"drun", "iv"},
    )


# ---------- Type B: player-leaves-behind ------------------------------------


def _scenario_b1() -> Scenario:
    """B1: player-leaves with accompanying-character distractor.

    Player walks from tavern to docks. Eira stays behind at the
    table (offstage). Tomas accompanies the player and is
    speaking at the docks (stay).
    """
    history = [
        _node("The tavern's back room is warm with peat-smoke and low talk.", location_id="tavern"),
        _node(
            "You'll want the docks before the tide turns. Won't be ships there much longer.",
            speaker_id="eira",
            location_id="tavern",
        ),
        _node("Tomas drains his cup and says nothing, watching you.", location_id="tavern"),
        _node(
            "I'll wait here. If you need me, I'll be at this table.",
            speaker_id="eira",
            location_id="tavern",
        ),
        _node("Right. Tomas?", speaker_id="player", location_id="tavern"),
        _node(
            "Aye, I'll come. Someone has to keep you out of trouble down there.",
            speaker_id="tomas",
            location_id="tavern",
        ),
        _node(
            "You and Tomas push out into the cold. The tavern door swings closed behind you.",
            location_id="docks",
        ),
        _node(
            "The wind off the water hits hard. Tomas pulls his collar up and looks for the harbour-master.",
            speaker_id="tomas",
            location_id="docks",
        ),
        _node(
            "There — the man with the lamp. He's the one she meant.",
            speaker_id="tomas",
            location_id="docks",
        ),
        _node(
            "You follow Tomas down a slick gangway past stacks of crab pots.", location_id="docks"
        ),
        _node(
            "Mind your step. Plank's loose in the middle.", speaker_id="tomas", location_id="docks"
        ),
    ]
    return Scenario(
        name="B1-player-leaves",
        description="Player and Tomas move tavern→docks; Eira stays at the table.",
        world=_world(),
        history=history,
        on_stage_ids=["eira", "tomas"],
        player_id="player",
        character_names={"eira": "Eira", "tomas": "Tomas", "player": "You"},
        expected_offstage={"eira"},
        must_stay={"tomas"},
    )


def _scenario_b2() -> Scenario:
    """B2: player exits a meeting; multi-character left behind.

    Player and Reeve leave the office together. Vella stays in
    her office (offstage). Reeve walks the player home and bids
    them goodnight at the door (offstage at the very end). Player
    ends alone at home. Trap: Reeve is on stage for ~3 beats AFTER
    the location change, then explicitly leaves.
    """
    history = [
        _node(
            "Magistrate Vella's office is small, lit by a single green-shaded lamp. Constable Reeve stands by the doorframe, arms crossed.",
            location_id="office",
        ),
        _node(
            "I've told you what I can. The rest is on the books in the archive. Come back if the constable agrees.",
            speaker_id="vella",
            location_id="office",
        ),
        _node("Reeve pushes off the doorframe.", location_id="office"),
        _node(
            "I'll walk you out. Streets aren't friendly tonight.",
            speaker_id="reeve",
            location_id="office",
        ),
        _node(
            "You nod to Vella and step into the corridor with Reeve at your shoulder.",
            location_id="street",
        ),
        _node(
            "Listen — what she said about the archive. There's a clerk there who owes me. Use my name.",
            speaker_id="reeve",
            location_id="street",
        ),
        _node(
            "You note the name and turn the corner toward your house. Reeve walks the last two blocks beside you in silence.",
            location_id="street",
        ),
        _node(
            "Goodnight. Lock the door behind you, and don't open it before sunrise. I mean it.",
            speaker_id="reeve",
            location_id="home",
        ),
        _node(
            "Reeve waits until you're inside, then walks back the way you came. The door clicks shut behind you.",
            location_id="home",
        ),
        _node(
            "The fire's gone out. You sit alone and think over what Vella said.", location_id="home"
        ),
    ]
    return Scenario(
        name="B2-player-leaves",
        description="Player exits office to home; both Vella (left in office) and Reeve (escorted then left) are offstage.",
        world=_world(),
        history=history,
        on_stage_ids=["vella", "reeve"],
        player_id="player",
        character_names={"vella": "Vella", "reeve": "Reeve", "player": "You"},
        expected_offstage={"vella", "reeve"},
        must_stay=set(),
    )


def _scenario_b3() -> Scenario:
    """B3: multi-stop journey — characters left at each stop.

    Player walks inn→road→camp. Innkeeper Pol stays at inn
    (offstage). Coachman Rivet drives partway then returns to
    inn (offstage). At camp the player meets Captain Suun. None
    of inn/road characters accompany the player to camp. Suun
    is on stage at the end (must stay).
    """
    history = [
        _node(
            "The inn's common room is half-empty at this hour. Pol wipes down the bar with a rag.",
            location_id="inn",
        ),
        _node(
            "Rivet'll have the trap out front in five minutes. He's a good driver — won't take you all the way, mind, only as far as the second waypost.",
            speaker_id="pol",
            location_id="inn",
        ),
        _node("From there it's a walk?", speaker_id="player", location_id="inn"),
        _node(
            "From there it's a walk. About an hour. Stay on the path; the moor's nasty in the dark.",
            speaker_id="pol",
            location_id="inn",
        ),
        _node("You shoulder your pack and step out into the yard.", location_id="inn"),
        _node(
            "Rivet whistles to the horse and snaps the reins. The trap rocks out onto the moor road.",
            location_id="road",
        ),
        _node(
            "Mind the dip after the milestone — wheel got stuck there last week.",
            speaker_id="rivet",
            location_id="road",
        ),
        _node(
            "You ride in silence as the moor opens out under a low grey sky.", location_id="road"
        ),
        _node(
            "Right. This is the second waypost. Watch yourself out there.",
            speaker_id="rivet",
            location_id="road",
        ),
        _node(
            "You climb down. Rivet turns the trap in a wide arc and rattles back the way you came.",
            location_id="road",
        ),
        _node(
            "An hour later, the camp's bonfire shows yellow against the moor. A figure in a dark coat is silhouetted against it.",
            location_id="camp",
        ),
        _node(
            "You're late. I was about to send a runner back for you.",
            speaker_id="suun",
            location_id="camp",
        ),
        _node(
            "Captain Suun nods to the bench by the fire. There's tea in the pot, surprisingly.",
            location_id="camp",
        ),
        _node(
            "Sit. Tell me everything the magistrate said.", speaker_id="suun", location_id="camp"
        ),
    ]
    return Scenario(
        name="B3-multi-stop",
        description="Player journeys inn→road→camp; Pol and Rivet are left at earlier stops; Suun is met at camp.",
        world=_world(),
        history=history,
        on_stage_ids=["pol", "rivet", "suun"],
        player_id="player",
        character_names={"pol": "Pol", "rivet": "Rivet", "suun": "Captain Suun", "player": "You"},
        expected_offstage={"pol", "rivet"},
        must_stay={"suun"},
    )


# ---------- Type C: negative control (no one should be offstaged) -----------


def _scenario_c1() -> Scenario:
    """C1: long single-room conversation; one character is mostly silent.

    Player and two characters in a study. Doctor Aldwin does most
    of the talking. Mrs Crale speaks once at the start, then is
    silent for ~10 beats while listening. NEITHER should be
    offstaged. Tests against an old idle-pruning heuristic.
    """
    history = [
        _node(
            "Doctor Aldwin's study is lined floor-to-ceiling with cracked leather books. A coal fire hisses in the grate.",
            location_id="study",
        ),
        _node(
            "Mrs Crale lifted the kettle off the hob herself before you arrived; she insists.",
            speaker_id="aldwin",
            location_id="study",
        ),
        _node("Tea. Don't argue. Sit.", speaker_id="crale", location_id="study"),
        _node(
            "You take the offered cup and settle into the chair opposite Aldwin.",
            location_id="study",
        ),
        _node(
            "So. The constable's notes. Show me what you've got.",
            speaker_id="aldwin",
            location_id="study",
        ),
        _node(
            "You spread the papers across the desk between you.",
            speaker_id="player",
            location_id="study",
        ),
        _node(
            "Three deaths inside two weeks. All the same — purple under the nails, foam at the mouth. The constable wrote it as plague.",
            speaker_id="aldwin",
            location_id="study",
        ),
        _node("It's not plague.", speaker_id="player", location_id="study"),
        _node(
            "No. It isn't. Plague kills one in a household, then half. This is one in a household, and only one. Selective.",
            speaker_id="aldwin",
            location_id="study",
        ),
        _node(
            "Aldwin pulls a heavy reference book from the shelf behind him and starts thumbing through it.",
            location_id="study",
        ),
        _node("Foxglove? Yew? Something cleverer?", speaker_id="player", location_id="study"),
        _node(
            "Cleverer. Foxglove looks like heart failure, not plague. Yew looks like seizures. This is something with the symptom profile of plague but the selectivity of poison.",
            speaker_id="aldwin",
            location_id="study",
        ),
        _node(
            "Mrs Crale watches the fire, listening, the cup forgotten in her hands.",
            location_id="study",
        ),
        _node("Could it be in the water?", speaker_id="player", location_id="study"),
        _node(
            "Possible. But all three victims drank from different wells. So either it's in something they all ate, or they all drank from the same source on the same day.",
            speaker_id="aldwin",
            location_id="study",
        ),
        _node("Aldwin closes the book and rubs his eyes.", location_id="study"),
        _node(
            "I need to see one of the bodies. Can you get me into the morgue?",
            speaker_id="aldwin",
            location_id="study",
        ),
    ]
    return Scenario(
        name="C1-quiet-listener",
        description="Long study scene; Crale silent for ~12 beats but still in the room; nothing should be offstaged.",
        world=_world(),
        history=history,
        on_stage_ids=["aldwin", "crale"],
        player_id="player",
        character_names={"aldwin": "Doctor Aldwin", "crale": "Mrs Crale", "player": "You"},
        expected_offstage=set(),
        must_stay={"aldwin", "crale"},
    )


def _scenario_c2() -> Scenario:
    """C2: long single-character conversation, no scene change.

    Player and one character on a park bench. Many beats of
    dialogue. No one comes or goes. Empty offstage list is the
    correct answer. Tests against a "if many beats have passed,
    something must be prunable" failure mode.
    """
    history = [
        _node(
            "The park bench is wet from the morning rain. You sit on the dry side; Edmund leans on his cane on the other.",
            location_id="park",
        ),
        _node(
            "Thirty years on the bench, and I never saw a case like the Whitcomb girl. Are you sure?",
            speaker_id="edmund",
            location_id="park",
        ),
        _node(
            "As sure as I can be without the body. The pattern matches.",
            speaker_id="player",
            location_id="park",
        ),
        _node(
            "Edmund stares at the gravel between his shoes for a long moment.", location_id="park"
        ),
        _node(
            "And the constable buried the question under 'plague.'",
            speaker_id="edmund",
            location_id="park",
        ),
        _node(
            "He had reason to. The town would have rioted.", speaker_id="player", location_id="park"
        ),
        _node("It will riot anyway, when this gets out.", speaker_id="edmund", location_id="park"),
        _node("Edmund taps the cane against the gravel three times, slowly.", location_id="park"),
        _node(
            "I'll write the warrant. But you'll need a witness who isn't me.",
            speaker_id="edmund",
            location_id="park",
        ),
        _node("Who would you suggest?", speaker_id="player", location_id="park"),
        _node(
            "Aldwin. He's already half-convinced — and he can't be bribed because he doesn't want anything anymore.",
            speaker_id="edmund",
            location_id="park",
        ),
        _node(
            "Some pigeons land near the bench. Edmund waves them off without breaking eye contact.",
            location_id="park",
        ),
        _node("How long do I have?", speaker_id="player", location_id="park"),
        _node(
            "Until the next death. Which, by your count, is overdue.",
            speaker_id="edmund",
            location_id="park",
        ),
    ]
    return Scenario(
        name="C2-long-talk",
        description="Long park bench conversation, one NPC, no movement, no entries/exits. Empty offstage.",
        world=_world(),
        history=history,
        on_stage_ids=["edmund"],
        player_id="player",
        character_names={"edmund": "Edmund", "player": "You"},
        expected_offstage=set(),
        must_stay={"edmund"},
    )


# ---------- Type D: step-out-and-return -------------------------------------


def _scenario_d1() -> Scenario:
    """D1: dedicated step-out-and-return at the END of history.

    Vintner Brae steps out at the very last beat to fetch a
    bottle from the cellar. The narration explicitly notes she
    will return shortly — she's NOT leaving the scene. Trap: the
    last narrative beat says "Brae goes downstairs" which a
    hyperliteral reading would treat as departure. Apprentice
    Rook is silent in the corner the whole scene. Both must stay.
    """
    history = [
        _node(
            "Brae's back-room tasting bench is lit by a single oil lamp. Bottles on every shelf, dust on most.",
            location_id="cellar",
        ),
        _node(
            "Rook, the cork-puller. Top shelf, behind the wax-stamps. — Yes, that one.",
            speaker_id="brae",
            location_id="cellar",
        ),
        _node(
            "Rook fetches a small iron tool from the shelf and sets it on the bench, then steps back to the wall.",
            location_id="cellar",
        ),
        _node(
            "So. The vintage your magistrate keeps asking about. You think it's tied to the deaths.",
            speaker_id="brae",
            location_id="cellar",
        ),
        _node(
            "All three victims drank from the '78 — it's the only common food or drink we've found.",
            speaker_id="player",
            location_id="cellar",
        ),
        _node(
            "Brae pours a thimble-glass of dark wine and turns it in the lamplight.",
            location_id="cellar",
        ),
        _node(
            "The '78 was a small batch. Maybe four casks total. I sold three; the fourth went to the Whitcomb estate as a wedding gift.",
            speaker_id="brae",
            location_id="cellar",
        ),
        _node("And the three you sold?", speaker_id="player", location_id="cellar"),
        _node(
            "Aldwin's house. The constable's house. And — ah. The dead girl's father.",
            speaker_id="brae",
            location_id="cellar",
        ),
        _node("Brae sets the glass down very carefully.", location_id="cellar"),
        _node(
            "There's a sample bottle from that batch in the lower cellar. Untouched, sealed, my own mark on the wax. If anything's in it, it's in that bottle too.",
            speaker_id="brae",
            location_id="cellar",
        ),
        _node("Get it.", speaker_id="player", location_id="cellar"),
        _node("Won't be a minute. Rook, watch the lamp.", speaker_id="brae", location_id="cellar"),
        _node(
            "Brae takes a key from a peg and steps through the inner door into the lower cellar. You hear her boots on the stairs.",
            location_id="cellar",
        ),
    ]
    return Scenario(
        name="D1-step-out-end",
        description="Brae steps into the lower cellar at the end to fetch a bottle, will return shortly; Rook silent in corner.",
        world=_world(),
        history=history,
        on_stage_ids=["brae", "rook"],
        player_id="player",
        character_names={"brae": "Brae", "rook": "Rook", "player": "You"},
        expected_offstage=set(),
        must_stay={"brae", "rook"},
    )


# ---------- Type E: mixed ---------------------------------------------------


def _scenario_e1() -> Scenario:
    """E1: explicit-leave + player-move + accompanying character.

    Triple combo. In the warehouse: foreman Lask is dismissed
    and exits (offstage). Then player and Inspector Dane exit
    the warehouse for the dock. Watchman Orth was on stage in
    the warehouse, doesn't accompany (offstage — left behind).
    Dane is at the dock with the player at the end (stay).
    """
    history = [
        _node(
            "The warehouse is cold. Crates marked HEMP RIGGING are stacked along the long wall. Foreman Lask wipes his hands on a rag.",
            location_id="warehouse",
        ),
        _node(
            "Watchman Orth stands near the loading door, arms folded, eyeing you.",
            location_id="warehouse",
        ),
        _node(
            "So. The inspector wants to look at the crates. The night-manifest is in the office; want me to fetch it?",
            speaker_id="lask",
            location_id="warehouse",
        ),
        _node(
            "Lask, leave us. I want to talk to the man without you in the room.",
            speaker_id="dane",
            location_id="warehouse",
        ),
        _node(
            "Lask shrugs, drops the rag on a barrel, and walks out through the office door.",
            location_id="warehouse",
        ),
        _node(
            "Now. Show me the third crate from the left, top row.",
            speaker_id="dane",
            location_id="warehouse",
        ),
        _node(
            "You and Dane prise the lid up on the indicated crate. The hemp inside is bone-dry — and there are tin canisters under the rope.",
            location_id="warehouse",
        ),
        _node(
            "That's not import duty stamped on those. That's salvage-tax. Smuggled.",
            speaker_id="dane",
            location_id="warehouse",
        ),
        _node(
            "We need to see the dock to know where it came in. Now, before they reload.",
            speaker_id="dane",
            location_id="warehouse",
        ),
        _node(
            "You and Dane head for the loading door; Orth steps aside with a tight expression but doesn't follow.",
            location_id="warehouse",
        ),
        _node(
            "The dock is a narrow strip of planking under a low fog. A barge sits half-loaded at the far end.",
            location_id="dock",
        ),
        _node(
            "That's the one. Look — the same canisters, lined up on the deck, waiting to be re-stacked.",
            speaker_id="dane",
            location_id="dock",
        ),
        _node("Dane crouches to read the burn-mark on a canister.", location_id="dock"),
        _node(
            "Foreign make. Northern foundry. Means this isn't a local job.",
            speaker_id="dane",
            location_id="dock",
        ),
    ]
    return Scenario(
        name="E1-mixed",
        description="Lask dismissed and exits; player+Dane move warehouse→dock; Orth stays at warehouse; Dane on stage at end.",
        world=_world(),
        history=history,
        on_stage_ids=["lask", "orth", "dane"],
        player_id="player",
        character_names={"lask": "Lask", "orth": "Orth", "dane": "Inspector Dane", "player": "You"},
        expected_offstage={"lask", "orth"},
        must_stay={"dane"},
    )


SCENARIOS: list[Scenario] = [
    _scenario_a1(),
    _scenario_a2(),
    _scenario_a3(),
    _scenario_b1(),
    _scenario_b2(),
    _scenario_b3(),
    _scenario_c1(),
    _scenario_c2(),
    _scenario_d1(),
    _scenario_e1(),
]


# ---------- Grading ---------------------------------------------------------


def _score_scenario(s: Scenario, predicted: list[str], weight: int) -> tuple[int, str]:
    """Score a single scenario.

    Full credit: prediction exactly matches expected offstage set
    (including the empty set for negative controls).

    Partial credit (40 % of weight): the model got SOME of the
    expected ids and didn't offstage anything that must stay.

    Mixed (40 % of weight): the model got at least one expected
    id but ALSO offstaged a must-stay character or hallucinated
    an unknown id.

    Zero: predicted nothing when something was expected, or
    predicted only wrong ids.
    """
    pred = set(predicted)
    expected = s.expected_offstage
    forbidden = s.must_stay
    correct_hits = pred & expected
    missed = expected - pred
    bad_removals = pred & forbidden
    extras = pred - expected - forbidden
    partial = max(1, round(weight * 0.4))

    if pred == expected:
        return weight, f"PERFECT: {sorted(pred)}"
    if correct_hits and not bad_removals and not extras and missed:
        return partial, f"PARTIAL — got {sorted(correct_hits)}, missed {sorted(missed)}"
    if correct_hits and (bad_removals or extras):
        return partial, (
            f"MIXED — got {sorted(correct_hits)} but also removed/hallucinated "
            f"{sorted(bad_removals | extras)} (must-stay={sorted(forbidden)})"
        )
    if not pred and expected:
        return 0, f"MISSED — predicted nothing, expected {sorted(expected)}"
    if bad_removals or extras:
        return 0, (
            f"WRONG — predicted {sorted(pred)}; expected {sorted(expected)}; "
            f"must-stay={sorted(forbidden)}"
        )
    return 0, f"WRONG — predicted {sorted(pred)} expected {sorted(expected)}"


# ---------- LLM call --------------------------------------------------------


def _load_settings() -> Settings:
    raw = settings_path().read_text(encoding="utf-8")
    return Settings.model_validate_json(raw)


async def _drain(stream) -> str:
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    return "".join(chunks)


async def _run_one(
    scenario: Scenario, client: OpenAiCompatibleLlmClient, llm_settings, weight: int
) -> tuple[int, str]:
    msgs = world_refresh.build(
        world=scenario.world,
        history=scenario.history,
        player_typed_quotes=[],
        on_stage_ids=scenario.on_stage_ids,
        player_character_id=scenario.player_id,
        character_names=scenario.character_names,
    )
    log(f"[harness] {scenario.name}: prompt_chars={sum(len(m['content']) for m in msgs)}")
    raw = await _drain(
        await client.complete(
            msgs,
            model=llm_settings.model,
            temperature=0.0,
            max_tokens=2048,
            stream=False,
        )
    )
    try:
        parsed = parse_json_object(raw, LlmSummaryResult)
    except Exception as exc:
        log(f"[harness] {scenario.name}: PARSE FAIL — {exc}; raw[:300]={raw[:300]!r}")
        return 0, f"PARSE FAIL: {exc}"
    predicted = list(parsed.characters_to_offstage)
    score, why = _score_scenario(scenario, predicted, weight)
    log(f"[harness] {scenario.name}: predicted={predicted} score={score}/{weight} — {why}")
    return score, why


async def _async_main() -> int:
    settings = _load_settings()
    deterministic_llm = settings.llm.model_copy(update={"temperature": 0.0})
    client = OpenAiCompatibleLlmClient(deterministic_llm)
    log(f"[harness] model={deterministic_llm.model} base_url={deterministic_llm.base_url}")
    n = len(SCENARIOS)
    # Distribute 100 points evenly; use list to track each scenario's max.
    base_weight = 100 // n
    weights = [base_weight] * n
    # Assign any remainder to the first scenarios so total == 100.
    for i in range(100 - sum(weights)):
        weights[i] += 1
    log(f"[harness] running {n} scenarios; weights={weights}")
    try:
        results: list[tuple[Scenario, int, int, str]] = []
        # Run scenarios concurrently so a 10-scenario pass doesn't
        # serialise into a 60-second wall-clock penalty per
        # autoresearch iteration. Each scenario is an independent
        # LLM call; the order doesn't affect any global state.
        tasks = [
            _run_one(s, client, deterministic_llm, weights[i]) for i, s in enumerate(SCENARIOS)
        ]
        scored = await asyncio.gather(*tasks)
        for (s, weight), (score, why) in zip(
            zip(SCENARIOS, weights, strict=True), scored, strict=True
        ):
            results.append((s, score, weight, why))
        total = sum(score for _, score, _, _ in results)
        log("[harness] === Summary ===")
        for s, score, weight, why in results:
            log(f"[harness]   {s.name}: {score}/{weight} — {why}")
        log(f"[harness] TOTAL: {total}/100")
        return total
    finally:
        await client.aclose()


def main() -> None:
    total = asyncio.run(_async_main())
    print(total)


if __name__ == "__main__":
    main()

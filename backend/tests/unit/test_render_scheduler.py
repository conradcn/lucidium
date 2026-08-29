"""Per-entity redraw scheduler: scoring, priority tiers, and the
render/apply behaviour.

Pins the contract described in ``render_scheduler``:
  * staleness score rises by 1 per turn, +1000 on an explicit request;
  * the worker draws the highest-scoring stale entity, characters ahead
    of the background on a tie, and every on-screen entity ahead of any
    speculative pre-render;
  * a completed render supersedes the presented one (prepended), and the
    ONLY discard is an entity that vanished — the speculative-not-picked
    case.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lucidium.api.messages import MessageType
from lucidium.domain.character import Character, CharacterImage, CharacterKind
from lucidium.domain.dialog import DialogNode, DialogNodeState, DialogTree
from lucidium.domain.environment import Environment
from lucidium.domain.game import Game
from lucidium.domain.settings import ImageSettings, Settings
from lucidium.domain.world import WorldState
from lucidium.orchestration.render_scheduler import EXPLICIT_BOOST, RenderScheduler


def _world() -> WorldState:
    return WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink wash",
    )


def _npc(name: str, *, seed: int = 7) -> Character:
    return Character(
        name=name,
        description=f"the {name}",
        gender="male",
        age=40,
        ethnicity="local",
        skin="tan",
        hair_color="grey",
        hairstyle="short",
        eye_color="brown",
        build="stocky",
        bust="n/a",
        outfit="oilskin",
        pose="leaning",
        expression="watchful",
        seed=seed,
        kind=CharacterKind.human,
    )


def _env() -> Environment:
    return Environment(location_label="harbor", prompt="stone harbor", prompt_hash="x")


class _Img:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def generate(self, workflow, params, *, seed):
        self.calls.append((workflow, dict(params), seed))
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class _Session:
    def __init__(self, game: Game, saves_root: Path) -> None:
        self.game = game
        self.settings = Settings(image=ImageSettings())
        self.saves_root = saves_root
        self.emitted: list[tuple] = []
        self._img = _Img()
        self._asset_tasks: list = []

    def emit(self, message_type, payload) -> None:
        self.emitted.append((message_type, payload))

    def install_game(self, g: Game) -> None:
        self.game = g

    async def commit(self) -> None:
        self.commit_blocking()

    def commit_blocking(self) -> None:
        pass

    def _image_factory(self) -> _Img:
        return self._img


def _game(
    chars: list[Character],
    env: Environment | None,
    *,
    on_stage: list[str],
    location_id: str | None,
) -> Game:
    node = DialogNode(
        parent_id=None,
        speaker_id=None,
        text="x",
        location_id=location_id,
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )
    return Game(
        world=_world(),
        characters={c.id: c for c in chars},
        environments={env.id: env} if env else {},
        dialog_tree=DialogTree(
            nodes={node.id: node},
            root_id=node.id,
            committed_path=[node.id],
        ),
        current_node_id=node.id,
        on_stage=list(on_stage),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_redraw_priority_orders_by_change_kind(tmp_path: Path) -> None:
    """Redraw priority follows the requested order: new character >
    new scene > clothing/appearance change > expression-only change."""
    from lucidium.orchestration.render_scheduler import (
        _PRIORITY_CLOTHING,
        _PRIORITY_EXPRESSION,
        _PRIORITY_NEW_CHARACTER,
        _PRIORITY_NEW_SCENE,
    )

    base_snap = {
        "outfit": "oilskin",
        "pose": "leaning",
        "expression": "watchful",
        "hair_color": "grey",
        "eye_color": "brown",
        "effects": "",
    }

    def _img(name: str) -> CharacterImage:
        p = tmp_path / name
        p.write_bytes(b"x")
        return CharacterImage(
            path=str(p),
            prompt_hash="old",
            identity_hash="id",
            attributes_snapshot=dict(base_snap),
        )

    new_char = _npc("New", seed=1)  # no images -> brand-new character
    clothing = _npc("Cloth", seed=2).model_copy(
        update={"outfit": "a red cloak", "images": [_img("cloth.png")]}
    )
    expr = _npc("Expr", seed=3).model_copy(
        update={"expression": "grinning", "images": [_img("expr.png")]}
    )
    env = _env()
    game = _game(
        [new_char, clothing, expr],
        env,
        on_stage=[new_char.id, clothing.id, expr.id],
        location_id=env.id,
    )
    sch = RenderScheduler(_Session(game, tmp_path))

    assert sch._redraw_priority(game, new_char.id, "portrait") == _PRIORITY_NEW_CHARACTER
    assert sch._redraw_priority(game, env.id, "background") == _PRIORITY_NEW_SCENE
    assert sch._redraw_priority(game, clothing.id, "portrait") == _PRIORITY_CLOTHING
    assert sch._redraw_priority(game, expr.id, "portrait") == _PRIORITY_EXPRESSION

    # With every entity stale and scored, the worker picks the highest
    # priority first: the brand-new character.
    sch.on_turn()
    assert sch._select() == ("portrait", new_char.id)


def test_on_turn_increments_staleness_each_turn(tmp_path: Path) -> None:
    npc = _npc("Hale")
    env = _env()
    game = _game([npc], env, on_stage=[npc.id], location_id=None)
    # location_id None -> active env resolves to None; set it so the env
    # is the active scene.
    game = _game([npc], env, on_stage=[npc.id], location_id=env.id)
    sch = RenderScheduler(_Session(game, tmp_path))

    sch.on_turn()
    assert sch._scores[npc.id] == 1
    assert sch._scores[env.id] == 1
    sch.on_turn()
    assert sch._scores[npc.id] == 2
    assert sch._scores[env.id] == 2


def test_explicit_request_boosts_and_marks(tmp_path: Path) -> None:
    npc = _npc("Hale")
    env = _env()
    game = _game([npc], env, on_stage=[npc.id], location_id=env.id)
    sch = RenderScheduler(_Session(game, tmp_path))

    sch.request_redraw(npc.id)
    assert sch._scores[npc.id] == EXPLICIT_BOOST
    assert npc.id in sch._explicit


def test_non_stale_entities_drop_out(tmp_path: Path) -> None:
    """An on-screen character that already has its current portrait does
    not accumulate score."""
    npc = _npc("Hale")
    env = _env()
    game = _game([npc], env, on_stage=[npc.id], location_id=env.id)
    sess = _Session(game, tmp_path)
    # Give the NPC its CURRENT portrait so it reads as fresh.
    from lucidium.orchestration.assets import _active_lighting, _portrait_prompt_hash

    lighting = _active_lighting(game)
    current = _portrait_prompt_hash(game.world, npc, lighting)
    fresh = npc.model_copy(
        update={
            "images": [CharacterImage(path="p.png", prompt_hash=current, attributes_snapshot={})],
        }
    )
    sess.game = _game([fresh], env, on_stage=[fresh.id], location_id=env.id)
    sch = RenderScheduler(sess)

    sch.on_turn()
    assert fresh.id not in sch._scores  # fresh portrait -> no redraw score
    assert env.id in sch._scores  # env still needs a first render


# ---------------------------------------------------------------------------
# Priority / selection
# ---------------------------------------------------------------------------


def test_select_prefers_character_over_background_on_tie(tmp_path: Path) -> None:
    npc = _npc("Hale")
    env = _env()
    game = _game([npc], env, on_stage=[npc.id], location_id=env.id)
    sch = RenderScheduler(_Session(game, tmp_path))
    sch._scores = {npc.id: 1, env.id: 1}

    assert sch._select() == ("portrait", npc.id)


def test_select_drains_in_descending_staleness_order(tmp_path: Path) -> None:
    """Among several stale on-screen characters, the worker draws the
    one that has been stale LONGEST (highest score) first, then the next,
    and so on — the core 'most stale wins' rule."""
    a = _npc("Aldo", seed=1)
    b = _npc("Bea", seed=2)
    c = _npc("Cole", seed=3)
    env = _env()
    game = _game([a, b, c], env, on_stage=[a.id, b.id, c.id], location_id=env.id)
    sch = RenderScheduler(_Session(game, tmp_path))
    # Different staleness ages; env is fresh-ish at 1 (lowest).
    sch._scores = {a.id: 2, b.id: 5, c.id: 3, env.id: 1}

    # Simulate the pump's pick-then-pop loop and record the order.
    order: list[str] = []
    for _ in range(4):
        pick = sch._select()
        assert pick is not None
        order.append(pick[1])
        sch._scores.pop(pick[1], None)

    assert order == [b.id, c.id, a.id, env.id], order


def test_select_explicit_beats_onscreen_staleness(tmp_path: Path) -> None:
    a = _npc("Aldo", seed=1)
    b = _npc("Bea", seed=2)
    env = _env()
    game = _game([a, b], env, on_stage=[a.id, b.id], location_id=env.id)
    sch = RenderScheduler(_Session(game, tmp_path))
    # ``a`` has high turn-staleness; ``b`` was explicitly requested.
    sch._scores = {a.id: 50, b.id: EXPLICIT_BOOST}
    sch._explicit = {b.id}

    assert sch._select() == ("portrait", b.id)


def test_speculative_sits_behind_every_onscreen_entity(tmp_path: Path) -> None:
    onscreen = _npc("Hale", seed=1)
    spec = _npc("Ghost", seed=2)
    env = _env()
    # ``spec`` exists in the cast but is NOT on stage (a not-yet-walked
    # branch character).
    game = _game([onscreen, spec], env, on_stage=[onscreen.id], location_id=env.id)
    sch = RenderScheduler(_Session(game, tmp_path))
    sch._scores = {onscreen.id: 1}  # low on-screen staleness
    sch._spec = {spec.id: 999}  # high speculative score

    # On-screen wins despite the far higher speculative score.
    assert sch._select() == ("portrait", onscreen.id)
    # Once nothing on-screen is pending, the speculative one is next.
    sch._scores = {}
    assert sch._select() == ("portrait", spec.id)


# ---------------------------------------------------------------------------
# Render + apply
# ---------------------------------------------------------------------------


async def _drain_pump(sch: RenderScheduler) -> None:
    """Run the scheduler's queue to empty.

    Awaits the pump task ``on_turn`` / ``request_*`` already started
    rather than calling ``_pump_loop`` a second time: two pumps on one
    scheduler render concurrently, and each ``_apply_*`` is a
    read-modify-write of ``session.game``, so the second one to install
    silently drops the first one's result. Production never has two
    (``_ensure_pump`` is idempotent) — only a test that starts its own
    can hit it.
    """
    if sch._pump is not None:
        await sch._pump
    else:
        await sch._pump_loop()


@pytest.mark.asyncio
async def test_pump_renders_supersedes_and_emits(tmp_path: Path) -> None:
    npc = _npc("Hale")
    env = _env()
    game = _game([npc], env, on_stage=[npc.id], location_id=env.id)
    sess = _Session(game, tmp_path)
    sch = RenderScheduler(sess)

    sch.on_turn()  # both stale -> scored
    await _drain_pump(sch)  # drain synchronously

    # The character gained its portrait and the env gained a background.
    assert sess.game.characters[npc.id].images, "portrait not applied"
    assert sess.game.environments[env.id].image_path, "background not applied"
    readies = [p for (mt, p) in sess.emitted if mt == MessageType.s2c_image_ready]
    kinds = {p.kind.value for p in readies}
    assert "portrait" in kinds and "background" in kinds, kinds


@pytest.mark.asyncio
async def test_pump_discards_speculative_render_when_branch_not_picked(
    tmp_path: Path,
) -> None:
    """A speculative character that is trimmed (branch not picked) before
    its render starts produces no render at all — the only discard."""
    spec = _npc("Ghost")
    # No on-stage characters, no active environment, so nothing else is
    # pending — the pump's only candidate would be the speculative char.
    game = _game([spec], None, on_stage=[], location_id=None)
    sess = _Session(game, tmp_path)
    sch = RenderScheduler(sess)

    sch.request_speculative([spec.id])
    # The branch is invalidated: the character is trimmed from the live
    # game and the scheduler drops the pending request.
    sess.install_game(_game([], None, on_stage=[], location_id=None))
    sch.on_invalidation()

    await _drain_pump(sch)
    assert sess._img.calls == [], "discarded speculative render should not run"


async def test_apply_background_suppresses_offscreen_image_ready(tmp_path: Path) -> None:
    """An automatic (turn-driven) background that lands after the player
    walked on must NOT announce itself — the env.image_path is set
    (cached for a return visit) but no image_ready swaps the backdrop to
    an off-screen room. An explicit Rerender announces regardless."""
    env = _env()
    other = Environment(location_label="cellar", prompt="dripping cellar", prompt_hash="y")
    # Active scene is ``other``; the render that just finished is for the
    # now-off-screen ``env``.
    game = _game([], other, on_stage=[], location_id=other.id)
    game = game.model_copy(update={"environments": {env.id: env, other.id: other}})
    sess = _Session(game, tmp_path)
    sch = RenderScheduler(sess)

    await sch._apply_background(env.id, Path("bg.png"), "newhash", explicit=False)
    # image_path updated (cached) but NO image_ready emitted.
    assert sess.game.environments[env.id].image_path == "bg.png"
    assert not any(mt == MessageType.s2c_image_ready for (mt, _p) in sess.emitted)

    # An explicit Rerender of the same off-screen env DOES announce.
    await sch._apply_background(env.id, Path("bg2.png"), "newhash2", explicit=True)
    assert any(mt == MessageType.s2c_image_ready for (mt, _p) in sess.emitted)


async def test_apply_portrait_drops_when_character_gone(tmp_path: Path) -> None:
    """An in-flight render whose character vanished by apply time is
    dropped without error (the speculative-not-picked discard, applied
    at landing rather than at queue time)."""
    npc = _npc("Hale")
    env = _env()
    game = _game([npc], env, on_stage=[npc.id], location_id=env.id)
    sess = _Session(game, tmp_path)
    sch = RenderScheduler(sess)
    # Character is gone from the live game.
    sess.install_game(_game([], env, on_stage=[], location_id=env.id))

    image = CharacterImage(path="late.png", prompt_hash="zzz", attributes_snapshot={})
    await sch._apply_portrait(npc.id, image)  # must be a no-op, not a crash

    assert npc.id not in sess.game.characters
    assert not any(mt == MessageType.s2c_image_ready for (mt, _p) in sess.emitted)

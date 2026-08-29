"""Per-entity redraw scheduler.

Each on-screen entity — a character on stage, plus the active-scene
background — carries a *redraw score*. The score rises by ``1`` for
every turn the entity has been stale (its on-disk image no longer
matches what its current attributes would render) and by ``1000`` the
moment the player explicitly asks for a redraw (the Cast / Environments
"Rerender" buttons). A single render worker repeatedly picks the
highest-scoring stale entity, renders its CURRENT state, and replaces
the presented image when the render lands.

Three properties fall out of "always render the entity's CURRENT state
at the moment it is picked":

  * **A completed render always supersedes the presented one** — it was
    the most-current state when it started, and the single worker means
    no newer render of the same entity could have landed in between.
    There is no stale-hash drop: a render is applied as long as it is
    newer than what is on screen.
  * **The only discard is speculative-and-not-picked.** A render is
    dropped solely when its entity has vanished from the live game by
    the time it lands — which happens exactly when a speculative branch
    is invalidated (the trim removes its characters). A render already
    in flight is never interrupted.
  * **An unstarted request never renders a stale image.** The worker
    recomputes the entity's state at pick time, so a request enqueued
    several edits ago still produces the image the entity needs *now*;
    an entity that is no longer stale is simply skipped.

Speculative pre-renders (characters that exist only in not-yet-walked
branches) sit in a strictly lower tier: every stale on-screen entity is
drained before the worker touches a speculative one.

The worker is transient: ``on_turn`` / ``request_redraw`` /
``request_speculative`` spawn a pump task if one is not already running;
the pump drains the score maps and exits when nothing is left to do. The
pump registers itself on ``session._asset_tasks`` so the existing
drain helpers (and tests) observe it like any other background asset
task.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain.character import CharacterImage
    from ..domain.game import Game
    from .session import Session

_log = logging.getLogger(__name__)

# Explicit player requests jump the queue: a single Rerender click
# outscores ~1000 turns of accumulated staleness, so it is always next.
EXPLICIT_BOOST = 1000

# Redraw-reason priority. Within a tier (explicit / on-screen), the
# REASON an entity is stale decides order before accumulated staleness
# score does — the player asked for exactly this ranking:
#   new character  >  new scene  >  clothing change  >  expression change
# Higher number = rendered first. Pose / hair / other appearance edits
# ride the CLOTHING tier (a bigger visual change than a bare expression
# swap); a background render is always the NEW_SCENE tier.
_PRIORITY_NEW_CHARACTER = 4
_PRIORITY_NEW_SCENE = 3
_PRIORITY_CLOTHING = 2
_PRIORITY_EXPRESSION = 1


class RenderScheduler:
    def __init__(self, session: Session) -> None:
        self._session = session
        # On-screen entity id -> accumulated redraw score.
        self._scores: dict[str, int] = {}
        # Ids the player explicitly asked to redraw. These bypass the
        # on-screen / active-scene gates (a Rerender on the PC or on a
        # background that is not the current scene must still render) and
        # carry the PC render authorisation.
        self._explicit: set[str] = set()
        # Speculative character id -> score (strictly lower tier).
        self._spec: dict[str, int] = {}
        self._pump: asyncio.Task[None] | None = None

    # ---- public API ---------------------------------------------------------

    def on_turn(self) -> None:
        """Advance the staleness clock by one turn. Called once per walk
        (the foreground/async asset hook). Every stale on-screen entity
        gains a point; entities that walked off stage drop out."""
        game = self._session.game
        if game is None:
            return
        from .assets import _active_lighting

        lighting = _active_lighting(game)
        char_ids, env_id = self._onscreen(game)
        onscreen = set(char_ids) | ({env_id} if env_id else set())
        # Stop tracking entities that are no longer on screen (unless the
        # player explicitly asked for them — those render regardless).
        for eid in list(self._scores):
            if eid not in onscreen and eid not in self._explicit:
                del self._scores[eid]
        for cid in char_ids:
            if self._portrait_stale(game, cid, lighting):
                self._scores[cid] = self._scores.get(cid, 0) + 1
        if env_id and self._background_stale(game, env_id):
            self._scores[env_id] = self._scores.get(env_id, 0) + 1
        self._ensure_pump()

    def request_redraw(self, entity_id: str) -> None:
        """Player explicitly asked to redraw a character or environment
        (the Rerender buttons). Adds the explicit boost and renders the
        entity even if it is the PC or an off-scene background."""
        self._scores[entity_id] = self._scores.get(entity_id, 0) + EXPLICIT_BOOST
        self._explicit.add(entity_id)
        self._ensure_pump()

    def request_speculative(self, character_ids: list[str]) -> None:
        """Pre-render portraits for characters in a not-yet-walked
        branch. Strictly lower priority than any on-screen work."""
        for cid in character_ids:
            self._spec[cid] = self._spec.get(cid, 0) + 1
        self._ensure_pump()

    def on_invalidation(self) -> None:
        """A speculative subtree was invalidated. Drop any pending
        request whose entity no longer exists on the live game — that is
        the speculative-and-not-picked discard. Renders already in
        flight are left to finish; they self-discard at apply time when
        they find their character gone."""
        game = self._session.game
        chars = game.characters if game is not None else {}
        envs = game.environments if game is not None else {}
        for cid in list(self._spec):
            if cid not in chars:
                del self._spec[cid]
        for eid in list(self._scores):
            if eid not in chars and eid not in envs:
                del self._scores[eid]
                self._explicit.discard(eid)

    # ---- staleness ----------------------------------------------------------

    def _onscreen(self, game: Game) -> tuple[list[str], str | None]:
        from .assets import _resolve_active_environment_id

        char_ids = [
            cid
            for cid in game.on_stage
            if (ch := game.characters.get(cid)) is not None and not ch.removed and not ch.is_player
        ]
        env_id = _resolve_active_environment_id(game)
        return char_ids, env_id

    def _portrait_stale(self, game: Game, cid: str, lighting: str) -> bool:
        from .assets import _long_context_image_model, _portrait_prompt_hash

        ch = game.characters.get(cid)
        if ch is None:
            return False
        long_context = _long_context_image_model(self._session)
        desired = _portrait_prompt_hash(game.world, ch, lighting, long_context=long_context)
        if not ch.images:
            return True
        return ch.images[0].prompt_hash != desired

    def _background_stale(self, game: Game, eid: str) -> bool:
        from .assets import _background_prompt_hash, _long_context_image_model

        env = game.environments.get(eid)
        if env is None:
            return False
        if not env.image_path or not Path(env.image_path).exists():
            return True
        long_context = _long_context_image_model(self._session, environment=True)
        return env.prompt_hash != _background_prompt_hash(
            game.world, env, long_context=long_context
        )

    # ---- pump ---------------------------------------------------------------

    def _ensure_pump(self) -> None:
        if not self._scores and not self._spec:
            return
        if self._pump is not None and not self._pump.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (e.g. a synchronous unit-test context that
            # only inspects state). Scores stay queued; the next call
            # made under a loop starts the pump.
            return
        self._pump = loop.create_task(self._pump_loop())
        tasks = [t for t in self._session._asset_tasks if not t.done()]
        tasks.append(self._pump)
        self._session._asset_tasks = tasks

    async def _pump_loop(self) -> None:
        while True:
            pick = self._select()
            if pick is None:
                return
            kind, eid = pick
            explicit = eid in self._explicit
            # One render attempt per scoring: drop the entity from the
            # maps BEFORE rendering so a render that leaves it still stale
            # (it changed mid-render) waits for the next turn rather than
            # spinning the pump. on_turn re-scores it if it stays stale.
            self._scores.pop(eid, None)
            self._spec.pop(eid, None)
            self._explicit.discard(eid)
            try:
                await self._render_and_apply(kind, eid, explicit=explicit)
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.warning("render failed for %s", eid, exc_info=True)

    def _select(self) -> tuple[str, str] | None:
        """Highest-priority stale work, or ``None`` when idle. Tiers, in
        order: explicit player requests, on-screen staleness, speculative
        pre-renders. Within a tier, higher score wins; characters
        outrank the background on a tie; ids break further ties."""
        game = self._session.game
        if game is None:
            return None
        from .assets import _active_lighting

        lighting = _active_lighting(game)
        char_ids, env_id = self._onscreen(game)
        onscreen = set(char_ids) | ({env_id} if env_id else set())

        explicit_best: tuple[tuple[int, int, str], str, str] | None = None
        onscreen_best: tuple[tuple[int, int, str], str, str] | None = None
        drop: list[str] = []
        for eid, score in self._scores.items():
            kind = self._classify(game, eid)
            if kind is None:
                drop.append(eid)
                continue
            if not self._stale(game, eid, kind, lighting):
                drop.append(eid)
                continue
            # Redraw-reason priority is the PRIMARY key (new character >
            # new scene > clothing > expression); accumulated staleness
            # ``score`` breaks ties within a reason tier; id breaks the
            # final tie (ascending — handled in ``_better``).
            priority = self._redraw_priority(game, eid, kind)
            rank = (priority, score, eid)
            if eid in self._explicit:
                explicit_best = _better(explicit_best, (rank, kind, eid))
            elif eid in onscreen:
                onscreen_best = _better(onscreen_best, (rank, kind, eid))
            else:
                # Scored but neither explicit nor on screen — stale
                # leftover from before the entity left; forget it.
                drop.append(eid)
        for eid in drop:
            self._scores.pop(eid, None)
            self._explicit.discard(eid)

        if explicit_best is not None:
            return (explicit_best[1], explicit_best[2])
        if onscreen_best is not None:
            return (onscreen_best[1], onscreen_best[2])

        spec_best: tuple[tuple[int, str], str] | None = None
        spec_drop: list[str] = []
        for cid, score in self._spec.items():
            ch = game.characters.get(cid)
            if ch is None or ch.is_player:
                spec_drop.append(cid)
                continue
            if not self._portrait_stale(game, cid, lighting):
                spec_drop.append(cid)
                continue
            key = (score, cid)
            if (
                spec_best is None
                or key[0] > spec_best[0][0]
                or (key[0] == spec_best[0][0] and cid < spec_best[0][1])
            ):
                spec_best = (key, cid)
        for cid in spec_drop:
            self._spec.pop(cid, None)
        if spec_best is not None:
            return ("portrait", spec_best[1])
        return None

    def _redraw_priority(self, game: Game, eid: str, kind: str) -> int:
        """Rank an entity by WHY it needs a redraw (see the module-level
        ``_PRIORITY_*`` constants). A background is always ``NEW_SCENE``.
        A portrait with no on-disk image yet is a ``NEW_CHARACTER``;
        otherwise the change is classified by diffing the current
        attributes against the last render's ``attributes_snapshot`` —
        an expression-only change is the lowest tier, anything else
        (outfit, pose, hair, effects, lighting) is the CLOTHING tier."""
        if kind == "background":
            return _PRIORITY_NEW_SCENE
        ch = game.characters.get(eid)
        if ch is None:
            return _PRIORITY_EXPRESSION
        latest = next((im for im in ch.images if Path(im.path).exists()), None)
        if latest is None:
            return _PRIORITY_NEW_CHARACTER
        snap = latest.attributes_snapshot or {}

        def _changed(field: str) -> bool:
            return str(snap.get(field, "")) != str(getattr(ch, field, "") or "")

        if any(
            _changed(f) for f in ("outfit", "pose", "hair_color", "eye_color", "effects", "decals")
        ):
            return _PRIORITY_CLOTHING
        if _changed("expression"):
            return _PRIORITY_EXPRESSION
        # Stale for a reason we don't snapshot (e.g. room lighting).
        # Treat as an appearance change rather than the lowest tier.
        return _PRIORITY_CLOTHING

    def _classify(self, game: Game, eid: str) -> str | None:
        if eid in game.characters:
            return "portrait"
        if eid in game.environments:
            return "background"
        return None

    def _stale(self, game: Game, eid: str, kind: str, lighting: str) -> bool:
        if kind == "portrait":
            return self._portrait_stale(game, eid, lighting)
        return self._background_stale(game, eid)

    # ---- render + apply -----------------------------------------------------

    async def _render_and_apply(self, kind: str, eid: str, *, explicit: bool) -> None:
        session = self._session
        game = session.game
        if game is None:
            return
        from .assets import (
            _active_lighting,
            _images_dir,
            check_minor_nudity_and_correct,
            render_background_to_disk,
            render_portrait_to_disk,
        )

        images_dir = _images_dir(game.id, session.saves_root)
        images_dir.mkdir(parents=True, exist_ok=True)
        client = session._image_factory()

        if kind == "portrait":
            # Minor-nudity safety pass mirrors ensure_portraits_for; it
            # may replace session.game, so re-read afterwards.
            check_minor_nudity_and_correct(session)
            game = session.game
            if game is None:
                return
            character = game.characters.get(eid)
            if character is None:
                return  # vanished — speculative-and-not-picked discard.
            if character.is_player and not explicit:
                return
            lighting = _active_lighting(game)
            image = await render_portrait_to_disk(
                session=session,
                image_client=client,
                character=character,
                lighting=lighting,
                images_dir=images_dir,
            )
            if image is None:
                return
            await self._apply_portrait(eid, image)
        else:
            env = game.environments.get(eid)
            if env is None:
                return
            result = await render_background_to_disk(
                session=session,
                image_client=client,
                env=env,
                images_dir=images_dir,
            )
            if result is None:
                return
            target, prompt_hash = result
            await self._apply_background(eid, target, prompt_hash, explicit=explicit)

    async def _apply_portrait(self, cid: str, image: CharacterImage) -> None:
        session = self._session
        game = session.game
        if game is None:
            return
        char = game.characters.get(cid)
        if char is None:
            return  # discarded: the character was trimmed while rendering.
        if char.images and char.images[0].prompt_hash == image.prompt_hash:
            return  # already presented — idempotent.
        new_chars = dict(game.characters)
        new_chars[cid] = char.model_copy(update={"images": [image, *char.images]})
        session.install_game(game.model_copy(update={"characters": new_chars}))
        await self._commit()
        self._emit_image(
            kind="portrait", target_id=cid, image_path=image.path, image_id=image.prompt_hash
        )

    async def _apply_background(
        self, eid: str, target: Path, prompt_hash: str, *, explicit: bool
    ) -> None:
        session = self._session
        game = session.game
        if game is None:
            return
        env = game.environments.get(eid)
        if env is None:
            return
        already = env.prompt_hash == prompt_hash and env.image_path == str(target)
        if not already:
            new_envs = dict(game.environments)
            new_envs[eid] = env.model_copy(
                update={"image_path": str(target), "prompt_hash": prompt_hash}
            )
            session.install_game(game.model_copy(update={"environments": new_envs}))
            await self._commit()
        # An automatic (turn-driven) background only announces itself when
        # it is the active scene — a render that lands after the player
        # walked on must not swap the backdrop to an off-screen room. An
        # explicit Rerender always announces (the player is looking at
        # that env in the editor).
        from .assets import _resolve_active_environment_id

        current = session.game
        # ``game`` above was non-None and install_game only ever replaces
        # the game, never clears it.
        assert current is not None
        if explicit or _resolve_active_environment_id(current) == eid:
            self._emit_image(
                kind="background", target_id=eid, image_path=str(target), image_id=prompt_hash
            )

    # ---- emit ---------------------------------------------------------------

    async def _commit(self) -> None:
        # Awaited rather than fired-and-forgotten so the save is on disk
        # before the matching ``s2c/image_ready`` tells the renderer the
        # image exists — a crash in between would otherwise leave a save
        # that doesn't mention a portrait the player already saw.
        try:
            await self._session.commit()
        except Exception:
            _log.warning("render scheduler commit failed", exc_info=True)

    def _emit_image(self, *, kind: str, target_id: str, image_path: str, image_id: str) -> None:
        session = self._session
        if session.emit is None or session.game is None:
            return
        from ..api.handlers import _full_state_message
        from ..api.messages import ImageKind, MessageType, S2CImageReady

        try:
            session.emit(MessageType.s2c_state_full, _full_state_message(session))
            session.emit(
                MessageType.s2c_image_ready,
                S2CImageReady(
                    kind=ImageKind.portrait if kind == "portrait" else ImageKind.background,
                    target_id=target_id,
                    image_path=image_path,
                    image_id=image_id,
                ),
            )
        except Exception:
            _log.warning("render scheduler emit failed", exc_info=True)


_Ranked = tuple[tuple[int, int, str], str, str]


def _better(current: _Ranked | None, candidate: _Ranked) -> _Ranked:
    """Pick the higher-ranked of two ``(rank, kind, eid)`` tuples.
    ``rank`` is ``(priority, score, eid)``; higher redraw-reason
    priority wins, then higher staleness score, then the LOWER id for
    determinism."""
    if current is None:
        return candidate
    cur_rank = current[0]
    cand_rank = candidate[0]
    # Compare priority then score (both higher-is-better); for the id
    # tiebreak, lower id wins, so invert that last element.
    cur_key = (cur_rank[0], cur_rank[1])
    cand_key = (cand_rank[0], cand_rank[1])
    if cand_key > cur_key:
        return candidate
    if cand_key == cur_key and cand_rank[2] < cur_rank[2]:
        return candidate
    return current

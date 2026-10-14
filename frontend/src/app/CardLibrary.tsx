import type { JSX, ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

import type { CardSummaryPayload } from "../shared/generated/CardSummaryPayload.schema";
import type { CardErrorPayload } from "../shared/generated/CardErrorPayload.schema";
import { assetUrl } from "./assetUrl";
import { ensureClient, send } from "./client";
import { canOpenFolder, openFolder } from "./desktopBridge";

interface Props {
  /** Called with the card's bare file name when the player picks it.
   * The parent sends the actual add message. */
  onAdd: (file: string) => void;
  /** True while the parent's add is in flight — disables the buttons. */
  busy: boolean;
  addLabel?: string;
  /** Extra controls rendered under the toolbar (e.g. "place on stage"). */
  children?: ReactNode;
}

/**
 * Browser for the SillyTavern character-card library
 * (``Settings.character_cards_dir``). Lists the cards the backend found
 * there with their avatars, filters by name / tag / creator, and hands
 * the chosen card's file name to ``onAdd``.
 *
 * Errors the backend reports while an add is in flight (missing card,
 * LLM unreachable during adaptation) are shown inline, since a failed
 * add otherwise has no visible effect.
 */
export function CardLibrary({
  onAdd,
  busy,
  addLabel = "Add",
  children,
}: Props): JSX.Element {
  const [cards, setCards] = useState<CardSummaryPayload[]>([]);
  const [failures, setFailures] = useState<CardErrorPayload[]>([]);
  const [cardsDir, setCardsDir] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [filter, setFilter] = useState("");
  const [error, setError] = useState("");
  const [addingFile, setAddingFile] = useState<string | null>(null);
  // Only errors that arrive while one of OUR adds is in flight belong
  // in this panel; anything else is some other screen's business.
  const awaitingRef = useRef(false);

  useEffect(() => {
    const client = ensureClient();
    const offList = client.on("s2c/cards/list", (p) => {
      setCards(p.cards ?? []);
      setFailures(p.errors ?? []);
      setCardsDir(p.cards_dir);
      setLoaded(true);
    });
    const offError = client.on("s2c/error", (p) => {
      if (!awaitingRef.current) return;
      awaitingRef.current = false;
      setError(p.message);
    });
    send("c2s/cards/list");
    return () => {
      offList();
      offError();
    };
  }, []);

  // The parent's add settled (its optimistic guard released). Clear the
  // in-flight marker; render-phase adjustment so there's no extra paint.
  const [seenBusy, setSeenBusy] = useState(busy);
  if (seenBusy !== busy) {
    setSeenBusy(busy);
    if (!busy) setAddingFile(null);
  }
  useEffect(() => {
    if (!busy) awaitingRef.current = false;
  }, [busy]);

  const refresh = (): void => {
    setError("");
    send("c2s/cards/list");
  };

  const add = (file: string): void => {
    setError("");
    setAddingFile(file);
    awaitingRef.current = true;
    onAdd(file);
  };

  const needle = filter.trim().toLowerCase();
  const visible = needle
    ? cards.filter((c) =>
        [c.name, c.creator ?? "", ...(c.tags ?? [])].some((s) =>
          s.toLowerCase().includes(needle),
        ),
      )
    : cards;
  const adding = addingFile ? cards.find((c) => c.file === addingFile) : undefined;

  return (
    <div className="card-library" data-testid="card-library">
      <div className="card-library__toolbar">
        <input
          type="search"
          placeholder="Filter by name, tag or creator..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label="Filter character cards"
          data-testid="card-library-filter"
        />
        <button type="button" onClick={refresh} title="Re-scan the card folder">
          Refresh
        </button>
        {canOpenFolder() && cardsDir ? (
          <button
            type="button"
            onClick={() => void openFolder(cardsDir)}
            title="Open the card folder to drop new cards in"
          >
            Open folder
          </button>
        ) : null}
      </div>
      {cardsDir ? (
        <p className="card-library__dir" title={cardsDir}>
          Cards from <code>{cardsDir}</code>
        </p>
      ) : null}
      {children}
      {error ? (
        <p className="card-library__error" role="alert">
          {error}
        </p>
      ) : null}
      {busy && adding ? (
        <p className="card-library__status" role="status">
          Adding {adding.name}
          {adding.has_lucidium_data ? "..." : " — adapting the card to this story..."}
        </p>
      ) : null}
      {!loaded ? (
        <p className="card-library__empty">Loading cards...</p>
      ) : cards.length === 0 ? (
        <p className="card-library__empty">
          No character cards yet. Drop SillyTavern cards (<code>.png</code>,{" "}
          <code>.json</code> or <code>.charx</code>) into the folder above, or
          pick a different folder in Settings, then press Refresh.
        </p>
      ) : visible.length === 0 ? (
        <p className="card-library__empty">No cards match “{filter}”.</p>
      ) : (
        <ul className="card-library__grid">
          {visible.map((card) => (
            <li
              key={card.file}
              className="card-library__card"
              data-testid={`card-library-card-${card.file}`}
            >
              {card.thumbnail_path ? (
                <img
                  className="card-library__thumb"
                  src={assetUrl(card.thumbnail_path) ?? ""}
                  alt=""
                  loading="lazy"
                />
              ) : (
                <div className="card-library__thumb card-library__thumb--empty" aria-hidden="true">
                  {card.name.slice(0, 1).toUpperCase()}
                </div>
              )}
              <div className="card-library__body">
                <div className="card-library__name">
                  {card.name}
                  {card.has_lucidium_data ? (
                    <span
                      className="card-library__badge"
                      title="Exported from Lucidium — imports with its exact look, no LLM call"
                    >
                      Lucidium
                    </span>
                  ) : null}
                </div>
                {card.creator ? (
                  <div className="card-library__meta">by {card.creator}</div>
                ) : null}
                {card.summary ? (
                  <div className="card-library__summary">{card.summary}</div>
                ) : null}
                {card.tags && card.tags.length > 0 ? (
                  <div className="card-library__meta">{card.tags.join(" · ")}</div>
                ) : null}
              </div>
              <button
                type="button"
                className="card-library__add"
                onClick={() => add(card.file)}
                disabled={busy}
                aria-label={`${addLabel} ${card.name}`}
              >
                {addLabel}
              </button>
            </li>
          ))}
        </ul>
      )}
      {failures.length > 0 ? (
        <details className="card-library__failures">
          <summary>
            {failures.length} file{failures.length === 1 ? "" : "s"} could not be read
          </summary>
          <ul>
            {failures.map((f) => (
              <li key={`${f.file}|${f.reason}`}>
                {f.file ? <code>{f.file}</code> : null} {f.reason}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
}

import type { JSX } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

import { useOptimisticAction } from "../../app/useOptimisticAction";

interface Props {
  options: string[];
  loading?: boolean;
  onAnswer: (name: string, isFreeText: boolean, pronouns: string) => void;
}

const PRESET_PRONOUNS: string[] = ["she/her", "he/him", "they/them"];

/**
 * Identity step — combined pronouns + name screen.
 *
 * Pronoun chips (she/her, he/him, they/them) or a custom free-text
 * entry; the chosen value is threaded into the player Character at
 * confirm time so the storyteller addresses the player correctly
 * from beat 1. The name input is pre-filled with the first
 * available option (LLM-generated or hardcoded default) so a
 * player who doesn't care about the name can just hit Continue.
 */
export function NameStep({ options, loading, onAnswer }: Props): JSX.Element {
  const defaultName = options[0] ?? "";
  const [name, setName] = useState(defaultName);
  const [nameTouched, setNameTouched] = useState(false);
  const [pronouns, setPronouns] = useState<string>(PRESET_PRONOUNS[0] ?? "she/her");
  const [customPronouns, setCustomPronouns] = useState("");
  const [pending, run] = useOptimisticAction();

  // When the LLM-generated name options arrive (or change), refresh
  // the pre-filled default — but only if the player hasn't typed a
  // name themselves yet. The ``nameTouched`` gate keeps a player's
  // own input from being clobbered the moment fresh suggestions
  // land.
  // Done as a render-phase adjustment rather than an effect (see
  // https://react.dev/learn/you-might-not-need-an-effect) so the
  // refreshed default is in the input on the very first paint after
  // the suggestions land, with no cascading second render.
  const [seenDefaultName, setSeenDefaultName] = useState(defaultName);
  if (seenDefaultName !== defaultName) {
    setSeenDefaultName(defaultName);
    if (!nameTouched && defaultName) setName(defaultName);
  }

  const taRef = useRef<HTMLTextAreaElement | null>(null);
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [name]);

  const effectivePronouns = useMemo(() => {
    if (pronouns === "__custom__") return customPronouns.trim();
    return pronouns;
  }, [pronouns, customPronouns]);

  const canSubmit =
    !pending
    && name.trim().length > 0
    && effectivePronouns.length > 0;

  const submit = (chosenName: string, isFreeText: boolean): void => {
    if (pending) return;
    const trimmedName = chosenName.trim();
    const trimmedPronouns = effectivePronouns;
    if (trimmedName.length === 0 || trimmedPronouns.length === 0) return;
    run(() => onAnswer(trimmedName, isFreeText, trimmedPronouns));
  };

  return (
    <div className="page-frame page-frame--wide">
      <div className="page-content interview-step-content">
        <h2 className="page-title">
          Their identity
          {loading ? (
            <span
              className="step-spinner"
              role="status"
              aria-label="Generating personalised suggestions"
              data-testid="interview-options-loading"
            />
          ) : null}
        </h2>

        <div className="identity-section">
          <label className="identity-label">Pronouns</label>
          <div
            className="option-grid option-grid--compact"
            data-testid="pronoun-options"
          >
            {PRESET_PRONOUNS.map((preset) => (
              <button
                key={preset}
                disabled={pending}
                className={pronouns === preset ? "is-selected" : undefined}
                onClick={() => setPronouns(preset)}
              >
                {preset}
              </button>
            ))}
            <button
              disabled={pending}
              className={pronouns === "__custom__" ? "is-selected" : undefined}
              onClick={() => setPronouns("__custom__")}
              data-testid="pronoun-custom-button"
            >
              custom…
            </button>
          </div>
          {pronouns === "__custom__" ? (
            <input
              type="text"
              className="interview-free-text"
              placeholder="e.g. xe/xir, ze/hir, or any form"
              value={customPronouns}
              onChange={(e) => setCustomPronouns(e.target.value)}
              disabled={pending}
              data-testid="pronoun-custom-input"
              autoFocus
            />
          ) : null}
        </div>

        <div className="identity-section">
          <label className="identity-label">Name</label>
          {loading ? (
            <p className="page-subtitle" style={{ opacity: 0.7 }}>
              Showing defaults while the engine thinks — tap a name to use it,
              or keep typing.
            </p>
          ) : null}
          <div
            className="option-grid option-grid--compact"
            data-testid="name-options"
          >
            {options.map((opt) => (
              <button
                key={opt}
                disabled={pending}
                onClick={() => submit(opt, false)}
              >
                {opt}
              </button>
            ))}
          </div>
          <div className="free-text-row">
            <textarea
              ref={taRef}
              className="interview-free-text"
              rows={1}
              placeholder="Or write your own..."
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                if (!nameTouched) setNameTouched(true);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit(name, true);
                }
              }}
              disabled={pending}
              data-testid="interview-free-text"
            />
            <button
              onClick={() => submit(name, true)}
              disabled={!canSubmit}
            >
              Continue
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

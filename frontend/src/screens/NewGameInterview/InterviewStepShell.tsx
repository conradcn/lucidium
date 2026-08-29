import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";

import { useOptimisticAction } from "../../app/useOptimisticAction";

interface Props {
  title: string;
  description?: string;
  options: string[];
  freeTextPlaceholder?: string;
  // Pack options into a tighter multi-column grid (used by Genre,
  // which has 16 short labels that overflow the default 14rem-min
  // single column).
  compact?: boolean;
  // ``loading=true`` means the options on screen are HARDCODED
  // defaults shown while the LLM-driven personalised list is
  // still en route. We render a small spinner + caption near the
  // title so the player knows fresh suggestions will swap in,
  // but the buttons stay clickable — picking a default is a
  // valid answer that just bypasses the wait.
  loading?: boolean;
  onAnswer: (value: string, isFreeText: boolean) => void;
}

/**
 * The shared layout for every interview step. Each option button
 * disables on click (and so does the free-text Continue), preventing
 * the player from sending two answers for the same step which would
 * race two parallel onboardings against each other on the backend.
 */
export function InterviewStepShell({
  title,
  description,
  options,
  freeTextPlaceholder = "Or write your own...",
  compact = false,
  loading = false,
  onAnswer,
}: Props): JSX.Element {
  const [free, setFree] = useState("");
  // ``options`` arriving fresh (next step) implicitly re-mounts this
  // shell because the parent switches case — so ``pending`` resets
  // for free, no need for a reset key.
  const [pending, run] = useOptimisticAction();
  // Slide-away animation. When the parent swaps from hardcoded
  // defaults to LLM-generated personalised options (the
  // ``loading`` flag flips and ``options`` changes), abruptly
  // replacing the buttons reads as a bait-and-switch — the
  // player just clicked toward an option that vanished. We
  // briefly hold the OLD options on screen with a slide-out
  // animation, then mount the NEW options with a slide-in.
  const [displayedOptions, setDisplayedOptions] = useState<string[]>(options);
  const [leaving, setLeaving] = useState(false);
  const sameContents =
    options.length === displayedOptions.length
    && options.every((o, i) => o === displayedOptions[i]);
  // Raising the slide-out flag is a render-phase adjustment, not part
  // of the effect body: a passive effect runs AFTER the browser may
  // have painted, so setting it there risked one frame of the old
  // options without the leaving class. The 220 ms hand-off timer
  // still lives in the effect below, unchanged.
  if (!sameContents && !leaving) setLeaving(true);
  useEffect(() => {
    if (sameContents) return;
    const handle = window.setTimeout(() => {
      setDisplayedOptions(options);
      setLeaving(false);
    }, 220);
    return () => window.clearTimeout(handle);
  }, [options, sameContents]);
  // Stable key for the grid that flips when the displayed set
  // changes — re-mount triggers the slide-in animation cleanly
  // without React reusing the old DOM with stale animation state.
  const gridKey = displayedOptions.join("|");
  // Auto-grow the multiline free-text field. Players write
  // sentence-length answers ("a tired locksmith with a haunted
  // cellar") and a single-line input clipped them. The textarea
  // grows with content, max ~6 lines so it doesn't push the
  // Continue button off the page on a long answer.
  const taRef = useRef<HTMLTextAreaElement | null>(null);
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [free]);

  const submit = (): void => {
    if (pending || free.trim().length === 0) return;
    run(() => onAnswer(free, true));
  };

  return (
    <div className="page-frame page-frame--wide">
      <div className="page-content interview-step-content">
        <h2 className="page-title">
          {title}
          {loading ? (
            <span
              className="step-spinner"
              role="status"
              aria-label="Generating personalised suggestions"
              data-testid="interview-options-loading"
            />
          ) : null}
        </h2>
        {description ? <p className="page-subtitle">{description}</p> : null}
        {loading ? (
          <p className="page-subtitle" style={{ opacity: 0.7 }}>
            Showing defaults while the engine thinks. Pick one to move on,
            or wait a moment for tailored suggestions.
          </p>
        ) : null}
        <div
          key={gridKey}
          className={
            (compact ? "option-grid option-grid--compact" : "option-grid")
            + (leaving ? " option-grid--leaving" : " option-grid--entering")
          }
          data-leaving={leaving ? "true" : "false"}
        >
          {displayedOptions.map((option) => (
            <button
              key={option}
              disabled={pending || leaving}
              onClick={() => run(() => onAnswer(option, false))}
            >
              {option}
            </button>
          ))}
        </div>
        <div className="free-text-row">
          <textarea
            ref={taRef}
            className="interview-free-text"
            rows={1}
            placeholder={freeTextPlaceholder}
            value={free}
            onChange={(e) => setFree(e.target.value)}
            // Enter alone → Continue (the most common case).
            // Shift+Enter inserts a newline so the player can
            // still write multi-line answers when they want to.
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            disabled={pending}
            data-testid="interview-free-text"
          />
          <button
            onClick={submit}
            disabled={pending || free.trim().length === 0}
          >
            Continue
          </button>
        </div>
      </div>
    </div>
  );
}

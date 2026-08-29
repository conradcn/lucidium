import type { JSX } from "react";

interface Props {
  onAcknowledge: () => void;
}

/**
 * Modal overlay shown on EVERY app startup. The warning text is the
 * point of the screen — short, sober, hand-written copy about the
 * risks of treating LLM storytellers as a relationship. It sits on
 * top of everything else (wizard, start screen, anything mid-load)
 * so the player has to read it before they can interact with the
 * engine.
 *
 * Dismissal is per-session only — there's no "don't show again"
 * checkbox. Re-displaying every launch is the whole point: the
 * warning has to feel slightly inconvenient or it stops registering.
 */
export function StartupWarning({ onAcknowledge }: Props): JSX.Element {
  return (
    <div
      className="startup-warning"
      role="dialog"
      aria-modal="true"
      aria-labelledby="startup-warning-title"
      data-testid="startup-warning"
    >
      <div className="startup-warning__card">
        <h2
          id="startup-warning-title"
          className="startup-warning__title"
        >
          A note before you start
        </h2>
        <div className="startup-warning__body">
          <p>
            Among their siblings of the AI summer, storytellers seem
            innocuous. They are not. This is Narcissus' pool, in glorious HD.
            It is designed to show you exactly what you want to see, for as
            long as you're interested in watching. If it even begins to
            convince you that you're special, or it becomes something you do
            when you're lonely,{" "}
            <strong>drop everything and talk to a human being.</strong>
          </p>
          <p>
            If you don't have one in your life, you should not touch this, or
            anything like it, with a ten-foot pole. "The time I met a bunch of
            people half-assing a shift at the food bank" or "the time I went
            to my local game store and learned how to play D&D" are going to
            be much fonder memories than "the sixtieth time the AI reassured
            me that I was God's chosen."
          </p>
          <p>
            <strong>
              Treat this entire class of software with the respect due a
              recreational drug. Adults should stare responsibly. Children,
              not at all.
            </strong>
          </p>
<p>
When all else fails, remember that a system that actually knew what it was talking about would react gracefully to "And then we all started shitting ourselves." If this is real, there is no harm in trying it to be sure.
</p>
        </div>
        <div className="startup-warning__actions">
          <button
            type="button"
            onClick={onAcknowledge}
            className="startup-warning__acknowledge"
            autoFocus
          >
            I understand
          </button>
        </div>
      </div>
    </div>
  );
}

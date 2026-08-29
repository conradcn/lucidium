import type { JSX } from "react";
import { useEffect, useRef } from "react";

interface Props {
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  ariaLabel?: string;
}

/**
 * Textarea that grows with its content. Removes the scrollbar and
 * the resize grip — the height is purely a function of the value.
 *
 * Implementation: on every value change we set ``height = auto``
 * (lets ``scrollHeight`` reflect the natural content height), then
 * snap ``height`` to ``scrollHeight``. This is the standard
 * auto-grow recipe; it works in every browser without measuring
 * fonts ourselves.
 */
export function AutoGrowTextarea({
  value,
  onChange,
  placeholder,
  ariaLabel,
}: Props): JSX.Element {
  const ref = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [value]);

  return (
    <textarea
      ref={ref}
      className="autogrow"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      aria-label={ariaLabel}
      rows={1}
    />
  );
}

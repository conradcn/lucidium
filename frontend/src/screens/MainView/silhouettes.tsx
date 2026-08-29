/**
 * Gendered placeholder silhouettes shown while a character's
 * portrait is still being rendered by ComfyUI. Three shapes:
 * feminine, masculine, neutral. Each is a single SVG path drawn at
 * the same canvas dimensions as a cropped portrait (roughly 570 ×
 * 1216 — head-and-shoulders narrows to a torso, then to legs) so
 * the placeholder reads as "person-shaped" instead of "broken
 * image".
 *
 * Style: filled with a soft purple gradient, no border. The
 * gradient matches the prior ``placeholder`` look so existing
 * screenshots stay visually similar — only the silhouette outline
 * is new.
 */
import type { JSX } from "react";

interface Props {
  gender?: string | undefined;
}

export function PortraitSilhouette({ gender }: Props): JSX.Element {
  const flavor = pickFlavor(gender);
  return (
    <svg
      className="character-silhouette"
      viewBox="0 0 100 220"
      preserveAspectRatio="xMidYMax meet"
      aria-hidden
    >
      <defs>
        <radialGradient id="silhouette-fill" cx="50%" cy="40%" r="60%">
          <stop offset="0%" stopColor="rgba(110, 95, 140, 0.55)" />
          <stop offset="60%" stopColor="rgba(75, 65, 110, 0.32)" />
          <stop offset="100%" stopColor="rgba(50, 40, 80, 0.05)" />
        </radialGradient>
      </defs>
      <path d={PATHS[flavor]} fill="url(#silhouette-fill)" />
    </svg>
  );
}

type Flavor = "feminine" | "masculine" | "neutral";

function pickFlavor(gender: string | undefined): Flavor {
  if (!gender) return "neutral";
  const g = gender.trim().toLowerCase();
  if (
    g === "female" ||
    g === "f" ||
    g === "woman" ||
    g === "girl" ||
    g.startsWith("fem")
  ) {
    return "feminine";
  }
  if (
    g === "male" ||
    g === "m" ||
    g === "man" ||
    g === "boy" ||
    g.startsWith("masc")
  ) {
    return "masculine";
  }
  return "neutral";
}

// Each path covers the same 100x220 viewBox: head at top, shoulders,
// torso, legs. Differences are subtle — narrower waist + wider hips
// for feminine, broader shoulders + straight torso for masculine, a
// midpoint for neutral. Coordinates were eyeballed against a real
// cropped portrait so the silhouette overlays the same canvas the
// real image will fill once it lands.
const PATHS: Record<Flavor, string> = {
  // Hourglass: narrow shoulders relative to hips, defined waist.
  feminine:
    "M50 8 C 42 8, 36 14, 36 23 C 36 32, 42 38, 50 38 " +
    "C 58 38, 64 32, 64 23 C 64 14, 58 8, 50 8 Z " +
    "M30 50 L70 50 L75 80 L72 100 L78 130 L80 175 L70 218 L62 218 L56 175 L50 145 L44 175 L38 218 L30 218 L20 175 L22 130 L28 100 L25 80 Z",
  // Triangle: broad shoulders, straight waist, narrower hips than feminine.
  masculine:
    "M50 8 C 42 8, 35 15, 35 24 C 35 33, 42 39, 50 39 " +
    "C 58 39, 65 33, 65 24 C 65 15, 58 8, 50 8 Z " +
    "M22 52 L78 52 L82 95 L78 145 L74 218 L62 218 L56 175 L50 150 L44 175 L38 218 L26 218 L22 145 L18 95 Z",
  // Midway between the two — used when gender is unknown / "unspecified".
  neutral:
    "M50 8 C 42 8, 36 14, 36 23 C 36 32, 42 38, 50 38 " +
    "C 58 38, 64 32, 64 23 C 64 14, 58 8, 50 8 Z " +
    "M28 51 L72 51 L77 90 L74 130 L76 218 L64 218 L58 175 L50 148 L42 175 L36 218 L24 218 L26 130 L23 90 Z",
};

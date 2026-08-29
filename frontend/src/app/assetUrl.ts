/**
 * Convert a backend disk path (e.g. ``C:\Users\...\portrait.png``)
 * into a URL the renderer can use as ``<img src>`` or
 * ``background-image: url(...)``.
 *
 * Electron blocks raw ``file://`` requests issued from a packaged
 * ``file://`` page (cross-origin), so the main process registers a
 * custom ``lucidium-asset`` scheme that reads the underlying file
 * from disk. See ``electron/main.ts``.
 *
 * In the test environment (Vitest / Playwright with Vite dev server),
 * no protocol is registered — but the tests don't rely on the image
 * actually decoding; they assert the URL is well-formed.
 */
export const ASSET_SCHEME = "lucidium-asset";
// Fixed authority. The URL spec requires standard-scheme URLs to have
// a non-empty authority — Chromium rejects ``lucidium-asset:///...``
// at fetch() time for "Failed to parse URL." A literal host like
// ``local`` keeps the URL valid; main.ts ignores it.
export const ASSET_HOST = "local";

export function assetUrl(diskPath: string | null | undefined): string | null {
  if (!diskPath) return null;
  // Pass through any URL that's already in a Web-safe scheme — data:,
  // http(s):, blob:, or our own scheme. Tests and bundled fixtures
  // sometimes substitute inline data: URLs to avoid touching disk.
  if (/^(data|https?|blob|lucidium-asset):/i.test(diskPath)) return diskPath;
  // Normalise Windows backslashes; percent-encode each segment so
  // spaces and other reserved characters survive URL parsing. Don't
  // encode the path separator itself.
  const forward = diskPath.replace(/\\/g, "/");
  const segments = forward
    .split("/")
    .filter((s) => s.length > 0)
    .map((seg) => encodeURIComponent(seg));
  return `${ASSET_SCHEME}://${ASSET_HOST}/${segments.join("/")}`;
}

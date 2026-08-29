import fs from "node:fs";
import path from "node:path";

/**
 * Path confinement for the ``lucidium-asset://`` scheme.
 *
 * The renderer turns a backend-supplied disk path into
 * ``lucidium-asset://local/C%3A/Users/.../portrait.png`` (see
 * ``src/app/assetUrl.ts``). Anything a *page* can construct reaches this
 * handler too, so the mapping back to a filesystem path has to be
 * treated as untrusted input:
 *
 *   * Chromium normalises ``..`` segments in a standard-scheme URL, but
 *     ``%2e%2e`` survives that pass and only becomes ``..`` when we
 *     percent-decode each segment. The traversal check must therefore
 *     run on the *resolved* path, never on the raw pathname.
 *   * Absolute paths are handed to us verbatim (``C:/Windows/win.ini``
 *     is a perfectly well-formed request), so a prefix check against the
 *     app's own asset roots is the only thing standing between a page
 *     and the rest of the disk.
 *   * Even inside a root, only image/audio media is ever legitimately
 *     served — an extension allowlist keeps ``settings.json`` and
 *     ``game.json`` out of the renderer's origin.
 */

const IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif", ".bmp"];
const AUDIO_EXTENSIONS = [".wav", ".mp3", ".ogg", ".oga", ".opus", ".flac", ".m4a", ".aac"];

export const ALLOWED_ASSET_EXTENSIONS: ReadonlySet<string> = new Set([
  ...IMAGE_EXTENSIONS,
  ...AUDIO_EXTENSIONS,
]);

/** Windows and macOS mount case-insensitive filesystems by default, so a
 * case-flipped root would otherwise slip past a plain prefix compare. */
const CASE_INSENSITIVE_FS = process.platform === "win32" || process.platform === "darwin";

function normaliseForCompare(p: string): string {
  return CASE_INSENSITIVE_FS ? p.toLowerCase() : p;
}

/**
 * Decode a ``lucidium-asset://`` URL back into the disk path it names.
 * Returns ``null`` when the URL isn't a shape we produce (bad
 * percent-encoding, embedded NUL, empty path).
 */
export function decodeAssetUrl(requestUrl: string): string | null {
  let url: URL;
  try {
    url = new URL(requestUrl);
  } catch {
    return null;
  }
  const pathname = url.pathname.startsWith("/") ? url.pathname.slice(1) : url.pathname;
  if (!pathname) return null;
  let segments: string[];
  try {
    segments = pathname.split("/").map((seg) => decodeURIComponent(seg));
  } catch {
    // Malformed percent-escape ("%zz"); nothing legitimate looks like this.
    return null;
  }
  let diskPath = segments.join("/");
  if (diskPath.includes("\0")) return null;
  // POSIX absolute paths arrive as ``home/u/foo.png`` (the leading slash
  // was stripped above); restore it. Windows paths look like ``C:/...``
  // and don't need one.
  if (!/^[A-Za-z]:/.test(diskPath)) diskPath = `/${diskPath}`;
  return diskPath;
}

/**
 * Resolve ``diskPath`` and require that it lands inside one of ``roots``
 * with an allowlisted media extension. Returns the resolved path, or
 * ``null`` if the request must be refused.
 *
 * Symlinks are resolved when the target exists, so a link planted inside
 * a save folder can't point out of it.
 */
export function confineAssetPath(diskPath: string, roots: readonly string[]): string | null {
  if (!path.isAbsolute(diskPath)) return null;
  let resolved = path.resolve(diskPath);
  try {
    resolved = fs.realpathSync(resolved);
  } catch {
    // Doesn't exist (or isn't traversable) — keep the lexical resolution
    // and let the confinement check below decide; net.fetch will 404.
  }
  if (!ALLOWED_ASSET_EXTENSIONS.has(path.extname(resolved).toLowerCase())) return null;
  const candidate = normaliseForCompare(resolved);
  for (const root of roots) {
    let resolvedRoot = path.resolve(root);
    try {
      resolvedRoot = fs.realpathSync(resolvedRoot);
    } catch {
      /* root may not exist yet (no saves written); prefix check still holds */
    }
    const prefix = normaliseForCompare(
      resolvedRoot.endsWith(path.sep) ? resolvedRoot : resolvedRoot + path.sep,
    );
    if (candidate.startsWith(prefix)) return resolved;
  }
  return null;
}

/**
 * Full request -> disk path decision. ``null`` means "serve 403".
 */
export function resolveAssetRequest(requestUrl: string, roots: readonly string[]): string | null {
  const diskPath = decodeAssetUrl(requestUrl);
  if (diskPath === null) return null;
  return confineAssetPath(diskPath, roots);
}

/** The 403 every refused request gets — deliberately body-less and
 * uncacheable so a probing page learns nothing about the filesystem. */
export function forbiddenResponse(): Response {
  return new Response("Forbidden", {
    status: 403,
    headers: {
      "content-type": "text/plain",
      "cache-control": "no-store",
      // The scheme is CORS-enabled (see registerSchemesAsPrivileged in
      // main.ts); without this header Chromium turns the 403 into an
      // opaque network error, so the renderer cannot tell "refused" from
      // "the protocol handler crashed".
      "access-control-allow-origin": "*",
    },
  });
}
